"""OpenAI-compatible embedding provider (async, batched, retried)."""

from __future__ import annotations

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from pkb_agent.agent.errors import EmbeddingError
from pkb_agent.app.settings import Settings


def _is_retryable_embedding_error(error: BaseException) -> bool:
    """Retry transient transport/rate-limit/server failures, not invalid requests."""
    if isinstance(error, httpx.RequestError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and (
        error.response.status_code == 429 or error.response.status_code >= 500
    )


def _provider_error_detail(response: httpx.Response) -> str:
    """Extract a short provider error without echoing request input or credentials."""
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    value = payload.get("error")
    if isinstance(value, dict):
        value = value.get("message") or value.get("code")
    if not isinstance(value, str):
        value = payload.get("message") or payload.get("code")
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:500]


class EmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        batch_size: int = 64,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingProvider:
        return cls(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_api_base_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def __aenter__(self) -> EmbeddingProvider:
        self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        result: list[list[float]] = []
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception(_is_retryable_embedding_error),
                reraise=True,
            ):
                with attempt:
                    resp = await client.post(
                        f"{self._base_url}/embeddings",
                        json={
                            "model": self._model,
                            "input": texts,
                            "dimensions": self._dimensions,
                            "encoding_format": "float",
                        },
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    data = payload.get("data") or []
                    embeds = [
                        item["embedding"]
                        for item in sorted(data, key=lambda x: x.get("index", 0))
                    ]
                    if len(embeds) != len(texts):
                        raise EmbeddingError(
                            f"embedding count mismatch: got {len(embeds)}, "
                            f"expected {len(texts)}"
                        )
                    for vec in embeds:
                        if len(vec) != self._dimensions:
                            raise EmbeddingError(
                                f"embedding dimension mismatch: got {len(vec)}, "
                                f"expected {self._dimensions}"
                            )
                    result = embeds
        except httpx.HTTPStatusError as e:
            detail = _provider_error_detail(e.response)
            suffix = f": {detail}" if detail else ""
            raise EmbeddingError(
                "embedding API rejected request "
                f"(HTTP {e.response.status_code}, model={self._model!r}, "
                f"batch_items={len(texts)}, dimensions={self._dimensions}){suffix}"
            ) from e
        except httpx.HTTPError as e:
            raise EmbeddingError(f"embedding request failed: {e}") from e
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"embedding failed: {e}") from e
        if not result:
            raise EmbeddingError("embedding returned no result")
        return result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            out.extend(await self._embed_batch(batch))
        return out

    async def embed_one(self, text: str) -> list[float]:
        res = await self.embed([text])
        return res[0]
