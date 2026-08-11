"""DashScope native embedding provider (async, batched, retried)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import dashscope
import requests
from dashscope.common.error import ServiceUnavailableError, TimeoutException
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from pkb_agent.agent.errors import EmbeddingError
from pkb_agent.app.settings import Settings


class _RetryableDashScopeResponse(Exception):
    """Wrap a retryable non-200 DashScope response for tenacity."""

    def __init__(self, response: Any) -> None:
        self.response = response
        super().__init__("retryable DashScope embedding response")


def _is_retryable_embedding_error(error: BaseException) -> bool:
    """Retry transient transport/rate-limit/server failures, not bad input."""
    return isinstance(
        error,
        (
            _RetryableDashScopeResponse,
            requests.RequestException,
            ServiceUnavailableError,
            TimeoutException,
            TimeoutError,
            ConnectionError,
        ),
    )


def _response_value(response: Any, name: str) -> Any:
    if isinstance(response, Mapping):
        return response.get(name)
    return getattr(response, name, None)


def _response_status(response: Any) -> int | None:
    value = _response_value(response, "status_code")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _provider_error_detail(response: Any) -> str:
    """Extract a short DashScope error without echoing text or credentials."""
    values = [_response_value(response, "code"), _response_value(response, "message")]
    detail = ": ".join(str(value) for value in values if value)
    return " ".join(detail.split())[:500]


class EmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        batch_size: int = 64,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._timeout = max(1, timeout)

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingProvider:
        return cls(
            api_key=settings.embedding_api_key,
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

    async def __aenter__(self) -> EmbeddingProvider:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Keep the provider lifecycle compatible with existing callers."""

    def _response_error(self, response: Any, batch_size: int) -> EmbeddingError:
        status_code = _response_status(response)
        status = str(status_code) if status_code is not None else "unknown"
        detail = _provider_error_detail(response)
        suffix = f": {detail}" if detail else ""
        return EmbeddingError(
            "DashScope embedding API rejected request "
            f"(HTTP {status}, model={self._model!r}, batch_items={batch_size}, "
            f"dimensions={self._dimensions}){suffix}"
        )

    def _parse_embeddings(self, response: Any, texts: list[str]) -> list[list[float]]:
        output = _response_value(response, "output")
        embeddings = _response_value(output, "embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError("DashScope embedding response contained no embeddings")
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
            )

        ordered: list[list[float] | None] = [None] * len(texts)
        for item in embeddings:
            if not isinstance(item, Mapping):
                raise EmbeddingError("DashScope embedding response contained an invalid item")
            text_index = item.get("text_index")
            vector = item.get("embedding")
            if (
                not isinstance(text_index, int)
                or text_index < 0
                or text_index >= len(texts)
                or ordered[text_index] is not None
            ):
                raise EmbeddingError("DashScope embedding response contained invalid text indexes")
            if not isinstance(vector, list):
                raise EmbeddingError("DashScope embedding response contained an invalid vector")
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"embedding dimension mismatch: got {len(vector)}, "
                    f"expected {self._dimensions}"
                )
            ordered[text_index] = vector

        if any(vector is None for vector in ordered):
            raise EmbeddingError("DashScope embedding response omitted an input text")
        return [vector for vector in ordered if vector is not None]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception(_is_retryable_embedding_error),
                reraise=True,
            ):
                with attempt:
                    resp = await asyncio.to_thread(
                        dashscope.TextEmbedding.call,
                        model=self._model,
                        input=texts,
                        api_key=self._api_key,
                        dimension=self._dimensions,
                        request_timeout=self._timeout,
                    )
                    status_code = _response_status(resp)
                    if status_code != HTTPStatus.OK:
                        if status_code == HTTPStatus.TOO_MANY_REQUESTS or (
                            status_code is not None and status_code >= 500
                        ):
                            raise _RetryableDashScopeResponse(resp)
                        raise self._response_error(resp, len(texts))
                    result = self._parse_embeddings(resp, texts)
        except _RetryableDashScopeResponse as e:
            raise self._response_error(e.response, len(texts)) from e
        except requests.RequestException as e:
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
