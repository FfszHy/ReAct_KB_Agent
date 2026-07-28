"""DeepSeek Chat Completions client with native tool calling.

Uses ``httpx.AsyncClient`` against the DeepSeek ``/chat/completions`` endpoint.
DeepSeek follows the OpenAI-compatible protocol and supports a ``tools`` array
with JSON-schema function definitions plus ``tool_choice``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pkb_agent.agent.errors import (
    ConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)
from pkb_agent.app.settings import Settings
from pkb_agent.llm.schemas import (
    ChatCompletion,
    Message,
    build_chat_request,
    parse_chat_completion,
)


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ConfigError("deepseek_api_key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> DeepSeekClient:
        settings.require_llm()
        return cls(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )

    # ------------------------------------------------------------------
    @property
    def model(self) -> str:
        return self._model

    async def __aenter__(self) -> DeepSeekClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Lazy client for non-context-manager usage.
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatCompletion:
        body = build_chat_request(
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=self._model_temperature(temperature),
            max_tokens=max_tokens,
            response_format=response_format,
            tool_choice=tool_choice,
            stream=False,
        )
        client = self._ensure_client()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError, LLMRateLimitError)),
            reraise=True,
        ):
            with attempt:
                return await self._do_request(client, body)
        # Unreachable, but satisfies type checkers.
        raise LLMError("exhausted retries")

    def _model_temperature(self, requested: float | None) -> float:
        # DeepSeek recommends temperature in [0,1]; clamp.
        if requested is None:
            requested = 0.2
        return max(0.0, min(1.0, float(requested)))

    async def _do_request(self, client: httpx.AsyncClient, body: dict[str, Any]) -> ChatCompletion:
        try:
            resp = await client.post("/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"deepseek request failed: {e}") from e

        if resp.status_code == 429 or resp.status_code >= 500:
            raise LLMRateLimitError(f"deepseek returned {resp.status_code}: {resp.text[:300]}")

        if resp.status_code != 200:
            raise LLMError(f"deepseek returned {resp.status_code}: {resp.text[:500]}")

        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"deepseek returned non-JSON: {e}") from e

        try:
            return parse_chat_completion(payload)
        except Exception as e:
            raise LLMResponseError(f"failed to parse deepseek response: {e}") from e
