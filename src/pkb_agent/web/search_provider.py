"""Web search providers.

Abstract :class:`SearchProvider` plus concrete implementations for Tavily,
Serper (Google), and Bing. Each provider lazily creates its own
``httpx.AsyncClient`` and supports async context-manager usage and ``close()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from pkb_agent.agent.errors import ConfigError, WebError
from pkb_agent.app.settings import Settings

DEFAULT_TIMEOUT: float = 20.0


@dataclass
class SearchResult:
    """A single search hit."""

    title: str
    url: str
    snippet: str
    source: str  # provider name


class SearchProvider(ABC):
    """Abstract web search provider."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    async def __aenter__(self) -> SearchProvider:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


class TavilyProvider(SearchProvider):
    """Tavily search API (POST https://api.tavily.com/search)."""

    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        if not api_key:
            raise ConfigError("tavily api_key is required")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self._client

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        body = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        client = self._ensure_client()
        try:
            resp = await client.post(self.URL, json=body)
        except httpx.HTTPError as exc:
            raise WebError(f"tavily request failed: {exc}") from exc

        if resp.status_code != 200:
            raise WebError(f"tavily returned {resp.status_code}: {resp.text[:300]}")

        data = _safe_json(resp)
        results: list[SearchResult] = []
        for item in data.get("results", []) or []:
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                    source="tavily",
                )
            )
        return results

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class SerperProvider(SearchProvider):
    """Serper.dev (Google) search API (POST https://google.serper.dev/search)."""

    URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        if not api_key:
            raise ConfigError("serper api_key is required")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
            )
        return self._client

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        body = {"q": query, "num": max_results}
        client = self._ensure_client()
        try:
            resp = await client.post(self.URL, json=body)
        except httpx.HTTPError as exc:
            raise WebError(f"serper request failed: {exc}") from exc

        if resp.status_code != 200:
            raise WebError(f"serper returned {resp.status_code}: {resp.text[:300]}")

        data = _safe_json(resp)
        results: list[SearchResult] = []
        for item in data.get("organic", []) or []:
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("link", "")),
                    snippet=str(item.get("snippet", "")),
                    source="serper",
                )
            )
        return results

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class BingProvider(SearchProvider):
    """Bing Web Search API v7 (GET https://api.bing.microsoft.com/v7.0/search)."""

    URL = "https://api.bing.microsoft.com/v7.0/search"

    def __init__(self, api_key: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        if not api_key:
            raise ConfigError("bing api_key is required")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={"Ocp-Apim-Subscription-Key": self._api_key},
            )
        return self._client

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        params: dict[str, str | int] = {"q": query, "count": max_results}
        client = self._ensure_client()
        try:
            resp = await client.get(self.URL, params=params)
        except httpx.HTTPError as exc:
            raise WebError(f"bing request failed: {exc}") from exc

        if resp.status_code != 200:
            raise WebError(f"bing returned {resp.status_code}: {resp.text[:300]}")

        data = _safe_json(resp)
        web_pages = data.get("webPages", {}) or {}
        results: list[SearchResult] = []
        for item in web_pages.get("value", []) or []:
            results.append(
                SearchResult(
                    title=str(item.get("name", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("snippet", "")),
                    source="bing",
                )
            )
        return results

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        payload = resp.json()
    except ValueError as exc:
        raise WebError(f"provider returned non-JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WebError("provider returned non-object JSON")
    return payload


def make_search_provider(settings: Settings) -> SearchProvider:
    """Select a search provider by ``settings.web_search_provider``."""
    name = (settings.web_search_provider or "").strip().lower()
    api_key = settings.web_search_api_key or ""

    if name == "tavily":
        return TavilyProvider(api_key)
    if name == "serper":
        return SerperProvider(api_key)
    if name == "bing":
        return BingProvider(api_key)

    if not name:
        raise ConfigError("web_search_provider is not configured")
    raise ConfigError(f"unknown web_search_provider: {name!r}")
