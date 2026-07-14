"""HTTP page fetcher with SSRF safety and content sanitization.

:class:`Fetcher` performs async GET requests (following redirects), enforces
URL safety via :func:`pkb_agent.security.url_safety.assert_safe_url`, and
converts HTML/JSON/text responses into a text payload suitable for the agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from pkb_agent.app.settings import Settings
from pkb_agent.security.url_safety import assert_safe_url
from pkb_agent.web.sanitizer import sanitize_html, truncate_text

_ACCEPTABLE_CONTENT_TYPES: frozenset[str] = frozenset(
    {"text/html", "application/json", "text/plain"}
)


@dataclass
class FetchedPage:
    """Outcome of a fetch operation."""

    url: str
    final_url: str
    title: str
    text: str
    status_code: int
    content_type: str
    truncated: bool
    error: str | None = None


class Fetcher:
    """Async HTTP fetcher with content-aware extraction."""

    def __init__(
        self,
        *,
        timeout: int = 20,
        max_chars: int = 8000,
        user_agent: str = "pkb-agent/0.1",
    ) -> None:
        self._timeout = timeout
        self._max_chars = max_chars
        self._user_agent = user_agent
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> Fetcher:
        return cls(
            timeout=settings.web_fetch_timeout,
            max_chars=settings.web_fetch_max_chars,
        )

    # ------------------------------------------------------------------
    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            )
        return self._client

    async def __aenter__(self) -> Fetcher:
        self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    async def fetch(self, url: str) -> FetchedPage:
        # Unsafe URLs raise UnsafeUrlError (caller handles).
        assert_safe_url(url)

        client = self._ensure_client()
        try:
            resp = await client.get(url)
        except httpx.HTTPError as exc:
            return FetchedPage(
                url=url,
                final_url=url,
                title="",
                text="",
                status_code=0,
                content_type="",
                truncated=False,
                error=f"request failed: {exc}",
            )

        content_type = _extract_content_type(resp.headers.get("content-type", ""))
        final_url = str(resp.url)

        if content_type == "text/html":
            text, title = sanitize_html(resp.text)
        elif content_type == "application/json":
            text = _pretty_json(resp)
            title = ""
        elif content_type == "text/plain":
            text = resp.text
            title = ""
        else:
            return FetchedPage(
                url=url,
                final_url=final_url,
                title="",
                text="",
                status_code=resp.status_code,
                content_type=content_type,
                truncated=False,
                error=f"unsupported content type: {content_type!r}",
            )

        text, truncated = truncate_text(text, self._max_chars)
        return FetchedPage(
            url=url,
            final_url=final_url,
            title=title or "",
            text=text,
            status_code=resp.status_code,
            content_type=content_type,
            truncated=truncated,
        )


def _extract_content_type(header: str) -> str:
    """Return the normalized media type (without parameters)."""
    if not header:
        return ""
    return header.split(";", 1)[0].strip().lower()


def _pretty_json(resp: httpx.Response) -> str:
    """Render a JSON response as pretty-printed text, falling back to raw body."""
    try:
        payload: Any = resp.json()
    except ValueError:
        return resp.text
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return resp.text
