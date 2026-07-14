"""Web fetch tool: download and sanitize a web page."""

from __future__ import annotations

from typing import Any

from pkb_agent.agent.errors import UnsafeUrlError
from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = (
        "Download a web page and return its main text content (HTML stripped). "
        "Use for reading a URL found via web_search. Only http/https URLs are "
        "allowed; private/internal hosts are blocked."
    )
    params = [
        ToolParam("url", "string", "The http(s) URL to fetch."),
        ToolParam(
            "max_chars",
            "integer",
            "Max characters of text to return.",
            required=False,
            default=8000,
        ),
    ]
    permission = "ask"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        url = str(args.get("url", "")).strip()
        if not url:
            return ToolResult.failure("url must not be empty")
        fetcher = ctx.service("fetcher")
        try:
            page = await fetcher.fetch(url)
        except UnsafeUrlError as e:
            return ToolResult.failure(f"unsafe url: {e.reason}")
        except Exception as e:
            return ToolResult.failure(f"web_fetch failed: {e}")

        if page.error:
            return ToolResult.failure(
                f"fetch failed (status={page.status_code}): {page.error}"
            )

        text = page.text
        truncated = page.truncated
        max_chars = args.get("max_chars")
        if max_chars is not None:
            try:
                limit = int(max_chars)
                if len(text) > limit:
                    text = text[:limit].rstrip() + "…"
                    truncated = True
            except (TypeError, ValueError):
                pass
        return ToolResult.success(
            {
                "url": page.url,
                "final_url": page.final_url,
                "title": page.title,
                "status_code": page.status_code,
                "content_type": page.content_type,
                "truncated": truncated,
                "text": text,
            },
            truncated=truncated,
        )
