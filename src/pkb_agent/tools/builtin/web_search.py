"""Web search tool."""

from __future__ import annotations

from typing import Any

from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the public web for fresh information. Returns titles, URLs and "
        "short snippets. Use web_fetch to retrieve full page content."
    )
    params = [
        ToolParam("query", "string", "The web search query."),
        ToolParam(
            "max_results",
            "integer",
            "Max number of results to return.",
            required=False,
            default=5,
        ),
    ]
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult.failure("query must not be empty")
        max_results = int(args.get("max_results") or ctx.settings.web_max_results)
        try:
            provider = ctx.service("search_provider")
            results = await provider.search(query, max_results=max_results)
        except Exception as e:
            return ToolResult.failure(f"web_search failed: {e}")

        return ToolResult.success(
            {
                "query": query,
                "provider": getattr(provider, "name", ctx.settings.web_search_provider),
                "count": len(results),
                "results": [
                    {
                        "rank": i + 1,
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                    }
                    for i, r in enumerate(results)
                ],
            }
        )
