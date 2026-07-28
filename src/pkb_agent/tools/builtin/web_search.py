"""Web search tool."""

from __future__ import annotations

import asyncio
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
            "queries",
            "array",
            "Optional focused queries produced by the retrieval planner.",
            required=False,
            items={"type": "string"},
        ),
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
        queries = _queries_from_args(args, max_queries=_max_queries(ctx))
        if not queries:
            return ToolResult.failure("query must not be empty")
        max_results = int(args.get("max_results") or ctx.settings.web_max_results)
        try:
            provider = ctx.service("search_provider")
            batches = await asyncio.gather(
                *(provider.search(query, max_results=max_results) for query in queries)
            )
        except Exception as e:
            return ToolResult.failure(f"web_search failed: {e}")

        results = _merge_results(batches, max_results=max_results)

        return ToolResult.success(
            {
                "query": queries[0],
                "queries": queries,
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


def _queries_from_args(args: dict[str, Any], *, max_queries: int) -> list[str]:
    raw_queries = args.get("queries")
    values = raw_queries if isinstance(raw_queries, (list, tuple)) and raw_queries else [args.get("query")]
    queries: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = str(value or "").strip()
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def _max_queries(ctx: ToolContext) -> int:
    configured = getattr(ctx.settings, "prompts_query_rewrite_max_queries", 3)
    try:
        return min(max(int(configured), 1), 10)
    except (TypeError, ValueError):
        return 3


def _merge_results(batches: list[Any], *, max_results: int) -> list[Any]:
    """Deduplicate URLs while preserving the planner's query priority order."""
    results: list[Any] = []
    seen_urls: set[str] = set()
    for batch in batches:
        for result in batch:
            url = str(getattr(result, "url", ""))
            key = url.casefold()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            results.append(result)
            if len(results) >= max_results:
                return results
    return results
