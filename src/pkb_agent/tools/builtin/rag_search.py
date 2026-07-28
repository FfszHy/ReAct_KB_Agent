"""RAG hybrid search tool."""

from __future__ import annotations

import asyncio
from typing import Any

from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult

_MAX_CONTENT_PREVIEW = 800


class RagSearchTool(BaseTool):
    name = "rag_search"
    description = (
        "Hybrid (vector + full-text) search over your personal knowledge base. "
        "Returns ranked chunks with content previews and source metadata. Use "
        "this to find relevant notes/documents before answering."
    )
    params = [
        ToolParam("query", "string", "The search query (keywords or natural language)."),
        ToolParam(
            "queries",
            "array",
            "Optional focused queries produced by the retrieval planner.",
            required=False,
            items={"type": "string"},
        ),
        ToolParam("top_k", "integer", "Max number of chunks to return.", required=False, default=6),
    ]
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        queries = _queries_from_args(args, max_queries=_max_queries(ctx))
        if not queries:
            return ToolResult.failure("query must not be empty")
        top_k = int(args.get("top_k") or ctx.settings.rag_top_k)
        try:
            retriever = ctx.service("retriever")
            batches = await asyncio.gather(
                *(retriever.search(query, top_k=top_k, user_id=ctx.user_id) for query in queries)
            )
        except Exception as e:
            return ToolResult.failure(f"rag_search failed: {e}")

        hits = _merge_hits(batches, top_k=top_k)

        if not hits:
            return ToolResult.success(
                {
                    "query": queries[0],
                    "queries": queries,
                    "count": 0,
                    "results": [],
                    "note": "no matching chunks found",
                }
            )

        results = []
        for i, h in enumerate(hits):
            preview = h.content
            if len(preview) > _MAX_CONTENT_PREVIEW:
                preview = preview[:_MAX_CONTENT_PREVIEW] + "…"
            results.append(
                {
                    "rank": i + 1,
                    "chunk_id": h.chunk_id,
                    "document_id": h.document_id,
                    "title": h.doc_title,
                    "source_uri": h.source_uri,
                    "chunk_index": h.chunk_index,
                    "score": round(h.score, 4),
                    "vector_score": round(h.vector_score, 4),
                    "fts_score": round(h.fts_score, 4),
                    "content_preview": preview,
                }
            )
        return ToolResult.success(
            {"query": queries[0], "queries": queries, "count": len(results), "results": results}
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


def _merge_hits(batches: list[Any], *, top_k: int) -> list[Any]:
    """Keep the highest-scoring occurrence of each chunk across rewritten queries."""
    best_by_chunk: dict[str, Any] = {}
    for batch in batches:
        for hit in batch:
            previous = best_by_chunk.get(hit.chunk_id)
            if previous is None or hit.score > previous.score:
                best_by_chunk[hit.chunk_id] = hit
    return sorted(best_by_chunk.values(), key=lambda hit: hit.score, reverse=True)[:top_k]
