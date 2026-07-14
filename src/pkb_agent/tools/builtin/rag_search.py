"""RAG hybrid search tool."""

from __future__ import annotations

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
        ToolParam("top_k", "integer", "Max number of chunks to return.", required=False, default=6),
    ]
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult.failure("query must not be empty")
        top_k = int(args.get("top_k") or ctx.settings.rag_top_k)
        try:
            retriever = ctx.service("retriever")
            hits = await retriever.search(query, top_k=top_k, user_id=ctx.user_id)
        except Exception as e:
            return ToolResult.failure(f"rag_search failed: {e}")

        if not hits:
            return ToolResult.success(
                {"query": query, "count": 0, "results": [], "note": "no matching chunks found"}
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
        return ToolResult.success({"query": query, "count": len(results), "results": results})
