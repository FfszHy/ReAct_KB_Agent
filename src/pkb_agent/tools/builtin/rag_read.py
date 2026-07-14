"""RAG read tool: fetch full chunk/document text by id."""

from __future__ import annotations

from typing import Any

from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult


class RagReadTool(BaseTool):
    name = "rag_read"
    description = (
        "Read the full text of a knowledge-base chunk (by chunk_id) or all "
        "chunks of a document (by document_id). Use after rag_search to get "
        "complete content."
    )
    params = [
        ToolParam("chunk_id", "string", "Chunk id to read.", required=False),
        ToolParam("document_id", "string", "Document id whose chunks to read.", required=False),
    ]
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        chunk_id = args.get("chunk_id")
        document_id = args.get("document_id")
        if not chunk_id and not document_id:
            return ToolResult.failure("provide either chunk_id or document_id")

        chunks_repo = ctx.repo("chunks")
        try:
            if chunk_id:
                row = await _to_thread(chunks_repo.get_chunk_with_doc, str(chunk_id))
                if not row:
                    return ToolResult.failure(f"chunk not found: {chunk_id}")
                return ToolResult.success(
                    {
                        "kind": "chunk",
                        "chunk_id": row.get("id"),
                        "document_id": row.get("document_id"),
                        "chunk_index": row.get("chunk_index"),
                        "content": row.get("content"),
                        "doc_title": (row.get("document") or {}).get("title"),
                        "source_uri": (row.get("document") or {}).get("source_uri"),
                        "meta": row.get("meta"),
                    }
                )
            rows = await _to_thread(chunks_repo.list_by_document, str(document_id))
            if not rows:
                return ToolResult.failure(f"no chunks for document: {document_id}")
            return ToolResult.success(
                {
                    "kind": "document",
                    "document_id": str(document_id),
                    "chunk_count": len(rows),
                    "chunks": [
                        {
                            "chunk_index": r.get("chunk_index"),
                            "chunk_id": r.get("id"),
                            "content": r.get("content"),
                            "token_count": r.get("token_count"),
                        }
                        for r in rows
                    ],
                }
            )
        except Exception as e:
            return ToolResult.failure(f"rag_read failed: {e}")


async def _to_thread(func, *args, **kwargs):
    import asyncio

    if kwargs:
        return await asyncio.to_thread(lambda: func(*args, **kwargs))
    return await asyncio.to_thread(func, *args)
