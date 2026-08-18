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
        "complete content. Prefer the raw chunk_id returned by a tool result; "
        "a final citation id such as kb:<chunk_id> is accepted for recovery."
    )
    params = [
        ToolParam(
            "chunk_id",
            "string",
            "Raw chunk id from rag_search; kb:<chunk_id> is also accepted.",
            required=False,
        ),
        ToolParam("document_id", "string", "Document id whose chunks to read.", required=False),
    ]
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        chunk_id = _normalize_chunk_id(args.get("chunk_id"))
        document_id = _normalize_chunk_id(args.get("document_id"))
        if not chunk_id and not document_id:
            return ToolResult.failure("provide either chunk_id or document_id")

        chunks_repo = ctx.repo("chunks")
        try:
            if chunk_id:
                row = await _to_thread(chunks_repo.get_chunk_with_doc, str(chunk_id))
                if not row:
                    return ToolResult.failure(f"chunk not found: {chunk_id}")
                return ToolResult.success(_chunk_payload(row))
            rows = await _to_thread(chunks_repo.list_by_document, str(document_id))
            if not rows:
                # Models occasionally place a chunk ID into ``document_id``.
                # Probe the ID as a chunk only after the legitimate document
                # lookup misses, preserving ordinary document-read behavior.
                row = await _to_thread(chunks_repo.get_chunk_with_doc, str(document_id))
                if row:
                    return ToolResult.success(_chunk_payload(row))
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


def _normalize_chunk_id(value: Any) -> str | None:
    """Accept the runtime's ``kb:<chunk_id>`` citation form as read input.

    The prompt tells the model to use a raw ID for tools and reserve citation
    IDs for final JSON. This small compatibility layer avoids a needless
    database syntax error when a model nevertheless copies a displayed
    evidence ID (for example ``kb:2d0…``) into ``rag_read``.
    """
    if not isinstance(value, str):
        return None
    chunk_id = value.strip()
    if chunk_id.startswith("kb:"):
        chunk_id = chunk_id.removeprefix("kb:").strip()
    return chunk_id or None


def _chunk_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "chunk",
        "chunk_id": row.get("id"),
        "document_id": row.get("document_id"),
        "chunk_index": row.get("chunk_index"),
        "content": row.get("content"),
        "doc_title": (row.get("document") or {}).get("title"),
        "source_uri": (row.get("document") or {}).get("source_uri"),
        "meta": row.get("meta"),
    }


async def _to_thread(func, *args, **kwargs):
    import asyncio

    if kwargs:
        return await asyncio.to_thread(lambda: func(*args, **kwargs))
    return await asyncio.to_thread(func, *args)
