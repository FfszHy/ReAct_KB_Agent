"""Knowledge-base catalog tool for listing a user's ingested documents."""

from __future__ import annotations

import asyncio
from typing import Any

from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult

_DEFAULT_LIMIT = 10
_MAX_LIMIT = 20


class RagListDocumentsTool(BaseTool):
    """List source-level metadata without relying on semantic chunk retrieval."""

    name = "rag_list_documents"
    description = (
        "List documents currently stored in the calling user's knowledge base. "
        "Use this for catalog questions such as which materials, files, or notes are in "
        "the knowledge base; it is not a semantic search. Results are paginated and "
        "include citation-eligible source metadata."
    )
    params = [
        ToolParam(
            "limit",
            "integer",
            "Maximum documents to return per page (1-20).",
            required=False,
            default=_DEFAULT_LIMIT,
        ),
        ToolParam(
            "offset",
            "integer",
            "Zero-based document offset for the next catalog page.",
            required=False,
            default=0,
        ),
    ]
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            limit = _page_value(args.get("limit"), default=_DEFAULT_LIMIT, minimum=1, maximum=_MAX_LIMIT)
            offset = _page_value(args.get("offset"), default=0, minimum=0)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        try:
            documents_repo = ctx.repo("documents")
            rows = await asyncio.to_thread(
                documents_repo.list,
                ctx.user_id or "default",
                limit,
                offset,
            )
        except Exception as exc:
            return ToolResult.failure(f"rag_list_documents failed: {exc}")

        items = [item for row in rows if (item := _serialize_document(row)) is not None]
        # A full page can still be the final page, so the caller follows the
        # cursor once more before concluding that the catalog is exhausted.
        next_offset = offset + len(rows) if len(rows) == limit else None
        return ToolResult.success(
            {
                "count": len(items),
                "offset": offset,
                "next_offset": next_offset,
                "items": items,
                "note": (
                    "no documents found for this user"
                    if not items and offset == 0
                    else "document catalog page"
                ),
            }
        )


def _page_value(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("pagination values must be integers")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("pagination values must be integers") from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        upper = f" and {maximum}" if maximum is not None else ""
        raise ValueError(f"pagination value must be between {minimum}{upper}")
    return parsed


def _serialize_document(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    document_id = _text(row.get("id"))
    if document_id is None:
        return None

    item: dict[str, Any] = {"document_id": document_id}
    for key in ("title", "source_uri", "source_type", "created_at"):
        value = _text(row.get(key))
        if value is not None:
            item[key] = value
    for key in ("chunk_count", "char_count"):
        value = row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            item[key] = value
    return item


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
