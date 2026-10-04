"""Knowledge-base catalog tool for listing a user's ingested documents."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pkb_agent.agent.verification import extract_evidence
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

        rows = rows or []
        budget = int(getattr(ctx.settings, "agent_tool_result_max_chars", 6000))
        items: list[dict[str, Any]] = []
        consumed = 0
        for row in rows:
            item = _serialize_document(row)
            if item is None:
                consumed += 1
                continue
            next_offset = _next_offset(offset, consumed + 1, len(rows), limit)
            candidate = _page([*items, item], offset, next_offset)
            if _observation_size(candidate, ctx) > budget:
                if items:
                    break
                # Even one unusually long title/URI can exceed the observation
                # budget. Preserve its document ID, mark shortened metadata,
                # and advance the cursor instead of repeating this row forever.
                item = _fit_single_item(item, offset, next_offset, budget, ctx)
                if item is None:
                    return ToolResult.failure(
                        "catalog observation budget is too small for one document ID; "
                        "increase agent_tool_result_max_chars before retrying"
                    )
            items.append(item)
            consumed += 1
        page = _page(items, offset, _next_offset(offset, consumed, len(rows), limit))
        return ToolResult.success(page, truncated=page["truncated"])


def _next_offset(offset: int, consumed: int, fetched: int, limit: int) -> int | None:
    # Continue from rows actually represented on this page, not all rows fetched.
    # A full storage page needs one more query to establish end-of-catalog.
    return offset + consumed if consumed < fetched or fetched == limit else None


def _page(items: list[dict[str, Any]], offset: int, next_offset: int | None) -> dict[str, Any]:
    return {
        "count": len(items),
        "offset": offset,
        "next_offset": next_offset,
        "items": items,
        "truncated": any(item.get("truncated", False) for item in items),
        "note": "no documents found for this user" if not items and offset == 0 else "document catalog page",
    }


def _observation_size(page: dict[str, Any], ctx: ToolContext) -> int:
    # Account for the exact citation descriptor prepended by the runtime too;
    # budgeting the items alone still leaves a truncated, invalid JSON result.
    evidence = extract_evidence("rag_list_documents", page, settings=ctx.settings)
    payload = {"citation_evidence": [item.to_prompt_dict() for item in evidence], **page}
    return len(json.dumps(payload, ensure_ascii=False, default=str, indent=2))


def _fit_single_item(
    item: dict[str, Any], offset: int, next_offset: int | None, budget: int, ctx: ToolContext
) -> dict[str, Any] | None:
    shortened = {**item, "truncated": True}
    while _observation_size(_page([shortened], offset, next_offset), ctx) > budget:
        fields = [
            key for key, value in shortened.items()
            if key != "document_id" and isinstance(value, str)
        ]
        if not fields:
            return None
        key = max(fields, key=lambda name: len(shortened[name]))
        value = shortened[key]
        if len(value) <= 4:
            del shortened[key]
        else:
            shortened[key] = value[:len(value) // 2] + "…"
    return shortened


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
