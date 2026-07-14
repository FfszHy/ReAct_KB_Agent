"""Citation building and formatting for retrieved hits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Citation:
    index: int
    kind: str
    source_id: str
    title: str | None
    locator: str | None
    snippet: str | None = None


def _get(hit: Any, name: str, default: Any = None) -> Any:
    if isinstance(hit, dict):
        return hit.get(name, default)
    return getattr(hit, name, default)


def build_citations(hits: list[Any], *, kind: str = "kb") -> list[Citation]:
    citations: list[Citation] = []
    for i, hit in enumerate(hits or [], start=1):
        source_id = (
            _get(hit, "chunk_id")
            or _get(hit, "document_id")
            or _get(hit, "source_id")
            or ""
        )
        title = _get(hit, "doc_title") or _get(hit, "title")
        locator = (
            _get(hit, "source_uri")
            or _get(hit, "locator")
            or _get(hit, "document_id")
        )
        snippet = _get(hit, "content") or _get(hit, "snippet")
        citations.append(
            Citation(
                index=i,
                kind=kind,
                source_id=str(source_id),
                title=title,
                locator=locator,
                snippet=snippet,
            )
        )
    return citations


def format_inline(citations: list[Citation]) -> str:
    lines: list[str] = []
    for c in citations:
        title = c.title or "(untitled)"
        lines.append(f"[{c.index}] {title} ({c.source_id})")
    return "\n".join(lines)


def citation_map(citations: list[Citation]) -> dict[int, Citation]:
    return {c.index: c for c in citations}
