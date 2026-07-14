"""Text chunking: natural-boundary splitting with token-aware sizing and overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ENCODING: Any = None
_ENCODING_LOADED = False


def _get_encoding() -> Any:
    global _ENCODING, _ENCODING_LOADED
    if not _ENCODING_LOADED:
        _ENCODING_LOADED = True
        try:
            import tiktoken

            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODING = None
    return _ENCODING


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_encoding()
    if enc is None:
        return len(text) // 4
    return len(enc.encode(text))


@dataclass
class Chunk:
    content: str
    chunk_index: int
    token_count: int
    char_start: int
    char_end: int


_BOUNDARY_RE = re.compile(r"\n\s*\n|(?<=[.!?。！？])\s+")  # noqa: RUF001


def _unit_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of natural-boundary units covering ``text``."""
    bounds = {0}
    for m in _BOUNDARY_RE.finditer(text):
        bounds.add(m.end())
    ordered = sorted(bounds)
    spans: list[tuple[int, int]] = []
    for i, start in enumerate(ordered):
        end = ordered[i + 1] if i + 1 < len(ordered) else len(text)
        if start < end and text[start:end].strip():
            spans.append((start, end))
    return spans


def _overlap_start(
    units: list[tuple[int, int, int]], i: int, j: int, overlap: int
) -> int:
    """Return the start index of the next chunk, carrying a trailing overlap window."""
    if overlap <= 0:
        return j
    total = 0
    idx = j - 1
    while idx > i and total + units[idx][2] <= overlap:
        total += units[idx][2]
        idx -= 1
    next_i = idx + 1
    if next_i <= i:
        next_i = i + 1
    return next_i


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        return []
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    stripped = text.strip()
    spans = _unit_spans(stripped)
    if not spans:
        return []
    units = [(s, e, count_tokens(stripped[s:e])) for s, e in spans]
    n = len(units)
    chunks: list[Chunk] = []
    i = 0
    while i < n:
        cur_start = units[i][0]
        cur_tokens = units[i][2]
        last_end = units[i][1]
        j = i + 1
        while j < n and cur_tokens + units[j][2] <= chunk_size:
            cur_tokens += units[j][2]
            last_end = units[j][1]
            j += 1
        chunks.append(
            Chunk(
                content=stripped[cur_start:last_end],
                chunk_index=len(chunks),
                token_count=cur_tokens,
                char_start=cur_start,
                char_end=last_end,
            )
        )
        if j >= n:
            break
        next_i = _overlap_start(units, i, j, overlap)
        if next_i <= i:
            next_i = i + 1
        i = next_i
    return chunks


def split_into_chunks(
    text: str, chunk_size: int, chunk_overlap: int, base_index: int = 0
) -> list[Chunk]:
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if base_index:
        chunks = [
            Chunk(
                content=c.content,
                chunk_index=c.chunk_index + base_index,
                token_count=c.token_count,
                char_start=c.char_start,
                char_end=c.char_end,
            )
            for c in chunks
        ]
    return chunks
