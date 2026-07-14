"""Reciprocal Rank Fusion for hybrid (vector + FTS) retrieval."""

from __future__ import annotations

from typing import Any


def rrf_fuse(
    vector_hits: list[dict],
    fts_hits: list[dict],
    k: int = 60,
    vector_weight: float = 0.6,
    fts_weight: float = 0.4,
) -> list[dict]:
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}
    vector_score_map: dict[str, float] = {}
    fts_score_map: dict[str, float] = {}

    for rank, hit in enumerate(vector_hits or [], start=1):
        cid = hit.get("chunk_id")
        if cid is None:
            continue
        cid = str(cid)
        scores[cid] = scores.get(cid, 0.0) + vector_weight / (k + rank)
        if cid not in merged:
            merged[cid] = dict(hit)
        vector_score_map[cid] = float(
            hit.get("vector_score", hit.get("score", 0.0)) or 0.0
        )

    for rank, hit in enumerate(fts_hits or [], start=1):
        cid = hit.get("chunk_id")
        if cid is None:
            continue
        cid = str(cid)
        scores[cid] = scores.get(cid, 0.0) + fts_weight / (k + rank)
        if cid not in merged:
            merged[cid] = dict(hit)
        else:
            for key, val in hit.items():
                if key not in merged[cid]:
                    merged[cid][key] = val
        fts_score_map[cid] = float(
            hit.get("fts_score", hit.get("score", 0.0)) or 0.0
        )

    result: list[dict[str, Any]] = []
    for cid, score in scores.items():
        item = merged[cid]
        item["score"] = score
        item["vector_score"] = vector_score_map.get(
            cid, float(item.get("vector_score", 0.0) or 0.0)
        )
        item["fts_score"] = fts_score_map.get(
            cid, float(item.get("fts_score", 0.0) or 0.0)
        )
        result.append(item)

    result.sort(key=lambda d: d["score"], reverse=True)
    return result


def dedupe_hits(hits: list[dict], key: str = "chunk_id") -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for hit in hits or []:
        k_val = hit.get(key)
        if k_val is None:
            out.append(hit)
            continue
        s = str(k_val)
        if s in seen:
            continue
        seen.add(s)
        out.append(hit)
    return out
