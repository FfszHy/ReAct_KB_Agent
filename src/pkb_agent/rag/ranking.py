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


def weighted_score_fuse(
    vector_hits: list[dict],
    fts_hits: list[dict],
    *,
    vector_weight: float = 0.6,
    fts_weight: float = 0.4,
) -> list[dict]:
    """Fuse normalized source scores as the non-RRF hybrid baseline.

    Database cosine similarity and ``ts_rank`` are not on a common numeric
    scale.  Normalizing each candidate list independently gives an explicit,
    reproducible "Vector + FTS" baseline to compare with rank-only RRF.
    """
    vector_scores = _normalized_score_map(vector_hits or [], "vector_score")
    fts_scores = _normalized_score_map(fts_hits or [], "fts_score")
    merged: dict[str, dict[str, Any]] = {}
    raw_vector: dict[str, float] = {}
    raw_fts: dict[str, float] = {}
    for hit in vector_hits or []:
        key = _chunk_key(hit)
        if key is None:
            continue
        merged.setdefault(key, dict(hit))
        raw_vector[key] = _score(hit, "vector_score")
    for hit in fts_hits or []:
        key = _chunk_key(hit)
        if key is None:
            continue
        if key not in merged:
            merged[key] = dict(hit)
        else:
            for field, value in hit.items():
                if field not in merged[key]:
                    merged[key][field] = value
        raw_fts[key] = _score(hit, "fts_score")

    result: list[dict[str, Any]] = []
    for key, item in merged.items():
        item["vector_score"] = raw_vector.get(key, float(item.get("vector_score", 0.0) or 0.0))
        item["fts_score"] = raw_fts.get(key, float(item.get("fts_score", 0.0) or 0.0))
        item["score"] = (
            vector_weight * vector_scores.get(key, 0.0)
            + fts_weight * fts_scores.get(key, 0.0)
        )
        result.append(item)
    return sorted(result, key=lambda item: item["score"], reverse=True)


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


def _normalized_score_map(hits: list[dict], field: str) -> dict[str, float]:
    raw = {key: _score(hit, field) for hit in hits if (key := _chunk_key(hit)) is not None}
    if not raw:
        return {}
    floor = min(raw.values())
    ceiling = max(raw.values())
    if ceiling == floor:
        return {key: 1.0 for key in raw}
    return {key: (value - floor) / (ceiling - floor) for key, value in raw.items()}


def _chunk_key(hit: dict) -> str | None:
    value = hit.get("chunk_id")
    return str(value) if value is not None else None


def _score(hit: dict, field: str) -> float:
    try:
        return float(hit.get(field, hit.get("score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0
