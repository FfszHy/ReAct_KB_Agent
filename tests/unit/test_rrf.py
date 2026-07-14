from __future__ import annotations

import pytest

from pkb_agent.rag.ranking import dedupe_hits, rrf_fuse

# ---- rrf_fuse: empty / single-list ----------------------------------------


def test_rrf_empty_lists():
    assert rrf_fuse([], []) == []


def test_rrf_none_lists():
    assert rrf_fuse(None, None) == []  # type: ignore[arg-type]


def test_rrf_vector_only():
    hits = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    out = rrf_fuse(hits, [])
    assert len(out) == 2
    assert {h["chunk_id"] for h in out} == {"a", "b"}


def test_rrf_fts_only():
    hits = [{"chunk_id": "x"}]
    out = rrf_fuse([], hits)
    assert len(out) == 1
    assert out[0]["chunk_id"] == "x"


# ---- formula --------------------------------------------------------------


def test_rrf_formula_vector_weight():
    # score = vector_weight / (k + rank), rank starts at 1
    out = rrf_fuse([{"chunk_id": "a"}], [], k=60, vector_weight=0.6, fts_weight=0.4)
    assert out[0]["score"] == pytest.approx(0.6 / 61)


def test_rrf_formula_fts_weight():
    out = rrf_fuse([], [{"chunk_id": "a"}], k=60, vector_weight=0.6, fts_weight=0.4)
    assert out[0]["score"] == pytest.approx(0.4 / 61)


def test_rrf_custom_k():
    out = rrf_fuse([{"chunk_id": "a"}], [], k=10, vector_weight=1.0, fts_weight=0.0)
    assert out[0]["score"] == pytest.approx(1.0 / 11)


def test_rrf_dedup_by_id_sums_scores():
    vec = [{"chunk_id": "a"}]
    fts = [{"chunk_id": "a"}]
    out = rrf_fuse(vec, fts, k=60, vector_weight=0.6, fts_weight=0.4)
    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(0.6 / 61 + 0.4 / 61)


def test_rrf_skips_none_chunk_id():
    vec = [{"chunk_id": None}, {"chunk_id": "a"}]
    out = rrf_fuse(vec, [], k=60, vector_weight=1.0, fts_weight=0.0)
    assert len(out) == 1
    assert out[0]["chunk_id"] == "a"


def test_rrf_str_chunk_id_normalization():
    # int chunk_id should be normalized to str key (dedup across lists).
    vec = [{"chunk_id": 1}]
    fts = [{"chunk_id": "1"}]
    out = rrf_fuse(vec, fts, k=60, vector_weight=0.6, fts_weight=0.4)
    assert len(out) == 1


# ---- ordering -------------------------------------------------------------


def test_rrf_ordering_desc_by_score():
    # 'a' ranks first in vector; with default weights 'a' should beat 'b'.
    vec = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    fts = [{"chunk_id": "b"}, {"chunk_id": "a"}]
    out = rrf_fuse(vec, fts, k=60, vector_weight=0.6, fts_weight=0.4)
    assert out[0]["chunk_id"] == "a"
    assert out[0]["score"] > out[1]["score"]
    assert out == sorted(out, key=lambda d: d["score"], reverse=True)


# ---- result fields --------------------------------------------------------


def test_rrf_score_fields_populated():
    vec = [{"chunk_id": "a", "vector_score": 0.9}]
    fts = [{"chunk_id": "a", "fts_score": 0.5}]
    out = rrf_fuse(vec, fts)
    item = out[0]
    assert item["vector_score"] == pytest.approx(0.9)
    assert item["fts_score"] == pytest.approx(0.5)
    assert "score" in item


def test_rrf_merges_missing_keys_from_fts():
    vec = [{"chunk_id": "a", "vector_score": 0.9}]
    fts = [{"chunk_id": "a", "fts_score": 0.5, "extra": "val"}]
    out = rrf_fuse(vec, fts)
    assert out[0]["extra"] == "val"


# ---- dedupe_hits ----------------------------------------------------------


def test_dedupe_hits_basic():
    hits = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "a"}]
    out = dedupe_hits(hits)
    assert out == [{"chunk_id": "a"}, {"chunk_id": "b"}]


def test_dedupe_hits_custom_key():
    hits = [{"id": "x"}, {"id": "y"}, {"id": "x"}]
    out = dedupe_hits(hits, key="id")
    assert out == [{"id": "x"}, {"id": "y"}]


def test_dedupe_hits_none_key_preserved():
    hits = [{"chunk_id": None}, {"chunk_id": None}]
    out = dedupe_hits(hits)
    assert len(out) == 2


def test_dedupe_hits_empty():
    assert dedupe_hits([]) == []
    assert dedupe_hits(None) == []  # type: ignore[arg-type]
