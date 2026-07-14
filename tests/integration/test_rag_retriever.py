"""Integration tests for the hybrid RAG retriever.

The :class:`Retriever` orchestrates an :class:`EmbeddingProvider` (async) and a
:class:`ChunksRepository` (sync, run via ``asyncio.to_thread``). Both are
replaced by mocks so the test runs fully offline: no embedding API and no
Supabase RPC. The RRF fusion + ``SearchHit`` mapping logic uses the real
implementation under test.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pkb_agent.rag.retriever import Retriever, SearchHit

FIXED_VECTOR = [0.1, 0.2, 0.3]


def _make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one = AsyncMock(return_value=FIXED_VECTOR)
    return embedder


def _make_chunks_repo(
    vector_hits: list | None = None, fts_hits: list | None = None
) -> MagicMock:
    repo = MagicMock()
    repo.vector_search = MagicMock(return_value=vector_hits or [])
    repo.fts_search = MagicMock(return_value=fts_hits or [])
    return repo


def _hit(chunk_id: str, *, content: str = "c", document_id: str = "d", **extra):
    base = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_index": 0,
        "content": content,
    }
    base.update(extra)
    return base


# --------------------------------------------------------------------------- #
# search: input validation
# --------------------------------------------------------------------------- #


async def test_search_empty_query_returns_empty_without_calls():
    embedder = _make_embedder()
    repo = _make_chunks_repo()
    retriever = Retriever(embedder, repo)

    assert await retriever.search("") == []
    assert await retriever.search("   ") == []
    embedder.embed_one.assert_not_called()
    repo.vector_search.assert_not_called()


# --------------------------------------------------------------------------- #
# search: embedding + dispatch
# --------------------------------------------------------------------------- #


async def test_search_embeds_query_then_calls_vector_search():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[_hit("a")])
    retriever = Retriever(embedder, repo)

    await retriever.search("hello world")

    embedder.embed_one.assert_awaited_once_with("hello world")
    repo.vector_search.assert_called_once()
    # vector_search(embedding, match_count, user_id)
    args = repo.vector_search.call_args.args
    assert args[0] == FIXED_VECTOR
    assert args[1] == 18  # match_count = top_k(6) * 3
    assert args[2] is None  # user_id


async def test_search_calls_both_vector_and_fts_for_long_query():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[_hit("a")], fts_hits=[_hit("b")])
    retriever = Retriever(embedder, repo)

    await retriever.search("hello world")

    repo.vector_search.assert_called_once()
    repo.fts_search.assert_called_once()
    fts_args = repo.fts_search.call_args.args
    assert fts_args[0] == "hello world"
    assert fts_args[1] == 18


async def test_search_short_query_skips_fts():
    # min_query_len default is 2; a single-char query must skip FTS.
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[_hit("a")])
    retriever = Retriever(embedder, repo, min_query_len=5)

    await retriever.search("abc")  # len 3 < 5

    repo.vector_search.assert_called_once()
    repo.fts_search.assert_not_called()


# --------------------------------------------------------------------------- #
# search: fusion + ranking
# --------------------------------------------------------------------------- #


async def test_search_fuses_via_rrf_and_returns_top_k():
    vec = [_hit("a"), _hit("b"), _hit("c")]
    fts = [_hit("b"), _hit("d")]
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=vec, fts_hits=fts)
    retriever = Retriever(embedder, repo, top_k=2)

    hits = await retriever.search("query text")

    assert len(hits) == 2
    # 'b' appears in BOTH lists, so RRF sums its vector + fts contributions,
    # boosting it above 'a' which only appears in vector.
    assert hits[0].chunk_id == "b"
    assert hits[1].chunk_id == "a"
    assert hits[0].score > hits[1].score


async def test_search_only_vector_hits():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[_hit("a"), _hit("b")], fts_hits=[])
    retriever = Retriever(embedder, repo)

    hits = await retriever.search("query")

    assert [h.chunk_id for h in hits] == ["a", "b"]


async def test_search_only_fts_hits():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[], fts_hits=[_hit("x"), _hit("y")])
    retriever = Retriever(embedder, repo)

    hits = await retriever.search("query")

    assert [h.chunk_id for h in hits] == ["x", "y"]


async def test_search_empty_results_from_both():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[], fts_hits=[])
    retriever = Retriever(embedder, repo)

    assert await retriever.search("query") == []


async def test_search_respects_custom_top_k_and_match_count():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[_hit(str(i)) for i in range(10)])
    retriever = Retriever(embedder, repo, top_k=6)

    hits = await retriever.search("query", top_k=3)

    assert len(hits) == 3
    args = repo.vector_search.call_args.args
    assert args[1] == 9  # match_count = top_k(3) * 3


async def test_search_passes_user_id_to_both_searches():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[_hit("a")], fts_hits=[_hit("b")])
    retriever = Retriever(embedder, repo)

    await retriever.search("query", user_id="alice")

    assert repo.vector_search.call_args.args[2] == "alice"
    assert repo.fts_search.call_args.args[2] == "alice"


async def test_search_hit_maps_score_and_content_fields():
    embedder = _make_embedder()
    repo = _make_chunks_repo(
        vector_hits=[_hit("a", content="hello", doc_title="Doc", source_uri="uri")]
    )
    retriever = Retriever(embedder, repo)

    hits = await retriever.search("query")

    hit = hits[0]
    assert isinstance(hit, SearchHit)
    assert hit.chunk_id == "a"
    assert hit.content == "hello"
    assert hit.doc_title == "Doc"
    assert hit.source_uri == "uri"
    assert hit.score > 0.0


# --------------------------------------------------------------------------- #
# vector_only
# --------------------------------------------------------------------------- #


async def test_vector_only_never_calls_fts():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[_hit("a"), _hit("b")])
    retriever = Retriever(embedder, repo)

    hits = await retriever.vector_only("query")

    assert [h.chunk_id for h in hits] == ["a", "b"]
    repo.vector_search.assert_called_once()
    repo.fts_search.assert_not_called()


async def test_vector_only_respects_top_k():
    embedder = _make_embedder()
    repo = _make_chunks_repo(vector_hits=[_hit(str(i)) for i in range(8)])
    retriever = Retriever(embedder, repo, top_k=3)

    hits = await retriever.vector_only("query")

    assert len(hits) == 3


# --------------------------------------------------------------------------- #
# SearchHit.from_dict mapping (unit-level but kept here for cohesion)
# --------------------------------------------------------------------------- #


def test_search_hit_from_dict_uses_chunk_meta_fallback():
    hit = SearchHit.from_dict(
        {"chunk_id": "c1", "document_id": "d1", "content": "x", "chunk_meta": {"k": "v"}}
    )
    assert hit.meta == {"k": "v"}


def test_search_hit_from_dict_score_falls_back_to_vector_score():
    hit = SearchHit.from_dict({"chunk_id": "a", "vector_score": 0.9})
    assert hit.score == pytest.approx(0.9)
