"""Hybrid retriever: vector + FTS search fused via RRF."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from pkb_agent.app.settings import Settings
from pkb_agent.rag.embeddings import EmbeddingProvider
from pkb_agent.rag.ranking import rrf_fuse, weighted_score_fuse
from pkb_agent.storage.repositories.chunks import ChunksRepository


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    meta: dict
    doc_title: str | None = None
    source_uri: str | None = None
    vector_score: float = 0.0
    fts_score: float = 0.0
    score: float = 0.0
    source: str = "rag"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SearchHit:
        meta = d.get("chunk_meta")
        if meta is None:
            meta = d.get("meta") or {}
        raw_score = d.get("score")
        if raw_score is None:
            raw_score = d.get("vector_score")
        if raw_score is None:
            raw_score = d.get("fts_score")
        return cls(
            chunk_id=str(d.get("chunk_id") or d.get("id") or ""),
            document_id=str(d.get("document_id") or ""),
            chunk_index=int(d.get("chunk_index") or 0),
            content=str(d.get("content") or ""),
            meta=meta if isinstance(meta, dict) else {},
            doc_title=d.get("doc_title"),
            source_uri=d.get("source_uri"),
            vector_score=float(d.get("vector_score") or 0.0),
            fts_score=float(d.get("fts_score") or 0.0),
            score=float(raw_score or 0.0),
        )


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        chunks_repo: ChunksRepository,
        *,
        top_k: int = 6,
        rrf_k: int = 60,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
        min_query_len: int = 2,
    ) -> None:
        self._embedder = embedder
        self._chunks_repo = chunks_repo
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._fts_weight = fts_weight
        self._min_query_len = min_query_len

    @classmethod
    def from_settings(
        cls, settings: Settings, embedder: EmbeddingProvider, chunks_repo: ChunksRepository
    ) -> Retriever:
        return cls(
            embedder=embedder,
            chunks_repo=chunks_repo,
            top_k=settings.rag_top_k,
            rrf_k=settings.rag_rrf_k,
            vector_weight=settings.rag_vector_weight,
            fts_weight=settings.rag_fts_weight,
            min_query_len=settings.rag_min_query_len,
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> list[SearchHit]:
        if not query or not query.strip():
            return []
        k = top_k or self._top_k
        match_count = max(k * 3, 1)
        embedding = await self._embedder.embed_one(query)
        do_fts = len(query.strip()) >= self._min_query_len
        if do_fts:
            fts_query = _fts_query(query)
            vector_hits, fts_hits = await asyncio.gather(
                asyncio.to_thread(
                    self._chunks_repo.vector_search, embedding, match_count, user_id
                ),
                asyncio.to_thread(
                    self._chunks_repo.fts_search, fts_query, match_count, user_id
                ),
            )
        else:
            vector_hits = await asyncio.to_thread(
                self._chunks_repo.vector_search, embedding, match_count, user_id
            )
            fts_hits = []

        fused = rrf_fuse(
            vector_hits,
            fts_hits,
            k=self._rrf_k,
            vector_weight=self._vector_weight,
            fts_weight=self._fts_weight,
        )
        return [SearchHit.from_dict(h) for h in fused[:k]]

    async def weighted_hybrid(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> list[SearchHit]:
        """Fuse vector and FTS scores after per-list normalization.

        This supplies the plain ``Vector + FTS`` ablation.  Production RRF
        remains :meth:`search`, so the benchmark can demonstrate the impact of
        rank fusion independently from adding lexical retrieval itself.
        """
        if not query or not query.strip():
            return []
        k = top_k or self._top_k
        match_count = max(k * 3, 1)
        embedding = await self._embedder.embed_one(query)
        do_fts = len(query.strip()) >= self._min_query_len
        if do_fts:
            vector_hits, fts_hits = await asyncio.gather(
                asyncio.to_thread(
                    self._chunks_repo.vector_search, embedding, match_count, user_id
                ),
                asyncio.to_thread(
                    self._chunks_repo.fts_search, _fts_query(query), match_count, user_id
                ),
            )
        else:
            vector_hits = await asyncio.to_thread(
                self._chunks_repo.vector_search, embedding, match_count, user_id
            )
            fts_hits = []
        fused = weighted_score_fuse(
            vector_hits,
            fts_hits,
            vector_weight=self._vector_weight,
            fts_weight=self._fts_weight,
        )
        return [SearchHit.from_dict(hit) for hit in fused[:k]]

    async def vector_only(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> list[SearchHit]:
        if not query or not query.strip():
            return []
        k = top_k or self._top_k
        match_count = max(k * 3, 1)
        embedding = await self._embedder.embed_one(query)
        hits = await asyncio.to_thread(
            self._chunks_repo.vector_search, embedding, match_count, user_id
        )
        return [SearchHit.from_dict(h) for h in hits[:k]]

    async def fts_only(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> list[SearchHit]:
        """Run lexical retrieval without embedding or RRF.

        This is intentionally a first-class method instead of an evaluator-only
        repository call.  It keeps the lexical baseline subject to the same
        input validation and result shape as the production retriever.
        """
        if not query or not query.strip() or len(query.strip()) < self._min_query_len:
            return []
        k = top_k or self._top_k
        match_count = max(k * 3, 1)
        hits = await asyncio.to_thread(
            self._chunks_repo.fts_search, _fts_query(query), match_count, user_id
        )
        return [SearchHit.from_dict(h) for h in hits[:k]]


_FTS_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "use",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "would",
    }
)
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _fts_query(query: str) -> str:
    """Turn a natural-language question into a forgiving OR lexical query.

    ``websearch_to_tsquery`` treats plain adjacent words as conjunctions.  A
    user question consequently returns zero results when even one filler word
    is absent from a source.  This small deterministic projection preserves
    meaningful terms and joins them with ``OR``; non-Latin input falls back to
    the original query so it remains usable with a language-appropriate DB
    text-search configuration.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for token in _FTS_TOKEN_RE.findall(query):
        normalized = token.casefold()
        if len(normalized) < 2 or normalized in _FTS_STOP_WORDS or normalized == "fastapi":
            continue
        if normalized not in seen:
            seen.add(normalized)
            terms.append(token)
        if len(terms) >= 10:
            break
    return " OR ".join(terms) if terms else query
