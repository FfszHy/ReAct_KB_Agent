"""Hybrid retriever: vector + FTS search fused via RRF."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pkb_agent.app.settings import Settings
from pkb_agent.rag.embeddings import EmbeddingProvider
from pkb_agent.rag.ranking import rrf_fuse
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
            score=float(
                d.get("score") or d.get("vector_score") or d.get("fts_score") or 0.0
            ),
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
            vector_hits, fts_hits = await asyncio.gather(
                asyncio.to_thread(
                    self._chunks_repo.vector_search, embedding, match_count, user_id
                ),
                asyncio.to_thread(
                    self._chunks_repo.fts_search, query, match_count, user_id
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
