"""Memory manager: write notes with embeddings and semantic search."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pkb_agent.memory.policy import MemoryPolicy
from pkb_agent.rag.embeddings import EmbeddingProvider
from pkb_agent.storage.repositories.memory import MemoryRepository

if TYPE_CHECKING:
    from pkb_agent.app.settings import Settings


@dataclass
class MemoryNote:
    id: str
    scope: str
    kind: str
    content: str
    meta: dict = field(default_factory=dict)
    score: float = 0.0
    created_at: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> MemoryNote:
        created_at = d.get("created_at")
        if created_at is not None:
            created_at = str(created_at)
        return cls(
            id=d.get("id", ""),
            scope=d.get("scope", "long"),
            kind=d.get("kind", "fact"),
            content=d.get("content", ""),
            meta=d.get("meta") or {},
            score=float(d.get("score") or 0.0),
            created_at=created_at,
        )


class MemoryManager:
    def __init__(
        self,
        repo: MemoryRepository,
        embedder: EmbeddingProvider,
        policy: MemoryPolicy,
        *,
        max_results: int = 5,
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._policy = policy
        self._max_results = max_results

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        repo: MemoryRepository,
        embedder: EmbeddingProvider,
    ) -> MemoryManager:
        return cls(
            repo=repo,
            embedder=embedder,
            policy=MemoryPolicy.from_settings(settings),
            max_results=settings.memory_max_results,
        )

    async def write(
        self,
        *,
        user_id: str = "default",
        content: str,
        kind: str = "fact",
        scope: str = "long",
        run_id: str | None = None,
        meta: dict | None = None,
    ) -> dict:
        ok, reason = self._policy.validate(content=content, kind=kind, scope=scope)
        if not ok:
            raise ValueError(f"invalid memory: {reason}")
        embedding = await self._embedder.embed_one(content)
        row = await asyncio.to_thread(
            self._repo.create,
            user_id=user_id,
            scope=scope,
            kind=kind,
            content=content,
            embedding=embedding,
            run_id=run_id,
            meta=meta,
        )
        return row

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        user_id: str | None = None,
        scope: str | None = None,
    ) -> list[MemoryNote]:
        query_embedding = await self._embedder.embed_one(query)
        limit = max_results if max_results is not None else self._max_results
        rows = await asyncio.to_thread(
            self._repo.search,
            query_embedding,
            match_count=limit,
            user_id=user_id,
            scope=scope,
        )
        return [MemoryNote.from_dict(r) for r in rows]
