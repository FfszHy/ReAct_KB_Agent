"""Ingestion pipeline: chunk -> embed -> store documents/chunks/embeddings."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from pkb_agent.agent.errors import EmbeddingError, StorageError
from pkb_agent.app.settings import Settings
from pkb_agent.rag.chunking import split_text
from pkb_agent.rag.embeddings import EmbeddingProvider
from pkb_agent.storage.repositories.chunks import ChunksRepository
from pkb_agent.storage.repositories.documents import DocumentsRepository


def _normalize(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        documents_repo: DocumentsRepository,
        chunks_repo: ChunksRepository,
        embedder: EmbeddingProvider,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self._documents_repo = documents_repo
        self._chunks_repo = chunks_repo
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        documents_repo: DocumentsRepository,
        chunks_repo: ChunksRepository,
        embedder: EmbeddingProvider,
    ) -> IngestionPipeline:
        return cls(
            documents_repo=documents_repo,
            chunks_repo=chunks_repo,
            embedder=embedder,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    async def ingest_text(
        self,
        text: str,
        *,
        title: str,
        source_uri: str | None = None,
        source_type: str = "text",
        user_id: str = "default",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = _normalize(text)
        content_hash = _content_hash(normalized)

        existing = await asyncio.to_thread(
            self._documents_repo.get_by_hash, user_id, content_hash
        )
        if existing:
            return existing

        chunks = split_text(normalized, self._chunk_size, self._chunk_overlap)
        chunk_count = len(chunks)

        document = await asyncio.to_thread(
            self._documents_repo.create,
            user_id=user_id,
            title=title,
            source_uri=source_uri,
            source_type=source_type,
            content_hash=content_hash,
            char_count=len(normalized),
            chunk_count=chunk_count,
            meta=meta,
        )
        doc_id = document.get("id")
        if not doc_id:
            raise StorageError("document creation returned no id")

        rows = [
            {
                "document_id": doc_id,
                "user_id": user_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "meta": {
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            }
            for chunk in chunks
        ]

        created_chunks = await asyncio.to_thread(self._chunks_repo.create_chunks, rows)
        if len(created_chunks) != chunk_count:
            document = await asyncio.to_thread(
                self._documents_repo.update,  # type: ignore[arg-type]
                doc_id,
                chunk_count=len(created_chunks),
            ) or document

        try:
            texts = [c["content"] for c in created_chunks]
            embeddings = await self._embedder.embed(texts)
            if len(embeddings) != len(created_chunks):
                raise EmbeddingError(
                    f"embedding count mismatch: got {len(embeddings)}, "
                    f"expected {len(created_chunks)}"
                )
            emb_rows = [
                {
                    "chunk_id": created_chunks[i]["id"],
                    "embedding": embeddings[i],
                    "model": self._embedder.model,
                    "dimensions": self._embedder.dimensions,
                }
                for i in range(len(created_chunks))
            ]
            await asyncio.to_thread(self._chunks_repo.create_embeddings, emb_rows)
        except EmbeddingError:
            await asyncio.to_thread(
                self._documents_repo.update,
                doc_id,
                chunk_count=len(created_chunks),
            )
            raise

        return document

    async def ingest_file(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        p = Path(path)
        text = await asyncio.to_thread(p.read_text, "utf-8")
        kwargs.setdefault("title", p.stem)
        kwargs.setdefault("source_uri", str(p))
        kwargs.setdefault("source_type", "file")
        return await self.ingest_text(text, **kwargs)
