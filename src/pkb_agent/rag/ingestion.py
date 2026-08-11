"""Ingestion pipeline: chunk -> embed -> store documents/chunks/embeddings."""

from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
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


def _extract_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_pdf_bytes(content: bytes) -> str:
    """Extract text from an uploaded PDF without materialising it on disk."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def read_file_text(path: Path) -> str:
    """Read text content from a file, dispatching by extension."""
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path)
    return path.read_text("utf-8")


def read_upload_text(content: bytes, filename: str | None = None) -> str:
    """Decode a browser upload into ingestible text.

    PDF extraction shares the same parser as CLI ingestion. Text files are
    decoded as UTF-8 with replacement so an individual malformed byte does not
    discard an otherwise useful source; binary formats remain unsupported.
    """
    suffix = Path(filename or "upload.txt").suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_bytes(content)
    return content.decode("utf-8", errors="replace")


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
        chunks = split_text(normalized, self._chunk_size, self._chunk_overlap)
        chunk_count = len(chunks)

        existing = await asyncio.to_thread(
            self._documents_repo.get_by_hash, user_id, content_hash
        )
        if existing:
            doc_id = existing.get("id")
            if not doc_id:
                raise StorageError("existing document has no id")
            existing_chunks = await asyncio.to_thread(self._chunks_repo.list_by_document, doc_id)
            if len(existing_chunks) == chunk_count:
                await self._embed_missing_chunks(existing_chunks)
                return existing
            # A prior ingest could have stopped after creating the document or
            # chunks. Replace only this document's dependent chunks, then
            # retain its stable document id and content hash.
            await asyncio.to_thread(self._chunks_repo.delete_by_document, doc_id)
            document = await asyncio.to_thread(
                self._documents_repo.update,
                doc_id,
                title=title,
                source_uri=source_uri,
                source_type=source_type,
                char_count=len(normalized),
                chunk_count=chunk_count,
                meta=meta,
            ) or existing
        else:
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
            await self._embed_missing_chunks(created_chunks)
        except EmbeddingError:
            await asyncio.to_thread(
                self._documents_repo.update,
                doc_id,
                chunk_count=len(created_chunks),
            )
            raise

        return document

    async def _embed_missing_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Embed only chunks without a vector for the active model and dimensions."""
        chunk_ids = [str(chunk["id"]) for chunk in chunks if chunk.get("id")]
        persisted = await asyncio.to_thread(
            self._chunks_repo.get_embeddings_by_chunk_ids, chunk_ids
        )
        missing_chunks = [
            chunk
            for chunk in chunks
            if not self._has_current_embedding(persisted.get(str(chunk.get("id") or "")))
        ]
        if not missing_chunks:
            return
        texts = [str(chunk["content"]) for chunk in missing_chunks]
        embeddings = await self._embedder.embed(texts)
        if len(embeddings) != len(missing_chunks):
            raise EmbeddingError(
                f"embedding count mismatch: got {len(embeddings)}, "
                f"expected {len(missing_chunks)}"
            )
        emb_rows = [
            {
                "chunk_id": missing_chunks[index]["id"],
                "embedding": embeddings[index],
                "model": self._embedder.model,
                "dimensions": self._embedder.dimensions,
            }
            for index in range(len(missing_chunks))
        ]
        await asyncio.to_thread(self._chunks_repo.create_embeddings, emb_rows)

    def _has_current_embedding(self, row: dict[str, Any] | None) -> bool:
        if not row or row.get("model") != self._embedder.model:
            return False
        try:
            return int(row.get("dimensions")) == self._embedder.dimensions
        except (TypeError, ValueError):
            return False

    async def ingest_file(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        p = Path(path)
        text = await asyncio.to_thread(read_file_text, p)
        if not kwargs.get("title"):
            kwargs["title"] = p.stem
        kwargs.setdefault("source_uri", str(p))
        kwargs.setdefault("source_type", "file")
        return await self.ingest_text(text, **kwargs)

    async def ingest_upload(
        self,
        content: bytes,
        *,
        filename: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ingest an uploaded file while retaining source/version metadata."""
        text = await asyncio.to_thread(read_upload_text, content, filename)
        if not kwargs.get("title"):
            kwargs["title"] = Path(filename or "Untitled upload").stem or "Untitled upload"
        kwargs.setdefault("source_uri", f"upload://{filename or 'untitled'}")
        kwargs.setdefault("source_type", "file")
        metadata = dict(kwargs.pop("meta", {}) or {})
        metadata.setdefault("source_status", "ready")
        metadata.setdefault("uploaded_filename", filename or "untitled")
        kwargs["meta"] = metadata
        return await self.ingest_text(text, **kwargs)
