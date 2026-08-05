"""Pinned corpus manifests and ingestion for reproducible benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from pkb_agent.app.settings import Settings
from pkb_agent.evaluation.dataset import DatasetError, _ensure_unique, _required_text
from pkb_agent.rag.embeddings import EmbeddingProvider
from pkb_agent.rag.ingestion import IngestionPipeline
from pkb_agent.storage.repositories.chunks import ChunksRepository
from pkb_agent.storage.repositories.documents import DocumentsRepository
from pkb_agent.storage.supabase_client import SupabaseClient


@dataclass(frozen=True)
class CorpusDocument:
    key: str
    title: str
    source_url: str
    sha256: str | None = None

    @classmethod
    def from_dict(cls, value: Any, *, context: str) -> CorpusDocument:
        if not isinstance(value, dict):
            raise DatasetError(f"{context}: corpus document must be an object")
        sha256 = value.get("sha256")
        if sha256 is not None:
            sha256 = _required_text(sha256, f"{context}.sha256").lower()
            if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
                raise DatasetError(f"{context}.sha256 must be a lowercase SHA-256 hash")
        return cls(
            key=_required_text(value.get("key"), f"{context}.key"),
            title=_required_text(value.get("title"), f"{context}.title"),
            source_url=_required_text(value.get("source_url"), f"{context}.source_url"),
            sha256=sha256,
        )


@dataclass(frozen=True)
class CorpusManifest:
    id: str
    title: str
    revision: str
    documents: tuple[CorpusDocument, ...]
    source_repository: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> CorpusManifest:
        source_path = Path(path)
        try:
            raw = json.loads(source_path.read_text("utf-8"))
        except FileNotFoundError as exc:
            raise DatasetError(f"corpus manifest not found: {source_path}") from exc
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON in {source_path}: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise DatasetError(f"{source_path}: corpus manifest must be an object")
        documents_raw = raw.get("documents")
        if not isinstance(documents_raw, list) or not documents_raw:
            raise DatasetError(f"{source_path}.documents must be a non-empty array")
        documents = tuple(
            CorpusDocument.from_dict(value, context=f"{source_path}.documents[{index}]")
            for index, value in enumerate(documents_raw)
        )
        _ensure_unique((document.key for document in documents), f"{source_path}.documents")
        repo = raw.get("source_repository")
        if repo is not None:
            repo = _required_text(repo, f"{source_path}.source_repository")
        return cls(
            id=_required_text(raw.get("id"), f"{source_path}.id"),
            title=_required_text(raw.get("title"), f"{source_path}.title"),
            revision=_required_text(raw.get("revision"), f"{source_path}.revision"),
            documents=documents,
            source_repository=repo,
        )

    def source_uri(self, document_key: str) -> str:
        return f"eval://{self.id}/{document_key}"


@dataclass(frozen=True)
class FetchedCorpusDocument:
    document: CorpusDocument
    text: str
    sha256: str


async def fetch_corpus(
    manifest: CorpusManifest,
    *,
    timeout_seconds: float = 30.0,
    trust_env: bool = False,
) -> tuple[FetchedCorpusDocument, ...]:
    """Fetch and verify a manifest's immutable source revision.

    ``revision`` in the manifest pins the upstream release/tag.  A fetch writes
    actual content hashes into the run lock so later runs can additionally pin
    exact bytes even when the upstream documentation host changes formatting.
    """
    timeout = httpx.Timeout(timeout_seconds)
    # Do not accidentally route public corpus downloads through a stale editor
    # proxy.  It mirrors the Supabase client's explicit proxy posture; callers
    # that genuinely need a corporate proxy can opt in.
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        trust_env=trust_env,
    ) as client:
        results = await asyncio.gather(
            *(_fetch_one(client, document) for document in manifest.documents)
        )
    return tuple(results)


async def _fetch_one(client: httpx.AsyncClient, document: CorpusDocument) -> FetchedCorpusDocument:
    try:
        response = await client.get(document.source_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DatasetError(f"failed to fetch corpus source {document.key}: {exc}") from exc
    text = response.text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if document.sha256 is not None and digest != document.sha256:
        raise DatasetError(
            f"corpus source hash mismatch for {document.key}: expected {document.sha256}, got {digest}"
        )
    return FetchedCorpusDocument(document=document, text=text, sha256=digest)


async def ingest_corpus(
    manifest: CorpusManifest,
    settings: Settings,
    *,
    user_id: str,
    timeout_seconds: float = 30.0,
    trust_env: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Fetch a benchmark corpus and ingest it with stable evaluator URIs."""
    fetched = await fetch_corpus(
        manifest,
        timeout_seconds=timeout_seconds,
        trust_env=trust_env,
    )
    settings.require_supabase()
    settings.require_embedding()
    client = SupabaseClient.from_settings(settings)
    embedder = EmbeddingProvider.from_settings(settings)
    documents = DocumentsRepository(client)
    chunks = ChunksRepository(client)
    pipeline = IngestionPipeline.from_settings(settings, documents, chunks, embedder)
    records: list[dict[str, Any]] = []
    try:
        for item in fetched:
            created = await pipeline.ingest_text(
                item.text,
                title=item.document.title,
                source_uri=manifest.source_uri(item.document.key),
                source_type="eval_corpus",
                user_id=user_id,
                meta={
                    "eval_corpus_id": manifest.id,
                    "eval_document_key": item.document.key,
                    "upstream_url": item.document.source_url,
                    "upstream_revision": manifest.revision,
                    "upstream_sha256": item.sha256,
                },
            )
            records.append(
                {
                    "key": item.document.key,
                    "document_id": created.get("id"),
                    "title": item.document.title,
                    "source_uri": manifest.source_uri(item.document.key),
                    "sha256": item.sha256,
                    "chunk_count": created.get("chunk_count"),
                }
            )
    finally:
        await embedder.close()
        client.close()
    return tuple(records)


def make_corpus_lock(manifest: CorpusManifest, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return a portable lock file that captures what was actually ingested."""
    return {
        "schema_version": 1,
        "corpus_id": manifest.id,
        "revision": manifest.revision,
        "generated_at": datetime.now(UTC).isoformat(),
        "documents": list(records),
    }


def write_json(path: str | Path, payload: Any) -> Path:
    """Write deterministic UTF-8 JSON used by CLI artifacts and tests."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")
    return output
