"""Chunks + embeddings repository, including vector/FTS retrieval via rpc()."""

from __future__ import annotations

from typing import Any

from pkb_agent.agent.errors import StorageError
from pkb_agent.storage.supabase_client import SupabaseClient, format_vector

_CHUNKS = "document_chunks"
_EMBEDDINGS = "chunk_embeddings"


class ChunksRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._client = client

    # ------------------------------------------------------------------ chunks
    def create_chunks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        try:
            return self._client.table(_CHUNKS).insert(rows).execute().data or []
        except Exception as e:
            raise StorageError(f"create_chunks failed: {e}") from e

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        data = self._client.table(_CHUNKS).select("*").eq("id", chunk_id).execute().data
        return data[0] if data else None

    def get_chunk_with_doc(self, chunk_id: str) -> dict[str, Any] | None:
        """Return a chunk joined with its document's title/source_uri."""
        data = (
            self._client.table(_CHUNKS)
            .select("*, documents(title, source_uri, source_type)")
            .eq("id", chunk_id)
            .execute()
            .data
        )
        if not data:
            return None
        row = data[0]
        doc = row.pop("documents", None) or {}
        if isinstance(doc, list):
            doc = doc[0] if doc else {}
        row["document"] = doc
        return row

    def list_by_document(self, document_id: str) -> list[dict[str, Any]]:
        return (
            self._client.table(_CHUNKS)
            .select("*")
            .eq("document_id", document_id)
            .order("chunk_index", desc=False)
            .execute()
            .data
        )

    def delete_by_document(self, document_id: str) -> None:
        # document_chunks cascade-deletes embeddings; chunks cascade from documents.
        self._client.table(_CHUNKS).delete().eq("document_id", document_id).execute()

    # --------------------------------------------------------------- embeddings
    def create_embeddings(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """rows: chunk_id, embedding (list[float]), model, dimensions."""
        if not rows:
            return []
        payload = [
            {
                "chunk_id": r["chunk_id"],
                "embedding": format_vector(r["embedding"]),
                "model": r["model"],
                "dimensions": r["dimensions"],
            }
            for r in rows
        ]
        try:
            # upsert so re-ingestion overwrites stale vectors.
            return (
                self._client.table(_EMBEDDINGS)
                .upsert(payload, on_conflict="chunk_id")
                .execute()
                .data
                or []
            )
        except Exception as e:
            raise StorageError(f"create_embeddings failed: {e}") from e

    def get_embeddings_by_chunk_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return persisted embedding metadata keyed by chunk id."""
        if not chunk_ids:
            return {}
        try:
            rows = (
                self._client.table(_EMBEDDINGS)
                .select("chunk_id,model,dimensions")
                .in_("chunk_id", chunk_ids)
                .execute()
                .data
                or []
            )
        except Exception as e:
            raise StorageError(f"get embeddings failed: {e}") from e
        return {
            str(row["chunk_id"]): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("chunk_id"), str)
        }

    # ------------------------------------------------------------------ search
    def vector_search(
        self,
        query_embedding: list[float],
        match_count: int = 6,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "p_embedding": format_vector(query_embedding),
            "p_match_count": match_count,
        }
        if user_id is not None:
            params["p_user_id"] = user_id
        return self._client.rpc("rag_vector_search", params).data or []

    def fts_search(
        self,
        query_text: str,
        match_count: int = 6,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "p_query": query_text,
            "p_match_count": match_count,
        }
        if user_id is not None:
            params["p_user_id"] = user_id
        return self._client.rpc("rag_fts_search", params).data or []

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        match_count: int = 6,
        user_id: str | None = None,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "p_query": query_text,
            "p_embedding": format_vector(query_embedding),
            "p_match_count": match_count,
            "p_vector_weight": vector_weight,
            "p_fts_weight": fts_weight,
        }
        if user_id is not None:
            params["p_user_id"] = user_id
        return self._client.rpc("rag_hybrid_search", params).data or []
