"""Documents repository (top-level ingested sources)."""

from __future__ import annotations

from typing import Any

from pkb_agent.agent.errors import StorageError
from pkb_agent.storage.supabase_client import SupabaseClient

_TABLE = "documents"


class DocumentsRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._client = client

    def create(
        self,
        *,
        user_id: str = "default",
        title: str,
        source_uri: str | None = None,
        source_type: str = "text",
        content_hash: str | None = None,
        char_count: int = 0,
        chunk_count: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "user_id": user_id,
            "title": title,
            "source_uri": source_uri,
            "source_type": source_type,
            "content_hash": content_hash,
            "char_count": char_count,
            "chunk_count": chunk_count,
            "meta": meta or {},
        }
        try:
            data = self._client.table(_TABLE).insert(row).execute().data
        except Exception as e:
            raise StorageError(f"create document failed: {e}") from e
        return data[0] if data else {}

    def get(self, document_id: str) -> dict[str, Any] | None:
        data = self._client.table(_TABLE).select("*").eq("id", document_id).execute().data
        return data[0] if data else None

    def get_by_hash(self, user_id: str, content_hash: str) -> dict[str, Any] | None:
        data = (
            self._client.table(_TABLE)
            .select("*")
            .eq("user_id", user_id)
            .eq("content_hash", content_hash)
            .execute()
            .data
        )
        return data[0] if data else None

    def list(
        self, user_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        q = self._client.table(_TABLE).select("*").order("created_at", desc=True)
        if user_id is not None:
            q = q.eq("user_id", user_id)
        return q.limit(limit).offset(offset).execute().data

    def update(self, document_id: str, **fields: Any) -> dict[str, Any] | None:
        data = self._client.table(_TABLE).update(fields).eq("id", document_id).execute().data
        return data[0] if data else None

    def delete(self, document_id: str) -> None:
        self._client.table(_TABLE).delete().eq("id", document_id).execute()
