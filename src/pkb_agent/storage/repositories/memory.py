"""Memory repository: task_memory notes + semantic recall via rpc()."""

from __future__ import annotations

from typing import Any

from pkb_agent.agent.errors import StorageError
from pkb_agent.storage.supabase_client import SupabaseClient, format_vector

_TABLE = "task_memory"


class MemoryRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._client = client

    def create(
        self,
        *,
        user_id: str = "default",
        scope: str = "long",
        kind: str = "fact",
        content: str,
        embedding: list[float] | None = None,
        run_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "user_id": user_id,
            "scope": scope,
            "kind": kind,
            "content": content,
            "run_id": run_id,
            "meta": meta or {},
        }
        if embedding is not None:
            row["embedding"] = format_vector(embedding)
        try:
            data = self._client.table(_TABLE).insert(row).execute().data
        except Exception as e:
            raise StorageError(f"memory create failed: {e}") from e
        return data[0] if data else {}

    def search(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        user_id: str | None = None,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "p_embedding": format_vector(query_embedding),
            "p_match_count": match_count,
        }
        if user_id is not None:
            params["p_user_id"] = user_id
        if scope is not None:
            params["p_scope"] = scope
        return self._client.rpc("memory_vector_search", params).data or []

    def get(self, memory_id: str) -> dict[str, Any] | None:
        data = self._client.table(_TABLE).select("*").eq("id", memory_id).execute().data
        return data[0] if data else None

    def list(
        self,
        user_id: str | None = None,
        scope: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        q = self._client.table(_TABLE).select("*").order("created_at", desc=True)
        if user_id is not None:
            q = q.eq("user_id", user_id)
        if scope is not None:
            q = q.eq("scope", scope)
        return q.limit(limit).execute().data

    def delete(self, memory_id: str) -> None:
        self._client.table(_TABLE).delete().eq("id", memory_id).execute()
