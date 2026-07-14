"""Permissions repository: optional DB-backed tool permission overrides."""

from __future__ import annotations

from typing import Any

from pkb_agent.agent.errors import StorageError
from pkb_agent.storage.supabase_client import SupabaseClient

_TABLE = "tool_permissions"


class PermissionsRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._client = client

    def get(self, tool_name: str) -> dict[str, Any] | None:
        data = self._client.table(_TABLE).select("*").eq("tool_name", tool_name).execute().data
        return data[0] if data else None

    def upsert(
        self,
        *,
        tool_name: str,
        permission: str = "allow",
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {"tool_name": tool_name, "permission": permission, "constraints": constraints or {}}
        try:
            data = (
                self._client.table(_TABLE)
                .upsert(row, on_conflict="tool_name")
                .execute()
                .data
            )
        except Exception as e:
            raise StorageError(f"permissions upsert failed: {e}") from e
        return data[0] if data else {}

    def list_all(self) -> list[dict[str, Any]]:
        return self._client.table(_TABLE).select("*").execute().data or []

    def delete(self, tool_name: str) -> None:
        self._client.table(_TABLE).delete().eq("tool_name", tool_name).execute()
