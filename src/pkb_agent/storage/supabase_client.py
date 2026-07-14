"""Supabase client wrapper + pgvector formatting helpers.

Repository methods are intentionally *synchronous* (the underlying
``supabase-py`` client is synchronous). They are invoked from async tools; the
brief blocking on DB I/O is acceptable for a single-user CLI. Slow network I/O
(LLM, embeddings, web) lives in async clients elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pkb_agent.agent.errors import ConfigError, StorageError

if TYPE_CHECKING:
    from pkb_agent.app.settings import Settings


def format_vector(vec: list[float]) -> str:
    """Render a vector as a pgvector text literal, e.g. ``[0.1,0.2,0.3]``.

    PostgREST expects vector params in this string form when calling an rpc whose
    argument is typed ``vector(n)``.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class SupabaseClient:
    """Thin wrapper around the Supabase Python client."""

    def __init__(self, url: str, key: str) -> None:
        try:
            from supabase import create_client
        except ImportError as e:  # pragma: no cover
            raise ConfigError("supabase package is not installed") from e
        self._client = create_client(url, key)
        self._url = url

    @classmethod
    def from_settings(
        cls, settings: Settings, *, use_service_role: bool = True
    ) -> SupabaseClient:
        key = (
            settings.supabase_service_role_key
            if use_service_role
            else settings.supabase_anon_key
        )
        if not settings.supabase_url or not key:
            raise ConfigError("supabase_url and a key (service role or anon) are required")
        return cls(settings.supabase_url, key)

    @property
    def client(self):
        return self._client

    @property
    def url(self) -> str:
        return self._url

    def table(self, name: str):
        return self._client.table(name)

    def rpc(self, name: str, params: dict[str, Any]):
        try:
            return self._client.rpc(name, params).execute()
        except Exception as e:
            raise StorageError(f"rpc '{name}' failed: {e}") from e
