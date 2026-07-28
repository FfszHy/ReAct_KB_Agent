"""Connection-policy tests for the Supabase wrapper."""

from __future__ import annotations

import sys
import types

from pkb_agent.storage import supabase_client
from pkb_agent.storage.supabase_client import SupabaseClient


def test_supabase_client_ignores_environment_proxy_by_default(monkeypatch):
    captured: dict[str, object] = {}

    class FakeHttpClient:
        def __init__(self, **kwargs) -> None:
            captured["httpx_kwargs"] = kwargs

        def close(self) -> None:
            captured["closed"] = True

    class FakeClientOptions:
        def __init__(self, **kwargs) -> None:
            captured["client_options"] = kwargs

    def create_client(url: str, key: str, *, options):
        captured["url"] = url
        captured["key"] = key
        return object()

    fake_supabase = types.ModuleType("supabase")
    fake_supabase.ClientOptions = FakeClientOptions
    fake_supabase.create_client = create_client
    monkeypatch.setitem(sys.modules, "supabase", fake_supabase)
    monkeypatch.setattr(supabase_client.httpx, "Client", FakeHttpClient)

    client = SupabaseClient("https://example.supabase.co", "service-key")

    assert captured["httpx_kwargs"] == {"trust_env": False}
    assert captured["client_options"] == {"httpx_client": client._http_client}
    client.close()
    assert captured["closed"] is True
