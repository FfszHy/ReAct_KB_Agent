"""Connection-policy tests for the Supabase wrapper."""

from __future__ import annotations

import ssl
import sys
import types
from unittest.mock import MagicMock

import httpx
import pytest

from pkb_agent.agent.errors import StorageError
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

    httpx_kwargs = captured["httpx_kwargs"]
    assert isinstance(httpx_kwargs, dict)
    assert httpx_kwargs["trust_env"] is False
    assert httpx_kwargs["timeout"].read == 60
    assert captured["client_options"] == {"httpx_client": client._http_client}
    client.close()
    assert captured["closed"] is True


def _rpc_client(effects):
    client = SupabaseClient.__new__(SupabaseClient)
    client._client = MagicMock()
    client._client.rpc.return_value.execute.side_effect = effects
    return client


@pytest.mark.parametrize("name", ["rag_vector_search", "rag_fts_search", "rag_hybrid_search"])
def test_read_rpc_recovers_after_transient_disconnect(monkeypatch, caplog, name):
    response = object()
    client = _rpc_client([
        httpx.RemoteProtocolError("Server disconnected; private-query-marker"),
        ssl.SSLError("UNEXPECTED_EOF; private-query-marker"),
        response,
    ])
    delays = []
    monkeypatch.setattr(supabase_client.time, "sleep", delays.append)

    assert client.rpc(name, {"query": "private-query-marker"}) is response

    assert client._client.rpc.call_count == 3
    assert delays == [0.25, 0.5]
    assert "RemoteProtocolError" in caplog.text
    assert "SSLError" in caplog.text
    assert "attempt=1/3" in caplog.text
    assert "attempt=2/3" in caplog.text
    assert "private-query-marker" not in caplog.text


@pytest.mark.parametrize("error_type", [httpx.ConnectError, TimeoutError, ConnectionError, ssl.SSLError])
def test_read_rpc_stops_after_three_attempts_and_preserves_last_cause(monkeypatch, error_type):
    failures = [error_type(f"failure-{index}") for index in range(3)]
    client = _rpc_client(failures)
    delays = []
    monkeypatch.setattr(supabase_client.time, "sleep", delays.append)

    with pytest.raises(StorageError) as caught:
        client.rpc("rag_vector_search", {"p_embedding": "private-vector"})

    assert caught.value.__cause__ is failures[-1]
    assert client._client.rpc.call_count == 3
    assert delays == [0.25, 0.5]
    assert "private-vector" not in str(caught.value)


def test_unknown_or_mutating_rpc_is_never_retried(monkeypatch):
    failure = httpx.RemoteProtocolError("connection lost after possible commit")
    client = _rpc_client([failure])
    delays = []
    monkeypatch.setattr(supabase_client.time, "sleep", delays.append)

    with pytest.raises(StorageError) as caught:
        client.rpc("create_memory_entry", {"value": "payload"})

    assert caught.value.__cause__ is failure
    assert client._client.rpc.call_count == 1
    assert delays == []


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("invalid RPC arguments"),
        RuntimeError("business rule rejected"),
        httpx.HTTPStatusError(
            "authentication failed",
            request=httpx.Request("POST", "https://example.com/rpc"),
            response=httpx.Response(401),
        ),
    ],
)
def test_read_rpc_does_not_retry_business_or_authentication_errors(monkeypatch, failure):
    client = _rpc_client([failure])
    delays = []
    monkeypatch.setattr(supabase_client.time, "sleep", delays.append)

    with pytest.raises(StorageError) as caught:
        client.rpc("rag_fts_search", {"p_query": "question"})

    assert caught.value.__cause__ is failure
    assert client._client.rpc.call_count == 1
    assert delays == []
