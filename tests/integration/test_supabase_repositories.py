"""Integration tests for the Supabase-backed repositories.

These tests exercise the real repository classes against a *fake* Supabase
client that captures the fluent ``.table().<op>().execute()`` and
``.rpc().execute()`` call chains. No real database connection is made; every
external boundary is replaced by an in-memory fake so the tests run fully
offline in CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from pkb_agent.agent.errors import StorageError
from pkb_agent.storage.repositories.chunks import ChunksRepository
from pkb_agent.storage.repositories.documents import DocumentsRepository
from pkb_agent.storage.repositories.memory import MemoryRepository
from pkb_agent.storage.repositories.permissions import PermissionsRepository
from pkb_agent.storage.repositories.traces import TracesRepository
from pkb_agent.storage.supabase_client import format_vector

# --------------------------------------------------------------------------- #
# Fake Supabase client
# --------------------------------------------------------------------------- #


class _FakeResult:
    """Mimics the object returned by supabase-py's ``.execute()``."""

    def __init__(self, data: Any) -> None:
        self.data = data


class _FakeQueryBuilder:
    """Fluent builder that records operations and returns queued data on execute."""

    def __init__(self, client: _FakeSupabaseClient, table_name: str) -> None:
        self._client = client
        self._table = table_name

    def insert(self, row: Any) -> _FakeQueryBuilder:
        self._client.calls.append({"op": "insert", "table": self._table, "row": row})
        return self

    def upsert(self, payload: Any, on_conflict: str | None = None) -> _FakeQueryBuilder:
        self._client.calls.append(
            {
                "op": "upsert",
                "table": self._table,
                "payload": payload,
                "on_conflict": on_conflict,
            }
        )
        return self

    def select(self, cols: str = "*") -> _FakeQueryBuilder:
        self._client.calls.append({"op": "select", "table": self._table, "cols": cols})
        return self

    def eq(self, col: str, val: Any) -> _FakeQueryBuilder:
        self._client.calls.append(
            {"op": "eq", "table": self._table, "col": col, "val": val}
        )
        return self

    def in_(self, col: str, values: list[Any]) -> _FakeQueryBuilder:
        self._client.calls.append(
            {"op": "in", "table": self._table, "col": col, "values": values}
        )
        return self

    def order(self, col: str, desc: bool = False) -> _FakeQueryBuilder:
        self._client.calls.append(
            {"op": "order", "table": self._table, "col": col, "desc": desc}
        )
        return self

    def limit(self, n: int) -> _FakeQueryBuilder:
        self._client.calls.append({"op": "limit", "table": self._table, "n": n})
        return self

    def offset(self, n: int) -> _FakeQueryBuilder:
        self._client.calls.append({"op": "offset", "table": self._table, "n": n})
        return self

    def update(self, fields: Any) -> _FakeQueryBuilder:
        self._client.calls.append({"op": "update", "table": self._table, "fields": fields})
        return self

    def delete(self) -> _FakeQueryBuilder:
        self._client.calls.append({"op": "delete", "table": self._table})
        return self

    def execute(self) -> _FakeResult:
        return self._client._next_result()


class _FakeSupabaseClient:
    """In-memory stand-in for :class:`SupabaseClient`.

    ``table(name)`` returns a chainable query builder; ``rpc(name, params)``
    returns an already-executed result (matching ``SupabaseClient.rpc`` which
    calls ``.execute()`` internally). Returned ``.data`` values are pulled from a
    FIFO queue populated via :meth:`queue`; exceptions can be injected via
    :meth:`raise_on_next`.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._data_queue: list[Any] = []
        self._exc_queue: list[BaseException] = []

    def queue(self, data: Any) -> _FakeSupabaseClient:
        self._data_queue.append(data)
        return self

    def raise_on_next(self, exc: BaseException) -> _FakeSupabaseClient:
        self._exc_queue.append(exc)
        return self

    def _next_result(self) -> _FakeResult:
        if self._exc_queue:
            raise self._exc_queue.pop(0)
        data = self._data_queue.pop(0) if self._data_queue else []
        return _FakeResult(data)

    def table(self, name: str) -> _FakeQueryBuilder:
        return _FakeQueryBuilder(self, name)

    def rpc(self, name: str, params: dict[str, Any]) -> _FakeResult:
        self.calls.append({"op": "rpc", "name": name, "params": params})
        return self._next_result()

    # Convenience accessors for assertions -----------------------------------
    def calls_for(self, op: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["op"] == op]

    def find_call(self, op: str, table: str | None = None) -> dict[str, Any] | None:
        for c in self.calls:
            if c["op"] == op and (table is None or c.get("table") == table):
                return c
        return None


# --------------------------------------------------------------------------- #
# format_vector
# --------------------------------------------------------------------------- #


def test_format_vector_basic_floats():
    assert format_vector([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_format_vector_coerces_integers_to_float_repr():
    # repr(float(1)) == "1.0"
    assert format_vector([1, 2, 3]) == "[1.0,2.0,3.0]"
    assert format_vector([]) == "[]"
    assert format_vector([-1.5, 0.0]) == "[-1.5,0.0]"


# --------------------------------------------------------------------------- #
# DocumentsRepository
# --------------------------------------------------------------------------- #


def test_documents_create_inserts_row_and_returns_first():
    client = _FakeSupabaseClient().queue([{"id": "d1", "title": "T"}])
    repo = DocumentsRepository(client)

    row = repo.create(
        title="My Doc",
        source_type="text",
        char_count=10,
        chunk_count=2,
        meta={"k": "v"},
    )

    assert row == {"id": "d1", "title": "T"}
    call = client.find_call("insert", "documents")
    assert call is not None
    assert call["row"]["title"] == "My Doc"
    assert call["row"]["char_count"] == 10
    assert call["row"]["meta"] == {"k": "v"}
    assert call["row"]["user_id"] == "default"


def test_documents_create_wraps_exception_as_storage_error():
    client = _FakeSupabaseClient().raise_on_next(RuntimeError("db down"))
    repo = DocumentsRepository(client)

    with pytest.raises(StorageError, match="create document failed"):
        repo.create(title="x")


def test_documents_get_by_hash_filters_user_and_hash():
    client = _FakeSupabaseClient().queue([{"id": "d1", "content_hash": "h"}])
    repo = DocumentsRepository(client)

    row = repo.get_by_hash(user_id="alice", content_hash="h")

    assert row == {"id": "d1", "content_hash": "h"}
    cols = {c["col"]: c["val"] for c in client.calls_for("eq")}
    assert cols["user_id"] == "alice"
    assert cols["content_hash"] == "h"


def test_documents_list_applies_user_filter_limit_offset():
    client = _FakeSupabaseClient().queue([{"id": "d1"}, {"id": "d2"}])
    repo = DocumentsRepository(client)

    rows = repo.list(user_id="alice", limit=5, offset=10)

    assert rows == [{"id": "d1"}, {"id": "d2"}]
    order_call = client.find_call("order", "documents")
    assert order_call is not None
    assert order_call["desc"] is True
    assert client.find_call("limit", "documents")["n"] == 5
    assert client.find_call("offset", "documents")["n"] == 10


# --------------------------------------------------------------------------- #
# ChunksRepository
# --------------------------------------------------------------------------- #


def test_chunks_create_embeddings_formats_vectors_and_upserts():
    client = _FakeSupabaseClient().queue([{"chunk_id": "c1"}])
    repo = ChunksRepository(client)

    out = repo.create_embeddings(
        [
            {
                "chunk_id": "c1",
                "embedding": [0.1, 0.2, 0.3],
                "model": "qwen3.7-text-embedding",
                "dimensions": 3,
            }
        ]
    )

    assert out == [{"chunk_id": "c1"}]
    call = client.find_call("upsert", "chunk_embeddings")
    assert call is not None
    assert call["on_conflict"] == "chunk_id"
    payload = call["payload"][0]
    assert payload["chunk_id"] == "c1"
    # embedding must be rendered as a pgvector text literal, not a raw list.
    assert payload["embedding"] == "[0.1,0.2,0.3]"
    assert payload["dimensions"] == 3


def test_chunks_get_embeddings_by_chunk_ids_returns_metadata_by_chunk():
    client = _FakeSupabaseClient().queue(
        [
            {"chunk_id": "c1", "model": "qwen3.7-text-embedding", "dimensions": 1536},
            {"chunk_id": "c2", "model": "qwen3.7-text-embedding", "dimensions": 1536},
        ]
    )
    repo = ChunksRepository(client)

    rows = repo.get_embeddings_by_chunk_ids(["c1", "c2"])

    assert rows["c1"]["dimensions"] == 1536
    in_call = client.find_call("in", "chunk_embeddings")
    assert in_call == {"op": "in", "table": "chunk_embeddings", "col": "chunk_id", "values": ["c1", "c2"]}


def test_chunks_vector_search_rpc_params_with_formatted_embedding():
    client = _FakeSupabaseClient().queue([{"chunk_id": "c1", "content": "hi"}])
    repo = ChunksRepository(client)

    out = repo.vector_search([0.4, 0.5], match_count=6, user_id="alice")

    assert out == [{"chunk_id": "c1", "content": "hi"}]
    rpc = client.find_call("rpc")
    assert rpc is not None
    assert rpc["name"] == "rag_vector_search"
    assert rpc["params"]["p_embedding"] == "[0.4,0.5]"
    assert rpc["params"]["p_match_count"] == 6
    assert rpc["params"]["p_user_id"] == "alice"


def test_chunks_fts_search_rpc_params():
    client = _FakeSupabaseClient().queue([{"chunk_id": "c2"}])
    repo = ChunksRepository(client)

    out = repo.fts_search("hello world", match_count=3)

    assert out == [{"chunk_id": "c2"}]
    rpc = client.find_call("rpc")
    assert rpc["name"] == "rag_fts_search"
    assert rpc["params"]["p_query"] == "hello world"
    assert "p_user_id" not in rpc["params"]


def test_chunks_get_chunk_with_doc_flattens_documents_key():
    client = _FakeSupabaseClient().queue(
        [
            {
                "id": "c1",
                "content": "body",
                "documents": {"title": "Doc", "source_uri": "u", "source_type": "text"},
            }
        ]
    )
    repo = ChunksRepository(client)

    row = repo.get_chunk_with_doc("c1")

    assert row is not None
    assert row["document"] == {"title": "Doc", "source_uri": "u", "source_type": "text"}
    assert "documents" not in row  # popped into "document"
    select_call = client.find_call("select", "document_chunks")
    assert "documents(title, source_uri, source_type)" in select_call["cols"]


# --------------------------------------------------------------------------- #
# TracesRepository
# --------------------------------------------------------------------------- #


def test_traces_create_run_upserts_row_and_returns_first():
    client = _FakeSupabaseClient().queue([{"id": "r1", "status": "running"}])
    repo = TracesRepository(client)

    row = repo.create_run(run_id="r1", user_id="alice", question="why?")

    assert row == {"id": "r1", "status": "running"}
    call = client.find_call("upsert", "agent_runs")
    assert call["on_conflict"] == "id"
    assert call["payload"] == {
        "id": "r1",
        "user_id": "alice",
        "question": "why?",
        "status": "running",
    }


def test_traces_finish_run_only_includes_provided_fields():
    client = _FakeSupabaseClient().queue([{"id": "r1"}])
    repo = TracesRepository(client)

    repo.finish_run("r1", status="finished")

    update_call = client.find_call("update", "agent_runs")
    assert update_call is not None
    assert update_call["fields"] == {"status": "finished"}


def test_traces_finish_run_persists_answer_contract_and_verification_audit():
    client = _FakeSupabaseClient().queue([{"id": "r1"}])
    repo = TracesRepository(client)

    repo.finish_run(
        "r1",
        status="finished",
        final_answer="Use SSO.",
        answer_payload={"status": "grounded", "answer": "Use SSO.", "claims": [], "citations": []},
        verification={"status": "verified", "cited_evidence_count": 1},
    )

    fields = client.find_call("update", "agent_runs")["fields"]
    assert fields["answer_payload"]["answer"] == "Use SSO."
    assert fields["verification"]["status"] == "verified"


def test_traces_store_prompt_and_rewrite_provenance_when_provided():
    client = _FakeSupabaseClient().queue([{"id": "r1"}]).queue([{"id": "s1"}]).queue([{"id": "t1"}])
    repo = TracesRepository(client)

    repo.create_run(
        run_id="r1",
        question="why?",
        prompt_context={"prompts": [{"id": "system_react", "sha256": "abc"}]},
    )
    repo.add_step(
        run_id="r1",
        step_index=0,
        tool_name="rag_search",
        tool_args={"query": "auth decision"},
        original_tool_args={"query": "What did we decide about auth?"},
        prompt_context={"status": "applied"},
    )
    repo.add_tool_call(
        run_id="r1",
        step_index=0,
        tool_name="rag_search",
        arguments={"query": "auth decision"},
        original_arguments={"query": "What did we decide about auth?"},
        prompt_context={"status": "applied"},
    )

    run_row = client.find_call("upsert", "agent_runs")["payload"]
    step_row = client.find_call("upsert", "agent_steps")["payload"]
    call_row = client.find_call("upsert", "tool_calls")["payload"]
    assert run_row["prompt_context"]["prompts"][0]["id"] == "system_react"
    assert step_row["id"]
    assert call_row["id"]
    assert step_row["original_tool_args"]["query"].startswith("What did")
    assert call_row["original_arguments"]["query"].startswith("What did")


# --------------------------------------------------------------------------- #
# MemoryRepository
# --------------------------------------------------------------------------- #


def test_memory_create_with_embedding_formats_vector():
    client = _FakeSupabaseClient().queue([{"id": "m1"}])
    repo = MemoryRepository(client)

    row = repo.create(content="remember this", embedding=[0.1, 0.2], scope="short")

    assert row == {"id": "m1"}
    call = client.find_call("insert", "task_memory")
    assert call["row"]["embedding"] == "[0.1,0.2]"
    assert call["row"]["scope"] == "short"


def test_memory_search_rpc_params():
    client = _FakeSupabaseClient().queue([{"id": "m1", "content": "x"}])
    repo = MemoryRepository(client)

    out = repo.search([0.7, 0.8], match_count=5, user_id="alice", scope="short")

    assert out == [{"id": "m1", "content": "x"}]
    rpc = client.find_call("rpc")
    assert rpc["name"] == "memory_vector_search"
    assert rpc["params"]["p_embedding"] == "[0.7,0.8]"
    assert rpc["params"]["p_user_id"] == "alice"
    assert rpc["params"]["p_scope"] == "short"


# --------------------------------------------------------------------------- #
# PermissionsRepository
# --------------------------------------------------------------------------- #


def test_permissions_upsert_uses_on_conflict_tool_name():
    client = _FakeSupabaseClient().queue([{"tool_name": "echo"}])
    repo = PermissionsRepository(client)

    row = repo.upsert(tool_name="echo", permission="ask", constraints={"max_top_k": 5})

    assert row == {"tool_name": "echo"}
    call = client.find_call("upsert", "tool_permissions")
    assert call is not None
    assert call["on_conflict"] == "tool_name"
    assert call["payload"]["permission"] == "ask"
    assert call["payload"]["constraints"] == {"max_top_k": 5}
