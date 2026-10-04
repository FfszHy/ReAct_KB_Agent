from __future__ import annotations

from types import SimpleNamespace

from pkb_agent.agent.verification import extract_evidence
from pkb_agent.tools.base import ToolContext
from pkb_agent.tools.builtin.rag_read import RagReadTool


class _FakeChunksRepository:
    def __init__(self) -> None:
        self.chunk_calls: list[str] = []
        self.document_calls: list[str] = []

    def get_chunk_with_doc(self, chunk_id: str):
        self.chunk_calls.append(chunk_id)
        return {
            "id": chunk_id,
            "document_id": "document-1",
            "chunk_index": 0,
            "content": "The stored evidence.",
            "document": {"title": "Fixture", "source_uri": "eval://fixture"},
        }

    def list_by_document(self, document_id: str):
        self.document_calls.append(document_id)
        return []


async def test_rag_read_normalizes_a_runtime_citation_id_to_its_raw_chunk_id():
    repo = _FakeChunksRepository()
    ctx = ToolContext(
        settings=SimpleNamespace(),
        supabase=None,
        trace=None,
        repositories={"chunks": repo},
    )

    result = await RagReadTool().execute(ctx, {"chunk_id": " kb:chunk-1 "})

    assert result.ok is True
    assert repo.chunk_calls == ["chunk-1"]
    assert result.data["chunk_id"] == "chunk-1"
    evidence = extract_evidence("rag_read", result.data, settings=ctx.settings)[0]
    assert evidence.title == "Fixture"
    assert evidence.locator == "eval://fixture"


async def test_rag_read_rejects_an_empty_citation_id_before_storage():
    repo = _FakeChunksRepository()
    ctx = ToolContext(
        settings=SimpleNamespace(),
        supabase=None,
        trace=None,
        repositories={"chunks": repo},
    )

    result = await RagReadTool().execute(ctx, {"chunk_id": "kb:"})

    assert result.ok is False
    assert result.error == "provide either chunk_id or document_id"
    assert repo.chunk_calls == []


async def test_rag_read_recovers_when_a_chunk_id_is_sent_as_document_id():
    repo = _FakeChunksRepository()
    ctx = ToolContext(
        settings=SimpleNamespace(),
        supabase=None,
        trace=None,
        repositories={"chunks": repo},
    )

    result = await RagReadTool().execute(ctx, {"document_id": "kb:chunk-1"})

    assert result.ok is True
    assert repo.document_calls == ["chunk-1"]
    assert repo.chunk_calls == ["chunk-1"]
    assert result.data["kind"] == "chunk"
    assert result.data["chunk_id"] == "chunk-1"


async def test_direct_document_read_preserves_each_chunks_source_without_a_search():
    class DocumentRepository(_FakeChunksRepository):
        def list_by_document(self, document_id: str):
            return [
                {
                    "id": f"chunk-{index}",
                    "document_id": document_id,
                    "chunk_index": index,
                    "content": f"Stored evidence {index}.",
                    "document": {"title": "Fixture", "source_uri": "eval://fixture"},
                }
                for index in range(2)
            ]

    ctx = ToolContext(
        settings=SimpleNamespace(),
        supabase=None,
        trace=None,
        repositories={"chunks": DocumentRepository()},
    )

    result = await RagReadTool().execute(ctx, {"document_id": "document-1"})
    evidence = extract_evidence("rag_read", result.data, settings=ctx.settings)

    assert result.ok is True
    assert len(evidence) == 2
    assert {item.title for item in evidence} == {"Fixture"}
    assert {item.locator for item in evidence} == {"eval://fixture"}
    assert {item.metadata["document_id"] for item in evidence} == {"document-1"}


async def test_read_does_not_invent_missing_source_metadata():
    class MissingMetadataRepository(_FakeChunksRepository):
        def get_chunk_with_doc(self, chunk_id: str):
            return {"id": chunk_id, "document_id": "document-1", "content": "Evidence."}

        def list_by_document(self, document_id: str):
            return [self.get_chunk_with_doc("chunk-1")]

    ctx = ToolContext(
        settings=SimpleNamespace(),
        supabase=None,
        trace=None,
        repositories={"chunks": MissingMetadataRepository()},
    )
    for arguments in ({"chunk_id": "chunk-1"}, {"document_id": "document-1"}):
        result = await RagReadTool().execute(ctx, arguments)
        evidence = extract_evidence("rag_read", result.data, settings=ctx.settings)[0]
        assert evidence.title is None
        assert evidence.locator == "document-1"
