from __future__ import annotations

from types import SimpleNamespace

from pkb_agent.tools.base import ToolContext
from pkb_agent.tools.builtin import build_builtin_tools
from pkb_agent.tools.builtin.rag_list_documents import RagListDocumentsTool


class _FakeDocumentsRepository:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str | None, int, int]] = []

    def list(self, user_id: str | None, limit: int, offset: int):
        self.calls.append((user_id, limit, offset))
        return self.rows


def test_rag_list_documents_is_registered_as_a_builtin_tool():
    assert "rag_list_documents" in [tool.name for tool in build_builtin_tools()]


async def test_rag_list_documents_lists_only_the_active_users_catalog_page():
    repo = _FakeDocumentsRepository(
        [
            {
                "id": "doc-1",
                "title": "项目计划",
                "source_uri": "file://project-plan.md",
                "source_type": "file",
                "chunk_count": 3,
                "char_count": 1200,
                "created_at": "2026-08-11T00:00:00+00:00",
                "meta": {"should_not": "leak"},
            },
            {"id": "doc-2", "title": "会议纪要", "chunk_count": 1},
        ]
    )
    ctx = ToolContext(
        settings=SimpleNamespace(),
        supabase=None,
        trace=None,
        repositories={"documents": repo},
        user_id="alice",
    )

    result = await RagListDocumentsTool().execute(ctx, {"limit": 2, "offset": 4})

    assert result.ok is True
    assert repo.calls == [("alice", 2, 4)]
    assert result.data == {
        "count": 2,
        "offset": 4,
        "next_offset": 6,
        "items": [
            {
                "document_id": "doc-1",
                "title": "项目计划",
                "source_uri": "file://project-plan.md",
                "source_type": "file",
                "chunk_count": 3,
                "char_count": 1200,
                "created_at": "2026-08-11T00:00:00+00:00",
            },
            {"document_id": "doc-2", "title": "会议纪要", "chunk_count": 1},
        ],
        "note": "document catalog page",
    }


async def test_rag_list_documents_rejects_an_oversized_page_before_querying_storage():
    repo = _FakeDocumentsRepository([])
    ctx = ToolContext(
        settings=SimpleNamespace(),
        supabase=None,
        trace=None,
        repositories={"documents": repo},
        user_id="alice",
    )

    result = await RagListDocumentsTool().execute(ctx, {"limit": 21})

    assert result.ok is False
    assert result.error == "pagination value must be between 1 and 20"
    assert repo.calls == []
