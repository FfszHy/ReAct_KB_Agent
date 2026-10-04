from __future__ import annotations

import json
from types import SimpleNamespace

from pkb_agent.agent.runtime import _attach_citation_evidence, _data_observation
from pkb_agent.agent.verification import extract_evidence
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
        "truncated": False,
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


class _PagedDocumentsRepository(_FakeDocumentsRepository):
    def list(self, user_id, limit, offset):
        self.calls.append((user_id, limit, offset))
        return self.rows[offset:offset + limit]


def _runtime_observation(result, ctx):
    records = extract_evidence("rag_list_documents", result.data, settings=ctx.settings)
    payload = _attach_citation_evidence(result.data, records)
    text = _data_observation(payload, max_chars=ctx.settings.agent_tool_result_max_chars)
    # This is the actual runtime wrapper, including its citation descriptors.
    # A clipped observation would no longer parse as a complete page.
    assert "[truncated]" not in text
    assert len(text) <= ctx.settings.agent_tool_result_max_chars
    return json.loads(text)


async def test_budget_reduces_page_size_without_skipping_rows_or_losing_the_tail():
    rows = [
        {
            "id": f"document-{index}",
            "title": f"文档 {index} " + "长标题" * 180,
            "source_uri": f"https://example.com/long-document-{index}.md",
            "source_type": "text",
            "chunk_count": 12,
            "char_count": 4500,
            "created_at": "2026-10-04T00:00:00Z",
        }
        for index in range(41)
    ]
    repo = _PagedDocumentsRepository(rows)
    ctx = ToolContext(
        settings=SimpleNamespace(agent_tool_result_max_chars=6000),
        supabase=None, trace=None, repositories={"documents": repo}, user_id="alice",
    )
    offset = 0
    observed_items = []
    pages = []
    for _ in range(len(rows) + 1):
        result = await RagListDocumentsTool().execute(ctx, {"limit": 20, "offset": offset})
        assert result.ok
        page = _runtime_observation(result, ctx)
        pages.append(page)
        observed_items.extend(page["items"])
        assert page["count"] == len(page["items"])
        assert result.truncated is False
        if page["next_offset"] is None:
            break
        assert page["next_offset"] == offset + page["count"]
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    else:
        raise AssertionError("catalog cursor did not reach its tail")

    assert 0 < pages[0]["count"] < 20
    assert pages[-1]["next_offset"] is None
    assert [item["document_id"] for item in observed_items] == [row["id"] for row in rows]
    assert [item["title"] for item in observed_items] == [row["title"] for row in rows]
    assert all(limit == 20 for _, limit, _ in repo.calls)


async def test_one_extreme_title_is_marked_shortened_and_cursor_still_advances():
    rows = [
        {"id": "first-id", "title": "很长" * 10_000, "source_uri": "https://example.com/" + "x" * 10_000},
        {"id": "second-id", "title": "Tail document " + "x" * 400},
    ]
    repo = _PagedDocumentsRepository(rows)
    ctx = ToolContext(
        settings=SimpleNamespace(agent_tool_result_max_chars=1000),
        supabase=None, trace=None, repositories={"documents": repo}, user_id="alice",
    )

    first = await RagListDocumentsTool().execute(ctx, {"limit": 20, "offset": 0})
    page = _runtime_observation(first, ctx)

    assert first.truncated is True
    assert page["truncated"] is True
    assert page["items"][0]["truncated"] is True
    assert page["items"][0]["document_id"] == "first-id"
    assert page["next_offset"] == 1
    tail = await RagListDocumentsTool().execute(ctx, {"limit": 20, "offset": page["next_offset"]})
    tail_page = _runtime_observation(tail, ctx)
    assert tail_page["items"][0]["document_id"] == "second-id"
    assert tail_page["next_offset"] is None
    assert tail.truncated is False


async def test_impossibly_small_budget_reports_a_configuration_error():
    repo = _PagedDocumentsRepository([{"id": "document-id", "title": "Title"}])
    ctx = ToolContext(
        settings=SimpleNamespace(agent_tool_result_max_chars=100),
        supabase=None, trace=None, repositories={"documents": repo}, user_id="alice",
    )

    result = await RagListDocumentsTool().execute(ctx, {"limit": 20})

    assert result.ok is False
    assert "increase agent_tool_result_max_chars before retrying" in result.error
    assert len(repo.calls) == 1
