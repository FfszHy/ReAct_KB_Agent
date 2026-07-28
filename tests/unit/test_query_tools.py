from __future__ import annotations

from types import SimpleNamespace

from pkb_agent.rag.retriever import SearchHit
from pkb_agent.tools.base import ToolContext
from pkb_agent.tools.builtin.rag_search import RagSearchTool
from pkb_agent.tools.builtin.web_search import WebSearchTool
from pkb_agent.web.search_provider import SearchResult


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str, *, top_k: int, user_id: str | None):
        self.calls.append(query)
        common = SearchHit(
            chunk_id="common",
            document_id="d1",
            chunk_index=0,
            content="common",
            meta={},
            score=0.5,
        )
        unique = SearchHit(
            chunk_id=f"{query}-id",
            document_id="d1",
            chunk_index=1,
            content=query,
            meta={},
            score=0.9 if query == "auth decision" else 0.8,
        )
        return [common, unique]


class _FakeSearchProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str, *, max_results: int):
        self.calls.append(query)
        return [
            SearchResult(title="common", url="https://example.com/common", snippet="x", source="fake"),
            SearchResult(
                title=query,
                url=f"https://example.com/{query.replace(' ', '-')}",
                snippet=query,
                source="fake",
            ),
        ]


async def test_rag_search_executes_rewritten_queries_and_merges_duplicate_chunks():
    retriever = _FakeRetriever()
    ctx = ToolContext(
        settings=SimpleNamespace(rag_top_k=2),
        supabase=None,
        trace=None,
        services={"retriever": retriever},
        user_id="alice",
    )

    result = await RagSearchTool().execute(
        ctx,
        {"query": "auth decision", "queries": ["auth decision", "sso design"], "top_k": 2},
    )

    assert result.ok is True
    assert retriever.calls == ["auth decision", "sso design"]
    assert result.data["queries"] == ["auth decision", "sso design"]
    assert [item["chunk_id"] for item in result.data["results"]] == [
        "auth decision-id",
        "sso design-id",
    ]


async def test_web_search_executes_rewritten_queries_and_deduplicates_urls():
    provider = _FakeSearchProvider()
    ctx = ToolContext(
        settings=SimpleNamespace(web_max_results=3, web_search_provider="fake"),
        supabase=None,
        trace=None,
        services={"search_provider": provider},
    )

    result = await WebSearchTool().execute(
        ctx,
        {"query": "auth decision", "queries": ["auth decision", "sso design"], "max_results": 3},
    )

    assert result.ok is True
    assert provider.calls == ["auth decision", "sso design"]
    assert result.data["queries"] == ["auth decision", "sso design"]
    assert result.data["count"] == 3
    assert [item["url"] for item in result.data["results"]] == [
        "https://example.com/common",
        "https://example.com/auth-decision",
        "https://example.com/sso-design",
    ]
