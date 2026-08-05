from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from pkb_agent.app.settings import Settings
from pkb_agent.evaluation import corpus as corpus_module
from pkb_agent.evaluation.corpus import CorpusDocument, CorpusManifest
from pkb_agent.evaluation.dataset import (
    BenchmarkMeta,
    EvalCase,
    EvalDataset,
    KeyFact,
    RelevantDocument,
)
from pkb_agent.evaluation.metrics import percentile, summarize_records
from pkb_agent.evaluation.report import write_report
from pkb_agent.evaluation.runner import (
    RetrieverStrategy,
    attach_audits,
    document_key_from_uri,
    make_audit_template,
)
from pkb_agent.prompts.query_rewriter import QueryPlan
from pkb_agent.rag.retriever import SearchHit


def _dataset() -> EvalDataset:
    cases = (
        EvalCase(
            id="a",
            question="a?",
            answerable=True,
            relevant_documents=(RelevantDocument("doc-a", 2), RelevantDocument("doc-b", 1)),
            key_facts=(KeyFact("fact-a", "A fact", ("doc-a",)),),
        ),
        EvalCase(
            id="b",
            question="b?",
            answerable=True,
            relevant_documents=(RelevantDocument("doc-c", 2),),
        ),
        EvalCase(id="negative", question="negative?", answerable=False),
    )
    return EvalDataset(
        root=Path("."),
        meta=BenchmarkMeta("unit", "Unit benchmark", "1", "corpus.json", default_top_k=2),
        cases=cases,
    )


def test_checked_in_fastapi_dataset_has_frozen_50_30_split_and_valid_document_keys():
    dataset = EvalDataset.load("data/evals/fastapi-0.115")
    manifest = CorpusManifest.load(dataset.root / dataset.meta.corpus_manifest)
    keys = {document.key for document in manifest.documents}

    assert len(dataset.for_split("dev")) == 50
    assert len(dataset.for_split("test")) == 30
    assert sum(not case.answerable for case in dataset.cases) == 8
    assert {
        relevant.key
        for case in dataset.cases
        for relevant in case.relevant_documents
    }.issubset(keys)


def test_retrieval_metrics_collapse_duplicate_documents_and_calculate_aggregate_values():
    dataset = _dataset()
    records = [
        {
            "kind": "retrieval",
            "strategy": "rrf",
            "case_id": "a",
            "retrieved_document_keys": ["doc-a", "doc-a", "noise"],
            "duration_ms": 10,
            "success": True,
        },
        {
            "kind": "retrieval",
            "strategy": "rrf",
            "case_id": "b",
            "retrieved_document_keys": ["noise", "doc-c"],
            "duration_ms": 30,
            "success": True,
        },
        {
            "kind": "retrieval",
            "strategy": "rrf",
            "case_id": "negative",
            "retrieved_document_keys": [],
            "duration_ms": 50,
            "success": True,
        },
    ]

    metrics = summarize_records(dataset, records)["strategies"]["rrf"]

    assert metrics["retrieval"]["evaluated_cases"] == 2
    assert metrics["retrieval"]["recall_at_k"] == 0.75
    assert metrics["retrieval"]["mrr_at_k"] == 0.75
    assert metrics["engineering"]["p50_latency_ms"] == 30.0
    assert metrics["engineering"]["p95_latency_ms"] == 50.0


def test_answer_metrics_keep_automatic_alignment_separate_from_human_audit():
    dataset = _dataset()
    records = [
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "a",
            "cited_document_keys": ["doc-a", "noise"],
            "answer_status": "grounded",
            "verification_status": "verified",
            "duration_ms": 10,
            "known_llm_cost": 0.01,
            "tools": ["rag_search"],
            "tool_errors": [],
            "success": True,
            "audit": {
                "citation_support": [{"supported": True}, {"supported": False}],
                "fact_support": [{"supported": True}],
                "factually_consistent": True,
            },
        },
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "negative",
            "cited_document_keys": [],
            "answer_status": "insufficient_evidence",
            "verification_status": "refused",
            "duration_ms": 20,
            "known_llm_cost": 0.02,
            "tools": ["rag_search"],
            "tool_errors": [],
            "success": False,
        },
    ]

    answer = summarize_records(dataset, records)["strategies"]["agent"]["answer"]

    assert answer["automatic"]["citation_alignment_precision"] == 0.5
    assert answer["automatic"]["evidence_document_coverage"] == 0.5
    assert answer["automatic"]["refusal_correctness"] == 1.0
    assert answer["human_audit"]["citation_precision"] == 0.5
    assert answer["human_audit"]["key_fact_coverage"] == 1.0
    assert answer["human_audit"]["fact_consistency"] == 1.0


def test_report_and_audit_artifacts_are_portable(tmp_path: Path):
    dataset = _dataset()
    records = [
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "a",
            "citations": [{"id": "kb:1", "document_key": "doc-a"}],
            "cited_document_keys": ["doc-a"],
            "answer_status": "grounded",
            "verification_status": "verified",
            "duration_ms": 12,
            "known_llm_cost": 0.01,
            "tools": [],
            "tool_errors": [],
            "success": True,
        }
    ]
    template = make_audit_template(dataset.cases, records)
    merged = attach_audits(records, [{"case_id": "a", "factually_consistent": True}])
    paths = write_report(tmp_path / "report", summarize_records(dataset, merged))

    assert template[0]["citation_support"][0]["supports_claim"] is None
    assert merged[0]["audit"]["factually_consistent"] is True
    assert all(path.exists() for path in paths.values())
    assert "Retrieval ablation" in paths["chart"].read_text("utf-8")
    assert json.loads(paths["summary"].read_text("utf-8"))["benchmark"]["id"] == "unit"


@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [([1, 2, 3, 4], 50, 2.0), ([1, 2, 3, 4], 95, 4.0), ([], 50, None)],
)
def test_percentile_nearest_rank(values, pct, expected):
    assert percentile(values, pct) == expected


def test_document_key_from_eval_uri_only_accepts_stable_eval_sources():
    assert document_key_from_uri("eval://fastapi-0.115/tutorial/body") == "tutorial/body"
    assert document_key_from_uri("https://example.com/doc") is None


async def test_corpus_fetch_retries_a_transient_read_timeout(monkeypatch):
    url = "https://example.test/source.md"

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def get(self, requested_url: str):
            self.calls += 1
            request = httpx.Request("GET", requested_url)
            if self.calls == 1:
                raise httpx.ReadTimeout("slow source", request=request)
            return httpx.Response(200, text="verified text", request=request)

    client = Client()
    sleep = AsyncMock()
    monkeypatch.setattr(corpus_module.asyncio, "sleep", sleep)

    fetched = await corpus_module._fetch_one(
        client,  # type: ignore[arg-type]
        CorpusDocument(key="source", title="Source", source_url=url),
        max_attempts=3,
    )

    assert fetched.text == "verified text"
    assert client.calls == 2
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.parametrize(
    ("strategy_name", "method_name"),
    [("rewrite_rrf", "search"), ("rewrite_hybrid", "weighted_hybrid")],
)
async def test_rewrite_strategy_retains_completed_rewrite_cost_when_retrieval_fails(
    strategy_name: str, method_name: str
):
    class FailingRetriever:
        def __init__(self) -> None:
            self.called: list[str] = []

        async def search(self, *_args, **_kwargs):
            self.called.append("search")
            raise RuntimeError("database unavailable")

        async def weighted_hybrid(self, *_args, **_kwargs):
            self.called.append("weighted_hybrid")
            raise RuntimeError("database unavailable")

    class Planner:
        async def rewrite(self, **_kwargs):
            return QueryPlan(
                original_query="question",
                queries=("keywords",),
                status="applied",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

    settings = Settings(
        observability_cache_miss_input_token_cost_per_million=1.0,
        observability_output_token_cost_per_million=2.0,
    )
    retriever = FailingRetriever()
    strategy = RetrieverStrategy(
        strategy_name,
        retriever,  # type: ignore[arg-type]
        user_id="eval",
        planner=Planner(),  # type: ignore[arg-type]
        settings=settings,
    )

    attempt = await strategy.retrieve("question", top_k=6)

    assert attempt.error == "RuntimeError: database unavailable"
    assert attempt.queries == ("keywords",)
    assert attempt.known_llm_cost == pytest.approx(0.00002)
    assert retriever.called == [method_name]


async def test_rewrite_hybrid_uses_weighted_hybrid_for_each_rewritten_query():
    class HybridRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def weighted_hybrid(self, query: str, **_kwargs):
            self.queries.append(query)
            if query == "first rewrite":
                return [
                    SearchHit(
                        "chunk-a",
                        "doc-a",
                        0,
                        "",
                        {},
                        source_uri="eval://unit/doc-a",
                        score=0.8,
                    )
                ]
            return [
                SearchHit(
                    "chunk-b",
                    "doc-b",
                    0,
                    "",
                    {},
                    source_uri="eval://unit/doc-b",
                    score=0.9,
                )
            ]

        async def search(self, *_args, **_kwargs):
            raise AssertionError("rewrite_hybrid must not use RRF search")

    class Planner:
        async def rewrite(self, **_kwargs):
            return QueryPlan(
                original_query="question",
                queries=("first rewrite", "second rewrite"),
                status="applied",
                usage=None,
            )

    retriever = HybridRetriever()
    strategy = RetrieverStrategy(
        "rewrite_hybrid",
        retriever,  # type: ignore[arg-type]
        user_id="eval",
        planner=Planner(),  # type: ignore[arg-type]
    )

    attempt = await strategy.retrieve("question", top_k=6)

    assert set(retriever.queries) == {"first rewrite", "second rewrite"}
    assert attempt.queries == ("first rewrite", "second rewrite")
    assert [hit.document_id for hit in attempt.hits] == ["doc-b", "doc-a"]
