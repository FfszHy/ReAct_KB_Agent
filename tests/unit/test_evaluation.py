from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from pkb_agent.agent.state import AgentRunState, AgentStatus
from pkb_agent.app.settings import Settings
from pkb_agent.evaluation import corpus as corpus_module
from pkb_agent.evaluation.corpus import CorpusDocument, CorpusManifest
from pkb_agent.evaluation.dataset import (
    AgentExpectation,
    BenchmarkMeta,
    EvalCase,
    EvalDataset,
    KeyFact,
    RelevantDocument,
)
from pkb_agent.evaluation.metrics import percentile, summarize_records
from pkb_agent.evaluation.report import write_report
from pkb_agent.evaluation.runner import (
    RetrievalAttempt,
    RetrieverStrategy,
    attach_audits,
    document_key_from_uri,
    make_audit_template,
    run_agent_evaluation,
    run_retrieval_evaluation,
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


def test_checked_in_multidomain_dataset_has_pinned_40_document_frozen_test():
    dataset = EvalDataset.load("data/evals/tech-multidomain-v1")
    manifest = CorpusManifest.load(dataset.root / dataset.meta.corpus_manifest)
    keys = {document.key for document in manifest.documents}
    test_cases = dataset.for_split("test")
    test_tags = {tag for case in test_cases for tag in case.tags}

    assert len(manifest.documents) == 40
    assert all(document.sha256 for document in manifest.documents)
    assert sum(document.key.startswith("pydantic-") for document in manifest.documents) == 10
    assert sum(document.key.startswith("kubernetes-") for document in manifest.documents) == 10
    assert sum(document.key.startswith("sqlalchemy-") for document in manifest.documents) == 10
    assert len(dataset.for_split("dev")) == 80
    assert len(test_cases) == 120
    assert sum(case.answerable for case in test_cases) == 88
    assert sum(not case.answerable for case in test_cases) == 32
    assert {"multi-document", "paraphrase", "term-ambiguity", "version-trap", "permission", "tool-selection"}.issubset(test_tags)
    assert {
        relevant.key
        for case in dataset.cases
        for relevant in case.relevant_documents
    }.issubset(keys)


def test_checked_in_multidomain_v1_1_corrects_the_fastapi_alias_source():
    dataset = EvalDataset.load("data/evals/tech-multidomain-v1.1")
    manifest = CorpusManifest.load(dataset.root / dataset.meta.corpus_manifest)
    test_cases = {case.id: case for case in dataset.for_split("test")}
    query_alias = test_cases["test-query-02"]

    assert dataset.meta.version == "1.1.0"
    assert len(manifest.documents) == 41
    assert len(dataset.for_split("dev")) == 80
    assert len(test_cases) == 120
    assert manifest.source_uri("tutorial/query-params-str-validations") == (
        "eval://tech-multidomain-v1.1/tutorial/query-params-str-validations"
    )
    assert next(
        document.sha256
        for document in manifest.documents
        if document.key == "tutorial/query-params-str-validations"
    ) == "64b083625942724f7269a4f955df186916668d6c2a518467e26686a2905bff37"
    assert query_alias.relevance_by_document == {"tutorial/query-params-str-validations": 2}
    assert query_alias.key_facts[0].source_documents == (
        "tutorial/query-params-str-validations",
    )


def test_checked_in_approved_web_dataset_reuses_pinned_documents_with_external_locators():
    dataset = EvalDataset.load("data/evals/tech-web-approved-v1")
    manifest = CorpusManifest.load(dataset.root / dataset.meta.corpus_manifest)
    keys = {document.key for document in manifest.documents}
    cases = dataset.for_split("test")

    assert len(dataset.for_split("dev")) == 0
    assert len(cases) == 4
    assert all("web-approved" in case.tags for case in cases)
    assert all(case.agent and case.agent.required_all_tools == ("web_fetch",) for case in cases)
    assert {
        document.key
        for case in cases
        for document in case.relevant_documents
    }.issubset(keys)
    assert all(
        document.source_url and document.source_url.startswith("https://raw.githubusercontent.com/")
        for case in cases
        for document in case.relevant_documents
    )


def test_dataset_loads_declared_split_includes(tmp_path: Path):
    included = tmp_path / "included.dev.jsonl"
    included.write_text(
        json.dumps(
            {
                "id": "included-case",
                "question": "What is included?",
                "answerable": True,
                "relevant_documents": [{"key": "doc-a", "grade": 2}],
                "key_facts": [],
            }
        )
        + "\n",
        "utf-8",
    )
    (tmp_path / "benchmark.json").write_text(
        json.dumps(
            {
                "id": "composed",
                "title": "Composed benchmark",
                "version": "1",
                "corpus_manifest": "corpus.json",
                "split_includes": {"dev": ["included.dev.jsonl"]},
            }
        ),
        "utf-8",
    )
    (tmp_path / "questions.dev.jsonl").write_text(
        json.dumps(
            {
                "id": "local-case",
                "question": "What is local?",
                "answerable": False,
                "relevant_documents": [],
                "key_facts": [],
            }
        )
        + "\n",
        "utf-8",
    )

    dataset = EvalDataset.load(tmp_path, splits=("dev",))

    assert [case.id for case in dataset.for_split("dev")] == ["local-case", "included-case"]


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


def test_retrieval_summary_reports_bootstrap_intervals_and_case_outcomes(tmp_path: Path):
    dataset = EvalDataset(
        root=Path("."),
        meta=BenchmarkMeta("unit", "Unit benchmark", "1", "corpus.json", default_top_k=2),
        cases=tuple(
            EvalCase(
                id=case_id,
                question=f"{case_id}?",
                answerable=True,
                relevant_documents=(RelevantDocument(f"doc-{case_id}", 2),),
            )
            for case_id in ("win", "loss", "tie")
        ),
    )
    records = [
        {
            "kind": "retrieval",
            "strategy": strategy,
            "case_id": case_id,
            "retrieved_document_keys": hits,
            "duration_ms": 1,
            "success": True,
        }
        for strategy, result_by_case in {
            "vector": {
                "win": ["noise"],
                "loss": ["doc-loss"],
                "tie": ["doc-tie"],
            },
            "hybrid": {
                "win": ["doc-win"],
                "loss": ["noise"],
                "tie": ["doc-tie"],
            },
        }.items()
        for case_id, hits in result_by_case.items()
    ]
    records.append(
        {
            "kind": "retrieval",
            "strategy": "hybrid",
            "case_id": "win",
            "retrieved_document_keys": ["doc-win"],
            "duration_ms": 1,
            "success": True,
        }
    )

    summary = summarize_records(dataset, records)
    intervals = summary["strategies"]["hybrid"]["retrieval"]["confidence_intervals"]
    comparison = summary["retrieval_comparison"]["comparisons"]["hybrid"]
    report = write_report(tmp_path / "report", summary)["markdown"].read_text("utf-8")

    assert intervals["case_count"] == 3
    assert summary["strategies"]["hybrid"]["retrieval"]["evaluated_records"] == 4
    assert intervals["ndcg_at_k"]["lower"] <= intervals["ndcg_at_k"]["estimate"]
    assert intervals["ndcg_at_k"]["estimate"] <= intervals["ndcg_at_k"]["upper"]
    assert (comparison["wins"], comparison["losses"], comparison["ties"]) == (1, 1, 1)
    assert [item["outcome"] for item in comparison["case_outcomes"]] == ["loss", "tie", "win"]
    assert "Bootstrap uncertainty" in report
    assert "Case-level wins, losses, and ties" in report


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
    assert answer["automatic"]["grounded_answer_rate"] == 1.0
    assert answer["automatic"]["refusal_correctness"] == 1.0
    assert answer["automatic"]["terminal_outcome_accuracy"] == 1.0
    assert answer["automatic"]["answerable_case_count"] == 1
    assert answer["automatic"]["unanswerable_case_count"] == 1
    assert answer["human_audit"]["citation_precision"] == 0.5
    assert answer["human_audit"]["key_fact_coverage"] == 1.0
    assert answer["human_audit"]["fact_consistency"] == 1.0


def test_answer_metrics_keep_refusal_correctness_conditional_on_unanswerable_cases():
    dataset = _dataset()
    records = [
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "a",
            "answer_status": "grounded",
            "verification_status": "verified",
        },
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "negative",
            "answer_status": "grounded",
            "verification_status": "verified",
        },
    ]

    automatic = summarize_records(dataset, records)["strategies"]["agent"]["answer"]["automatic"]

    assert automatic["grounded_answer_rate"] == 1.0
    assert automatic["refusal_correctness"] == 0.0
    assert automatic["terminal_outcome_accuracy"] == 0.5


def test_agent_metrics_score_denied_confirmation_after_selecting_protected_tool():
    dataset = EvalDataset(
        root=Path("."),
        meta=BenchmarkMeta("unit", "Unit benchmark", "1", "corpus.json"),
        cases=(
            EvalCase(
                id="permission",
                question="Fetch the protected page.",
                answerable=False,
                agent=AgentExpectation(
                    required_all_tools=("web_fetch",),
                    expected_outcome="refuse",
                    permission_outcome="confirmation_required",
                ),
            ),
        ),
    )
    records = [
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "permission",
            "tools": ["web_fetch"],
            "tool_errors": ["web_fetch requires confirmation (not granted)"],
            "answer_status": "insufficient_evidence",
            "verification_status": "refused",
            "duration_ms": 1,
            "success": False,
        }
    ]

    agent = summarize_records(dataset, records)["strategies"]["agent"]["agent"]

    assert agent["tool_selection_correctness"] == 1.0
    assert agent["permission_refusal_handling"] == 1.0
    assert agent["task_success_rate"] == 1.0


def test_engineering_metrics_keep_refusal_terminals_out_of_execution_failures():
    dataset = _dataset()
    records = [
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "negative",
            "answer_status": "insufficient_evidence",
            "verification_status": "refused",
            "duration_ms": 10,
            "error": None,
            "execution_success": True,
            "success": False,
        },
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "a",
            "answer_status": "insufficient_evidence",
            "verification_status": "refused",
            "duration_ms": 20,
            "error": "exceeded max_steps=20",
            "execution_success": False,
            "success": False,
        },
    ]

    engineering = summarize_records(dataset, records)["strategies"]["agent"]["engineering"]

    assert engineering["execution_failure_rate"] == 0.5
    assert engineering["refusal_terminal_rate"] == 0.5
    assert engineering["failure_rate"] == 0.5


def test_engineering_metrics_expose_tool_and_nonfatal_trace_failures():
    records = [
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "a",
            "tools": ["rag_search", "rag_read"],
            "tool_errors": ["rag_read failed: temporary timeout"],
            "budget_finalized": True,
            "trace_write_failure_count": 2,
            "duration_ms": 10,
            "execution_success": True,
        },
        {
            "kind": "agent",
            "strategy": "agent",
            "case_id": "negative",
            "tools": ["rag_search"],
            "tool_errors": [],
            "budget_finalized": False,
            "trace_write_failure_count": 0,
            "duration_ms": 20,
            "execution_success": True,
        },
    ]

    engineering = summarize_records(_dataset(), records)["strategies"]["agent"]["engineering"]

    assert engineering["tool_error_run_rate"] == 0.5
    assert engineering["tool_call_failure_rate"] == pytest.approx(1 / 3)
    assert engineering["budget_finalization_rate"] == 0.5
    assert engineering["trace_write_failure_count"] == 2
    assert engineering["trace_write_failure_run_rate"] == 0.5


async def test_agent_runner_marks_normal_refusal_as_an_execution_success():
    state = AgentRunState(question="negative?")
    state.status = AgentStatus.FINISHED
    state.answer_payload = {"answer": "Insufficient evidence.", "status": "insufficient_evidence"}
    state.verification = {"status": "refused"}
    state.mark_finished()

    class RefusingRuntime:
        async def run(self, _question: str) -> AgentRunState:
            return state

    records = await run_agent_evaluation(
        _dataset().cases[2:],
        RefusingRuntime(),  # type: ignore[arg-type]
    )

    assert records[0]["execution_success"] is True
    assert records[0]["success"] is False


async def test_agent_runner_maps_approved_web_citation_to_its_annotated_source_url():
    source_url = "https://raw.githubusercontent.com/example/project/v1/docs/page.md"
    case = EvalCase(
        id="web",
        question="Fetch the approved page.",
        answerable=True,
        relevant_documents=(RelevantDocument("web-page", 2, source_url),),
    )
    state = AgentRunState(question=case.question)
    state.status = AgentStatus.FINISHED
    state.answer_payload = {
        "answer": "The approved page says hello.",
        "status": "grounded",
        "claims": [],
        "citations": [
            {
                "id": "web:fixture",
                "locator": f"{source_url}#section",
                "title": "Fixture page",
            }
        ],
    }
    state.verification = {"status": "verified"}
    state.mark_finished()

    class WebRuntime:
        async def run(self, _question: str) -> AgentRunState:
            return state

    records = await run_agent_evaluation([case], WebRuntime())  # type: ignore[arg-type]

    assert records[0]["cited_document_keys"] == ["web-page"]
    assert records[0]["citations"][0]["document_key"] == "web-page"


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
    assert "Execution failure rate" in paths["markdown"].read_text("utf-8")
    assert "Refusal terminal rate" in paths["markdown"].read_text("utf-8")
    assert "Tool-error runs" in paths["markdown"].read_text("utf-8")
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


async def test_retrieval_runner_reports_per_case_progress():
    class Strategy:
        name = "vector"

        async def retrieve(self, question: str, *, top_k: int) -> RetrievalAttempt:
            return RetrievalAttempt(queries=(question,))

    cases = _dataset().cases[:2]
    progress: list[tuple[int, int, str, bool]] = []

    records = await run_retrieval_evaluation(
        cases,
        Strategy(),  # type: ignore[arg-type]
        top_k=2,
        on_progress=lambda completed, total, record: progress.append(
            (completed, total, str(record["case_id"]), bool(record["success"]))
        ),
    )

    assert len(records) == 2
    assert progress == [(1, 2, "a", True), (2, 2, "b", True)]


async def test_agent_runner_reports_progress_when_a_case_errors():
    class FailingRuntime:
        async def run(self, _question: str):
            raise RuntimeError("offline")

    progress: list[tuple[int, int, str, bool]] = []

    records = await run_agent_evaluation(
        _dataset().cases[:2],
        FailingRuntime(),  # type: ignore[arg-type]
        on_progress=lambda completed, total, record: progress.append(
            (completed, total, str(record["case_id"]), bool(record["success"]))
        ),
    )

    assert len(records) == 2
    assert progress == [(1, 2, "a", False), (2, 2, "b", False)]


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
