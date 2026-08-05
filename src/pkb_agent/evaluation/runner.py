"""Live evaluation runners and portable JSONL artifacts."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pkb_agent.agent.runtime import AgentRuntime
from pkb_agent.app.settings import Settings
from pkb_agent.evaluation.dataset import EvalCase
from pkb_agent.prompts.query_rewriter import QueryPlanner
from pkb_agent.rag.retriever import Retriever, SearchHit


class RetrievalStrategy(Protocol):
    """A named async retrieval implementation that returns the same hit shape."""

    name: str

    async def retrieve(self, question: str, *, top_k: int) -> RetrievalAttempt: ...


class RetrievalAttempt:
    def __init__(
        self,
        *,
        hits: Iterable[SearchHit] = (),
        queries: Iterable[str] = (),
        known_llm_cost: float | None = None,
        error: str | None = None,
    ) -> None:
        self.hits = tuple(hits)
        self.queries = tuple(queries)
        self.known_llm_cost = known_llm_cost
        self.error = error


class RetrieverStrategy:
    """Expose the production retrieval variants needed for a fair ablation."""

    _MODES: ClassVar[set[str]] = {
        "vector",
        "fts",
        "hybrid",
        "rrf",
        "rewrite_rrf",
        "rewrite_hybrid",
    }

    def __init__(
        self,
        name: str,
        retriever: Retriever,
        *,
        user_id: str,
        planner: QueryPlanner | None = None,
        settings: Settings | None = None,
        candidate_multiplier: int = 3,
    ) -> None:
        if name not in self._MODES:
            raise ValueError(f"unsupported retrieval strategy: {name}")
        if name in {"rewrite_rrf", "rewrite_hybrid"} and planner is None:
            raise ValueError(f"{name} requires a QueryPlanner")
        self.name = name
        self._retriever = retriever
        self._user_id = user_id
        self._planner = planner
        self._settings = settings
        self._candidate_multiplier = max(int(candidate_multiplier), 1)

    async def retrieve(self, question: str, *, top_k: int) -> RetrievalAttempt:
        candidate_k = max(top_k * self._candidate_multiplier, top_k)
        queries: tuple[str, ...] = ()
        known_llm_cost: float | None = None
        try:
            if self.name == "vector":
                hits = await self._retriever.vector_only(question, top_k=candidate_k, user_id=self._user_id)
                return RetrievalAttempt(hits=_unique_document_hits(hits, top_k=top_k), queries=(question,))
            if self.name == "fts":
                hits = await self._retriever.fts_only(question, top_k=candidate_k, user_id=self._user_id)
                return RetrievalAttempt(hits=_unique_document_hits(hits, top_k=top_k), queries=(question,))
            if self.name == "hybrid":
                hits = await self._retriever.weighted_hybrid(
                    question,
                    top_k=candidate_k,
                    user_id=self._user_id,
                )
                return RetrievalAttempt(hits=_unique_document_hits(hits, top_k=top_k), queries=(question,))
            if self.name == "rrf":
                hits = await self._retriever.search(question, top_k=candidate_k, user_id=self._user_id)
                return RetrievalAttempt(hits=_unique_document_hits(hits, top_k=top_k), queries=(question,))

            assert self._planner is not None
            plan = await self._planner.rewrite(
                tool_name="rag_search",
                query=question,
                question=question,
            )
            queries = plan.queries or (question,)
            known_llm_cost = _usage_cost(plan.usage, self._settings)
            search = (
                self._retriever.search
                if self.name == "rewrite_rrf"
                else self._retriever.weighted_hybrid
            )
            batches = await asyncio.gather(
                *(
                    search(query, top_k=candidate_k, user_id=self._user_id)
                    for query in queries
                )
            )
            return RetrievalAttempt(
                hits=_unique_document_hits(_merge_hits(batches, top_k=candidate_k), top_k=top_k),
                queries=queries,
                known_llm_cost=known_llm_cost,
            )
        except Exception as exc:
            return RetrievalAttempt(
                queries=queries,
                known_llm_cost=known_llm_cost,
                error=f"{type(exc).__name__}: {exc}",
            )


async def run_retrieval_evaluation(
    cases: Iterable[EvalCase],
    strategy: RetrievalStrategy,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run one strategy over a fixed suite, retaining raw rankings for review."""
    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        attempt = await strategy.retrieve(case.question, top_k=top_k)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        records.append(
            {
                "schema_version": 1,
                "kind": "retrieval",
                "strategy": strategy.name,
                "case_id": case.id,
                "split": case.split,
                "question": case.question,
                "answerable": case.answerable,
                "top_k": top_k,
                "queries": list(attempt.queries),
                "retrieved_document_keys": _document_keys_from_hits(attempt.hits),
                "hits": [_serialize_hit(hit) for hit in attempt.hits],
                "duration_ms": duration_ms,
                "known_llm_cost": attempt.known_llm_cost,
                "error": attempt.error,
                "success": attempt.error is None,
            }
        )
    return records


async def run_agent_evaluation(
    cases: Iterable[EvalCase],
    runtime: AgentRuntime,
    *,
    strategy_name: str = "agent",
) -> list[dict[str, Any]]:
    """Execute end-to-end AgentRuntime cases and preserve trace-derived labels."""
    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            state = await runtime.run(case.question)
        except Exception as exc:
            records.append(
                {
                    "schema_version": 1,
                    "kind": "agent",
                    "strategy": strategy_name,
                    "case_id": case.id,
                    "split": case.split,
                    "question": case.question,
                    "answerable": case.answerable,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "known_llm_cost": None,
                    "tools": [],
                    "tool_errors": [],
                    "cited_document_keys": [],
                    "citations": [],
                    "answer_status": "",
                    "verification_status": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "success": False,
                }
            )
            continue

        payload = state.answer_payload if isinstance(state.answer_payload, dict) else {}
        citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
        citation_rows = [item for item in citations if isinstance(item, dict)]
        records.append(
            {
                "schema_version": 1,
                "kind": "agent",
                "strategy": strategy_name,
                "case_id": case.id,
                "split": case.split,
                "question": case.question,
                "answerable": case.answerable,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "known_llm_cost": _float_or_none(state.usage.get("actual_cost")),
                "tools": [step.tool_call.name for step in state.steps if step.tool_call is not None],
                "tool_errors": [step.error for step in state.steps if isinstance(step.error, str)],
                "cited_document_keys": _document_keys_from_citations(citation_rows),
                "citations": [_serialize_citation(item) for item in citation_rows],
                "answer_status": str(payload.get("status") or ""),
                "verification_status": str(state.verification.get("status") or ""),
                "answer": payload.get("answer"),
                "claims": payload.get("claims") if isinstance(payload.get("claims"), list) else [],
                "error": state.error,
                "success": bool(state.metrics().get("run_succeeded")),
            }
        )
    return records


def write_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write raw results as JSONL so experiments can be re-scored later."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
    return output


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(source.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid run JSONL at {source}:{line_no}: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"run JSONL row must be an object: {source}:{line_no}")
        records.append(item)
    return records


def make_audit_template(cases: Iterable[EvalCase], records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create blank, human-reviewable answer/citation judgments for agent runs."""
    by_id = {case.id: case for case in cases}
    template: list[dict[str, Any]] = []
    for record in records:
        if record.get("kind") != "agent":
            continue
        case = by_id.get(str(record.get("case_id") or ""))
        if case is None:
            continue
        citations = record.get("citations")
        citation_rows = citations if isinstance(citations, list) else []
        template.append(
            {
                "case_id": case.id,
                "citation_support": [
                    {
                        "citation_id": item.get("id"),
                        "document_key": item.get("document_key"),
                        "supports_claim": None,
                    }
                    for item in citation_rows
                    if isinstance(item, Mapping)
                ],
                "fact_support": [
                    {"fact_id": fact.id, "fact": fact.text, "supported": None}
                    for fact in case.key_facts
                ],
                "factually_consistent": None,
                "reviewer_note": "",
            }
        )
    return template


def attach_audits(
    records: Iterable[Mapping[str, Any]], audits: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join reviewer labels by case ID without changing original raw artifacts."""
    by_id: dict[str, dict[str, Any]] = {}
    for audit in audits:
        case_id = audit.get("case_id")
        if isinstance(case_id, str) and case_id.strip():
            by_id[case_id.strip()] = dict(audit)
    merged: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        audit = by_id.get(str(item.get("case_id") or ""))
        if audit is not None:
            item["audit"] = audit
        merged.append(item)
    return merged


def document_key_from_uri(value: Any) -> str | None:
    """Map evaluator corpus URIs to stable labels; ignore all other sources."""
    if not isinstance(value, str) or not value.startswith("eval://"):
        return None
    remainder = value[len("eval://") :]
    corpus, separator, key = remainder.partition("/")
    if not corpus or not separator or not key:
        return None
    return key.strip() or None


def _document_keys_from_hits(hits: Iterable[SearchHit]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        key = document_key_from_uri(hit.source_uri)
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _document_keys_from_citations(citations: Iterable[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        key = document_key_from_uri(citation.get("locator"))
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _serialize_hit(hit: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "document_key": document_key_from_uri(hit.source_uri),
        "chunk_index": hit.chunk_index,
        "title": hit.doc_title,
        "source_uri": hit.source_uri,
        "score": hit.score,
        "vector_score": hit.vector_score,
        "fts_score": hit.fts_score,
    }


def _serialize_citation(citation: Mapping[str, Any]) -> dict[str, Any]:
    locator = citation.get("locator")
    return {
        "id": citation.get("id"),
        "document_key": document_key_from_uri(locator),
        "locator": locator,
        "title": citation.get("title"),
    }


def _merge_hits(batches: Iterable[Iterable[SearchHit]], *, top_k: int) -> list[SearchHit]:
    best_by_chunk: dict[str, SearchHit] = {}
    for batch in batches:
        for hit in batch:
            previous = best_by_chunk.get(hit.chunk_id)
            if previous is None or hit.score > previous.score:
                best_by_chunk[hit.chunk_id] = hit
    return sorted(best_by_chunk.values(), key=lambda hit: hit.score, reverse=True)[:top_k]


def _unique_document_hits(hits: Iterable[SearchHit], *, top_k: int) -> list[SearchHit]:
    """Keep the first ranked chunk per source for document-level metrics."""
    result: list[SearchHit] = []
    seen: set[str] = set()
    for hit in hits:
        key = hit.document_id or hit.source_uri or hit.chunk_id
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)
        if len(result) >= top_k:
            break
    return result


def _usage_cost(usage: Mapping[str, Any] | None, settings: Settings | None) -> float | None:
    if not usage or settings is None:
        return None
    try:
        prompt = max(int(usage.get("prompt_tokens", 0)), 0)
        completion = max(int(usage.get("completion_tokens", 0)), 0)
        hit = max(int(usage.get("prompt_cache_hit_tokens", 0)), 0)
        miss = max(int(usage.get("prompt_cache_miss_tokens", 0)), 0)
    except (TypeError, ValueError):
        return None
    miss += max(prompt - hit - miss, 0)
    miss_rate = (
        settings.observability_input_token_cost_per_million
        if settings.observability_input_token_cost_per_million is not None
        else settings.observability_cache_miss_input_token_cost_per_million
    )
    return round(
        (
            hit * settings.observability_cache_hit_input_token_cost_per_million
            + miss * miss_rate
            + completion * settings.observability_output_token_cost_per_million
        )
        / 1_000_000,
        8,
    )


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
