"""Transparent metrics for retrieval, grounded answers, agent policy and cost.

The module keeps automatic metrics deliberately conservative.  A cited document
matching an expected source is *alignment*, not proof that every sentence is
true.  Semantic citation support and fact consistency are reported only from
explicit human-audit labels, so the dashboard cannot quietly turn a heuristic
into a quality claim.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from pkb_agent.evaluation.dataset import EvalCase, EvalDataset


def percentile(values: Iterable[float | int], percentile_value: float) -> float | None:
    """Nearest-rank percentile, with a documented and dependency-free rule."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    index = max(math.ceil(percentile_value / 100 * len(ordered)) - 1, 0)
    return ordered[index]


def summarize_records(
    dataset: EvalDataset,
    records: Iterable[Mapping[str, Any]],
    *,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Aggregate portable run records into strategy-by-strategy metrics."""
    k = top_k or dataset.meta.default_top_k
    if k < 1:
        raise ValueError("top_k must be positive")
    cases = {case.id: case for case in dataset.cases}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    recorded_case_ids: set[str] = set()
    recorded_splits: set[str] = set()
    for record in records:
        strategy = str(record.get("strategy") or "unknown")
        grouped[strategy].append(record)
        case_id = record.get("case_id")
        if isinstance(case_id, str) and case_id:
            recorded_case_ids.add(case_id)
        split = record.get("split")
        if isinstance(split, str) and split:
            recorded_splits.add(split)

    strategies: dict[str, Any] = {}
    for name, rows in sorted(grouped.items()):
        retrieval_rows = [row for row in rows if row.get("kind") == "retrieval"]
        agent_rows = [row for row in rows if row.get("kind") == "agent"]
        payload: dict[str, Any] = {
            "record_count": len(rows),
            "engineering": _engineering_metrics(rows),
        }
        if retrieval_rows:
            payload["retrieval"] = _retrieval_metrics(cases, retrieval_rows, k=k)
        if agent_rows:
            payload["answer"] = _answer_metrics(cases, agent_rows)
            payload["agent"] = _agent_metrics(cases, agent_rows)
        strategies[name] = payload
    return {
        "schema_version": 1,
        "benchmark": {
            "id": dataset.meta.id,
            "title": dataset.meta.title,
            "version": dataset.meta.version,
            "language": dataset.meta.language,
            "default_top_k": k,
            "case_count": len(dataset.cases),
            "splits": _split_counts(dataset.cases),
            "recorded_case_count": len(recorded_case_ids),
            "recorded_splits": sorted(recorded_splits),
        },
        "strategies": strategies,
    }


def _retrieval_metrics(
    cases: Mapping[str, EvalCase], rows: Iterable[Mapping[str, Any]], *, k: int) -> dict[str, Any]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    evaluated = 0
    missing_cases = 0
    for row in rows:
        case = cases.get(str(row.get("case_id") or ""))
        if case is None or not case.answerable or not case.relevant_documents:
            if case is None:
                missing_cases += 1
            continue
        expected = case.relevance_by_document
        ranked = _unique_texts(row.get("retrieved_document_keys"))[:k]
        retrieved_expected = set(ranked) & set(expected)
        recalls.append(len(retrieved_expected) / len(expected))
        reciprocal_ranks.append(_reciprocal_rank(ranked, expected))
        ndcgs.append(_ndcg(ranked, expected, k=k))
        evaluated += 1
    return {
        "definition": (
            "Document-level macro metrics; each strategy retrieves a wider chunk candidate pool, "
            "then duplicate chunks from the same source are collapsed before ranking. Recall@K is "
            "the fraction of annotated relevant documents retrieved."
        ),
        "k": k,
        "evaluated_cases": evaluated,
        "skipped_unanswerable_cases": sum(
            1
            for row in rows
            if (case := cases.get(str(row.get("case_id") or ""))) is not None and not case.answerable
        ),
        "unknown_case_records": missing_cases,
        "recall_at_k": _mean(recalls),
        "mrr_at_k": _mean(reciprocal_ranks),
        "ndcg_at_k": _mean(ndcgs),
    }


def _answer_metrics(cases: Mapping[str, EvalCase], rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    alignment_scores: list[float] = []
    document_coverages: list[float] = []
    refusal_correct: list[float] = []
    cited_total = 0
    cited_aligned = 0
    human_citation: list[bool] = []
    human_facts: list[bool] = []
    human_consistency: list[bool] = []
    for row in rows:
        case = cases.get(str(row.get("case_id") or ""))
        if case is None:
            continue
        cited = _unique_texts(row.get("cited_document_keys"))
        expected = set(case.relevance_by_document)
        if case.answerable:
            if cited:
                aligned = sum(key in expected for key in cited)
                cited_total += len(cited)
                cited_aligned += aligned
                alignment_scores.append(aligned / len(cited))
            if expected:
                document_coverages.append(len(set(cited) & expected) / len(expected))
        answer_status = str(row.get("answer_status") or "")
        verification_status = str(row.get("verification_status") or "")
        refusal = answer_status == "insufficient_evidence" or verification_status == "refused"
        correct_refusal_behavior = refusal if not case.answerable else answer_status == "grounded"
        refusal_correct.append(float(correct_refusal_behavior))
        audit = row.get("audit")
        if isinstance(audit, Mapping):
            human_citation.extend(_bool_judgments(audit.get("citation_support"), "supported"))
            human_facts.extend(_bool_judgments(audit.get("fact_support"), "supported"))
            consistent = audit.get("factually_consistent")
            if isinstance(consistent, bool):
                human_consistency.append(consistent)

    return {
        "automatic": {
            "note": (
                "Alignment uses annotated source documents only; it is not a semantic claim-support "
                "judgment. Complete audit labels to populate the human metrics."
            ),
            "citation_alignment_precision": _ratio(cited_aligned, cited_total),
            "citation_alignment_precision_macro": _mean(alignment_scores),
            "evidence_document_coverage": _mean(document_coverages),
            "refusal_correctness": _mean(refusal_correct),
            "cited_documents": cited_total,
        },
        "human_audit": {
            "citation_precision": _mean_bool(human_citation),
            "key_fact_coverage": _mean_bool(human_facts),
            "fact_consistency": _mean_bool(human_consistency),
            "citation_judgment_count": len(human_citation),
            "fact_judgment_count": len(human_facts),
            "answer_judgment_count": len(human_consistency),
        },
    }


def _agent_metrics(cases: Mapping[str, EvalCase], rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    tool_selection: list[bool] = []
    permission_handling: list[bool] = []
    task_success: list[bool] = []
    evaluated = 0
    for row in rows:
        case = cases.get(str(row.get("case_id") or ""))
        if case is None or case.agent is None:
            continue
        expected = case.agent
        observed = set(_unique_texts(row.get("tools")))
        all_required = set(expected.required_all_tools).issubset(observed)
        one_required = not expected.required_any_tools or bool(
            observed & set(expected.required_any_tools)
        )
        no_forbidden = not (observed & set(expected.forbidden_tools))
        selection_ok = all_required and one_required and no_forbidden
        tool_selection.append(selection_ok)

        errors = " ".join(_unique_texts(row.get("tool_errors"))).casefold()
        answer_status = str(row.get("answer_status") or "")
        verification_status = str(row.get("verification_status") or "")
        is_refusal = answer_status == "insufficient_evidence" or verification_status == "refused"
        if expected.permission_outcome == "denied":
            permission_ok = "permission denied" in errors and is_refusal
            permission_handling.append(permission_ok)
        elif expected.permission_outcome == "confirmation_required":
            permission_ok = "requires confirmation" in errors and is_refusal
            permission_handling.append(permission_ok)
        else:
            permission_ok = True

        outcome_ok = (
            answer_status == "grounded"
            if expected.expected_outcome == "answer"
            else is_refusal
        )
        task_success.append(selection_ok and permission_ok and outcome_ok)
        evaluated += 1
    return {
        "definition": (
            "Tool selection permits multiple valid plans: required-any, required-all and forbidden "
            "tool labels are evaluated instead of exact action-sequence match."
        ),
        "evaluated_cases": evaluated,
        "tool_selection_correctness": _mean_bool(tool_selection),
        "permission_refusal_handling": _mean_bool(permission_handling),
        "task_success_rate": _mean_bool(task_success),
    }


def _engineering_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    durations = [_as_float(row.get("duration_ms")) for row in rows]
    durations = [value for value in durations if value is not None]
    costs = [_as_float(row.get("known_llm_cost")) for row in rows]
    costs = [value for value in costs if value is not None]
    failures = sum(
        bool(row.get("error")) or row.get("success") is False
        for row in rows
    )
    return {
        "record_count": len(rows),
        "p50_latency_ms": percentile(durations, 50),
        "p95_latency_ms": percentile(durations, 95),
        "mean_latency_ms": _mean(durations),
        "mean_known_llm_cost": _mean(costs),
        "total_known_llm_cost": round(sum(costs), 8) if costs else 0.0,
        "cost_record_count": len(costs),
        "failure_rate": _ratio(failures, len(rows)),
    }


def _reciprocal_rank(ranked: list[str], relevance: Mapping[str, int]) -> float:
    for rank, key in enumerate(ranked, start=1):
        if relevance.get(key, 0) > 0:
            return 1.0 / rank
    return 0.0


def _ndcg(ranked: list[str], relevance: Mapping[str, int], *, k: int) -> float:
    dcg = sum(
        (2 ** relevance.get(key, 0) - 1) / math.log2(rank + 1)
        for rank, key in enumerate(ranked[:k], start=1)
    )
    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
    )
    return dcg / ideal if ideal else 0.0


def _split_counts(cases: Iterable[EvalCase]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case.split] += 1
    return dict(sorted(counts.items()))


def _unique_texts(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        text = item.strip()
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _bool_judgments(value: Any, key: str) -> list[bool]:
    if not isinstance(value, list):
        return []
    result: list[bool] = []
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get(key), bool):
            result.append(item[key])
    return result


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else None


def _mean_bool(values: Iterable[bool]) -> float | None:
    values = list(values)
    return _mean(float(value) for value in values) if values else None


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)
