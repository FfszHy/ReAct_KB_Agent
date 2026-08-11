"""Transparent metrics for retrieval, grounded answers, agent policy and cost.

The module keeps automatic metrics deliberately conservative.  A cited document
matching an expected source is *alignment*, not proof that every sentence is
true.  Semantic citation support and fact consistency are reported only from
explicit human-audit labels, so the dashboard cannot quietly turn a heuristic
into a quality claim.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from pkb_agent.evaluation.dataset import EvalCase, EvalDataset

_BOOTSTRAP_RESAMPLES = 2_000
_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
_PRIMARY_COMPARISON_METRIC = "ndcg_at_k"


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
    retrieval_case_scores: dict[str, dict[str, dict[str, float]]] = {}
    for name, rows in sorted(grouped.items()):
        retrieval_rows = [row for row in rows if row.get("kind") == "retrieval"]
        agent_rows = [row for row in rows if row.get("kind") == "agent"]
        payload: dict[str, Any] = {
            "record_count": len(rows),
            "engineering": _engineering_metrics(rows),
        }
        if retrieval_rows:
            retrieval, case_scores = _retrieval_metrics(cases, retrieval_rows, k=k)
            payload["retrieval"] = retrieval
            retrieval_case_scores[name] = case_scores
        if agent_rows:
            payload["answer"] = _answer_metrics(cases, agent_rows)
            payload["agent"] = _agent_metrics(cases, agent_rows)
        strategies[name] = payload
    summary: dict[str, Any] = {
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
    if retrieval_case_scores:
        summary["retrieval_comparison"] = _pairwise_retrieval_comparison(retrieval_case_scores)
    return summary


def _retrieval_metrics(
    cases: Mapping[str, EvalCase], rows: Iterable[Mapping[str, Any]], *, k: int
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Score unique answerable cases, averaging repeated executions within case."""
    per_case_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    missing_cases = 0
    skipped_unanswerable_ids: set[str] = set()
    for row in rows:
        case = cases.get(str(row.get("case_id") or ""))
        if case is None or not case.answerable or not case.relevant_documents:
            if case is None:
                missing_cases += 1
            elif not case.answerable:
                skipped_unanswerable_ids.add(case.id)
            continue
        expected = case.relevance_by_document
        ranked = _unique_texts(row.get("retrieved_document_keys"))[:k]
        retrieved_expected = set(ranked) & set(expected)
        per_case_scores[case.id].append(
            {
                "recall_at_k": len(retrieved_expected) / len(expected),
                "mrr_at_k": _reciprocal_rank(ranked, expected),
                "ndcg_at_k": _ndcg(ranked, expected, k=k),
            }
        )

    case_scores = {
        case_id: {
            metric: round(sum(row[metric] for row in scores) / len(scores), 6)
            for metric in ("recall_at_k", "mrr_at_k", "ndcg_at_k")
        }
        for case_id, scores in per_case_scores.items()
    }
    recalls = [scores["recall_at_k"] for scores in case_scores.values()]
    reciprocal_ranks = [scores["mrr_at_k"] for scores in case_scores.values()]
    ndcgs = [scores["ndcg_at_k"] for scores in case_scores.values()]
    metrics = {
        "definition": (
            "Document-level macro metrics; each strategy retrieves a wider chunk candidate pool, "
            "then duplicate chunks from the same source are collapsed before ranking. Recall@K is "
            "the fraction of annotated relevant documents retrieved. Repeated runs are averaged "
            "within each case before aggregation."
        ),
        "k": k,
        "evaluated_cases": len(case_scores),
        "evaluated_records": sum(len(scores) for scores in per_case_scores.values()),
        "skipped_unanswerable_cases": len(skipped_unanswerable_ids),
        "unknown_case_records": missing_cases,
        "recall_at_k": _mean(recalls),
        "mrr_at_k": _mean(reciprocal_ranks),
        "ndcg_at_k": _mean(ndcgs),
        "confidence_intervals": _bootstrap_confidence_intervals(case_scores),
    }
    return metrics, case_scores


def _bootstrap_confidence_intervals(
    case_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Return deterministic percentile bootstrap intervals over unique cases."""
    intervals = {
        metric: _bootstrap_interval(
            [scores[metric] for scores in case_scores.values()],
            seed=f"retrieval:{metric}:{','.join(sorted(case_scores))}",
        )
        for metric in ("recall_at_k", "mrr_at_k", "ndcg_at_k")
    }
    return {
        "method": "nonparametric percentile bootstrap over unique answerable cases",
        "confidence_level": _BOOTSTRAP_CONFIDENCE_LEVEL,
        "resamples": _BOOTSTRAP_RESAMPLES,
        "case_count": len(case_scores),
        **intervals,
    }


def _bootstrap_interval(values: Iterable[float], *, seed: str) -> dict[str, float] | None:
    values = list(values)
    if not values:
        return None
    estimate = _mean(values)
    if len(values) == 1:
        assert estimate is not None
        return {"estimate": estimate, "lower": estimate, "upper": estimate}
    rng = random.Random(seed)
    sample_size = len(values)
    samples = [
        sum(values[rng.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(_BOOTSTRAP_RESAMPLES)
    ]
    lower_tail = (1 - _BOOTSTRAP_CONFIDENCE_LEVEL) / 2 * 100
    upper_tail = (1 + _BOOTSTRAP_CONFIDENCE_LEVEL) / 2 * 100
    lower = percentile(samples, lower_tail)
    upper = percentile(samples, upper_tail)
    assert estimate is not None and lower is not None and upper is not None
    return {
        "estimate": estimate,
        "lower": round(lower, 6),
        "upper": round(upper, 6),
    }


def _pairwise_retrieval_comparison(
    strategies: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Compare every retrieval strategy with Vector on the same unique cases."""
    names = sorted(strategies)
    baseline = "vector" if "vector" in strategies else names[0]
    baseline_scores = strategies[baseline]
    comparisons: dict[str, Any] = {}
    for name in names:
        if name == baseline:
            continue
        candidate_scores = strategies[name]
        case_ids = sorted(set(baseline_scores) & set(candidate_scores))
        outcomes: list[dict[str, Any]] = []
        deltas: list[float] = []
        wins = losses = ties = 0
        for case_id in case_ids:
            baseline_score = baseline_scores[case_id][_PRIMARY_COMPARISON_METRIC]
            candidate_score = candidate_scores[case_id][_PRIMARY_COMPARISON_METRIC]
            delta = candidate_score - baseline_score
            if delta > 1e-12:
                outcome = "win"
                wins += 1
            elif delta < -1e-12:
                outcome = "loss"
                losses += 1
            else:
                outcome = "tie"
                ties += 1
            deltas.append(delta)
            outcomes.append(
                {
                    "case_id": case_id,
                    "outcome": outcome,
                    "candidate_score": candidate_score,
                    "baseline_score": baseline_score,
                }
            )
        comparisons[name] = {
            "evaluated_cases": len(case_ids),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "mean_delta": _mean(deltas),
            "delta_confidence_interval": _bootstrap_interval(
                deltas,
                seed=f"comparison:{baseline}:{name}:{','.join(case_ids)}",
            ),
            "case_outcomes": outcomes,
        }
    return {
        "baseline_strategy": baseline,
        "metric": _PRIMARY_COMPARISON_METRIC,
        "unit": "unique answerable case; repeated runs are averaged within case",
        "comparisons": comparisons,
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
    durations = [
        value
        for row in rows
        if (value := _as_float(row.get("duration_ms"))) is not None
    ]
    costs = [
        value
        for row in rows
        if (value := _as_float(row.get("known_llm_cost"))) is not None
    ]
    execution_failures = sum(_is_execution_failure(row) for row in rows)
    agent_rows = [row for row in rows if row.get("kind") == "agent"]
    refusal_terminals = sum(
        _is_refusal_terminal(row) and not _is_execution_failure(row)
        for row in agent_rows
    )
    return {
        "definition": (
            "Execution failure means a recorded runtime error or explicit execution failure. "
            "A normal insufficient-evidence refusal is reported separately and is not an "
            "engineering failure; its correctness is scored by the answer and agent metrics."
        ),
        "record_count": len(rows),
        "p50_latency_ms": percentile(durations, 50),
        "p95_latency_ms": percentile(durations, 95),
        "mean_latency_ms": _mean(durations),
        "mean_known_llm_cost": _mean(costs),
        "total_known_llm_cost": round(sum(costs), 8) if costs else 0.0,
        "cost_record_count": len(costs),
        "execution_failure_rate": _ratio(execution_failures, len(rows)),
        "refusal_terminal_rate": _ratio(refusal_terminals, len(agent_rows)) if agent_rows else None,
        # Kept as a compatibility alias for existing consumers.  Its meaning
        # is now aligned with the report label instead of conflating a safe
        # refusal with an execution failure.
        "failure_rate": _ratio(execution_failures, len(rows)),
    }


def _is_execution_failure(row: Mapping[str, Any]) -> bool:
    """Classify runtime failures without treating a safe Agent refusal as one."""
    if bool(row.get("error")):
        return True
    if "execution_success" in row:
        return row.get("execution_success") is False
    # Retrieval artifacts before the explicit execution field used ``success``
    # for this meaning.  Older Agent records used it for a grounded-answer
    # outcome, so they deliberately fall through as non-failures when no error
    # was recorded.
    return row.get("kind") != "agent" and row.get("success") is False


def _is_refusal_terminal(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("answer_status") or "") == "insufficient_evidence"
        or str(row.get("verification_status") or "") == "refused"
    )


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
