"""Dependency-free static reports for benchmark artifacts."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_report(output_dir: str | Path, summary: Mapping[str, Any]) -> dict[str, Path]:
    """Write a shareable Markdown/HTML/SVG evaluation dashboard."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_payload = dict(summary)
    summary_payload["generated_at"] = datetime.now(UTC).isoformat()
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    chart_path = root / "comparison.svg"
    chart_path.write_text(_comparison_svg(summary_payload), "utf-8")
    markdown_path = root / "report.md"
    markdown_path.write_text(_markdown(summary_payload), "utf-8")
    html_path = root / "index.html"
    html_path.write_text(_html(summary_payload), "utf-8")
    return {
        "summary": summary_path,
        "chart": chart_path,
        "markdown": markdown_path,
        "html": html_path,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    benchmark = _mapping(summary.get("benchmark"))
    strategies = _mapping(summary.get("strategies"))
    k = benchmark.get("default_top_k", 6)
    lines = [
        f"# {benchmark.get('title', 'Evaluation report')}",
        "",
        f"Benchmark: `{benchmark.get('id', 'unknown')}` · version `{benchmark.get('version', 'unknown')}` "
        f"· corpus benchmark has {benchmark.get('case_count', 0)} cases; this run has "
        f"{benchmark.get('recorded_case_count', 0)} unique cases "
        f"({', '.join(benchmark.get('recorded_splits', [])) or 'unknown split'}) · "
        f"generated {summary.get('generated_at', '')}",
        "",
        "## Retrieval and engineering comparison",
        "",
        f"| Strategy | Recall@{k} | MRR@{k} | NDCG@{k} | p50 | p95 | Known LLM cost | Execution failure rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, raw in strategies.items():
        item = _mapping(raw)
        retrieval = _mapping(item.get("retrieval"))
        engineering = _mapping(item.get("engineering"))
        lines.append(
            "| {name} | {recall} | {mrr} | {ndcg} | {p50} | {p95} | {cost} | {failure} |".format(
                name=name,
                recall=_number(retrieval.get("recall_at_k")),
                mrr=_number(retrieval.get("mrr_at_k")),
                ndcg=_number(retrieval.get("ndcg_at_k")),
                p50=_ms(engineering.get("p50_latency_ms")),
                p95=_ms(engineering.get("p95_latency_ms")),
                cost=_number(engineering.get("mean_known_llm_cost"), digits=6),
                failure=_percent(
                    engineering.get("execution_failure_rate", engineering.get("failure_rate"))
                ),
            )
        )
    retrieval_rows = [
        (name, _mapping(item)) for name, item in strategies.items() if _mapping(item).get("retrieval")
    ]
    if retrieval_rows:
        lines.extend(
            [
                "",
                "## Bootstrap uncertainty",
                "",
                "95% nonparametric percentile bootstrap intervals over unique answerable cases; "
                "repeated executions are averaged within each case.",
                "",
                f"| Strategy | Recall@{k} 95% CI | MRR@{k} 95% CI | NDCG@{k} 95% CI | Cases |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, item in retrieval_rows:
            retrieval = _mapping(item.get("retrieval"))
            intervals = _mapping(retrieval.get("confidence_intervals"))
            lines.append(
                "| {name} | {recall} | {mrr} | {ndcg} | {cases} |".format(
                    name=name,
                    recall=_interval(intervals.get("recall_at_k")),
                    mrr=_interval(intervals.get("mrr_at_k")),
                    ndcg=_interval(intervals.get("ndcg_at_k")),
                    cases=intervals.get("case_count", retrieval.get("evaluated_cases", "—")),
                )
            )
    comparison = _mapping(summary.get("retrieval_comparison"))
    comparisons = _mapping(comparison.get("comparisons"))
    if comparisons:
        baseline = comparison.get("baseline_strategy", "baseline")
        metric = comparison.get("metric", "ndcg_at_k")
        lines.extend(
            [
                "",
                "## Case-level wins, losses, and ties",
                "",
                f"Each strategy is compared with `{baseline}` on `{metric}`. "
                "A tie means the per-case score is identical.",
                "",
                "| Strategy | Wins | Losses | Ties | Mean delta | 95% CI of delta | Cases |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, raw in comparisons.items():
            item = _mapping(raw)
            lines.append(
                "| {name} | {wins} | {losses} | {ties} | {delta} | {interval} | {cases} |".format(
                    name=name,
                    wins=item.get("wins", 0),
                    losses=item.get("losses", 0),
                    ties=item.get("ties", 0),
                    delta=_number(item.get("mean_delta")),
                    interval=_interval(item.get("delta_confidence_interval")),
                    cases=item.get("evaluated_cases", 0),
                )
            )
        for name, raw in comparisons.items():
            item = _mapping(raw)
            outcomes = item.get("case_outcomes")
            if not isinstance(outcomes, list):
                continue
            lines.extend(
                [
                    "",
                    "<details>",
                    f"<summary>{name} vs {baseline}: per-case {metric}</summary>",
                    "",
                    "| Case | Outcome | Strategy | Baseline |",
                    "| --- | --- | ---: | ---: |",
                ]
            )
            for raw_outcome in outcomes:
                outcome = _mapping(raw_outcome)
                lines.append(
                    "| {case_id} | {outcome} | {candidate} | {baseline_score} |".format(
                        case_id=outcome.get("case_id", "—"),
                        outcome=outcome.get("outcome", "—"),
                        candidate=_number(outcome.get("candidate_score")),
                        baseline_score=_number(outcome.get("baseline_score")),
                    )
                )
            lines.extend(["", "</details>"])
    agent_rows = [(name, _mapping(item)) for name, item in strategies.items() if _mapping(item).get("answer")]
    if agent_rows:
        lines.extend(
            [
                "",
                "## Grounded-answer and agent checks",
                "",
                "| Strategy | Citation alignment precision | Evidence document coverage | Refusal correctness | Human citation precision | Key-fact coverage | Fact consistency | Tool selection | Permission refusal | Task success |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, item in agent_rows:
            answer = _mapping(item.get("answer"))
            automatic = _mapping(answer.get("automatic"))
            human = _mapping(answer.get("human_audit"))
            agent = _mapping(item.get("agent"))
            lines.append(
                "| {name} | {align} | {coverage} | {refusal} | {citation} | {facts} | {consistency} | {tools} | {permission} | {success} |".format(
                    name=name,
                    align=_percent(automatic.get("citation_alignment_precision")),
                    coverage=_percent(automatic.get("evidence_document_coverage")),
                    refusal=_percent(automatic.get("refusal_correctness")),
                    citation=_percent(human.get("citation_precision")),
                    facts=_percent(human.get("key_fact_coverage")),
                    consistency=_percent(human.get("fact_consistency")),
                    tools=_percent(agent.get("tool_selection_correctness")),
                    permission=_percent(agent.get("permission_refusal_handling")),
                    success=_percent(agent.get("task_success_rate")),
                )
            )
        lines.extend(
            [
                "",
                "## Agent execution states",
                "",
                "| Strategy | Execution failure rate | Refusal terminal rate |",
                "| --- | ---: | ---: |",
            ]
        )
        for name, item in agent_rows:
            engineering = _mapping(item.get("engineering"))
            lines.append(
                "| {name} | {failure} | {refusal} |".format(
                    name=name,
                    failure=_percent(
                        engineering.get("execution_failure_rate", engineering.get("failure_rate"))
                    ),
                    refusal=_percent(engineering.get("refusal_terminal_rate")),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation safeguards",
            "",
            "- Retrieval metrics are document-level: a wider chunk candidate pool is deduplicated by source before scoring K documents.",
            "- Citation alignment is an automatic source-match proxy, not a semantic entailment claim.",
            "- Human-audit metrics remain blank until the generated audit JSONL is reviewed.",
            "- A refusal terminal is a normal insufficient-evidence completion, not an engineering failure; use refusal correctness and task success to judge whether it was appropriate.",
            "- Bootstrap intervals quantify uncertainty in this fixed benchmark, not generalization to arbitrary corpora.",
            "- Known LLM cost covers only usage emitted by the configured provider; it is not an invoice.",
            "",
        ]
    )
    return "\n".join(lines)


def _html(summary: Mapping[str, Any]) -> str:
    benchmark = _mapping(summary.get("benchmark"))
    markdown = _markdown(summary)
    escaped = html.escape(markdown)
    title = html.escape(str(benchmark.get("title", "Evaluation report")))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ max-width: 1080px; margin: 0 auto; padding: 40px 24px; background: #0b1020; color: #e8edf7; }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ color: #aab8d0; margin-top: 0; }}
    .card {{ margin-top: 24px; padding: 24px; border: 1px solid #273451; border-radius: 16px; background: #11192d; overflow-x: auto; }}
    img {{ width: 100%; min-width: 680px; background: #0b1020; border-radius: 12px; }}
    pre {{ white-space: pre-wrap; font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; color: #dce6f9; }}
    a {{ color: #76b7ff; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class=\"meta\">Static, reproducible benchmark report · raw records and summary.json live beside this file.</p>
  <section class=\"card\"><img src=\"comparison.svg\" alt=\"Retrieval comparison chart\"></section>
  <section class=\"card\"><pre>{escaped}</pre></section>
</body>
</html>
"""


def _comparison_svg(summary: Mapping[str, Any]) -> str:
    strategies = _mapping(summary.get("strategies"))
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for name, raw in strategies.items():
        item = _mapping(raw)
        retrieval = _mapping(item.get("retrieval"))
        if retrieval:
            rows.append(
                (
                    str(name),
                    _as_float(retrieval.get("recall_at_k")),
                    _as_float(retrieval.get("mrr_at_k")),
                    _as_float(retrieval.get("ndcg_at_k")),
                )
            )
    width = 920
    row_height = 92
    height = max(230 + len(rows) * row_height, 300)
    labels = [("Recall", "#66d9ef"), ("MRR", "#a6e22e"), ("NDCG", "#fd971f")]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#e8edf7}.sub{fill:#aab8d0;font-size:14px}.label{font-size:15px;font-weight:600}.value{font-size:13px;fill:#dce6f9}</style>",
        f'<rect width="{width}" height="{height}" fill="#0b1020" rx="18"/>',
        '<text x="42" y="48" font-size="24" font-weight="700">Retrieval ablation</text>',
        '<text x="42" y="74" class="sub">All metrics are document-level on the same frozen split and top-k.</text>',
    ]
    for index, (label, color) in enumerate(labels):
        x = 530 + index * 120
        parts.extend(
            [
                f'<rect x="{x}" y="38" width="12" height="12" fill="{color}" rx="3"/>',
                f'<text x="{x + 18}" y="49" class="sub">{label}</text>',
            ]
        )
    base_x = 270
    max_width = 600
    for row_index, (name, recall, mrr, ndcg) in enumerate(rows):
        y = 124 + row_index * row_height
        parts.append(f'<text x="42" y="{y + 26}" class="label">{html.escape(name)}</text>')
        values = (recall, mrr, ndcg)
        for metric_index, (value, (_, color)) in enumerate(zip(values, labels, strict=True)):
            y_bar = y + metric_index * 20
            parts.append(f'<rect x="{base_x}" y="{y_bar}" width="{max_width}" height="12" fill="#1c2944" rx="6"/>')
            if value is not None:
                clipped = max(0.0, min(value, 1.0))
                bar_width = max_width * clipped
                parts.append(f'<rect x="{base_x}" y="{y_bar}" width="{bar_width:.2f}" height="12" fill="{color}" rx="6"/>')
                parts.append(
                    f'<text x="{base_x + max_width + 12}" y="{y_bar + 11}" class="value">{value:.3f}</text>'
                )
            else:
                parts.append(f'<text x="{base_x + max_width + 12}" y="{y_bar + 11}" class="value">—</text>')
    if not rows:
        parts.append('<text x="42" y="132" class="sub">No retrieval records were found in this run.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any, *, digits: int = 3) -> str:
    parsed = _as_float(value)
    return f"{parsed:.{digits}f}" if parsed is not None else "—"


def _interval(value: Any) -> str:
    interval = _mapping(value)
    estimate = _as_float(interval.get("estimate"))
    lower = _as_float(interval.get("lower"))
    upper = _as_float(interval.get("upper"))
    if estimate is None or lower is None or upper is None:
        return "—"
    return f"{estimate:.3f} [{lower:.3f}, {upper:.3f}]"


def _percent(value: Any) -> str:
    parsed = _as_float(value)
    return f"{parsed * 100:.1f}%" if parsed is not None else "—"


def _ms(value: Any) -> str:
    parsed = _as_float(value)
    return f"{parsed:.0f} ms" if parsed is not None else "—"
