"""Reproducible, evidence-first evaluation for PKB-Agent.

The package deliberately has no dependency on a hosted evaluation service. A
benchmark is a versioned directory of JSONL annotations and a corpus manifest;
each run emits portable JSON plus a static HTML/SVG report.
"""

from pkb_agent.evaluation.dataset import EvalCase, EvalDataset
from pkb_agent.evaluation.metrics import summarize_records

__all__ = ["EvalCase", "EvalDataset", "summarize_records"]
