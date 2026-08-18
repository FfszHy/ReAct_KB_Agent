"""Agent loop state: run metadata, message history, and per-step records."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from pkb_agent.agent.verification import Evidence
from pkb_agent.llm.schemas import Message, ToolCall


class AgentStatus(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    MAX_STEPS = "max_steps"
    ABORTED = "aborted"


def _now() -> datetime:
    return datetime.now(UTC)


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


@dataclass
class AgentStep:
    """A single ReAct step: Thought -> Action -> Observation."""

    index: int
    thought: str | None = None
    tool_call: ToolCall | None = None
    execution_arguments: dict | None = None
    prompt_context: dict = field(default_factory=dict)
    observation: str | None = None
    status: str = "pending"  # pending | ok | error | skipped
    error: str | None = None
    started_at: datetime = field(default_factory=_now)
    ended_at: datetime | None = None

    def finish(self, status: str = "ok", observation: str | None = None) -> None:
        self.status = status
        if observation is not None:
            self.observation = observation
        self.ended_at = _now()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "thought": self.thought,
            "tool_call": (
                {"id": self.tool_call.id, "name": self.tool_call.name, "arguments": self.tool_call.arguments}
                if self.tool_call
                else None
            ),
            "execution_arguments": self.execution_arguments,
            "prompt_context": self.prompt_context,
            "observation": self.observation,
            "status": self.status,
            "error": self.error,
            "started_at": _utc_iso(self.started_at),
            "ended_at": _utc_iso(self.ended_at) if self.ended_at else None,
        }


@dataclass
class AgentRunState:
    """Full state of one agent run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    messages: list[Message] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    prompt_context: dict = field(default_factory=dict)
    status: AgentStatus = AgentStatus.RUNNING
    final_answer: str | None = None
    # Canonical JSON answer and its verification audit. Evidence records only
    # exist for sources retrieved during this run.
    answer_payload: dict | None = None
    verification: dict = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    verification_attempts: int = 0
    # ``budget_finalized`` means the runtime stopped further tool execution
    # deliberately and requested a final no-tool JSON answer. It is a normal
    # terminal path, not a max-step execution error.
    budget_finalized: bool = False
    # Trace persistence is intentionally non-fatal. Keep its degradation
    # visible in artifacts without letting observability decide run success.
    trace_write_failure_count: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    ended_at: datetime | None = None
    usage: dict = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            # This is calculated only from usage returned by completed model
            # requests. It is not a pre-run forecast.
            "actual_cost": 0.0,
            "cost_currency": "USD",
            "cost_status": "accruing",
            "cost_source": "completed_api_usage",
        }
    )
    tool_duration_ms: int = 0
    tool_call_count: int = 0
    successful_tool_call_count: int = 0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, thought: str | None = None, tool_call: ToolCall | None = None) -> AgentStep:
        step = AgentStep(index=self.step_count, thought=thought, tool_call=tool_call)
        self.steps.append(step)
        return step

    def add_usage(
        self,
        usage: dict | None,
        *,
        input_cost_per_million: float | None = None,
        cache_hit_input_cost_per_million: float = 0.0,
        cache_miss_input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        currency: str = "USD",
    ) -> None:
        """Accumulate returned usage and its cache-aware contract cost.

        Every value comes from a completed provider response. The total becomes
        final when :meth:`mark_finished` changes ``cost_status`` to
        ``settled``; the UI deliberately does not present it as a prediction.
        """
        raw = usage if isinstance(usage, dict) else {}
        prompt = _non_negative_int(raw.get("prompt_tokens"))
        completion = _non_negative_int(raw.get("completion_tokens"))
        total = _non_negative_int(raw.get("total_tokens"))
        if total == 0:
            total = prompt + completion
        cache_hit = _non_negative_int(raw.get("prompt_cache_hit_tokens"))
        cache_miss = _non_negative_int(raw.get("prompt_cache_miss_tokens"))
        # DeepSeek documents prompt_tokens as hit + miss. Older compatible
        # gateways may omit those details; treat any unclassified input as a
        # cache miss so the completed-usage calculation never understates the
        # contractual bill.
        unclassified = max(prompt - cache_hit - cache_miss, 0)
        cache_miss += unclassified
        self.usage["prompt_tokens"] = _non_negative_int(self.usage.get("prompt_tokens")) + prompt
        self.usage["prompt_cache_hit_tokens"] = (
            _non_negative_int(self.usage.get("prompt_cache_hit_tokens")) + cache_hit
        )
        self.usage["prompt_cache_miss_tokens"] = (
            _non_negative_int(self.usage.get("prompt_cache_miss_tokens")) + cache_miss
        )
        self.usage["completion_tokens"] = (
            _non_negative_int(self.usage.get("completion_tokens")) + completion
        )
        self.usage["total_tokens"] = _non_negative_int(self.usage.get("total_tokens")) + total
        # A legacy single input rate has precedence as the conservative miss
        # rate. It keeps existing deployments compatible with this expansion.
        miss_rate = (
            max(float(input_cost_per_million), 0.0)
            if input_cost_per_million is not None
            else max(float(cache_miss_input_cost_per_million), 0.0)
        )
        actual_cost = (
            self.usage["prompt_cache_hit_tokens"]
            * max(float(cache_hit_input_cost_per_million), 0.0)
            + self.usage["prompt_cache_miss_tokens"] * miss_rate
            + self.usage["completion_tokens"] * max(float(output_cost_per_million), 0.0)
        ) / 1_000_000
        self.usage["actual_cost"] = round(actual_cost, 8)
        self.usage["cost_currency"] = str(currency or "USD").upper()
        self.usage["cost_status"] = "accruing"
        self.usage["cost_source"] = "completed_api_usage"

    def record_tool_call(self, duration_ms: int, *, ok: bool) -> None:
        self.tool_call_count += 1
        self.tool_duration_ms += max(int(duration_ms), 0)
        if ok:
            self.successful_tool_call_count += 1

    def mark_finished(self) -> None:
        if self.ended_at is None:
            self.ended_at = _now()
        duration_ms = max(int((self.ended_at - self.created_at).total_seconds() * 1000), 0)
        self.usage["duration_ms"] = duration_ms
        self.usage["tool_duration_ms"] = self.tool_duration_ms
        self.usage["tool_call_count"] = self.tool_call_count
        self.usage["successful_tool_call_count"] = self.successful_tool_call_count
        self.usage["tool_success_rate"] = (
            round(self.successful_tool_call_count / self.tool_call_count * 100, 1)
            if self.tool_call_count
            else None
        )
        self.usage["cost_status"] = "settled"

    def metrics(self) -> dict:
        end = self.ended_at or _now()
        duration_ms = max(int((end - self.created_at).total_seconds() * 1000), 0)
        verification_status = str(self.verification.get("status") or "")
        # A run that deliberately refuses for lack of evidence is a completed
        # execution, even though it is not a grounded answer.  Keep this
        # separate from ``run_succeeded``, which intentionally remains the
        # stricter grounded-and-verified outcome used by the product UI.
        execution_succeeded = self.status == AgentStatus.FINISHED and self.error is None
        run_succeeded = self.status == AgentStatus.FINISHED and verification_status == "verified"
        tool_success_rate = (
            round(self.successful_tool_call_count / self.tool_call_count * 100, 1)
            if self.tool_call_count
            else None
        )
        return {
            "duration_ms": duration_ms,
            "tool_duration_ms": self.tool_duration_ms,
            "tool_call_count": self.tool_call_count,
            "successful_tool_call_count": self.successful_tool_call_count,
            "tool_success_rate": tool_success_rate,
            "budget_finalized": self.budget_finalized,
            "trace_write_failure_count": self.trace_write_failure_count,
            "execution_succeeded": execution_succeeded,
            "run_succeeded": run_succeeded,
            "run_success_rate": 100.0 if run_succeeded else 0.0,
            "usage": dict(self.usage),
        }

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "status": self.status.value,
            "step_count": self.step_count,
            "steps": [s.to_dict() for s in self.steps],
            "prompt_context": self.prompt_context,
            "final_answer": self.final_answer,
            "answer": self.answer_payload,
            "verification": self.verification,
            "budget_finalized": self.budget_finalized,
            "trace_write_failure_count": self.trace_write_failure_count,
            "usage": dict(self.usage),
            "metrics": self.metrics(),
            "retrieved_evidence": [
                _serialize_evidence(record) for record in self.evidence.values()
            ],
            "error": self.error,
            "created_at": _utc_iso(self.created_at),
            "ended_at": _utc_iso(self.ended_at) if self.ended_at else None,
        }


def _serialize_evidence(record: Evidence) -> dict:
    return record.to_dict()


def _non_negative_int(value: object) -> int:
    try:
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            return 0
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0
