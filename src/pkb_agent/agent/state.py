"""Agent loop state: run metadata, message history, and per-step records."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

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
    status: AgentStatus = AgentStatus.RUNNING
    final_answer: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, thought: str | None = None, tool_call: ToolCall | None = None) -> AgentStep:
        step = AgentStep(index=self.step_count, thought=thought, tool_call=tool_call)
        self.steps.append(step)
        return step

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "status": self.status.value,
            "step_count": self.step_count,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "error": self.error,
            "created_at": _utc_iso(self.created_at),
        }
