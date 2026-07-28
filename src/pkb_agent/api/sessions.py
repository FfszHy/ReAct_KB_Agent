"""Live run registry and event fan-out for Server-Sent Events."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pkb_agent.agent.runtime import AgentRuntime
from pkb_agent.api.approval import ApprovalBroker
from pkb_agent.app.settings import Settings

EventListener = Callable[[dict[str, Any]], Awaitable[None]]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunSession:
    run_id: str
    question: str
    user_id: str
    status: str = "queued"
    created_at: str = field(default_factory=_now_iso)
    history: list[dict[str, Any]] = field(default_factory=list)
    snapshot: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = None
    _subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _sequence: int = 0

    async def publish(self, event: dict[str, Any]) -> None:
        """Persist an event in the in-memory replay log and fan it out."""
        async with self._lock:
            self._sequence += 1
            payload = {
                **event,
                "sequence": self._sequence,
                "emitted_at": _now_iso(),
            }
            self.history.append(payload)
            # A run only produces tens of events; bounding history protects a
            # long-lived API process if a model behaves unexpectedly.
            if len(self.history) > 500:
                self.history = self.history[-500:]
            for queue in tuple(self._subscribers):
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(payload)

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Create a replaying subscription with no event gap at attach time."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=600)
        async with self._lock:
            for event in self.history:
                queue.put_nowait(event)
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    def public_view(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "user_id": self.user_id,
            "status": self.status,
            "created_at": self.created_at,
            "error": self.error,
            "snapshot": self.snapshot,
        }


class WorkbenchStore:
    """Coordinates isolated AgentRuntime instances without global runtime state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.approvals = ApprovalBroker(timeout_seconds=settings.api_approval_timeout_seconds)
        self._sessions: dict[str, RunSession] = {}
        self._lock = asyncio.Lock()

    async def start_run(self, *, question: str, user_id: str) -> RunSession:
        run_id = str(uuid.uuid4())
        session = RunSession(run_id=run_id, question=question, user_id=user_id)
        async with self._lock:
            self._sessions[run_id] = session

        async def confirm(tool_name: str, arguments: dict[str, Any]) -> bool:
            return await self.approvals.request(
                run_id=run_id,
                tool_name=tool_name,
                arguments=arguments,
                publish=session.publish,
            )

        async def execute() -> None:
            runtime: AgentRuntime | None = None
            try:
                session.status = "running"
                runtime = AgentRuntime.build(self.settings, user_id=user_id, confirm=confirm)
                async with runtime:
                    state = await runtime.run(
                        question,
                        user_id=user_id,
                        on_event=session.publish,
                        run_id=run_id,
                    )
                session.snapshot = state.to_dict()
                session.status = state.status.value
            except Exception as exc:
                session.status = "error"
                session.error = f"{type(exc).__name__}: {exc}"
                if not any(event.get("type") == "error" for event in session.history):
                    await session.publish({"type": "error", "error": session.error})
            finally:
                await self.approvals.cancel_run(run_id)
                await session.publish(
                    {
                        "type": "run_finished",
                        "status": session.status,
                        "state": session.snapshot,
                        "error": session.error,
                        "metrics": (session.snapshot or {}).get("metrics"),
                    }
                )

        session.task = asyncio.create_task(execute(), name=f"pkb-run-{run_id}")
        return session

    async def get(self, run_id: str) -> RunSession | None:
        async with self._lock:
            return self._sessions.get(run_id)

    async def approve(self, *, run_id: str, approval_id: str, approved: bool) -> bool:
        return await self.approvals.decide(
            run_id=run_id,
            approval_id=approval_id,
            approved=approved,
        )

    async def close(self) -> None:
        """Cancel live work cleanly when FastAPI shuts down."""
        async with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            await self.approvals.cancel_run(session.run_id)
            if session.task and not session.task.done():
                session.task.cancel()
        for session in sessions:
            if session.task:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await session.task

    async def recent_snapshots(self, *, user_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                dict(session.snapshot)
                for session in self._sessions.values()
                if session.user_id == user_id and session.snapshot is not None
            ]
