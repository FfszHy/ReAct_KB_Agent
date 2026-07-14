"""Trace recorder: async wrapper over TracesRepository with redaction."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pkb_agent.trace.redaction import redact_args, redact_result, redact_text

if TYPE_CHECKING:
    from pkb_agent.app.settings import Settings
    from pkb_agent.storage.repositories.traces import TracesRepository


class TraceRecorder:
    def __init__(
        self,
        repo: TracesRepository,
        *,
        enabled: bool = True,
        redact: bool = True,
    ) -> None:
        self._repo = repo
        self._enabled = enabled
        self._redact = redact

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        repo: TracesRepository,
    ) -> TraceRecorder:
        return cls(
            repo,
            enabled=settings.trace_enabled,
            redact=settings.trace_redact_secrets,
        )

    async def start_run(
        self,
        *,
        run_id: str,
        user_id: str = "default",
        question: str,
        status: str = "running",
    ) -> None:
        if not self._enabled:
            return None
        await asyncio.to_thread(
            self._repo.create_run,
            run_id=run_id,
            user_id=user_id,
            question=question,
            status=status,
        )

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        final_answer: str | None = None,
        error: str | None = None,
        step_count: int | None = None,
        usage: dict | None = None,
    ) -> None:
        if not self._enabled:
            return None
        if self._redact and error is not None:
            error = redact_text(error)
        await asyncio.to_thread(
            self._repo.finish_run,
            run_id,
            status=status,
            final_answer=final_answer,
            error=error,
            step_count=step_count,
            usage=usage,
        )

    async def add_step(
        self,
        *,
        run_id: str,
        step_index: int,
        thought: str | None = None,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        observation: str | None = None,
        status: str = "ok",
        error: str | None = None,
        started_at: datetime | str | None = None,
        ended_at: datetime | str | None = None,
    ) -> None:
        if not self._enabled:
            return None
        if self._redact:
            if tool_args is not None:
                tool_args = redact_args(tool_args)
            if observation is not None:
                observation = redact_text(observation)
            if error is not None:
                error = redact_text(error)
        await asyncio.to_thread(
            self._repo.add_step,
            run_id=run_id,
            step_index=step_index,
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            observation=observation,
            status=status,
            error=error,
            started_at=started_at,
            ended_at=ended_at,
        )

    async def add_tool_call(
        self,
        *,
        run_id: str,
        step_index: int,
        tool_name: str,
        arguments: dict | None = None,
        result: Any | None = None,
        ok: bool = True,
        truncated: bool = False,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        if not self._enabled:
            return None
        if self._redact:
            if arguments is not None:
                arguments = redact_args(arguments)
            if result is not None:
                result = redact_result(result)
            if error is not None:
                error = redact_text(error)
        await asyncio.to_thread(
            self._repo.add_tool_call,
            run_id=run_id,
            step_index=step_index,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            ok=ok,
            truncated=truncated,
            duration_ms=duration_ms,
            error=error,
        )
