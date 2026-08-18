"""Trace recorder: async wrapper over TracesRepository with redaction."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from pkb_agent.trace.redaction import redact_args, redact_result, redact_text

if TYPE_CHECKING:
    from pkb_agent.app.settings import Settings
    from pkb_agent.storage.repositories.traces import TracesRepository


_LOG = logging.getLogger(__name__)
_RETRYABLE_TRACE_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "server disconnected",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)


class TraceRecorder:
    def __init__(
        self,
        repo: TracesRepository,
        *,
        enabled: bool = True,
        redact: bool = True,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
        self._repo = repo
        self._enabled = enabled
        self._redact = redact
        self._retry_attempts = max(int(retry_attempts), 1)
        self._retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)

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
            retry_attempts=settings.trace_retry_attempts,
            retry_backoff_seconds=settings.trace_retry_backoff_seconds,
        )

    async def start_run(
        self,
        *,
        run_id: str,
        user_id: str = "default",
        question: str,
        status: str = "running",
        prompt_context: dict | None = None,
    ) -> bool | None:
        if not self._enabled:
            return None
        return await self._write(
            "start_run",
            self._repo.create_run,
            run_id=run_id,
            user_id=user_id,
            question=question,
            status=status,
            prompt_context=prompt_context,
        )

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        final_answer: str | None = None,
        answer_payload: dict | None = None,
        verification: dict | None = None,
        error: str | None = None,
        step_count: int | None = None,
        usage: dict | None = None,
    ) -> bool | None:
        if not self._enabled:
            return None
        if self._redact and error is not None:
            error = redact_text(error)
        if self._redact and answer_payload is not None:
            answer_payload = redact_result(answer_payload)
        if self._redact and verification is not None:
            verification = redact_result(verification)
        return await self._write(
            "finish_run",
            self._repo.finish_run,
            run_id,
            status=status,
            final_answer=final_answer,
            answer_payload=answer_payload,
            verification=verification,
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
        original_tool_args: dict | None = None,
        prompt_context: dict | None = None,
        observation: str | None = None,
        status: str = "ok",
        error: str | None = None,
        started_at: datetime | str | None = None,
        ended_at: datetime | str | None = None,
    ) -> bool | None:
        if not self._enabled:
            return None
        if self._redact:
            if tool_args is not None:
                tool_args = redact_args(tool_args)
            if original_tool_args is not None:
                original_tool_args = redact_args(original_tool_args)
            if prompt_context is not None:
                prompt_context = redact_args(prompt_context)
            if observation is not None:
                observation = redact_text(observation)
            if error is not None:
                error = redact_text(error)
        return await self._write(
            "add_step",
            self._repo.add_step,
            run_id=run_id,
            step_index=step_index,
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            original_tool_args=original_tool_args,
            prompt_context=prompt_context,
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
        original_arguments: dict | None = None,
        prompt_context: dict | None = None,
        result: Any | None = None,
        ok: bool = True,
        truncated: bool = False,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> bool | None:
        if not self._enabled:
            return None
        if self._redact:
            if arguments is not None:
                arguments = redact_args(arguments)
            if original_arguments is not None:
                original_arguments = redact_args(original_arguments)
            if prompt_context is not None:
                prompt_context = redact_args(prompt_context)
            if result is not None:
                result = redact_result(result)
            if error is not None:
                error = redact_text(error)
        return await self._write(
            "add_tool_call",
            self._repo.add_tool_call,
            run_id=run_id,
            step_index=step_index,
            tool_name=tool_name,
            arguments=arguments,
            original_arguments=original_arguments,
            prompt_context=prompt_context,
            result=result,
            ok=ok,
            truncated=truncated,
            duration_ms=duration_ms,
            error=error,
        )

    async def _write(
        self,
        operation: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Persist a trace event without turning observability into an outage.

        Repository writes are idempotent (stable IDs plus upsert for inserted
        rows), so a retry is safe even if the server committed just before a
        response timeout. Only transient transport-like failures are retried.
        A final failure is reported to the caller as ``False``; the runtime
        records it in the evaluation artifact and continues the answer path.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                await asyncio.to_thread(func, *args, **kwargs)
                return True
            except Exception as exc:  # repository implementations normalize differently
                last_error = exc
                if attempt >= self._retry_attempts or not _is_retryable_trace_error(exc):
                    break
                delay = min(self._retry_backoff_seconds * (2 ** (attempt - 1)), 5.0)
                if delay:
                    await asyncio.sleep(delay)
        _LOG.warning(
            "trace %s failed after %s attempt(s): %s",
            operation,
            attempt,
            last_error,
        )
        return False


def _is_retryable_trace_error(error: BaseException) -> bool:
    """Recognize transient storage transport failures through wrapped causes."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError, OSError),
        ):
            return True
        current = current.__cause__ or current.__context__
    text = str(error).casefold()
    return any(marker in text for marker in _RETRYABLE_TRACE_ERROR_MARKERS)
