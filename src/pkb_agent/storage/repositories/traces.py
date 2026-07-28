"""Trace repository: agent_runs, agent_steps, tool_calls."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pkb_agent.agent.errors import StorageError
from pkb_agent.storage.supabase_client import SupabaseClient

_RUNS = "agent_runs"
_STEPS = "agent_steps"
_TOOLCALLS = "tool_calls"


def _iso(dt: datetime | str | None) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return dt.isoformat()


class TracesRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._client = client

    # -------------------------------------------------------------------- runs
    def create_run(
        self,
        *,
        run_id: str,
        user_id: str = "default",
        question: str,
        status: str = "running",
        prompt_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": run_id,
            "user_id": user_id,
            "question": question,
            "status": status,
        }
        if prompt_context is not None:
            row["prompt_context"] = prompt_context
        try:
            data = self._client.table(_RUNS).insert(row).execute().data
        except Exception as e:
            raise StorageError(f"create_run failed: {e}") from e
        return data[0] if data else {}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        data = self._client.table(_RUNS).select("*").eq("id", run_id).execute().data
        return data[0] if data else None

    def list_runs(
        self, user_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        query = self._client.table(_RUNS).select("*").order("created_at", desc=True)
        if user_id is not None:
            query = query.eq("user_id", user_id)
        return query.limit(limit).offset(offset).execute().data or []

    def update_run(self, run_id: str, **fields: Any) -> dict[str, Any] | None:
        data = self._client.table(_RUNS).update(fields).eq("id", run_id).execute().data
        return data[0] if data else None

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        final_answer: str | None = None,
        answer_payload: dict[str, Any] | None = None,
        verification: dict[str, Any] | None = None,
        error: str | None = None,
        step_count: int | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        fields: dict[str, Any] = {"status": status}
        if final_answer is not None:
            fields["final_answer"] = final_answer
        if answer_payload is not None:
            fields["answer_payload"] = answer_payload
        if verification is not None:
            fields["verification"] = verification
        if error is not None:
            fields["error"] = error
        if step_count is not None:
            fields["step_count"] = step_count
        if usage is not None:
            fields["usage"] = usage
        return self.update_run(run_id, **fields)

    # ------------------------------------------------------------------- steps
    def add_step(
        self,
        *,
        run_id: str,
        step_index: int,
        thought: str | None = None,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        original_tool_args: dict[str, Any] | None = None,
        prompt_context: dict[str, Any] | None = None,
        observation: str | None = None,
        status: str = "pending",
        error: str | None = None,
        started_at: datetime | str | None = None,
        ended_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run_id": run_id,
            "step_index": step_index,
            "thought": thought,
            "tool_name": tool_name,
            "tool_args": tool_args or {},
            "observation": observation,
            "status": status,
            "error": error,
            "started_at": _iso(started_at),
            "ended_at": _iso(ended_at),
        }
        if original_tool_args is not None:
            row["original_tool_args"] = original_tool_args
        if prompt_context is not None:
            row["prompt_context"] = prompt_context
        try:
            data = self._client.table(_STEPS).insert(row).execute().data
        except Exception as e:
            raise StorageError(f"add_step failed: {e}") from e
        return data[0] if data else {}

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        return (
            self._client.table(_STEPS)
            .select("*")
            .eq("run_id", run_id)
            .order("step_index", desc=False)
            .execute()
            .data
            or []
        )

    # -------------------------------------------------------------- tool calls
    def add_tool_call(
        self,
        *,
        run_id: str,
        step_index: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        original_arguments: dict[str, Any] | None = None,
        prompt_context: dict[str, Any] | None = None,
        result: Any | None = None,
        ok: bool = True,
        truncated: bool = False,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run_id": run_id,
            "step_index": step_index,
            "tool_name": tool_name,
            "arguments": arguments or {},
            "result": result,
            "ok": ok,
            "truncated": truncated,
            "duration_ms": duration_ms,
            "error": error,
        }
        if original_arguments is not None:
            row["original_arguments"] = original_arguments
        if prompt_context is not None:
            row["prompt_context"] = prompt_context
        try:
            data = self._client.table(_TOOLCALLS).insert(row).execute().data
        except Exception as e:
            raise StorageError(f"add_tool_call failed: {e}") from e
        return data[0] if data else {}

    def list_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        return (
            self._client.table(_TOOLCALLS)
            .select("*")
            .eq("run_id", run_id)
            .order("step_index", desc=False)
            .execute()
            .data
            or []
        )
