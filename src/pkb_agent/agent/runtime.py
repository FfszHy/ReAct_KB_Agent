"""ReAct agent runtime.

The runtime orchestrates the DeepSeek model and the permissioned tool layer.
Core invariant: the runtime NEVER touches Supabase, the network, or memory
directly — it only calls the LLM (reasoning) and tools (world access). Every
tool call passes through validation, permission check, trace recording,
execution, result truncation, and error wrapping.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pkb_agent.agent.errors import (
    AgentError,
    ToolArgumentError,
    ToolError,
    ToolNotFoundError,
    ToolPermissionDenied,
)
from pkb_agent.agent.state import AgentRunState, AgentStatus
from pkb_agent.app.settings import Settings, get_settings, load_prompt, permissions_config_path
from pkb_agent.llm.deepseek_client import DeepSeekClient
from pkb_agent.llm.schemas import Message, ToolCall
from pkb_agent.memory.manager import MemoryManager
from pkb_agent.prompts import PromptComposer, PromptComposition, PromptRegistry
from pkb_agent.prompts.query_rewriter import QueryPlanner
from pkb_agent.rag.embeddings import EmbeddingProvider
from pkb_agent.rag.retriever import Retriever
from pkb_agent.storage.repositories.chunks import ChunksRepository
from pkb_agent.storage.repositories.documents import DocumentsRepository
from pkb_agent.storage.repositories.memory import MemoryRepository
from pkb_agent.storage.repositories.permissions import PermissionsRepository
from pkb_agent.storage.repositories.traces import TracesRepository
from pkb_agent.storage.supabase_client import SupabaseClient
from pkb_agent.tools.base import ToolContext
from pkb_agent.tools.builtin import build_builtin_tools
from pkb_agent.tools.permissions import PermissionManager
from pkb_agent.tools.registry import ToolRegistry
from pkb_agent.trace.recorder import TraceRecorder
from pkb_agent.web.fetcher import Fetcher
from pkb_agent.web.search_provider import make_search_provider

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class AgentRuntime:
    """Assembles all collaborators and runs the ReAct loop."""

    def __init__(
        self,
        settings: Settings,
        *,
        user_id: str = "default",
        confirm: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        self.settings = settings
        self._user_id = user_id
        self._confirm = confirm

        # Populated by _assemble().
        self.llm: DeepSeekClient
        self.registry: ToolRegistry
        self.permissions: PermissionManager
        self.ctx: ToolContext
        self.prompt_composer: PromptComposer | None = None
        self.query_planner: QueryPlanner | None = None
        self._permissions_repo: PermissionsRepository | None = None
        self._last_permission_override_refresh: float | None = None
        self._closables: list[Any] = []
        self._assembled = False

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        settings: Settings | None = None,
        *,
        user_id: str = "default",
        confirm: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> AgentRuntime:
        rt = cls(settings or get_settings(), user_id=user_id, confirm=confirm)
        rt._assemble()
        return rt

    def _assemble(self) -> None:
        s = self.settings
        # Fail before assembling external collaborators when the prompt contract
        # is invalid or a declared prompt file is missing.
        self.prompt_composer = PromptComposer(PromptRegistry.from_settings(s))

        # Storage
        sb = SupabaseClient.from_settings(s)
        repos: dict[str, Any] = {
            "documents": DocumentsRepository(sb),
            "chunks": ChunksRepository(sb),
            "traces": TracesRepository(sb),
            "memory": MemoryRepository(sb),
            "permissions": PermissionsRepository(sb),
        }
        self._permissions_repo = repos["permissions"]

        # Embeddings (shared by retriever + memory)
        embedder = EmbeddingProvider.from_settings(s)

        # Services
        retriever = Retriever.from_settings(s, embedder, repos["chunks"])
        memory_manager = MemoryManager.from_settings(s, repos["memory"], embedder)
        search_provider = make_search_provider(s)
        fetcher = Fetcher.from_settings(s)
        recorder = TraceRecorder.from_settings(s, repos["traces"])

        # LLM + tools + permissions
        self.llm = DeepSeekClient.from_settings(s)
        self.registry = ToolRegistry().register_all(build_builtin_tools())
        self.query_planner = QueryPlanner(
            self.llm,
            self.prompt_composer,
            enabled=s.prompts_query_rewrite_enabled,
            max_queries=s.prompts_query_rewrite_max_queries,
        )
        self.permissions = PermissionManager.from_yaml(
            permissions_config_path(), confirm=self._confirm
        )

        self.ctx = ToolContext(
            settings=s,
            supabase=sb,
            trace=recorder,
            repositories=repos,
            services={
                "retriever": retriever,
                "embedder": embedder,
                "memory_manager": memory_manager,
                "search_provider": search_provider,
                "fetcher": fetcher,
            },
            user_id=self._user_id,
            confirm_callback=self._confirm,
        )
        self._closables = [embedder, self.llm, search_provider, fetcher]
        self._assembled = True

    # ------------------------------------------------------------------
    async def __aenter__(self) -> AgentRuntime:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for c in self._closables:
            close = getattr(c, "close", None) or getattr(c, "aclose", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
        self._closables = []

    # ------------------------------------------------------------------
    async def run(
        self,
        question: str,
        *,
        user_id: str | None = None,
        on_event: EventCallback | None = None,
    ) -> AgentRunState:
        if not self._assembled:
            raise AgentError("runtime not assembled; call AgentRuntime.build()")
        uid = user_id or self._user_id or "default"
        self.ctx.user_id = uid

        state = AgentRunState(question=question)
        self.ctx.run_id = state.run_id

        composition = self._compose_system_prompt()
        state.prompt_context = composition.trace_context()
        state.messages.append(Message.system(composition.content))
        state.messages.append(Message.user(question))

        await _emit(on_event, {"type": "start", "run_id": state.run_id, "question": question})
        await self.ctx.trace.start_run(
            run_id=state.run_id,
            user_id=uid,
            question=question,
            status="running",
            prompt_context=state.prompt_context,
        )

        try:
            await self._loop(state, on_event)
        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error = f"{type(e).__name__}: {e}"
            await self.ctx.trace.finish_run(
                state.run_id,
                status="error",
                error=state.error,
                step_count=state.step_count,
            )
            await _emit(on_event, {"type": "error", "error": state.error})
            raise
        return state

    # ------------------------------------------------------------------
    async def _loop(self, state: AgentRunState, on_event: EventCallback | None) -> None:
        max_steps = self.settings.agent_max_steps
        while state.step_count < max_steps:
            completion = await self.llm.chat(
                state.messages,
                tools=self.registry.to_schemas(),
                temperature=self.settings.llm_temperature,
            )
            choice = completion.first
            msg = choice.message
            state.messages.append(msg)
            usage = completion.usage or {}

            if not msg.tool_calls:
                # Final answer.
                state.final_answer = msg.content or ""
                state.status = AgentStatus.FINISHED
                await self.ctx.trace.finish_run(
                    state.run_id,
                    status="finished",
                    final_answer=state.final_answer,
                    step_count=state.step_count,
                    usage=usage,
                )
                await _emit(
                    on_event,
                    {"type": "answer", "answer": state.final_answer, "steps": state.step_count},
                )
                return

            thought = msg.content
            for tc in msg.tool_calls:
                await self._execute_tool_call(state, tc, thought, on_event)

        # Exhausted step budget.
        state.status = AgentStatus.MAX_STEPS
        state.error = f"exceeded max_steps={max_steps}"
        await self.ctx.trace.finish_run(
            state.run_id,
            status="max_steps",
            error=state.error,
            step_count=state.step_count,
        )
        await _emit(on_event, {"type": "max_steps", "steps": state.step_count})

    # ------------------------------------------------------------------
    async def _execute_tool_call(
        self,
        state: AgentRunState,
        tc: ToolCall,
        thought: str | None,
        on_event: EventCallback | None,
    ) -> None:
        step = state.add_step(thought=thought, tool_call=tc)
        started = time.time()
        start_dt = datetime.now(UTC)
        observation_text = ""
        ok = True
        error: str | None = None
        truncated = False
        result_data: Any = None

        execution_args = dict(tc.arguments)
        prompt_context: dict[str, Any] = {}
        if self.query_planner is not None and isinstance(execution_args.get("query"), str):
            plan = await self.query_planner.rewrite(
                tool_name=tc.name,
                query=execution_args["query"],
                question=state.question,
            )
            execution_args = plan.apply_to_arguments(execution_args)
            prompt_context = plan.trace_context()

        step.execution_arguments = execution_args
        step.prompt_context = prompt_context
        original_tool_args = tc.arguments if prompt_context.get("status") != "skipped" else None
        await self._refresh_permission_overrides(on_event)
        permission_source = self.permissions.rule_source_for(tc.name)

        await _emit(
            on_event,
            {
                "type": "tool_call",
                "step": step.index,
                "tool": tc.name,
                "args": execution_args,
                "original_args": original_tool_args,
                "prompt_context": prompt_context,
                "permission_source": permission_source,
            },
        )

        try:
            tool = self.registry.require(tc.name)
            tool.validate_args(execution_args)
            self.permissions.check(tc.name, execution_args, self.ctx)
            result = await tool.execute(self.ctx, execution_args)
            ok = result.ok
            result_data = result.data
            truncated = result.truncated
            if not ok:
                error = result.error or "tool returned an error"
            observation_text = result.to_observation(self.settings.agent_tool_result_max_chars)
        except ToolPermissionDenied as e:
            ok = False
            error = str(e)
            observation_text = _error_observation(error)
        except ToolNotFoundError:
            ok = False
            error = f"unknown tool: {tc.name}"
            observation_text = _error_observation(error)
        except (ToolArgumentError, ToolError) as e:
            ok = False
            error = str(e)
            observation_text = _error_observation(error)
        except Exception as e:
            ok = False
            error = f"{type(e).__name__}: {e}"
            observation_text = _error_observation(error)

        duration_ms = int((time.time() - started) * 1000)
        end_dt = datetime.now(UTC)
        step.finish(status="ok" if ok else "error", observation=observation_text)

        # Trace (fire-and-forget failures are swallowed inside recorder? no — recorder raises on DB error; let it surface)
        await self.ctx.trace.add_step(
            run_id=state.run_id,
            step_index=step.index,
            thought=thought,
            tool_name=tc.name,
            tool_args=execution_args,
            original_tool_args=original_tool_args,
            prompt_context=prompt_context,
            observation=observation_text,
            status=step.status,
            error=error,
            started_at=start_dt,
            ended_at=end_dt,
        )
        await self.ctx.trace.add_tool_call(
            run_id=state.run_id,
            step_index=step.index,
            tool_name=tc.name,
            arguments=execution_args,
            original_arguments=original_tool_args,
            prompt_context=prompt_context,
            result=result_data,
            ok=ok,
            truncated=truncated,
            duration_ms=duration_ms,
            error=error,
        )

        # Append the tool response so the model can consume the observation.
        state.messages.append(Message.tool(tool_call_id=tc.id, content=observation_text, name=tc.name))

        await _emit(
            on_event,
            {
                "type": "tool_result",
                "step": step.index,
                "tool": tc.name,
                "ok": ok,
                "truncated": truncated,
                "duration_ms": duration_ms,
                "observation": observation_text,
            },
        )

    def _compose_system_prompt(self) -> PromptComposition:
        """Build the run-level system prompt once from active runtime capabilities."""
        if self.prompt_composer is not None:
            return self.prompt_composer.compose_system(self.registry.names())
        # Lightweight tests and third-party manual assembly remain compatible.
        return PromptComposition(
            content=load_prompt("system_react"),
            phase="system",
            manifest_version="legacy",
        )

    async def _refresh_permission_overrides(self, on_event: EventCallback | None) -> None:
        """Refresh DB rules before a tool call, retaining YAML as the fallback.

        The default refresh interval is zero, so a long-lived runtime observes
        permission edits immediately. A positive interval limits database reads
        while still replacing the complete override set atomically.
        """
        if not self.settings.permissions_db_overrides_enabled:
            self.permissions.clear_overrides()
            return
        if self._permissions_repo is None:
            return

        now = time.monotonic()
        try:
            interval = max(float(self.settings.permissions_refresh_seconds), 0.0)
        except (TypeError, ValueError):
            interval = 0.0
        if (
            interval > 0
            and self._last_permission_override_refresh is not None
            and now - self._last_permission_override_refresh < interval
        ):
            return

        self._last_permission_override_refresh = now
        try:
            rows = await asyncio.to_thread(self._permissions_repo.list_all)
            issues = self.permissions.replace_overrides(rows)
        except Exception as exc:
            # A failed refresh must never leave a stale dynamic allow/deny in
            # effect. The baseline YAML policy is the deterministic fallback.
            self.permissions.clear_overrides()
            await _emit(
                on_event,
                {
                    "type": "permission_overrides_error",
                    "error": type(exc).__name__,
                    "fallback": "yaml",
                },
            )
            return

        await _emit(
            on_event,
            {
                "type": "permission_overrides_refreshed",
                "count": len(self.permissions.overrides),
                "issues": len(issues),
            },
        )


def _error_observation(error: str) -> str:
    return json.dumps({"error": error}, ensure_ascii=False)


async def _emit(callback: EventCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    result = callback(event)
    if asyncio.iscoroutine(result):
        await result
