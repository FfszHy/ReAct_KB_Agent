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
from pkb_agent.agent.verification import AnswerVerifier, Evidence, extract_evidence, refusal_payload
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

# DeepSeek JSON Output guarantees JSON syntax for a non-tool completion.  The
# runtime still validates the required fields and runtime-issued citation IDs.
_JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}


class AgentRuntime:
    """Assembles all collaborators and runs the ReAct loop."""

    def __init__(
        self,
        settings: Settings,
        *,
        user_id: str = "default",
        confirm: Callable[[str, dict[str, Any]], bool | Awaitable[bool]] | None = None,
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
        self.answer_verifier = AnswerVerifier()
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
        confirm: Callable[[str, dict[str, Any]], bool | Awaitable[bool]] | None = None,
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
        self._closables = [sb, embedder, self.llm, search_provider, fetcher]
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
        run_id: str | None = None,
    ) -> AgentRunState:
        if not self._assembled:
            raise AgentError("runtime not assembled; call AgentRuntime.build()")
        uid = user_id or self._user_id or "default"
        self.ctx.user_id = uid

        state = AgentRunState(run_id=run_id, question=question) if run_id else AgentRunState(question=question)
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
            state.mark_finished()
            await self.ctx.trace.finish_run(
                state.run_id,
                status="error",
                error=state.error,
                step_count=state.step_count,
                usage=state.usage,
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
                max_tokens=self.settings.agent_json_output_max_tokens,
                response_format=_JSON_OBJECT_RESPONSE_FORMAT,
            )
            choice = completion.first
            msg = choice.message
            state.messages.append(msg)
            usage = completion.usage or {}
            state.add_usage(
                usage,
                input_cost_per_million=self.settings.observability_input_token_cost_per_million,
                cache_hit_input_cost_per_million=(
                    self.settings.observability_cache_hit_input_token_cost_per_million
                ),
                cache_miss_input_cost_per_million=(
                    self.settings.observability_cache_miss_input_token_cost_per_million
                ),
                output_cost_per_million=self.settings.observability_output_token_cost_per_million,
                currency=self.settings.observability_currency,
            )

            if not msg.tool_calls:
                finished = await self._handle_final_response(
                    state,
                    content=msg.content,
                    usage=usage,
                    on_event=on_event,
                )
                if finished:
                    return
                # The verifier requested a repair. The repair prompt is now in
                # the conversation and may lead to corrected JSON or more tool
                # retrieval on the next model turn.
                continue

            thought = msg.content
            for tool_position, tc in enumerate(msg.tool_calls):
                await self._execute_tool_call(
                    state,
                    tc,
                    thought,
                    on_event,
                    emit_plan=bool(thought) and tool_position == 0,
                )

        # Exhausted step budget.
        state.status = AgentStatus.MAX_STEPS
        state.error = f"exceeded max_steps={max_steps}"
        await self._finalize_refusal(
            state,
            reason="max_steps",
            errors=[state.error],
            on_event=on_event,
            trace_status="max_steps",
        )
        await _emit(on_event, {"type": "max_steps", "steps": state.step_count})

    # ------------------------------------------------------------------
    async def _execute_tool_call(
        self,
        state: AgentRunState,
        tc: ToolCall,
        thought: str | None,
        on_event: EventCallback | None,
        *,
        emit_plan: bool = False,
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

        if emit_plan:
            await _emit(
                on_event,
                {
                    "type": "plan",
                    "step": step.index,
                    "thought": thought,
                },
            )

        await _emit(
            on_event,
            {
                "type": "tool_call",
                "step": step.index,
                "tool": tc.name,
                "args": execution_args,
                "original_args": original_tool_args,
                "thought": thought,
                "prompt_context": prompt_context,
                "permission_source": permission_source,
            },
        )

        try:
            tool = self.registry.require(tc.name)
            tool.validate_args(execution_args)
            await self.permissions.check_async(tc.name, execution_args, self.ctx)
            result = await tool.execute(self.ctx, execution_args)
            ok = result.ok
            result_data = result.data
            truncated = result.truncated
            if not ok:
                error = result.error or "tool returned an error"
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

        end_dt = datetime.now(UTC)
        registered_evidence: list[Evidence] = []
        if ok:
            registered_evidence = extract_evidence(
                tc.name,
                result_data,
                settings=self.settings,
                observed_at=end_dt,
            )
            for evidence in registered_evidence:
                state.evidence[evidence.citation_id] = evidence
            if registered_evidence:
                result_data = _attach_citation_evidence(result_data, registered_evidence)
            observation_text = _data_observation(
                result_data,
                max_chars=self.settings.agent_tool_result_max_chars,
            )
        elif not observation_text:
            observation_text = _error_observation(error or "tool returned an error")

        duration_ms = int((time.time() - started) * 1000)
        step.finish(status="ok" if ok else "error", observation=observation_text)
        state.record_tool_call(duration_ms, ok=ok)

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
                "citation_evidence": [
                    evidence.to_prompt_dict() for evidence in registered_evidence
                ],
                "metrics": state.metrics(),
            },
        )

    async def _handle_final_response(
        self,
        state: AgentRunState,
        *,
        content: str | None,
        usage: dict[str, Any],
        on_event: EventCallback | None,
    ) -> bool:
        """Validate a model final response, repairing or refusing deterministically.

        Returning ``False`` keeps the ReAct loop alive after a repair prompt so
        the model may either correct its JSON or obtain additional evidence.
        """
        verdict = self.answer_verifier.validate(content, state.evidence)
        if verdict.valid:
            assert verdict.payload is not None
            state.status = AgentStatus.FINISHED
            state.final_answer = verdict.payload.answer
            state.answer_payload = verdict.payload.to_dict()
            state.verification = {
                "status": "verified"
                if verdict.payload.status == "grounded"
                else "insufficient_evidence",
                "repair_attempts": state.verification_attempts,
                "evidence_count": len(state.evidence),
                "cited_evidence_count": len(verdict.payload.citations),
                "errors": [],
            }
            state.mark_finished()
            await self.ctx.trace.finish_run(
                state.run_id,
                status="finished",
                final_answer=state.final_answer,
                answer_payload=state.answer_payload,
                verification=state.verification,
                step_count=state.step_count,
                usage=state.usage,
            )
            await _emit(
                on_event,
                {
                    "type": "answer",
                    "answer": state.final_answer,
                    "answer_payload": state.answer_payload,
                    "verification": state.verification,
                    "steps": state.step_count,
                    "metrics": state.metrics(),
                },
            )
            return True

        errors = list(verdict.errors) or ["unknown final-answer validation failure"]
        retry_limit = _answer_verification_retry_limit(self.settings)
        if state.verification_attempts < retry_limit:
            state.verification_attempts += 1
            state.messages.append(
                Message.user(_answer_repair_instruction(errors, state.evidence.values()))
            )
            await _emit(
                on_event,
                {
                    "type": "answer_verification_failed",
                    "errors": errors,
                    "retry": state.verification_attempts,
                    "retry_limit": retry_limit,
                    "evidence_count": len(state.evidence),
                },
            )
            return False

        state.status = AgentStatus.FINISHED
        await self._finalize_refusal(
            state,
            reason="answer_validation_failed",
            errors=errors,
            on_event=on_event,
            usage=usage,
            trace_status="finished",
        )
        return True

    async def _finalize_refusal(
        self,
        state: AgentRunState,
        *,
        reason: str,
        errors: list[str],
        on_event: EventCallback | None,
        trace_status: str,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Persist and emit a clear evidence-boundary refusal."""
        payload = refusal_payload(_refusal_message(reason))
        state.final_answer = payload.answer
        state.answer_payload = payload.to_dict()
        state.verification = {
            "status": "refused",
            "reason": reason,
            "repair_attempts": state.verification_attempts,
            "evidence_count": len(state.evidence),
            "cited_evidence_count": 0,
            "errors": errors,
        }
        state.mark_finished()
        await self.ctx.trace.finish_run(
            state.run_id,
            status=trace_status,
            final_answer=state.final_answer,
            answer_payload=state.answer_payload,
            verification=state.verification,
            error=state.error,
            step_count=state.step_count,
            usage=state.usage,
        )
        await _emit(
            on_event,
            {
                "type": "answer",
                "answer": state.final_answer,
                "answer_payload": state.answer_payload,
                "verification": state.verification,
                "steps": state.step_count,
                "metrics": state.metrics(),
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


def _attach_citation_evidence(data: Any, evidence: list[Evidence]) -> Any:
    """Expose only runtime-issued IDs to the model in the tool observation."""
    if not isinstance(data, dict):
        return data
    # Put IDs first: long fetched-page text may be truncated before the model
    # sees the rest of the observation.
    return {
        "citation_evidence": [record.to_prompt_dict() for record in evidence],
        **data,
    }


def _data_observation(data: Any, *, max_chars: int) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        text = str(data)
    if len(text) > max_chars:
        return text[: max(max_chars - 20, 0)].rstrip() + "\n…[truncated]"
    return text


def _answer_verification_retry_limit(settings: Settings) -> int:
    try:
        return min(max(int(settings.agent_answer_verification_max_retries), 0), 10)
    except (TypeError, ValueError):
        return 2


def _answer_repair_instruction(errors: list[str], evidence: Any) -> str:
    """Ask the model to repair the final JSON without weakening the contract."""
    records = [record.to_prompt_dict() for record in evidence]
    errors_text = "\n".join(f"- {error}" for error in errors)
    records_json = json.dumps(records, ensure_ascii=False)
    return (
        "Your previous final response was rejected by the answer verifier.\n"
        "Validation errors:\n"
        f"{errors_text}\n\n"
        "Return a replacement that follows the final JSON contract exactly. "
        "You may cite only IDs in this runtime-issued evidence list:\n"
        f"{records_json}\n\n"
        "If these sources are insufficient, either call a retrieval tool for more evidence "
        "or return the insufficient_evidence JSON shape. Do not explain the repair outside JSON."
    )


def _refusal_message(reason: str) -> str:
    if reason == "max_steps":
        return "无法在本轮允许的检索步骤内获得足以支撑回答的证据。"
    return "无法基于本轮检索到的证据生成可验证回答。"


async def _emit(callback: EventCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    result = callback(event)
    if asyncio.iscoroutine(result):
        await result
