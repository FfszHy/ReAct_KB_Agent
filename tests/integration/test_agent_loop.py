"""Integration tests for the ReAct agent loop and its building blocks.

``AgentRuntime.build()`` wires up real Supabase / DeepSeek / embedding
dependencies, so it cannot be used in an offline test. Instead we construct an
``AgentRuntime`` directly and inject mock collaborators (LLM, trace recorder,
tool registry, permissions) — exercising the real ``_loop`` /
``_execute_tool_call`` logic against canned LLM responses. A few lower-level
components (ToolRegistry, ToolResult, schema (de)serialization) are covered as
well for cohesion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from pkb_agent.agent.errors import AgentError, ToolNotFoundError
from pkb_agent.agent.runtime import AgentRuntime
from pkb_agent.agent.state import AgentStatus
from pkb_agent.agent.verification import ValidationResult, VerifiedAnswer
from pkb_agent.app.settings import Settings
from pkb_agent.llm.schemas import (
    ChatChoice,
    ChatCompletion,
    Message,
    ToolCall,
    build_chat_request,
    parse_chat_completion,
)
from pkb_agent.prompts import PromptComposer, PromptRegistry
from pkb_agent.prompts.query_rewriter import QueryPlan
from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.builtin.memory_write import MemoryWriteTool
from pkb_agent.tools.permissions import Permission, PermissionManager, ToolPermissionRule
from pkb_agent.tools.registry import ToolRegistry
from pkb_agent.tools.result import ToolResult

# --------------------------------------------------------------------------- #
# Concrete stub tools
# --------------------------------------------------------------------------- #


class _EchoTool(BaseTool):
    name = "echo"
    description = "echo back the message"
    params: ClassVar[list[ToolParam]] = [
        ToolParam(name="message", type="string", description="the message")
    ]

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult.success({"echo": args.get("message")})


class _FailTool(BaseTool):
    name = "fail"
    description = "always returns a failure"
    params: ClassVar[list[ToolParam]] = []

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult.failure("boom failure")


class _RecordingEchoTool(_EchoTool):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult.success({"echo": args.get("message")})


class _RecordingRagTool(BaseTool):
    name = "rag_search"
    description = "record search arguments"
    params: ClassVar[list[ToolParam]] = [
        ToolParam(name="query", type="string", description="search query"),
        ToolParam(name="queries", type="array", description="expanded queries", required=False),
    ]

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult.success({"ok": True})


class _EvidenceRagTool(BaseTool):
    name = "rag_search"
    description = "return a citation-eligible chunk"
    params: ClassVar[list[ToolParam]] = [
        ToolParam(name="query", type="string", description="search query")
    ]

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult.success(
            {
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "chunk_index": 0,
                        "title": "Architecture decision",
                        "source_uri": "kb://doc-1",
                        "content_preview": "Use SSO for authentication.",
                    }
                ]
            }
        )


class _RewriteStub:
    async def rewrite(self, *, tool_name: str, query: str, question: str = "") -> QueryPlan:
        assert tool_name == "rag_search"
        assert question
        return QueryPlan(
            original_query=query,
            queries=("auth architecture decision", "SSO design"),
            status="applied",
            prompt_context={"prompts": [{"id": "query_rewrite", "sha256": "test"}]},
        )


class _UsageRewriteStub(_RewriteStub):
    async def rewrite(self, *, tool_name: str, query: str, question: str = "") -> QueryPlan:
        plan = await super().rewrite(tool_name=tool_name, query=query, question=question)
        return QueryPlan(
            original_query=plan.original_query,
            queries=plan.queries,
            status=plan.status,
            prompt_context=plan.prompt_context,
            usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        )


class _PassthroughAnswerVerifier:
    """Keeps legacy loop tests focused on orchestration, not answer contracts."""

    def validate(self, content: str | None, evidence) -> ValidationResult:
        return ValidationResult(
            payload=VerifiedAnswer(
                answer=content or "",
                claims=(),
                citations=(),
                status="grounded",
            )
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _answer_completion(answer: str) -> ChatCompletion:
    return ChatCompletion(
        id="c1",
        model="deepseek-v4-flash",
        choices=[
            ChatChoice(index=0, message=Message.assistant(content=answer), finish_reason="stop")
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


def _tool_completion(thought: str, tc: ToolCall) -> ChatCompletion:
    return ChatCompletion(
        id="c2",
        model="deepseek-v4-flash",
        choices=[
            ChatChoice(
                index=0,
                message=Message.assistant(content=thought, tool_calls=[tc]),
                finish_reason="tool_calls",
            )
        ],
        usage={},
    )


def _make_trace() -> MagicMock:
    trace = MagicMock()
    trace.start_run = AsyncMock()
    trace.finish_run = AsyncMock()
    trace.add_step = AsyncMock()
    trace.add_tool_call = AsyncMock()
    return trace


def _make_runtime(
    *,
    llm_side_effect: list[Any],
    max_steps: int = 12,
    tools: list[BaseTool] | None = None,
    trace: MagicMock | None = None,
    strict_verification: bool = False,
) -> AgentRuntime:
    """Build an AgentRuntime with mocked dependencies (no real services)."""
    settings = Settings(agent_max_steps=max_steps, agent_tool_result_max_chars=1000,
                        agent_semantic_review_enabled=False)
    rt = AgentRuntime(settings)
    rt.llm = MagicMock()
    rt.llm.chat = AsyncMock(side_effect=llm_side_effect)
    registry = ToolRegistry()
    if tools:
        registry.register_all(tools)
    rt.registry = registry
    rt.permissions = PermissionManager()
    mock_trace = trace if trace is not None else _make_trace()
    rt.ctx = ToolContext(settings=settings, supabase=None, trace=mock_trace)
    if not strict_verification:
        rt.answer_verifier = _PassthroughAnswerVerifier()
    rt._assembled = True
    return rt


# --------------------------------------------------------------------------- #
# Component: ToolRegistry
# --------------------------------------------------------------------------- #


def test_tool_registry_register_get_require_and_schemas():
    registry = ToolRegistry()
    tool = _EchoTool()
    registry.register(tool)

    assert registry.get("echo") is tool
    assert registry.require("echo") is tool
    assert "echo" in registry
    assert len(registry) == 1
    schemas = registry.to_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "echo"
    assert "message" in schemas[0]["function"]["parameters"]["properties"]


def test_tool_registry_require_unknown_raises_not_found():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError, match="tool not found: ghost"):
        registry.require("ghost")


def test_runtime_can_restrict_the_tool_surface_for_an_evaluation_profile():
    rt = _make_runtime(
        llm_side_effect=[],
        tools=[_EchoTool(), _FailTool()],
    )

    rt.restrict_tools(["echo"])

    assert rt.registry.names() == ["echo"]
    with pytest.raises(ValueError, match="unknown tools"):
        rt.restrict_tools(["missing"])


async def test_runtime_prompt_explicitly_lists_only_the_restricted_tools():
    rt = _make_runtime(
        llm_side_effect=[_answer_completion("done")],
        tools=[_EchoTool(), _FailTool()],
    )
    rt.restrict_tools(["echo"])

    state = await rt.run("what can I use?")

    system_prompt = state.messages[0].content or ""
    assert "## Runtime tool allowlist" in system_prompt
    assert "`echo`" in system_prompt
    assert "`fail`" not in system_prompt
    assert state.prompt_context["available_tools"] == ["echo"]


# --------------------------------------------------------------------------- #
# Component: ToolResult
# --------------------------------------------------------------------------- #


def test_tool_result_success_to_observation_serializes_json():
    result = ToolResult.success({"answer": 42, "items": ["a", "b"]})
    obs = result.to_observation()
    parsed = json.loads(obs)
    assert parsed == {"answer": 42, "items": ["a", "b"]}
    assert result.ok is True


def test_tool_result_failure_to_observation_includes_error():
    result = ToolResult.failure("something broke")
    obs = result.to_observation()
    parsed = json.loads(obs)
    assert parsed == {"error": "something broke"}
    assert result.ok is False


def test_tool_result_to_observation_truncates_long_payload():
    big = {"data": "x" * 500}
    result = ToolResult.success(big)
    obs = result.to_observation(max_chars=100)
    assert len(obs) <= 100
    assert obs.endswith("…[truncated]")
    assert result.truncated is False  # truncated flag reflects tool-internal truncation


# --------------------------------------------------------------------------- #
# Component: schemas (ToolCall / parse_chat_completion)
# --------------------------------------------------------------------------- #


def test_tool_call_from_raw_parses_string_arguments():
    raw = {
        "id": "call_1",
        "function": {"name": "echo", "arguments": '{"message": "hello"}'},
    }
    tc = ToolCall.from_raw(raw)
    assert tc.id == "call_1"
    assert tc.name == "echo"
    assert tc.arguments == {"message": "hello"}


def test_parse_chat_completion_round_trip_with_tool_calls():
    payload = {
        "id": "chat-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "echo", "arguments": '{"message":"hi"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    completion = parse_chat_completion(payload)
    choice = completion.first
    assert choice.message.role == "assistant"
    assert choice.message.content == "let me check"
    assert len(choice.message.tool_calls) == 1
    tc = choice.message.tool_calls[0]
    assert tc.name == "echo"
    assert tc.arguments == {"message": "hi"}
    assert completion.usage == {"prompt_tokens": 5, "completion_tokens": 3}

    # build_chat_request must round-trip the message back to a serializable dict.
    body = build_chat_request(
        model="m",
        messages=[choice.message],
        temperature=0.2,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    assert body["messages"][0]["role"] == "assistant"
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "echo"
    assert body["max_tokens"] == 1024
    assert body["response_format"] == {"type": "json_object"}


def test_chat_completion_first_raises_when_no_choices():
    completion = ChatCompletion(id="x", model="m", choices=[])
    with pytest.raises(IndexError, match="no choices"):
        _ = completion.first


# --------------------------------------------------------------------------- #
# AgentRuntime loop: single-step answer
# --------------------------------------------------------------------------- #


async def test_run_single_step_answer_no_tools():
    trace = _make_trace()
    rt = _make_runtime(llm_side_effect=[_answer_completion("42 is the answer")], trace=trace)

    state = await rt.run("what is the answer?")

    assert state.status is AgentStatus.FINISHED
    assert state.final_answer == "42 is the answer"
    assert state.step_count == 0
    # Two seeding messages + the final assistant message.
    assert len(state.messages) == 3
    assert state.messages[-1].role == "assistant"
    rt.llm.chat.assert_awaited_once()
    chat_kwargs = rt.llm.chat.call_args.kwargs
    assert chat_kwargs["response_format"] == {"type": "json_object"}
    assert chat_kwargs["max_tokens"] == rt.settings.agent_json_output_max_tokens
    trace.finish_run.assert_awaited_once()
    finish_kwargs = trace.finish_run.call_args.kwargs
    assert finish_kwargs["status"] == "finished"
    assert finish_kwargs["final_answer"] == "42 is the answer"


async def test_run_composes_memory_policy_when_memory_write_is_registered():
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[_answer_completion("done")],
        tools=[MemoryWriteTool()],
        trace=trace,
    )
    root = Path(__file__).parents[2]
    rt.prompt_composer = PromptComposer(
        PromptRegistry(
            prompt_dir=root / "config" / "prompts",
            manifest_path=root / "config" / "prompts" / "manifest.yaml",
        )
    )

    state = await rt.run("hello")

    assert "What to write to memory" in (state.messages[0].content or "")
    start_kwargs = trace.start_run.call_args.kwargs
    assert [item["id"] for item in start_kwargs["prompt_context"]["prompts"]] == [
        "system_react",
        "memory_policy",
    ]


# --------------------------------------------------------------------------- #
# AgentRuntime loop: one tool call then answer
# --------------------------------------------------------------------------- #


async def test_run_one_tool_call_then_answer():
    tc = ToolCall(id="tc1", name="echo", arguments={"message": "hello"})
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[
            _tool_completion("I will echo", tc),
            _answer_completion("echoed: hello"),
        ],
        tools=[_EchoTool()],
        trace=trace,
    )

    state = await rt.run("echo hello")

    assert state.status is AgentStatus.FINISHED
    assert state.final_answer == "echoed: hello"
    assert state.step_count == 1
    # system, user, assistant(tool_call), tool(obs), assistant(answer)
    assert len(state.messages) == 5
    assert state.messages[2].role == "assistant"
    assert state.messages[2].tool_calls[0].name == "echo"
    assert state.messages[3].role == "tool"
    assert "hello" in state.messages[3].content
    # Tool call was recorded in the trace.
    trace.add_tool_call.assert_awaited_once()
    tool_call_kwargs = trace.add_tool_call.call_args.kwargs
    assert tool_call_kwargs["tool_name"] == "echo"
    assert tool_call_kwargs["ok"] is True


async def test_database_permission_overrides_refresh_before_each_tool_call():
    tool = _RecordingEchoTool()
    first = ToolCall(id="tc1", name="echo", arguments={"message": "first"})
    second = ToolCall(id="tc2", name="echo", arguments={"message": "second"})
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[
            _tool_completion("first", first),
            _tool_completion("second", second),
            _answer_completion("done"),
        ],
        tools=[tool],
        trace=trace,
    )
    rt.permissions = PermissionManager(
        rules={"echo": ToolPermissionRule(Permission.ALLOW)}
    )
    repo = MagicMock()
    repo.list_all.side_effect = [
        [{"tool_name": "echo", "permission": "deny", "constraints": {}}],
        [],
    ]
    rt._permissions_repo = repo
    events: list[dict[str, Any]] = []

    state = await rt.run("try twice", on_event=events.append)

    assert state.steps[0].status == "error"
    assert "permission denied" in (state.steps[0].error or "")
    assert state.steps[1].status == "ok"
    assert tool.calls == [{"message": "second"}]
    assert repo.list_all.call_count == 2
    sources = [event["permission_source"] for event in events if event["type"] == "tool_call"]
    assert sources == ["database", "yaml"]


async def test_database_permission_load_failure_falls_back_to_yaml():
    tool = _RecordingEchoTool()
    tc = ToolCall(id="tc1", name="echo", arguments={"message": "blocked"})
    rt = _make_runtime(
        llm_side_effect=[_tool_completion("try", tc), _answer_completion("done")],
        tools=[tool],
    )
    rt.permissions = PermissionManager(rules={"echo": ToolPermissionRule(Permission.DENY)})
    repo = MagicMock()
    repo.list_all.side_effect = RuntimeError("database offline")
    rt._permissions_repo = repo
    events: list[dict[str, Any]] = []

    state = await rt.run("try", on_event=events.append)

    assert state.steps[0].status == "error"
    assert tool.calls == []
    fallback = next(event for event in events if event["type"] == "permission_overrides_error")
    assert fallback["fallback"] == "yaml"


async def test_run_rewrites_retrieval_arguments_and_traces_both_versions():
    tool = _RecordingRagTool()
    tc = ToolCall(id="tc1", name="rag_search", arguments={"query": "What did we decide about auth?"})
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[_tool_completion("search", tc), _answer_completion("done")],
        tools=[tool],
        trace=trace,
    )
    rt.query_planner = _RewriteStub()  # type: ignore[assignment]

    state = await rt.run("What did we decide about auth?")

    assert state.steps[0].tool_call.arguments == {"query": "What did we decide about auth?"}
    assert state.steps[0].execution_arguments == {
        "query": "auth architecture decision",
        "queries": ["auth architecture decision", "SSO design"],
    }
    assert tool.calls == [state.steps[0].execution_arguments]
    trace_kwargs = trace.add_tool_call.call_args.kwargs
    assert trace_kwargs["original_arguments"] == {"query": "What did we decide about auth?"}
    assert trace_kwargs["arguments"] == state.steps[0].execution_arguments
    assert trace_kwargs["prompt_context"]["prompts"][0]["id"] == "query_rewrite"


async def test_run_cost_includes_query_rewrite_model_usage():
    tool = _RecordingRagTool()
    tc = ToolCall(id="tc1", name="rag_search", arguments={"query": "auth architecture"})
    rt = _make_runtime(
        llm_side_effect=[_tool_completion("search", tc), _answer_completion("done")],
        tools=[tool],
    )
    rt.query_planner = _UsageRewriteStub()  # type: ignore[assignment]

    state = await rt.run("What authentication approach should we use?")

    assert state.usage["prompt_tokens"] == 14
    assert state.usage["completion_tokens"] == 7
    assert state.usage["total_tokens"] == 21


async def test_run_repairs_invalid_citations_against_this_runs_evidence_ledger():
    tc = ToolCall(id="tc1", name="rag_search", arguments={"query": "authentication"})
    invalid = json.dumps(
        {
            "answer": "Use SSO.",
            "claims": [
                {"text": "Use SSO.", "kind": "fact", "citations": ["kb:invented"]}
            ],
            "citations": [{"id": "kb:invented"}],
        }
    )
    repaired = json.dumps(
        {
            "status": "grounded",
            "answer": "Use SSO for authentication.",
            "claims": [
                {"text": "Use SSO for authentication.", "kind": "fact", "citations": ["kb:chunk-1"]}
            ],
            "citations": [{"id": "kb:chunk-1"}],
        }
    )
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[
            _tool_completion("search", tc),
            _answer_completion(invalid),
            _answer_completion(repaired),
        ],
        tools=[_EvidenceRagTool()],
        trace=trace,
        strict_verification=True,
    )
    events: list[dict[str, Any]] = []

    state = await rt.run("What authentication approach should we use?", on_event=events.append)

    assert state.status is AgentStatus.FINISHED
    assert state.final_answer == "Use SSO for authentication."
    assert state.verification["status"] == "structure_verified"
    assert state.verification["repair_attempts"] == 1
    assert state.answer_payload["citations"][0]["source_id"] == "chunk-1"
    assert "citation_evidence" in (state.messages[3].content or "")
    assert "kb:chunk-1" in (state.messages[3].content or "")
    assert rt.llm.chat.await_count == 3
    failed = next(event for event in events if event["type"] == "answer_verification_failed")
    assert "was not retrieved in this run" in failed["errors"][0]
    finish_kwargs = trace.finish_run.call_args.kwargs
    assert finish_kwargs["answer_payload"]["status"] == "grounded"


async def test_run_refuses_when_final_answer_cannot_be_verified():
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[_answer_completion("this is not JSON")],
        trace=trace,
        strict_verification=True,
    )
    rt.settings.agent_answer_verification_max_retries = 0

    state = await rt.run("answer without evidence")

    assert state.status is AgentStatus.FINISHED
    assert state.answer_payload["status"] == "insufficient_evidence"
    assert state.verification["status"] == "refused"
    assert state.verification["reason"] == "answer_validation_failed"
    assert state.answer_payload["claims"] == []
    assert state.answer_payload["citations"] == []
    finish_kwargs = trace.finish_run.call_args.kwargs
    assert finish_kwargs["verification"]["status"] == "refused"


def _grounded_candidate(text: str) -> str:
    return json.dumps({
        "status": "grounded", "answer": text,
        "claims": [{"text": text, "kind": "fact", "citations": ["kb:chunk-1"]}],
        "citations": [{"id": "kb:chunk-1"}],
    })


def _supported_review(claim_count: int = 1) -> str:
    return json.dumps({"status": "passed", "errors": [], "checks": [
        {"claim_index": index, "supported": True,
         "reason": "The retrieved source directly recommends SSO for authentication.",
         "evidence": [{"id": "kb:chunk-1", "quote": "Use SSO for authentication."}]}
        for index in range(claim_count)
    ]})


async def test_uncovered_answer_text_is_repaired_before_semantic_review_and_publication():
    repaired = json.loads(_grounded_candidate("Use SSO for authentication."))
    repaired["claims"].append({
        "text": "Authentication is the use case covered by this recommendation.",
        "kind": "fact",
        "citations": ["kb:chunk-1"],
    })
    repaired["answer"] = "\n\n".join(claim["text"] for claim in repaired["claims"])
    invalid = {**repaired, "answer": "Only SSO works.\n" + repaired["answer"]}
    rt = _make_runtime(
        llm_side_effect=[
            _tool_completion("", ToolCall(id="t1", name="rag_search", arguments={"query": "auth"})),
            _answer_completion(json.dumps(invalid)),
            _answer_completion(json.dumps(repaired)),
            _answer_completion(_supported_review(2)),
        ], tools=[_EvidenceRagTool()], strict_verification=True,
    )
    rt.settings.agent_semantic_review_enabled = True
    events = []

    state = await rt.run("What does the document recommend?", on_event=events.append)

    assert state.final_answer == repaired["answer"]
    assert state.answer_payload["claims"] == repaired["claims"]
    assert state.verification["status"] == "verified"
    assert state.verification["repair_attempts"] == 1
    assert [review["status"] for review in state.semantic_reviews] == ["passed"]
    assert rt.llm.chat.await_count == 4  # no semantic review of the structurally invalid answer
    review_input = json.loads(rt.llm.chat.call_args.args[0][-1].content)
    assert review_input["candidate"]["answer"] == repaired["answer"]
    assert review_input["candidate"]["claims"] == [
        {"claim_index": index, **claim} for index, claim in enumerate(repaired["claims"])
    ]
    failed = next(event for event in events if event["type"] == "answer_verification_failed")
    assert "exact ordered spans" in failed["errors"][0]
    assert [event["answer"] for event in events if event["type"] == "answer"] == [repaired["answer"]]


async def test_semantic_failure_repairs_before_publishing_and_counts_review_cost():
    rt = _make_runtime(
        llm_side_effect=[
            _tool_completion("", ToolCall(id="t1", name="rag_search", arguments={"query": "auth"})),
            _answer_completion(_grounded_candidate("Only SSO can authenticate users.")),
            _answer_completion('{"status":"failed","errors":["Only is not supported by the source"]}'),
            _answer_completion(_grounded_candidate("Use SSO for authentication.")),
            _answer_completion(_supported_review()),
        ], tools=[_EvidenceRagTool()], strict_verification=True,
    )
    rt.settings.agent_semantic_review_enabled = True
    events = []
    state = await rt.run("What does the document recommend?", on_event=events.append)
    assert state.final_answer == "Use SSO for authentication."
    assert state.verification["status"] == "verified"
    assert state.verification["scope"] == "citation_integrity_and_model_review"
    assert state.verification["repair_attempts"] == 1
    assert [r["status"] for r in state.semantic_reviews] == ["failed", "passed"]
    assert state.usage["total_tokens"] == 60
    assert state.model_rounds == 3  # tool round + two candidates, excluding review calls
    assert [e["observed_tool_results"] for e in events if e["type"] == "model_turn"] == [0, 1, 1]
    assert [e["answer"] for e in events if e["type"] == "answer"] == [state.final_answer]
    assert "Only is not supported" in state.messages[-2].content


async def test_semantic_service_failure_stops_without_useless_answer_rewrites():
    from pkb_agent.agent.errors import LLMError

    rt = _make_runtime(
        llm_side_effect=[
            _tool_completion("", ToolCall(id="t1", name="rag_search", arguments={"query": "auth"})),
            _answer_completion(_grounded_candidate("Use SSO.")),
            LLMError("offline"),
        ], tools=[_EvidenceRagTool()], strict_verification=True,
    )
    rt.settings.agent_semantic_review_enabled = True
    state = await rt.run("What do we know?")
    assert state.verification["status"] == "refused"
    assert state.verification["reason"] == "semantic_review_unavailable"
    assert state.verification["repair_attempts"] == 0
    assert rt.llm.chat.await_count == 3
    assert not state.metrics()["run_succeeded"]


async def test_answer_length_limit_repairs_complete_json_without_truncating():
    def refusal(text):
        return json.dumps({"status": "insufficient_evidence", "answer": text,
                           "claims": [], "citations": []})
    rt = _make_runtime(llm_side_effect=[
        _answer_completion(refusal("There is not enough evidence to answer this question.")),
        _answer_completion(refusal("无法确定。")),
    ], strict_verification=True)
    state = await rt.run("回答控制在10字以内。")
    assert state.max_answer_chars == 10
    assert state.final_answer == "无法确定。"
    assert state.verification_attempts == 1
    assert "shorten it" in state.messages[-2].content


async def test_grounded_length_repair_preserves_complete_claim_and_citation_contract():
    rt = _make_runtime(llm_side_effect=[
        _tool_completion("", ToolCall(id="t1", name="rag_search", arguments={"query": "auth"})),
        _answer_completion(_grounded_candidate("Use SSO for authentication.")),
        _answer_completion(_grounded_candidate("Use SSO.")),
    ], tools=[_EvidenceRagTool()], strict_verification=True)
    events = []

    state = await rt.run("回答控制在10字以内。", on_event=events.append)

    assert state.max_answer_chars == 10
    assert state.final_answer == "Use SSO."
    assert state.answer_payload["claims"][0]["text"] == state.final_answer
    assert state.answer_payload["citations"][0]["id"] == "kb:chunk-1"
    assert state.verification["status"] == "structure_verified"
    assert state.verification_attempts == 1
    assert "shorten it" in state.messages[-2].content
    assert rt.llm.chat.await_count == 3
    assert [event["answer"] for event in events if event["type"] == "answer"] == ["Use SSO."]


async def test_refusal_never_publishes_uncited_claims_hidden_in_its_body():
    candidate = json.dumps({"status": "insufficient_evidence",
                            "answer": "No records, but measured p99 is 12 ms.",
                            "claims": [], "citations": []})
    rt = _make_runtime(llm_side_effect=[_answer_completion(candidate)], strict_verification=True)
    rt.settings.agent_semantic_review_enabled = True
    state = await rt.run("What is the measured p99?")
    assert "12" not in state.final_answer
    assert "无法" in state.final_answer
    assert state.verification["answer_origin"] == "runtime_boundary_template"
    assert state.verification["semantic_review"]["status"] == "not_required"
    assert rt.llm.chat.await_count == 1
    assert any(message.content == candidate for message in state.messages)


@pytest.mark.parametrize("limit", [1, 2, 5])
async def test_validation_failure_boundary_respects_answer_limit(limit):
    rt = _make_runtime(llm_side_effect=[_answer_completion("not JSON")], strict_verification=True)
    rt.settings.agent_answer_verification_max_retries = 0

    state = await rt.run("Give an answer.", max_answer_chars=limit)

    assert state.verification["reason"] == "answer_validation_failed"
    assert 0 < len(state.final_answer) <= limit
    assert state.verification["answer_chars"] == len(state.final_answer)
    assert state.verification["max_answer_chars"] == limit
    assert state.answer_payload["claims"] == []


async def test_truncated_json_or_semantic_failures_are_bounded_and_never_published():
    rt = _make_runtime(llm_side_effect=[
        _tool_completion("", ToolCall(id="t1", name="rag_search", arguments={"query": "auth"})),
        _answer_completion(_grounded_candidate("Only SSO works.")),
        _answer_completion('{"status":"failed","errors":["unsupported exclusive claim"]}'),
    ], tools=[_EvidenceRagTool()], strict_verification=True)
    rt.settings.agent_semantic_review_enabled = True
    rt.settings.agent_answer_verification_max_retries = 0
    state = await rt.run("What is supported?")
    assert state.verification["status"] == "refused"
    assert "Only SSO" not in state.final_answer
    assert state.semantic_reviews[-1]["status"] == "failed"

    cut_off = _answer_completion(_grounded_candidate("Use SSO."))
    cut_off.choices[0].finish_reason = "length"
    rt2 = _make_runtime(llm_side_effect=[cut_off], strict_verification=True)
    rt2.settings.agent_answer_verification_max_retries = 0
    state2 = await rt2.run("Answer briefly.")
    assert state2.verification["status"] == "refused"
    assert "output token limit" in state2.verification["errors"][0]


# --------------------------------------------------------------------------- #
# AgentRuntime loop: tool budget finalization
# --------------------------------------------------------------------------- #


async def test_run_tool_budget_finalizes_without_a_max_step_error():
    tc = ToolCall(id="tc1", name="echo", arguments={"message": "loop"})
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[_tool_completion("again", tc), _answer_completion("done")],
        tools=[_EchoTool()],
        max_steps=2,
        trace=trace,
    )

    state = await rt.run("keep looping")

    assert state.status is AgentStatus.FINISHED
    assert state.step_count == 1
    assert state.budget_finalized is True
    assert state.error is None
    finish_kwargs = trace.finish_run.call_args.kwargs
    assert finish_kwargs["status"] == "finished"
    assert rt.llm.chat.call_args_list[-1].kwargs["tools"] is None


async def test_tool_budget_closes_skipped_calls_in_a_multi_tool_completion():
    first = ToolCall(id="tc1", name="echo", arguments={"message": "first"})
    skipped = ToolCall(id="tc2", name="echo", arguments={"message": "second"})
    completion = ChatCompletion(
        id="batch",
        model="deepseek-v4-flash",
        choices=[
            ChatChoice(
                index=0,
                message=Message.assistant(content="batch", tool_calls=[first, skipped]),
                finish_reason="tool_calls",
            )
        ],
        usage={},
    )
    rt = _make_runtime(
        llm_side_effect=[completion, _answer_completion("done")],
        tools=[_EchoTool()],
        max_steps=2,
    )

    state = await rt.run("keep looping")

    assert state.status is AgentStatus.FINISHED
    assert state.step_count == 1
    skipped_message = next(message for message in state.messages if message.tool_call_id == "tc2")
    assert "tool budget exhausted" in (skipped_message.content or "")
    assert rt.llm.chat.call_args_list[-1].kwargs["tools"] is None


async def test_trace_write_failure_does_not_turn_a_completed_answer_into_an_execution_error():
    trace = _make_trace()
    trace.finish_run = AsyncMock(return_value=False)
    rt = _make_runtime(llm_side_effect=[_answer_completion("done")], trace=trace)

    state = await rt.run("answer despite trace failure")

    assert state.status is AgentStatus.FINISHED
    assert state.error is None
    assert state.trace_write_failure_count == 1
    assert state.metrics()["execution_succeeded"] is True


# --------------------------------------------------------------------------- #
# AgentRuntime loop: tool failure is captured, loop continues
# --------------------------------------------------------------------------- #


async def test_run_tool_failure_observation_recorded_and_continues():
    tc = ToolCall(id="tc1", name="fail", arguments={})
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[
            _tool_completion("trying fail", tc),
            _answer_completion("recovered"),
        ],
        tools=[_FailTool()],
        trace=trace,
    )

    state = await rt.run("use the failing tool")

    assert state.status is AgentStatus.FINISHED
    assert state.final_answer == "recovered"
    assert state.step_count == 1
    # The tool observation must carry the error payload.
    tool_msg = state.messages[3]
    assert tool_msg.role == "tool"
    assert "boom failure" in tool_msg.content
    assert state.steps[0].error == "boom failure"
    # Trace recorded the failure with ok=False.
    tool_call_kwargs = trace.add_tool_call.call_args.kwargs
    assert tool_call_kwargs["ok"] is False
    assert tool_call_kwargs["error"] is not None


# --------------------------------------------------------------------------- #
# AgentRuntime loop: unknown tool handled gracefully
# --------------------------------------------------------------------------- #


async def test_run_unknown_tool_handled_gracefully():
    tc = ToolCall(id="tc1", name="nonexistent_tool", arguments={})
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[
            _tool_completion("calling ghost", tc),
            _answer_completion("moved on"),
        ],
        tools=[_EchoTool()],
        trace=trace,
    )

    state = await rt.run("call a missing tool")

    assert state.status is AgentStatus.FINISHED
    assert state.final_answer == "moved on"
    tool_msg = state.messages[3]
    assert "unknown tool" in tool_msg.content
    tool_call_kwargs = trace.add_tool_call.call_args.kwargs
    assert tool_call_kwargs["ok"] is False


# --------------------------------------------------------------------------- #
# AgentRuntime loop: not assembled guard + events
# --------------------------------------------------------------------------- #


async def test_run_not_assembled_raises_agent_error():
    settings = Settings(agent_max_steps=3)
    rt = AgentRuntime(settings)  # _assembled stays False

    with pytest.raises(AgentError, match="not assembled"):
        await rt.run("anything")


async def test_run_emits_start_and_answer_events():
    events: list[dict[str, Any]] = []
    trace = _make_trace()
    rt = _make_runtime(llm_side_effect=[_answer_completion("done")], trace=trace)

    await rt.run("hi", on_event=lambda e: events.append(e))

    types = [e["type"] for e in events]
    assert types[0] == "start"
    assert "answer" in types
    answer_event = next(e for e in events if e["type"] == "answer")
    assert answer_event["answer"] == "done"
