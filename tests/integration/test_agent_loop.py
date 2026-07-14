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
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from pkb_agent.agent.errors import AgentError, ToolNotFoundError
from pkb_agent.agent.runtime import AgentRuntime
from pkb_agent.agent.state import AgentStatus
from pkb_agent.app.settings import Settings
from pkb_agent.llm.schemas import (
    ChatChoice,
    ChatCompletion,
    Message,
    ToolCall,
    build_chat_request,
    parse_chat_completion,
)
from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.permissions import PermissionManager
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
) -> AgentRuntime:
    """Build an AgentRuntime with mocked dependencies (no real services)."""
    settings = Settings(agent_max_steps=max_steps, agent_tool_result_max_chars=1000)
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
    body = build_chat_request(model="m", messages=[choice.message], temperature=0.2)
    assert body["messages"][0]["role"] == "assistant"
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "echo"


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
    trace.finish_run.assert_awaited_once()
    finish_kwargs = trace.finish_run.call_args.kwargs
    assert finish_kwargs["status"] == "finished"
    assert finish_kwargs["final_answer"] == "42 is the answer"


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


# --------------------------------------------------------------------------- #
# AgentRuntime loop: max steps exhausted
# --------------------------------------------------------------------------- #


async def test_run_max_steps_reached():
    tc = ToolCall(id="tc1", name="echo", arguments={"message": "loop"})
    trace = _make_trace()
    rt = _make_runtime(
        llm_side_effect=[_tool_completion("again", tc), _tool_completion("again", tc)],
        tools=[_EchoTool()],
        max_steps=2,
        trace=trace,
    )

    state = await rt.run("keep looping")

    assert state.status is AgentStatus.MAX_STEPS
    assert state.step_count == 2
    assert "max_steps=2" in (state.error or "")
    finish_kwargs = trace.finish_run.call_args.kwargs
    assert finish_kwargs["status"] == "max_steps"


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
