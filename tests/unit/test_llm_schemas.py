"""Private provider continuity fields must not become public run artifacts."""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from pkb_agent.agent.state import AgentRunState
from pkb_agent.llm.deepseek_client import DeepSeekClient
from pkb_agent.llm.schemas import Message, build_chat_request, parse_chat_completion

PRIVATE_CONTINUITY = "private-provider-continuity-fixture"


def _tool_response(reasoning=PRIVATE_CONTINUITY):
    return {
        "id": "tool-turn",
        "model": "deepseek-v4-flash",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": reasoning,
                "tool_calls": [{
                    "id": "call-1", "type": "function",
                    "function": {"name": "rag_search", "arguments": '{"query":"source"}'},
                }],
            },
        }],
        "usage": {},
    }


def test_provider_reasoning_is_retained_without_public_serialization_or_repr():
    completion = parse_chat_completion(_tool_response())
    message = completion.first.message

    assert message.reasoning_content == PRIVATE_CONTINUITY
    assert "reasoning_content" not in message.to_dict()
    assert PRIVATE_CONTINUITY not in repr(message)
    assert PRIVATE_CONTINUITY not in repr(completion)
    assert PRIVATE_CONTINUITY not in json.dumps(AgentRunState(messages=[message]).to_dict())

    request = build_chat_request(model="deepseek-v4-flash", messages=[message])
    assert request["messages"][0]["reasoning_content"] == PRIVATE_CONTINUITY
    assert request["messages"][0]["tool_calls"][0]["id"] == "call-1"
    # Provider serialization does not mutate the public representation.
    assert "reasoning_content" not in message.to_dict()


@pytest.mark.parametrize("reasoning", [None, {"unexpected": "object"}, 123])
def test_absent_or_invalid_reasoning_field_is_not_sent(reasoning):
    message = parse_chat_completion(_tool_response(reasoning)).first.message
    request = build_chat_request(model="model", messages=[message])
    assert message.reasoning_content is None
    assert "reasoning_content" not in request["messages"][0]


def test_empty_provider_reasoning_is_preserved_but_user_data_cannot_set_it():
    assistant = parse_chat_completion(_tool_response("")).first.message
    user = Message(role="user", content="question", reasoning_content=PRIVATE_CONTINUITY)
    request = build_chat_request(model="model", messages=[user, assistant])
    assert "reasoning_content" not in request["messages"][0]
    assert request["messages"][1]["reasoning_content"] == ""


def test_none_effort_explicitly_disables_thinking_without_sending_an_effort_value():
    request = build_chat_request(
        model="deepseek-v4-flash", messages=[Message.user("Review the claims.")],
        reasoning_effort="none",
    )

    assert request["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in request


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_configured_reasoning_effort_keeps_its_existing_provider_field(effort):
    request = build_chat_request(
        model="deepseek-v4-flash", messages=[Message.user("Review the claims.")],
        reasoning_effort=effort,
    )

    assert request["reasoning_effort"] == effort
    assert "thinking" not in request


def test_unspecified_effort_preserves_provider_default_for_the_main_agent():
    args = {"model": "deepseek-v4-flash", "messages": [Message.user("Find the source.")]}
    default_request = build_chat_request(**args)
    explicit_none_request = build_chat_request(**args, reasoning_effort=None)

    assert default_request == explicit_none_request
    assert "thinking" not in default_request
    assert "reasoning_effort" not in default_request


@respx.mock
async def test_thinking_tool_turn_round_trips_through_the_actual_http_request_builder():
    route = respx.post("https://api.deepseek.com/chat/completions").mock(side_effect=[
        Response(200, json=_tool_response()),
        Response(200, json={
            "id": "final", "model": "deepseek-v4-flash", "usage": {},
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Done."}}],
        }),
    ])
    messages = [Message.user("Find the source.")]
    async with DeepSeekClient(api_key="fixture-key") as client:
        first = await client.chat(messages)
        messages.extend([first.first.message, Message.tool("call-1", '{"results":[]}', "rag_search")])
        await client.chat(messages, reasoning_effort="low")

    sent = json.loads(route.calls[1].request.content)
    assert sent["messages"][1]["reasoning_content"] == PRIVATE_CONTINUITY
    assert sent["messages"][2]["tool_call_id"] == "call-1"
    assert sent["reasoning_effort"] == "low"
    assert all("reasoning_content" not in message.to_dict() for message in messages)
