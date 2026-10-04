from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from pkb_agent.llm.deepseek_client import DeepSeekClient
from pkb_agent.llm.schemas import Message


@respx.mock
@pytest.mark.parametrize("reasoning_effort", [None, "low"])
async def test_chat_sends_deepseek_json_output_options(reasoning_effort):
    route = respx.post("https://api.deepseek.com/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "completion-1",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )
    )

    async with DeepSeekClient(api_key="test-key") as client:
        await client.chat(
            [Message.system("Return json."), Message.user("test")],
            max_tokens=1024,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )

    body = json.loads(route.calls[0].request.content)
    assert body["max_tokens"] == 1024
    assert body["response_format"] == {"type": "json_object"}
    if reasoning_effort is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == reasoning_effort
