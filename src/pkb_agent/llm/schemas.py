"""DeepSeek chat-completions message & tool-call schemas.

These are lightweight pydantic/dataclass models used to build the request to the
DeepSeek Chat Completions API (which supports native ``tools`` calling) and to
parse the streaming-unfriendly but simple non-streaming JSON response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ToolCall:
        fn = raw.get("function", {}) or {}
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw": args_raw}
        else:
            arguments = args_raw or {}
        return cls(id=raw.get("id", ""), name=fn.get("name", ""), arguments=arguments)


@dataclass
class Message:
    """A chat message in the conversation history.

    Roles: ``system`` | ``user`` | ``assistant`` | ``tool``.
    For ``tool`` role, set ``tool_call_id`` and ``name``. For ``assistant`` with
    tool calls, set ``tool_calls``.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str | None = None) -> Message:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ChatChoice:
    index: int
    message: Message
    finish_reason: str


@dataclass
class ChatCompletion:
    id: str
    model: str
    choices: list[ChatChoice]
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def first(self) -> ChatChoice:
        if not self.choices:
            raise IndexError("chat completion has no choices")
        return self.choices[0]


def build_chat_request(
    *,
    model: str,
    messages: list[Message],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    tool_choice: str | dict | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Build the JSON body for ``POST /chat/completions``."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [m.to_dict() for m in messages],
        "temperature": temperature,
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def parse_chat_completion(payload: dict[str, Any]) -> ChatCompletion:
    choices: list[ChatChoice] = []
    for raw_choice in payload.get("choices", []) or []:
        msg_raw = raw_choice.get("message", {}) or {}
        tool_calls_raw = msg_raw.get("tool_calls") or []
        tool_calls = [ToolCall.from_raw(tc) for tc in tool_calls_raw]
        message = Message(
            role=msg_raw.get("role", "assistant"),
            content=msg_raw.get("content"),
            tool_calls=tool_calls,
        )
        choices.append(
            ChatChoice(
                index=raw_choice.get("index", 0),
                message=message,
                finish_reason=raw_choice.get("finish_reason", "stop"),
            )
        )
    return ChatCompletion(
        id=payload.get("id", ""),
        model=payload.get("model", ""),
        choices=choices,
        usage=payload.get("usage", {}) or {},
    )
