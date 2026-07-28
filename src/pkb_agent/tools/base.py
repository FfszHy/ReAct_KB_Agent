"""Tool base classes and the dependency-injection context.

Every tool receives a :class:`ToolContext` carrying all collaborators it may
need (settings, supabase client, repositories, services, trace recorder). Tools
never import global singletons — the runtime assembles the context and injects
it, which keeps tools testable and preserves the core invariant that the agent
runtime does not touch Supabase/network directly (only tools do).
"""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pkb_agent.agent.errors import ToolArgumentError
from pkb_agent.tools.result import ToolResult

if TYPE_CHECKING:
    from pkb_agent.app.settings import Settings
    from pkb_agent.storage.supabase_client import SupabaseClient
    from pkb_agent.trace.recorder import TraceRecorder


@dataclass
class ToolParam:
    """Declarative description of a single tool parameter -> JSON schema."""

    name: str
    type: str  # string | integer | number | boolean | array | object
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None  # for array types

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.type == "array" and self.items:
            schema["items"] = self.items
        return schema


@dataclass
class ToolContext:
    """Dependency-injection container handed to every tool ``execute`` call."""

    settings: Settings
    supabase: SupabaseClient
    trace: TraceRecorder
    repositories: dict[str, Any] = field(default_factory=dict)
    services: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    user_id: str | None = None
    # CLI callers remain synchronous; the FastAPI workbench supplies an async
    # callback that waits for an explicit browser approval without blocking the
    # event loop.
    confirm_callback: Callable[[str, dict[str, Any]], bool | Awaitable[bool]] | None = None

    def repo(self, name: str) -> Any:
        if name not in self.repositories:
            raise KeyError(f"repository not registered: {name}")
        return self.repositories[name]

    def service(self, name: str) -> Any:
        if name not in self.services:
            raise KeyError(f"service not registered: {name}")
        return self.services[name]


class BaseTool(abc.ABC):
    """Abstract base for all tools."""

    name: str = ""
    description: str = ""
    params: list[ToolParam] = []
    permission: str = "allow"  # allow | ask | deny (informational; enforced by PermissionManager)

    @abc.abstractmethod
    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        """Run the tool and return a :class:`ToolResult`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def to_schema(self) -> dict[str, Any]:
        """Produce the DeepSeek ``tools`` JSON-schema entry for this tool."""
        properties = {p.name: p.to_schema() for p in self.params}
        required = [p.name for p in self.params if p.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_args(self, args: dict[str, Any]) -> None:
        """Validate required arguments are present and basic types match."""
        for p in self.params:
            if p.name not in args:
                if p.required:
                    raise ToolArgumentError(self.name, p.name, "missing required argument")
                continue
            value = args[p.name]
            if not _type_matches(value, p.type):
                raise ToolArgumentError(
                    self.name, p.name, f"expected {p.type}, got {type(value).__name__}"
                )


_PY_TYPE_MAP = {
    "string": (str,),
    "integer": (int,),  # NOTE: bool is a subclass of int, handled below
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _type_matches(value: Any, expected: str) -> bool:
    if value is None:
        return True
    if expected == "integer" and isinstance(value, bool):
        return False
    if expected == "number" and isinstance(value, bool):
        return False
    allowed = _PY_TYPE_MAP.get(expected)
    if allowed is None:
        return True
    return isinstance(value, allowed)
