"""Tool registry: name -> BaseTool lookup + schema export for the LLM."""

from __future__ import annotations

from pkb_agent.agent.errors import ToolNotFoundError
from pkb_agent.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> ToolRegistry:
        if not tool.name:
            raise ValueError("tool must define a non-empty name")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return self

    def register_all(self, tools: list[BaseTool]) -> ToolRegistry:
        for t in tools:
            self.register(t)
        return self

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def require(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def to_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
