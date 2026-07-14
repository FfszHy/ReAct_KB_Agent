"""Tool result container: structured outcome + truncation helper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """The outcome of a tool execution.

    ``data`` is the structured payload returned to the agent (JSON-serialized
    into the observation string). ``error`` is set when ``ok`` is False.
    """

    ok: bool
    data: Any = None
    error: str | None = None
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def success(
        cls, data: Any, *, truncated: bool = False, meta: dict[str, Any] | None = None
    ) -> ToolResult:
        return cls(ok=True, data=data, truncated=truncated, meta=meta or {})

    @classmethod
    def failure(cls, error: str, *, meta: dict[str, Any] | None = None) -> ToolResult:
        return cls(ok=False, error=error, meta=meta or {})

    # ------------------------------------------------------------------
    def to_observation(self, max_chars: int | None = None) -> str:
        """Render the result as a string observation for the LLM."""
        if self.ok:
            payload: Any = self.data
        else:
            payload = {"error": self.error or "unknown error"}
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str, indent=2)
        except (TypeError, ValueError):
            text = str(payload)
        if max_chars is not None and len(text) > max_chars:
            text = text[: max(max_chars - 20, 0)].rstrip() + "\n…[truncated]"
        return text

    # ------------------------------------------------------------------
    @staticmethod
    def truncate(text: str, max_chars: int) -> tuple[str, bool]:
        """Truncate ``text`` to ``max_chars``; return (text, truncated)."""
        if len(text) <= max_chars:
            return text, False
        return text[: max(max_chars - 12, 0)].rstrip() + "…[truncated]", True
