"""Memory write tool."""

from __future__ import annotations

from typing import Any

from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult


class MemoryWriteTool(BaseTool):
    name = "memory_write"
    description = (
        "Persist a note for future recall. Use sparingly for "
        "stable facts, resolved preferences, or reusable decisions — not for "
        "ephemeral state."
    )
    params = [
        ToolParam("content", "string", "The self-contained note to remember."),
        ToolParam(
            "kind",
            "string",
            "Note category.",
            required=False,
            default="fact",
            enum=["preference", "fact", "decision", "reference", "note"],
        ),
        ToolParam(
            "scope",
            "string",
            "Memory scope. Defaults to the configured policy scope.",
            required=False,
            enum=["short", "long"],
        ),
    ]
    permission = "ask"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        content = str(args.get("content", "")).strip()
        if not content:
            return ToolResult.failure("content must not be empty")
        kind = str(args.get("kind") or "fact")
        try:
            manager = ctx.service("memory_manager")
            scope = str(args["scope"]) if args.get("scope") else manager.default_scope()
            row = await manager.write(
                user_id=ctx.user_id or "default",
                content=content,
                kind=kind,
                scope=scope,
                run_id=ctx.run_id,
            )
        except ValueError as e:
            return ToolResult.failure(str(e))
        except Exception as e:
            return ToolResult.failure(f"memory_write failed: {e}")

        return ToolResult.success(
            {
                "id": row.get("id"),
                "scope": scope,
                "kind": kind,
                "stored": True,
                "content_preview": content[:200],
            }
        )
