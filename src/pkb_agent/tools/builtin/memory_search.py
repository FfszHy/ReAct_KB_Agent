"""Memory search tool."""

from __future__ import annotations

from typing import Any

from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult


class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = (
        "Semantically recall previously stored memory notes (facts, "
        "preferences, decisions) across runs. Use to ground answers in what "
        "you already know."
    )
    params = [
        ToolParam("query", "string", "What to recall."),
        ToolParam("max_results", "integer", "Max notes to return.", required=False, default=5),
        ToolParam(
            "scope",
            "string",
            "Memory scope: 'short' (this run) or 'long' (persistent).",
            required=False,
            enum=["short", "long"],
        ),
    ]
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult.failure("query must not be empty")
        max_results = int(args.get("max_results") or ctx.settings.memory_max_results)
        scope = args.get("scope")
        try:
            manager = ctx.service("memory_manager")
            notes = await manager.search(
                query, max_results=max_results, user_id=ctx.user_id, scope=scope
            )
        except Exception as e:
            return ToolResult.failure(f"memory_search failed: {e}")

        return ToolResult.success(
            {
                "query": query,
                "count": len(notes),
                "notes": [
                    {
                        "id": n.id,
                        "scope": n.scope,
                        "kind": n.kind,
                        "content": n.content,
                        "score": round(n.score, 4),
                        "created_at": n.created_at,
                    }
                    for n in notes
                ],
            }
        )
