"""Now tool: current date/time."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pkb_agent.tools.base import BaseTool, ToolContext, ToolParam
from pkb_agent.tools.result import ToolResult


class NowTool(BaseTool):
    name = "now"
    description = "Return the current date and time (UTC and local). Useful for time-aware reasoning."
    params: list[ToolParam] = []
    permission = "allow"

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        utc_now = datetime.now(UTC)
        local_now = datetime.now().astimezone()
        return ToolResult.success(
            {
                "utc": utc_now.isoformat(),
                "local": local_now.isoformat(),
                "local_date": local_now.strftime("%Y-%m-%d"),
                "local_time": local_now.strftime("%H:%M:%S"),
                "timezone": local_now.tzname() or "local",
            }
        )
