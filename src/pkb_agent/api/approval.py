"""Human approval broker used by the SSE workbench."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

EventPublisher = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class PendingApproval:
    approval_id: str
    run_id: str
    tool_name: str
    arguments: dict[str, Any]
    future: asyncio.Future[bool]
    created_at: datetime


class ApprovalBroker:
    """Pause an ``ask``-permission tool until a browser resolves it.

    The broker owns only ephemeral approvals. Durable policy remains in the
    existing permission manager and its YAML/database rules.
    """

    def __init__(self, *, timeout_seconds: int = 300) -> None:
        self._timeout_seconds = timeout_seconds
        self._pending: dict[str, PendingApproval] = {}
        self._lock = asyncio.Lock()

    async def request(
        self,
        *,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        publish: EventPublisher,
    ) -> bool:
        approval_id = str(uuid.uuid4())
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        pending = PendingApproval(
            approval_id=approval_id,
            run_id=run_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            future=future,
            created_at=datetime.now(UTC),
        )
        async with self._lock:
            self._pending[approval_id] = pending

        await publish(
            {
                "type": "approval_required",
                "approval": _approval_payload(pending),
            }
        )
        resolution = "timeout"
        try:
            approved = await asyncio.wait_for(asyncio.shield(future), self._timeout_seconds)
            resolution = "approved" if approved else "rejected"
            return bool(approved)
        except TimeoutError:
            return False
        finally:
            async with self._lock:
                self._pending.pop(approval_id, None)
            await publish(
                {
                    "type": "approval_resolved",
                    "approval_id": approval_id,
                    "tool": tool_name,
                    "resolution": resolution,
                }
            )

    async def decide(self, *, run_id: str, approval_id: str, approved: bool) -> bool:
        """Resolve one pending approval, returning false when it no longer exists."""
        async with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None or pending.run_id != run_id:
                return False
            if pending.future.done():
                return False
            pending.future.set_result(bool(approved))
            return True

    async def cancel_run(self, run_id: str) -> None:
        """Unblock pending tools during application shutdown."""
        async with self._lock:
            pending = [item for item in self._pending.values() if item.run_id == run_id]
            for item in pending:
                if not item.future.done():
                    item.future.set_result(False)


def _approval_payload(pending: PendingApproval) -> dict[str, Any]:
    reason, impact = _reason_for(pending.tool_name, pending.arguments)
    return {
        "id": pending.approval_id,
        "run_id": pending.run_id,
        "tool": pending.tool_name,
        "arguments": pending.arguments,
        "reason": reason,
        "impact": impact,
        "requested_at": pending.created_at.isoformat(),
    }


def _reason_for(tool_name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    if tool_name == "web_fetch":
        url = str(arguments.get("url") or "")
        host = urlparse(url).hostname or "外部站点"
        return (
            "该操作会向外部网站发起请求并把页面内容引入本轮证据链。",
            f"将访问 {host}. 返回内容可能不可信。页面地址会暴露给该站点。",
        )
    if tool_name == "memory_write":
        scope = str(arguments.get("scope") or "long")
        return (
            "该操作会把信息写入 Agent 记忆。它可能影响之后的回答。",
            f"将写入 {scope} 期记忆。请确认内容准确且适合长期保留。内容不应包含敏感信息。",
        )
    return (
        "当前工具受到“需人工确认”策略保护。",
        "批准后 Agent 才会继续执行这一步。",
    )
