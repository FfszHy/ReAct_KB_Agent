"""Focused tests for the FastAPI workbench additions."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from pkb_agent.agent.errors import ToolPermissionDenied
from pkb_agent.agent.state import AgentRunState, AgentStatus
from pkb_agent.api.app import _serialize_document, create_app
from pkb_agent.api.approval import ApprovalBroker
from pkb_agent.api.sessions import RunSession
from pkb_agent.app.settings import Settings
from pkb_agent.tools.base import ToolContext
from pkb_agent.tools.permissions import Permission, PermissionManager, ToolPermissionRule


@pytest.mark.asyncio
async def test_async_ask_permission_waits_for_async_confirmation():
    manager = PermissionManager(rules={"web_fetch": ToolPermissionRule(Permission.ASK)})
    called: list[str] = []

    async def confirm(tool_name: str, args: dict) -> bool:
        called.append(f"{tool_name}:{args['url']}")
        await asyncio.sleep(0)
        return True

    ctx = ToolContext(settings=None, supabase=None, trace=None, confirm_callback=confirm)
    await manager.check_async("web_fetch", {"url": "https://example.com"}, ctx)
    assert called == ["web_fetch:https://example.com"]


def test_sync_permission_contract_denies_an_async_callback_safely():
    manager = PermissionManager(rules={"web_fetch": ToolPermissionRule(Permission.ASK)})

    async def confirm(tool_name: str, args: dict) -> bool:
        return True

    ctx = ToolContext(settings=None, supabase=None, trace=None, confirm_callback=confirm)
    with pytest.raises(ToolPermissionDenied, match="requires confirmation"):
        manager.check("web_fetch", {"url": "https://example.com"}, ctx)


@pytest.mark.asyncio
async def test_approval_broker_emits_reason_then_resolves():
    broker = ApprovalBroker(timeout_seconds=10)
    events: list[dict] = []

    async def publish(event: dict) -> None:
        events.append(event)

    task = asyncio.create_task(
        broker.request(
            run_id="run-1",
            tool_name="web_fetch",
            arguments={"url": "https://example.com/docs"},
            publish=publish,
        )
    )
    await asyncio.sleep(0)
    prompt = events[0]["approval"]
    assert "外部网站" in prompt["reason"]
    assert "example.com" in prompt["impact"]

    assert await broker.decide(run_id="run-1", approval_id=prompt["id"], approved=True)
    assert await task is True
    assert events[-1]["resolution"] == "approved"


@pytest.mark.asyncio
async def test_run_session_replays_events_to_late_sse_subscriber():
    session = RunSession(run_id="run-1", question="question", user_id="default")
    await session.publish({"type": "start"})
    queue = await session.subscribe()
    replayed = await queue.get()
    assert replayed["type"] == "start"
    assert replayed["sequence"] == 1

    await session.publish({"type": "tool_call", "tool": "rag_search"})
    live = await queue.get()
    assert live["tool"] == "rag_search"
    assert live["sequence"] == 2
    await session.unsubscribe(queue)


def test_run_state_accumulates_full_run_usage_and_metrics():
    state = AgentRunState(question="q")
    state.add_usage(
        {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
        input_cost_per_million=2,
        output_cost_per_million=8,
    )
    state.add_usage(
        {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        input_cost_per_million=2,
        output_cost_per_million=8,
    )
    state.record_tool_call(33, ok=True)
    state.record_tool_call(12, ok=False)
    state.status = AgentStatus.FINISHED
    state.verification = {"status": "verified"}
    state.mark_finished()

    metrics = state.metrics()
    assert metrics["usage"]["total_tokens"] == 185
    assert metrics["usage"]["actual_cost"] == pytest.approx(0.00058)
    assert metrics["usage"]["cost_status"] == "settled"
    assert metrics["tool_success_rate"] == 50.0
    assert metrics["run_succeeded"] is True
    assert state.usage["duration_ms"] == metrics["duration_ms"]


def test_run_state_uses_deepseek_cache_hit_and_miss_rates():
    state = AgentRunState(question="q")
    state.add_usage(
        {
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 110,
        },
        cache_hit_input_cost_per_million=0.02,
        cache_miss_input_cost_per_million=1.0,
        output_cost_per_million=2.0,
        currency="CNY",
    )

    assert state.usage["prompt_cache_hit_tokens"] == 80
    assert state.usage["prompt_cache_miss_tokens"] == 20
    assert state.usage["cost_currency"] == "CNY"
    assert state.usage["actual_cost"] == pytest.approx(0.0000416)


def test_fastapi_workbench_health_contract_is_available_without_runtime_secrets():
    with TestClient(create_app(Settings())) as client:
        response = client.get("/api/health")
        invalid_run = client.post("/api/runs", json={"question": "   "})
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "runtime": "python", "stream": "sse"}
    assert invalid_run.status_code == 422


def test_document_source_status_exposes_expired_webpage_age():
    document = _serialize_document(
        {
            "id": "document-1",
            "title": "Source page",
            "source_type": "url",
            "content_hash": "abcdef123456",
            "meta": {"expires_at": (datetime.now(UTC) - timedelta(days=14)).isoformat()},
        }
    )
    assert document["version"] == "vabcdef12"
    assert document["source_status"] == "expired"
    assert document["source_label"] == "该网页已过期 14 天"
