"""CLI contracts for bounded, observable knowledge-base answers."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from pkb_agent.app import cli


class _Runtime:
    def __init__(self) -> None:
        self.allowed_tools: list[str] | None = None
        self.build_options: dict = {}
        self.run_options: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def restrict_tools(self, names):
        self.allowed_tools = list(names)

    async def run(self, question, **kwargs):
        self.run_options = kwargs
        return SimpleNamespace(
            answer_payload=None,
            final_answer="No measurement evidence.",
            status=SimpleNamespace(value="finished"),
        )


def _stub_runtime(monkeypatch):
    runtime = _Runtime()
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace())

    def build(*args, **kwargs):
        runtime.build_options = kwargs
        runtime.allowed_tools = kwargs.get("allowed_tools")
        return runtime

    monkeypatch.setattr(cli.AgentRuntime, "build", build)
    return runtime


def test_kb_only_and_answer_limit_reach_the_runtime(monkeypatch):
    runtime = _stub_runtime(monkeypatch)

    result = CliRunner().invoke(
        cli.app, ["ask", "Read the sources", "--kb-only", "--max-answer-chars", "250"]
    )

    assert result.exit_code == 0, result.output
    assert runtime.allowed_tools == ["rag_search", "rag_read", "rag_list_documents"]
    assert runtime.build_options["allowed_tools"] == runtime.allowed_tools
    assert runtime.run_options["max_answer_chars"] == 250
    assert "仍会访问模型与知识库服务" in result.output


def test_default_ask_preserves_normal_tool_scope(monkeypatch):
    runtime = _stub_runtime(monkeypatch)

    result = CliRunner().invoke(cli.app, ["ask", "question"])

    assert result.exit_code == 0, result.output
    assert runtime.allowed_tools is None
    assert "allowed_tools" not in runtime.build_options
    assert "max_answer_chars" not in runtime.run_options


@pytest.mark.parametrize("limit", ["0", "-10", "invalid"])
def test_invalid_answer_limit_is_rejected_before_runtime_build(monkeypatch, limit):
    def fail_build(*args, **kwargs):
        pytest.fail("invalid CLI input must not build the network runtime")

    monkeypatch.setattr(cli.AgentRuntime, "build", fail_build)

    result = CliRunner().invoke(cli.app, ["ask", "question", "--max-answer-chars", limit])

    assert result.exit_code == 2


def _capture_console(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=160, color_system=None))
    return output


def test_model_round_describes_batch_and_observations_without_internal_thought(monkeypatch):
    output = _capture_console(monkeypatch)

    cli._render_event(
        {
            "type": "model_turn",
            "round": 2,
            "phase": "tools",
            "tools": ["rag_read", "rag_search"],
            "observed_tool_results": 1,
            "thought": "private model reasoning",
        }
    )
    cli._render_event(
        {"type": "model_turn", "round": 3, "phase": "answer", "observed_tool_results": 3}
    )

    rendered = output.getvalue()
    assert "ReAct 第 2 轮" in rendered
    assert "rag_read, rag_search" in rendered
    assert "已观察 1 次工具结果" in rendered
    assert "ReAct 第 3 轮 · 提交候选答案" in rendered
    assert "private model reasoning" not in rendered


@pytest.mark.parametrize("status,label", [
    ("running", "进行中"), ("passed", "通过"), ("failed", "未通过"), ("unavailable", "不可用"),
])
def test_semantic_review_events_are_visibly_separate_from_react_rounds(monkeypatch, status, label):
    output = _capture_console(monkeypatch)

    cli._render_event({"type": "answer_semantic_review", "attempt": 2, "status": status})

    assert f"AI 语义复核 · 第 2 次 · {label}" in output.getvalue()
    assert "ReAct" not in output.getvalue()


@pytest.mark.parametrize(
    "semantic_status,label",
    [("passed", "AI 语义复核通过"), ("disabled", "AI 语义复核未启用"), (None, "未记录通过的 AI 语义复核")],
)
def test_answer_rendering_distinguishes_citation_structure_from_semantic_review(
    monkeypatch, semantic_status, label
):
    output = _capture_console(monkeypatch)
    verification = {"status": "verified", "structure": "passed"}
    if semantic_status is not None:
        verification["semantic_review"] = {"status": semantic_status}

    cli._render_verified_answer(
        {
            "answer": "A supported answer.",
            "claims": [],
            "citations": [{"id": "kb:1", "title": "Example document", "locator": "kb://doc-1"}],
        },
        verification,
    )

    rendered = output.getvalue()
    assert "引用来源 / Citation sources" in rendered
    assert "Verified citations" not in rendered
    assert "答案结构与引用来源校验通过" in rendered
    assert label in rendered


@pytest.mark.parametrize("verification", [
    {"status": "structure_verified", "structure": "passed", "semantic_review": {"status": "disabled"}},
    {"status": "verified", "structure": "passed"},
])
def test_answer_without_recorded_review_has_no_green_success_panel(monkeypatch, verification):
    rendered = []
    monkeypatch.setattr(cli, "console", SimpleNamespace(print=rendered.append))

    cli._render_verified_answer(
        {"answer": "Candidate answer.", "status": "grounded", "claims": [], "citations": []},
        verification,
    )

    assert rendered[0].border_style == "yellow"
