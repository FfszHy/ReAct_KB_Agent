from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pkb_agent.agent.errors import ConfigError
from pkb_agent.agent.verification import AnswerVerifier, Evidence
from pkb_agent.llm.schemas import ChatChoice, ChatCompletion, Message
from pkb_agent.prompts import PromptComposer, PromptRegistry
from pkb_agent.prompts.query_rewriter import QueryPlanner


class _FakeLlm:
    def __init__(self, content: str, usage: dict | None = None) -> None:
        self.content = content
        self.usage = usage or {}
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return ChatCompletion(
            id="rewrite-1",
            model="fake",
            choices=[
                ChatChoice(
                    index=0,
                    message=Message.assistant(content=self.content),
                    finish_reason="stop",
                )
            ],
            usage=self.usage,
        )


def _write_manifest(tmp_path: Path, body: str, prompt_body: str = "system body") -> PromptRegistry:
    (tmp_path / "system.md").write_text(prompt_body, "utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(body, "utf-8")
    return PromptRegistry(prompt_dir=tmp_path, manifest_path=manifest)


def test_composer_activates_core_and_tool_scoped_system_prompts():
    root = Path(__file__).parents[2]
    registry = PromptRegistry(
        prompt_dir=root / "config" / "prompts",
        manifest_path=root / "config" / "prompts" / "manifest.yaml",
    )
    composer = PromptComposer(registry)

    without_memory = composer.compose_system({"rag_search"})
    with_memory = composer.compose_system({"rag_search", "memory_write"})

    assert [ref.id for ref in without_memory.references] == ["system_react"]
    assert [ref.id for ref in with_memory.references] == ["system_react", "memory_policy"]
    assert "What to write to memory" not in without_memory.content
    assert "What to write to memory" in with_memory.content
    assert "## Runtime tool allowlist" in without_memory.content
    assert "`rag_search`" in without_memory.content
    assert "`memory_write`" not in without_memory.content
    assert "`memory_write`" in with_memory.content
    assert without_memory.trace_context()["available_tools"] == ["rag_search"]
    assert "KB catalog page" in without_memory.content
    assert "A search miss" in without_memory.content
    assert "primary user-facing response, not a headline" in without_memory.content
    assert len(with_memory.references[0].sha256) == 64


def test_system_prompt_json_examples_follow_runtime_contract_and_exact_claim_coverage():
    root = Path(__file__).parents[2]
    composer = PromptComposer(
        PromptRegistry(
            prompt_dir=root / "config" / "prompts",
            manifest_path=root / "config" / "prompts" / "manifest.yaml",
        )
    )
    content = composer.compose_system({"rag_search", "rag_read"}).content
    examples = [json.loads(block) for block in re.findall(r"```json\s*(.*?)\s*```", content, re.DOTALL)]
    assert {example["status"] for example in examples} == {"grounded", "insufficient_evidence"}
    source = Evidence(
        citation_id="kb:<id from citation_evidence>",
        source_type="kb_chunk",
        source_id="example",
        review_text="Delivery is eventual.",
    )
    verifier = AnswerVerifier()

    for example in examples:
        verdict = verifier.validate(json.dumps(example), {source.citation_id: source})
        assert verdict.valid, verdict.errors
        if example["status"] == "grounded":
            assert len(example["claims"]) == 2
            assert example["answer"] == "\n".join(claim["text"] for claim in example["claims"])
            assert [claim["kind"] for claim in example["claims"]] == ["fact", "inference"]


def test_composer_selects_pre_tool_prompt_only_for_declared_tool():
    root = Path(__file__).parents[2]
    registry = PromptRegistry(
        prompt_dir=root / "config" / "prompts",
        manifest_path=root / "config" / "prompts" / "manifest.yaml",
    )
    composer = PromptComposer(registry)

    retrieval = composer.compose_pre_tool("rag_search")
    other = composer.compose_pre_tool("memory_search")

    assert [ref.id for ref in retrieval.references] == ["query_rewrite"]
    assert "Query rewriting guidance" in retrieval.content
    assert other.content == ""
    assert other.references == ()


def test_registry_rejects_prompt_file_outside_prompts_directory(tmp_path: Path):
    registry_path = tmp_path / "manifest.yaml"
    registry_path.write_text(
        "prompts:\n  unsafe:\n    file: ../outside.md\n    phase: system\n    always: true\n",
        "utf-8",
    )
    (tmp_path.parent / "outside.md").write_text("outside", "utf-8")

    with pytest.raises(ConfigError, match="escapes prompts directory"):
        PromptRegistry(prompt_dir=tmp_path, manifest_path=registry_path)


def test_registry_rejects_pre_tool_prompt_without_target_tools(tmp_path: Path):
    manifest = "prompts:\n  rewrite:\n    file: system.md\n    phase: pre_tool\n"
    with pytest.raises(ConfigError, match="must target at least one tool"):
        _write_manifest(tmp_path, manifest)


async def test_query_planner_uses_prompt_and_returns_deduplicated_plan():
    root = Path(__file__).parents[2]
    composer = PromptComposer(
        PromptRegistry(
            prompt_dir=root / "config" / "prompts",
            manifest_path=root / "config" / "prompts" / "manifest.yaml",
        )
    )
    llm = _FakeLlm('{"queries": [" auth approach decision ", "auth approach decision", "SSO design"]}')
    planner = QueryPlanner(llm, composer, max_queries=2)

    plan = await planner.rewrite(
        tool_name="rag_search",
        query="What did we decide about authentication?",
        question="What did we decide about authentication?",
    )

    assert plan.status == "applied"
    assert plan.queries == ("auth approach decision", "SSO design")
    assert plan.apply_to_arguments({"query": plan.original_query}) == {
        "query": "auth approach decision",
        "queries": ["auth approach decision", "SSO design"],
    }
    assert plan.trace_context()["prompts"][0]["id"] == "query_rewrite"
    assert "Runtime output contract" in llm.calls[0]["messages"][0].content


async def test_query_planner_falls_back_and_caches_invalid_model_response():
    root = Path(__file__).parents[2]
    composer = PromptComposer(
        PromptRegistry(
            prompt_dir=root / "config" / "prompts",
            manifest_path=root / "config" / "prompts" / "manifest.yaml",
        )
    )
    llm = _FakeLlm("this is not JSON", usage={"prompt_tokens": 12, "completion_tokens": 3})
    planner = QueryPlanner(llm, composer)

    first = await planner.rewrite(tool_name="web_search", query="fresh news", question="news")
    second = await planner.rewrite(tool_name="web_search", query="fresh news", question="news")

    assert first.status == "fallback"
    assert first.queries == ("fresh news",)
    assert first.usage == {"prompt_tokens": 12, "completion_tokens": 3}
    assert second.usage == {}
    assert second is not first
    assert len(llm.calls) == 1


async def test_query_planner_does_not_call_llm_for_unconfigured_tool():
    root = Path(__file__).parents[2]
    composer = PromptComposer(
        PromptRegistry(
            prompt_dir=root / "config" / "prompts",
            manifest_path=root / "config" / "prompts" / "manifest.yaml",
        )
    )
    llm = _FakeLlm('{"queries": ["ignored"]}')
    planner = QueryPlanner(llm, composer)

    plan = await planner.rewrite(tool_name="memory_search", query="preference", question="")

    assert plan.status == "skipped"
    assert plan.queries == ("preference",)
    assert llm.calls == []
