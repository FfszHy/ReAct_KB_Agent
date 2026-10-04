"""Run each demo/regression once through the real runtime; retain every result.

Usage: conda run --no-capture-output -n pkb-agent python scripts/preflight_agent_demo.py
This is a live API preflight, not a Ghostty recording or a quality benchmark.
"""

# ruff: noqa: RUF001 -- Preserve the original Chinese regression questions.

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pkb_agent.agent.runtime import AgentRuntime
from pkb_agent.app.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
KB_TOOLS = ("rag_search", "rag_read", "rag_list_documents")


def cases() -> list[dict]:
    commands = subprocess.check_output(
        ["bash", str(ROOT / "scripts/demo_ghostty.sh"), "--list"], text=True, cwd=ROOT
    )
    items = []
    for line in commands.splitlines():
        if "conda run " not in line:
            continue
        argv = shlex.split(line)
        items.append({"id": f"demo-{len(items) + 1}", "question": argv[argv.index("ask") + 1],
                      "kb_only": "--kb-only" in argv})
    items.extend([
        {"id": "field-exclude", "kb_only": True, "question":
         "仅依据知识库 Pydantic v2.10.6 文档：模型字段 secret: str = Field(exclude=True)，"
         "创建时给出非空 secret。调用 model_dump(include={'secret'}, exclude_none=False, "
         "exclude_unset=False) 能否恢复输出 secret？区分字段级排除与按值排除，"
         "给出处，不联网。回答控制在250字以内。"},
        {"id": "react-repair-scope", "kb_only": True, "question":
         "假设一个使用 Kubernetes 1.31 的服务把业务配置放在 ConfigMap 中。团队修改配置后，"
         "Deployment 显示正常，但有些 Pod 仍在使用旧值。同事认为“等一分钟就会全部自动更新”，"
         "准备不再处理。请仅依据知识库评估这项处理意见，给出一份简短排障建议：哪些结论现在能"
         "确定，哪些还缺现场信息，以及如何据此选择最小处理方案。请给出来源，区分文档事实与"
         "对这个假设场景的推断，控制在400字以内，不联网。"},
    ])
    return items


async def run(output: Path, selected: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in (
            "config/prompts/system_react.md", "src/pkb_agent/agent/verification.py",
            "src/pkb_agent/agent/semantic_review.py", "src/pkb_agent/agent/runtime.py",
            "src/pkb_agent/llm/schemas.py", "src/pkb_agent/app/settings.py", "config/default.yaml",
        )
    }
    (output / "source-hashes.json").write_text(json.dumps(snapshot, indent=2))
    for case in selected:
        settings = get_settings().model_copy(update={"permissions_db_overrides_enabled": False})
        review_config = {"effort": settings.agent_semantic_review_reasoning_effort,
                         "max_chars": settings.agent_semantic_review_max_chars,
                         "max_tokens": settings.agent_semantic_review_max_tokens}
        print(json.dumps({"case": case["id"], "effective_review_config": review_config}), flush=True)
        events = []

        def event(item, events=events, case_id=case["id"]):
            if item["type"] in ("model_turn", "answer_semantic_review", "answer_verification_failed"):
                events.append(item)
                print(json.dumps({"case": case_id, **item}, ensure_ascii=False), flush=True)

        try:
            async with AgentRuntime.build(settings, user_id="eval-tech-multidomain-v1.1",
                                          confirm=lambda *_: False,
                                          allowed_tools=KB_TOOLS if case["kb_only"] else None) as rt:
                state = await rt.run(case["question"], on_event=event)
                record = state.to_dict()
                record.pop("prompt_context", None)
                for step in record["steps"]:
                    step.pop("thought", None)
                    step.pop("prompt_context", None)
                # Keep attempted final JSON too: repairs must not erase failed candidates.
                record["answer_candidates"] = [m.content for m in state.messages
                                               if m.role == "assistant" and not m.tool_calls]
                record["events"] = events
                record["review_sources"] = [
                    {**source.to_dict(), "text": source.review_text or source.excerpt}
                    for source in state.evidence.values()
                ]
                record["preflight"] = {"case": case["id"], "kind": "live runtime; no recording",
                                       "kb_only": case["kb_only"], "approval_policy": "deny ASK",
                                       "model": settings.deepseek_model,
                                       "review_effort": settings.agent_semantic_review_reasoning_effort,
                                       "review_max_chars": settings.agent_semantic_review_max_chars,
                                       "review_max_tokens": settings.agent_semantic_review_max_tokens}
        except Exception as exc:
            record = {"preflight": case, "error_type": type(exc).__name__, "events": events}
        (output / f"{case['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(json.dumps({"case": case["id"], "verification": record.get("verification"),
                          "error_type": record.get("error_type")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/agent-fixes" /
                        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cases", help="Comma-separated case IDs; default runs all once")
    args = parser.parse_args()
    selected = cases()
    if args.cases:
        ids = set(args.cases.split(","))
        if ids - {item["id"] for item in selected}:
            parser.error("unknown case ID")
        selected = [item for item in selected if item["id"] in ids]
    print(f"Saving all results to {args.output}", flush=True)
    asyncio.run(run(args.output, selected))
