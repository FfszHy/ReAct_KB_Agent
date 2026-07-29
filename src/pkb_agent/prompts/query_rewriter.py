"""Prompt-backed query planning used immediately before retrieval tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from pkb_agent.llm.deepseek_client import DeepSeekClient
from pkb_agent.llm.schemas import Message
from pkb_agent.prompts.registry import PromptComposer

_MAX_QUERY_CHARS = 180
_MAX_PLANNED_QUERIES = 10
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class QueryPlan:
    """A retrieval query plan plus non-sensitive provenance for traces."""

    original_query: str
    queries: tuple[str, ...]
    status: str
    prompt_context: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None
    # Filled only for the call that actually asked the model. A cached plan
    # deliberately returns an empty usage payload so a repeated retrieval does
    # not charge the same completion twice in run accounting.
    usage: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def apply_to_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return executable tool arguments without mutating the model output."""
        if not self.queries:
            return dict(arguments)
        effective = dict(arguments)
        effective["query"] = self.queries[0]
        if len(self.queries) > 1:
            effective["queries"] = list(self.queries)
        else:
            effective.pop("queries", None)
        return effective

    def trace_context(self) -> dict[str, Any]:
        context = dict(self.prompt_context)
        context["status"] = self.status
        context["queries"] = list(self.queries)
        if self.fallback_reason:
            context["fallback_reason"] = self.fallback_reason
        return context


class QueryPlanner:
    """Use the ``pre_tool`` prompt phase to turn one query into focused queries.

    Rewriting is deliberately best-effort: a malformed/failed planning request
    always falls back to the original model-selected query and never blocks a
    retrieval tool call.
    """

    def __init__(
        self,
        llm: DeepSeekClient,
        composer: PromptComposer,
        *,
        enabled: bool = True,
        max_queries: int = 3,
    ) -> None:
        self._llm = llm
        self._composer = composer
        self._enabled = enabled
        self._max_queries = min(max(1, max_queries), _MAX_PLANNED_QUERIES)
        self._cache: dict[tuple[str, str, str], QueryPlan] = {}

    async def rewrite(self, *, tool_name: str, query: str, question: str = "") -> QueryPlan:
        original = _normalize_query(query)
        if not original:
            return QueryPlan(query, (), "skipped")

        composition = self._composer.compose_pre_tool(tool_name)
        if not self._enabled or not composition.content:
            return QueryPlan(original, (original,), "skipped")

        cache_key = (tool_name, original, question)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return replace(cached, usage={})

        prompt_context = composition.trace_context()
        system_prompt = (
            f"{composition.content}\n\n"
            "## Runtime output contract\n"
            "Return only a JSON object in this exact form: "
            '{"queries": ["focused query", "optional second query"]}. '
            f"Return at most {self._max_queries} non-empty strings."
        )
        user_prompt = (
            f"Original user question:\n{question or original}\n\n"
            f"Current retrieval query:\n{original}"
        )

        try:
            completion = await self._llm.chat(
                [Message.system(system_prompt), Message.user(user_prompt)],
                temperature=0.0,
                max_tokens=240,
            )
            usage = dict(completion.usage or {})
            content = completion.first.message.content or ""
            queries = _parse_queries(content, self._max_queries)
            if not queries:
                plan = QueryPlan(
                    original,
                    (original,),
                    "fallback",
                    prompt_context,
                    "invalid_model_output",
                    usage,
                )
            else:
                plan = QueryPlan(
                    original,
                    tuple(queries),
                    "applied",
                    prompt_context,
                    usage=usage,
                )
        except Exception as exc:
            plan = QueryPlan(
                original,
                (original,),
                "fallback",
                prompt_context,
                type(exc).__name__,
            )

        self._cache[cache_key] = plan
        return plan


def _parse_queries(content: str, max_queries: int) -> list[str]:
    candidate = content.strip()
    fenced = _JSON_FENCE_RE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    raw_queries: Any = payload.get("queries") if isinstance(payload, dict) else payload
    if not isinstance(raw_queries, list):
        return []

    queries: list[str] = []
    seen: set[str] = set()
    for raw in raw_queries:
        if not isinstance(raw, str):
            continue
        query = _normalize_query(raw)
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def _normalize_query(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:_MAX_QUERY_CHARS].strip()
