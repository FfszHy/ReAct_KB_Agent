"""Reproducible Agent tool and case profiles for evaluation runs.

The profiles deliberately separate knowledge-base quality, permission denial,
and a narrowly allowlisted approved-web acceptance check.  They are evaluation
controls only; production tool availability remains configured by the runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from pkb_agent.evaluation.dataset import EvalCase

ConfirmationCallback = Callable[[str, dict[str, object]], bool]


@dataclass(frozen=True)
class AgentEvaluationProfile:
    """A fixed tool surface, case filter, and confirmation policy."""

    name: str
    allowed_tools: tuple[str, ...] | None
    required_tag: str | None = None
    excluded_tag: str | None = None
    requires_web_allowlist: bool = False


_PROFILES: dict[str, AgentEvaluationProfile] = {
    # Preserves the pre-profile CLI behavior for existing workflows.
    "full": AgentEvaluationProfile(name="full", allowed_tools=None),
    # Primary RAG score: no mutable memory or external-web capability leaks
    # into answer quality, citations, latency, or refusal behavior.
    "kb_only": AgentEvaluationProfile(
        name="kb_only",
        allowed_tools=("rag_search", "rag_read"),
        excluded_tag="permission",
    ),
    # Explicit URL requests that must select a protected tool and then safely
    # refuse when confirmation is unavailable.
    "permission": AgentEvaluationProfile(
        name="permission",
        allowed_tools=("web_fetch",),
        required_tag="permission",
    ),
    # A separate operational acceptance suite.  The callback approves only a
    # caller-supplied exact host allowlist; it never grants global web access.
    "web_approved": AgentEvaluationProfile(
        name="web_approved",
        allowed_tools=("web_fetch",),
        required_tag="web-approved",
        requires_web_allowlist=True,
    ),
}


def get_agent_evaluation_profile(value: str) -> AgentEvaluationProfile:
    """Resolve a profile name with a clear error suitable for CLI rendering."""
    name = value.strip().casefold().replace("-", "_")
    try:
        return _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown agent profile {value!r}; choose one of: {choices}") from exc


def select_agent_cases(
    cases: Iterable[EvalCase], profile: AgentEvaluationProfile
) -> tuple[EvalCase, ...]:
    """Filter benchmark cases without changing the frozen source annotations."""
    selected: list[EvalCase] = []
    for case in cases:
        tags = set(case.tags)
        if profile.required_tag and profile.required_tag not in tags:
            continue
        if profile.excluded_tag and profile.excluded_tag in tags:
            continue
        selected.append(case)
    return tuple(selected)


def make_profile_confirmation(
    profile: AgentEvaluationProfile,
    allowed_web_hosts: Iterable[str] = (),
) -> ConfirmationCallback:
    """Return the non-interactive confirmation callback for an eval profile."""
    if not profile.requires_web_allowlist:
        return _deny_confirmation

    hosts = _normalise_hosts(allowed_web_hosts)
    if not hosts:
        raise ValueError(
            "web_approved requires at least one --web-allow-host for its approved fetches"
        )

    def confirm(tool_name: str, args: dict[str, object]) -> bool:
        if tool_name != "web_fetch":
            return False
        raw_url = args.get("url")
        if not isinstance(raw_url, str):
            return False
        try:
            host = (urlsplit(raw_url).hostname or "").strip().casefold().rstrip(".")
        except ValueError:
            return False
        return host in hosts

    return confirm


def _deny_confirmation(_tool_name: str, _args: dict[str, object]) -> bool:
    return False


def _normalise_hosts(values: Iterable[str]) -> frozenset[str]:
    hosts: set[str] = set()
    for value in values:
        host = value.strip().casefold().rstrip(".")
        if not host or "/" in host or ":" in host or "@" in host:
            raise ValueError(f"invalid web allowlist host: {value!r}")
        hosts.add(host)
    return frozenset(hosts)
