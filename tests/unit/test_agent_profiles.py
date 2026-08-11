from __future__ import annotations

import pytest

from pkb_agent.evaluation.agent_profiles import (
    get_agent_evaluation_profile,
    make_profile_confirmation,
    select_agent_cases,
)
from pkb_agent.evaluation.dataset import EvalCase


def test_profiles_select_disjoint_kb_permission_and_approved_web_cases():
    cases = (
        EvalCase(id="kb", question="KB?", answerable=False, tags=("unanswerable",)),
        EvalCase(id="permission", question="Fetch?", answerable=False, tags=("permission",)),
        EvalCase(id="web", question="Approved fetch?", answerable=False, tags=("web-approved",)),
    )

    kb_only = get_agent_evaluation_profile("kb-only")
    permission = get_agent_evaluation_profile("permission")
    web_approved = get_agent_evaluation_profile("web_approved")

    assert kb_only.allowed_tools == ("rag_search", "rag_read")
    assert [case.id for case in select_agent_cases(cases, kb_only)] == ["kb", "web"]
    assert [case.id for case in select_agent_cases(cases, permission)] == ["permission"]
    assert [case.id for case in select_agent_cases(cases, web_approved)] == ["web"]


def test_web_approved_confirmation_allows_only_exact_approved_host():
    profile = get_agent_evaluation_profile("web_approved")
    confirm = make_profile_confirmation(profile, ["raw.githubusercontent.com"])

    assert confirm(
        "web_fetch",
        {"url": "https://raw.githubusercontent.com/fastapi/fastapi/0.115.0/README.md"},
    )
    assert not confirm("web_fetch", {"url": "https://raw.githubusercontent.com.evil.test/file"})
    assert not confirm("web_search", {"query": "FastAPI"})


def test_web_approved_profile_requires_a_valid_allowlist_host():
    profile = get_agent_evaluation_profile("web_approved")

    with pytest.raises(ValueError, match="requires at least one"):
        make_profile_confirmation(profile)
    with pytest.raises(ValueError, match="invalid web allowlist host"):
        make_profile_confirmation(profile, ["https://raw.githubusercontent.com"])


def test_unknown_profile_has_actionable_error():
    with pytest.raises(ValueError, match="unknown agent profile"):
        get_agent_evaluation_profile("browser-everything")
