from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pkb_agent.agent.errors import ToolPermissionDenied
from pkb_agent.tools.base import ToolContext
from pkb_agent.tools.permissions import (
    Permission,
    PermissionManager,
    ToolPermissionRule,
)


@pytest.fixture
def make_ctx():
    """Build a minimal ToolContext (only confirm_callback matters for checks)."""

    def _make(confirm_callback=None):
        return ToolContext(
            settings=None,
            supabase=None,
            trace=None,
            confirm_callback=confirm_callback,
        )

    return _make


@pytest.fixture
def allow_manager():
    return PermissionManager()


# ---- Permission enum / defaults -------------------------------------------


def test_permission_enum_values():
    assert Permission.ALLOW == "allow"
    assert Permission.ASK == "ask"
    assert Permission.DENY == "deny"


def test_rule_for_unknown_defaults_to_allow(allow_manager):
    rule = allow_manager.rule_for("nonexistent_tool")
    assert rule.permission is Permission.ALLOW
    assert rule.constraints is None


def test_allow_policy_passes(allow_manager, make_ctx):
    # No rule -> ALLOW; check returns None (no raise).
    allow_manager.check("any_tool", {}, make_ctx())


def test_deny_policy_raises(make_ctx):
    mgr = PermissionManager(rules={"t": ToolPermissionRule(Permission.DENY)})
    with pytest.raises(ToolPermissionDenied):
        mgr.check("t", {}, make_ctx())


# ---- ASK permission + confirm callback ------------------------------------


def test_ask_denied_without_callback(make_ctx):
    mgr = PermissionManager(rules={"t": ToolPermissionRule(Permission.ASK)})
    with pytest.raises(ToolPermissionDenied, match="requires confirmation"):
        mgr.check("t", {}, make_ctx())


def test_ask_allowed_when_callback_true(make_ctx):
    mgr = PermissionManager(rules={"t": ToolPermissionRule(Permission.ASK)})
    ctx = make_ctx(confirm_callback=lambda name, args: True)
    mgr.check("t", {}, ctx)  # no raise


def test_ask_denied_when_callback_false(make_ctx):
    mgr = PermissionManager(rules={"t": ToolPermissionRule(Permission.ASK)})
    ctx = make_ctx(confirm_callback=lambda name, args: False)
    with pytest.raises(ToolPermissionDenied):
        mgr.check("t", {}, ctx)


def test_ask_denied_when_callback_raises(make_ctx):
    def boom(name, args):
        raise RuntimeError("boom")

    mgr = PermissionManager(rules={"t": ToolPermissionRule(Permission.ASK)})
    ctx = make_ctx(confirm_callback=boom)
    with pytest.raises(ToolPermissionDenied):
        mgr.check("t", {}, ctx)


# ---- Numeric constraints ---------------------------------------------------


def test_max_top_k_exceeded_raises(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_top_k": 20})
    mgr = PermissionManager(rules={"rag_search": rule})
    with pytest.raises(ToolPermissionDenied, match="top_k"):
        mgr.check("rag_search", {"top_k": 25}, make_ctx())


def test_max_top_k_at_limit_passes(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_top_k": 20})
    mgr = PermissionManager(rules={"rag_search": rule})
    mgr.check("rag_search", {"top_k": 20}, make_ctx())


def test_max_top_k_none_skipped(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_top_k": 20})
    mgr = PermissionManager(rules={"rag_search": rule})
    mgr.check("rag_search", {"top_k": None}, make_ctx())


def test_max_results_constraint_enforced(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_results": 10})
    mgr = PermissionManager(rules={"web_search": rule})
    with pytest.raises(ToolPermissionDenied):
        mgr.check("web_search", {"max_results": 50}, make_ctx())


def test_limit_constraint_uses_max_results(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_results": 10})
    mgr = PermissionManager(rules={"web_search": rule})
    with pytest.raises(ToolPermissionDenied, match="limit"):
        mgr.check("web_search", {"limit": 11}, make_ctx())


def test_max_chars_constraint_uses_max_fetch_chars(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_fetch_chars": 8000})
    mgr = PermissionManager(rules={"web_fetch": rule})
    with pytest.raises(ToolPermissionDenied, match="max_chars"):
        mgr.check("web_fetch", {"max_chars": 9000}, make_ctx())


def test_invalid_numeric_value_raises(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_top_k": 20})
    mgr = PermissionManager(rules={"rag_search": rule})
    with pytest.raises(ToolPermissionDenied, match="invalid"):
        mgr.check("rag_search", {"top_k": "not-a-number"}, make_ctx())


# ---- Content length constraint --------------------------------------------


def test_max_content_chars_exceeded_raises(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_content_chars": 10})
    mgr = PermissionManager(rules={"memory_write": rule})
    with pytest.raises(ToolPermissionDenied, match="content length"):
        mgr.check("memory_write", {"content": "x" * 11}, make_ctx())


def test_max_content_chars_at_limit_passes(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"max_content_chars": 10})
    mgr = PermissionManager(rules={"memory_write": rule})
    mgr.check("memory_write", {"content": "x" * 10}, make_ctx())


# ---- Allowed schemes ------------------------------------------------------


def test_allowed_schemes_https_passes(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"allowed_schemes": ["https", "http"]})
    mgr = PermissionManager(rules={"web_fetch": rule})
    mgr.check("web_fetch", {"url": "https://example.com"}, make_ctx())


def test_allowed_schemes_file_blocked(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"allowed_schemes": ["https", "http"]})
    mgr = PermissionManager(rules={"web_fetch": rule})
    with pytest.raises(ToolPermissionDenied, match="scheme"):
        mgr.check("web_fetch", {"url": "file:///etc/passwd"}, make_ctx())


def test_allowed_schemes_urls_list(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"allowed_schemes": ["https"]})
    mgr = PermissionManager(rules={"web_fetch": rule})
    with pytest.raises(ToolPermissionDenied):
        mgr.check("web_fetch", {"urls": ["https://a.com", "ftp://b.com"]}, make_ctx())


def test_allowed_schemes_case_insensitive(make_ctx):
    rule = ToolPermissionRule(Permission.ALLOW, {"allowed_schemes": ["HTTPS"]})
    mgr = PermissionManager(rules={"web_fetch": rule})
    mgr.check("web_fetch", {"url": "https://example.com"}, make_ctx())


# ---- from_yaml -------------------------------------------------------------


def test_from_yaml_loads_rules(tmp_path: Path):
    cfg = {
        "tools": {
            "rag_search": {"permission": "allow", "constraints": {"max_top_k": 20}},
            "memory_write": {
                "permission": "ask",
                "constraints": {"max_content_chars": 4000},
            },
            "blocked_tool": {"permission": "deny"},
        },
        "denylist": {"blocked_hosts": ["localhost", "127.0.0.1"]},
    }
    p = tmp_path / "permissions.yaml"
    p.write_text(yaml.safe_dump(cfg), "utf-8")

    mgr = PermissionManager.from_yaml(p)
    assert mgr.rule_for("rag_search").permission is Permission.ALLOW
    assert mgr.rule_for("rag_search").constraints == {"max_top_k": 20}
    assert mgr.rule_for("memory_write").permission is Permission.ASK
    assert mgr.rule_for("blocked_tool").permission is Permission.DENY
    assert "localhost" in mgr.blocked_hosts
    assert "127.0.0.1" in mgr.blocked_hosts


def test_from_yaml_missing_file_returns_empty(tmp_path: Path):
    mgr = PermissionManager.from_yaml(tmp_path / "nope.yaml")
    assert mgr.rules == {}
    assert mgr.rule_for("anything").permission is Permission.ALLOW


def test_from_yaml_empty_file_returns_empty(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("", "utf-8")
    mgr = PermissionManager.from_yaml(p)
    assert mgr.rules == {}
    assert mgr.blocked_hosts == set()


def test_blocked_hosts_normalized_lowercase():
    mgr = PermissionManager(blocked_hosts=["LOCALHOST", "Evil.COM"])
    assert "localhost" in mgr.blocked_hosts
    assert "evil.com" in mgr.blocked_hosts
    assert "LOCALHOST" not in mgr.blocked_hosts


# ---- database-backed overrides --------------------------------------------


def test_database_override_replaces_yaml_permission_and_constraints(make_ctx):
    mgr = PermissionManager(
        rules={"rag_search": ToolPermissionRule(Permission.ASK, {"max_top_k": 20})}
    )

    issues = mgr.replace_overrides(
        [{"tool_name": "rag_search", "permission": "allow", "constraints": {"max_top_k": 2}}]
    )

    assert issues == []
    assert mgr.rule_source_for("rag_search") == "database"
    assert mgr.rule_for("rag_search").permission is Permission.ALLOW
    mgr.check("rag_search", {"top_k": 2}, make_ctx())
    with pytest.raises(ToolPermissionDenied, match="top_k"):
        mgr.check("rag_search", {"top_k": 3}, make_ctx())


def test_empty_database_snapshot_returns_to_yaml_baseline(make_ctx):
    mgr = PermissionManager(rules={"tool": ToolPermissionRule(Permission.DENY)})
    mgr.replace_overrides([{"tool_name": "tool", "permission": "allow", "constraints": {}}])
    mgr.check("tool", {}, make_ctx())

    mgr.replace_overrides([])

    assert mgr.rule_source_for("tool") == "yaml"
    with pytest.raises(ToolPermissionDenied):
        mgr.check("tool", {}, make_ctx())


def test_invalid_database_override_is_ignored_and_yaml_rule_remains(make_ctx):
    mgr = PermissionManager(rules={"tool": ToolPermissionRule(Permission.DENY)})

    issues = mgr.replace_overrides(
        [{"tool_name": "tool", "permission": "sometimes", "constraints": {}}]
    )

    assert len(issues) == 1
    assert mgr.rule_source_for("tool") == "yaml"
    with pytest.raises(ToolPermissionDenied):
        mgr.check("tool", {}, make_ctx())
