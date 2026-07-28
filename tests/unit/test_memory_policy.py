from __future__ import annotations

import pytest

from pkb_agent.app.settings import Settings
from pkb_agent.memory.policy import (
    ALLOWED_KINDS,
    ALLOWED_SCOPES,
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_POLICY,
    MemoryPolicy,
    MemoryPolicyConfig,
)

# ---- defaults -------------------------------------------------------------


def test_default_max_content_chars():
    assert DEFAULT_MAX_CONTENT_CHARS == 4000


def test_allowed_kinds_contents():
    assert {"preference", "fact", "decision", "reference", "note"} == ALLOWED_KINDS


def test_allowed_scopes_contents():
    assert {"short", "long"} == ALLOWED_SCOPES


def test_default_scope():
    assert MemoryPolicy().default_scope() == "long"


def test_default_policy_singleton_validates():
    ok, reason = DEFAULT_POLICY.validate(content="hello", kind="fact", scope="long")
    assert ok is True
    assert reason == "ok"


# ---- validate: happy path -------------------------------------------------


@pytest.mark.parametrize("kind", ["preference", "fact", "decision", "reference", "note"])
def test_validate_all_kinds_allowed(kind):
    ok, _ = MemoryPolicy().validate(content="content", kind=kind, scope="long")
    assert ok is True


@pytest.mark.parametrize("scope", ["short", "long"])
def test_validate_all_scopes_allowed(scope):
    ok, _ = MemoryPolicy().validate(content="content", kind="fact", scope=scope)
    assert ok is True


# ---- validate: content edge cases -----------------------------------------


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_validate_empty_content_rejected(content):
    ok, reason = MemoryPolicy().validate(content=content)
    assert ok is False
    assert "non-empty" in reason


def test_validate_none_content_rejected():
    ok, _ = MemoryPolicy().validate(content=None)  # type: ignore[arg-type]
    assert ok is False


def test_validate_content_too_long_rejected():
    policy = MemoryPolicy(MemoryPolicyConfig(max_content_chars=10))
    ok, reason = policy.validate(content="x" * 11)
    assert ok is False
    assert "max length" in reason


def test_validate_content_at_limit_accepted():
    policy = MemoryPolicy(MemoryPolicyConfig(max_content_chars=10))
    ok, _ = policy.validate(content="x" * 10)
    assert ok is True


# ---- validate: invalid kind/scope ----------------------------------------


def test_validate_invalid_kind_rejected():
    ok, reason = MemoryPolicy().validate(content="x", kind="unknown")
    assert ok is False
    assert "kind" in reason


def test_validate_invalid_scope_rejected():
    ok, reason = MemoryPolicy().validate(content="x", scope="medium")
    assert ok is False
    assert "scope" in reason


# ---- custom config --------------------------------------------------------


def test_custom_config_allowed_kinds():
    cfg = MemoryPolicyConfig(allowed_kinds={"fact"})
    policy = MemoryPolicy(cfg)
    ok, _ = policy.validate(content="x", kind="fact")
    assert ok is True
    ok, _ = policy.validate(content="x", kind="note")
    assert ok is False


def test_custom_config_allowed_scopes():
    cfg = MemoryPolicyConfig(allowed_scopes={"short"})
    policy = MemoryPolicy(cfg)
    ok, _ = policy.validate(content="x", scope="short")
    assert ok is True
    ok, _ = policy.validate(content="x", scope="long")
    assert ok is False


# ---- from_settings --------------------------------------------------------


def test_from_settings_returns_policy():
    policy = MemoryPolicy.from_settings(None)  # type: ignore[arg-type]
    assert isinstance(policy, MemoryPolicy)
    ok, _ = policy.validate(content="x")
    assert ok is True


def test_from_settings_uses_memory_limits_and_default_scope():
    settings = Settings(
        memory_default_scope="short",
        memory_max_content_chars=10,
        memory_reject_secrets=True,
    )
    policy = MemoryPolicy.from_settings(settings)

    assert policy.default_scope() == "short"
    ok, reason = policy.validate(content="x" * 11)
    assert ok is False
    assert "max length" in reason


def test_secret_looking_content_is_rejected_unless_policy_disables_guard():
    content = "api_key=super-secret-value"

    ok, reason = MemoryPolicy().validate(content=content)
    assert ok is False
    assert "secret" in reason

    permissive = MemoryPolicy(MemoryPolicyConfig(reject_secrets=False))
    ok, _ = permissive.validate(content=content)
    assert ok is True
