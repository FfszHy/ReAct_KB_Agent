"""Trace-layer redaction entry point (delegates to security.secrets)."""

from __future__ import annotations

from typing import Any

from pkb_agent.security.secrets import redact_secrets, redact_value


def redact_text(text: str) -> str:
    return redact_secrets(text)


def redact(obj: Any) -> Any:
    return redact_value(obj)


def redact_args(args: dict) -> dict:
    redacted = redact_value(args)
    return redacted if isinstance(redacted, dict) else {}


def redact_result(result: Any) -> Any:
    return redact_value(result)
