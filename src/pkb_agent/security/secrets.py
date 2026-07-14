"""Secret detection and redaction.

Provides regex-based detection for common secret formats (Bearer tokens, API
keys like ``sk-...`` / ``AKIA...``, JWTs, PEM private keys) and a generic
``key=value`` redactor for suspicious key names (password, secret, token,
api_key, authorization, ...). Intended for trace/log redaction, not crypto.
"""

from __future__ import annotations

import re
from typing import Any

# ---- Regex patterns ----------------------------------------------------------

# ``Bearer <token>`` (Authorization header). Capture prefix to preserve "Bearer ".
BEARER_RE = re.compile(r"(?i)(Bearer\s+)([A-Za-z0-9\-._~+/]+={0,2})")

# OpenAI-style keys: sk-...
SK_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")

# AWS access key ids: AKIA...
AKIA_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")

# JWT: header.payload.signature (base64url, all three segments start with e/y).
JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
)

# PEM private key blocks (any algorithm).
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
    r"[\s\S]*?"
    r"-----END (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
)

# Long high-entropy token (hex or base64, 40+ chars).
LONG_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/=])")

# ``key=value`` / ``"key": "value"`` assignments with suspicious names.
# The value class includes ``]`` so that an already-redacted ``[REDACTED]``
# marker is consumed as a whole (avoids producing ``[REDACTED]]``).
SECRET_KV_RE = re.compile(
    r'(?i)\b(password|passwd|secret|token|api[_-]?key|apikey|authorization|access[_-]?token|'
    r'refresh[_-]?token|private[_-]?key|credential)\b\s*[:=]\s*["\']?'
    r'([^"\'\s,;}]+)["\']?'
)

# Key names that look like secrets.
SECRET_KEY_NAME_RE = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?key|apikey|auth(?:orization)?|"
    r"access[_-]?token|refresh[_-]?token|private[_-]?key|credential)"
)

_DEFAULT_REPLACEMENT = "[REDACTED]"


def is_secret_key_name(name: str) -> bool:
    """Return True if ``name`` looks like the name of a secret field."""
    if not isinstance(name, str) or not name:
        return False
    return bool(SECRET_KEY_NAME_RE.search(name))


def looks_like_secret(value: str) -> bool:
    """Return True if ``value`` matches a known secret format or is a long token."""
    if not isinstance(value, str) or len(value) < 8:
        return False
    if PRIVATE_KEY_RE.search(value):
        return True
    if JWT_RE.search(value):
        return True
    if BEARER_RE.search(value):
        return True
    if SK_KEY_RE.search(value):
        return True
    if AKIA_KEY_RE.search(value):
        return True
    # Treat a long high-entropy standalone string as a secret.
    return bool(LONG_TOKEN_RE.fullmatch(value.strip()))


def redact_secrets(text: str, *, replacement: str = _DEFAULT_REPLACEMENT) -> str:
    """Replace detected secret values in ``text`` with ``replacement``."""
    if not isinstance(text, str) or not text:
        return text

    out = text
    out = PRIVATE_KEY_RE.sub(replacement, out)
    out = JWT_RE.sub(replacement, out)
    # Preserve "Bearer " prefix, redact the token.
    out = BEARER_RE.sub(lambda m: m.group(1) + replacement, out)
    out = SK_KEY_RE.sub(replacement, out)
    out = AKIA_KEY_RE.sub(replacement, out)

    # key=value assignments: keep the key and separator, redact the value.
    def _kv_replace(m: re.Match[str]) -> str:
        full = m.group(0)
        val_start = m.start(2) - m.start(0)
        val_end = m.end(2) - m.start(0)
        return full[:val_start] + replacement + full[val_end:]

    out = SECRET_KV_RE.sub(_kv_replace, out)
    return out


def redact_value(obj: Any) -> Any:
    """Recursively redact strings in dict/list/str structures.

    Dict values whose key name looks like a secret are fully replaced; other
    strings are passed through :func:`redact_secrets`.
    """
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and is_secret_key_name(key):
                out[key] = _redact_scalar(value)
            else:
                out[key] = redact_value(value)
        return out
    if isinstance(obj, list):
        return [redact_value(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_value(v) for v in obj)
    return obj


def _redact_scalar(value: Any) -> Any:
    """Replace a value outright if it is a non-empty secret-like string."""
    if isinstance(value, str):
        if value:
            return _DEFAULT_REPLACEMENT
        return value
    if isinstance(value, dict):
        return {k: _redact_scalar(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_scalar(v) for v in value]
    return value
