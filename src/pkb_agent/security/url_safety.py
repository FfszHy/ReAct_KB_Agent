"""URL safety / SSRF protection.

Validates that a URL is safe to fetch: allowed scheme, no userinfo in the
netloc, no private/loopback/link-local/multicast/reserved/unspecified IP, and
no well-known SSRF metadata hostnames (cloud metadata endpoints, ``localhost``).
"""

from __future__ import annotations

import ipaddress
import urllib.parse

from pkb_agent.agent.errors import UnsafeUrlError

DEFAULT_BLOCKED_HOSTS: set[str] = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "[::1]",
    "::1",
    "169.254.169.254",
    "metadata.google.internal",
}

DEFAULT_ALLOWED_SCHEMES: set[str] = {"http", "https"}

_BLOCKED_SUFFIXES: tuple[str, ...] = (".local", ".internal")


def is_private_host(host: str) -> bool:
    """Return True if ``host`` parses to a private/reserved/loopback IP.

    Domain names return False; callers may still block them by name.
    """
    if not host:
        return False
    # Strip IPv6 brackets if present.
    cleaned = host.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    try:
        ip = ipaddress.ip_address(cleaned)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_safe_url(
    url: str,
    *,
    block_private: bool = True,
    allowed_schemes: set[str] | None = None,
    blocked_hosts: set[str] | None = None,
) -> tuple[bool, str]:
    """Validate ``url`` against SSRF rules. Returns ``(ok, reason)``."""
    if not isinstance(url, str) or not url or not url.strip():
        return False, "empty url"

    allowed = allowed_schemes if allowed_schemes is not None else DEFAULT_ALLOWED_SCHEMES
    blocked = blocked_hosts if blocked_hosts is not None else DEFAULT_BLOCKED_HOSTS

    try:
        parsed = urllib.parse.urlsplit(url.strip())
    except ValueError as exc:
        return False, f"malformed url: {exc}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in allowed:
        return False, f"scheme not allowed: {scheme!r}"

    # Reject userinfo in netloc (e.g. user:pass@host).
    if "@" in parsed.netloc:
        return False, "userinfo in netloc is not allowed"

    host = parsed.hostname
    if not host:
        return False, "missing host"

    host_lower = host.lower().strip()

    # Build candidate forms (plain and IPv6-bracketed) for blocked lookup.
    candidates = {host_lower}
    if ":" in host_lower:
        candidates.add(f"[{host_lower}]")
    if candidates & blocked:
        return False, f"blocked host: {host_lower}"

    # Block internal/local domain suffixes.
    if host_lower.endswith(_BLOCKED_SUFFIXES):
        return False, f"internal/local domain suffix: {host_lower}"

    if block_private and is_private_host(host_lower):
        return False, f"private/reserved/loopback ip: {host_lower}"

    return True, "ok"


def assert_safe_url(url: str, **kwargs: object) -> str:
    """Raise :class:`UnsafeUrlError` if ``url`` is unsafe; return ``url`` otherwise."""
    ok, reason = is_safe_url(url, **_filter_kwargs(kwargs))  # type: ignore[arg-type]
    if not ok:
        raise UnsafeUrlError(url, reason)
    return url


def _filter_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Keep only keyword arguments understood by :func:`is_safe_url`."""
    allowed_keys = {"block_private", "allowed_schemes", "blocked_hosts"}
    return {k: v for k, v in kwargs.items() if k in allowed_keys}


def normalize_url(url: str) -> str:
    """Strip whitespace and add ``https://`` if the scheme is missing.

    Only prepends the scheme when the leading segment looks like a domain
    (contains a ``.`` and no whitespace).
    """
    if not isinstance(url, str):
        return url  # type: ignore[return-value]
    cleaned = url.strip()
    if not cleaned:
        return cleaned
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme:
        return cleaned
    first_segment = parsed.path.split("/", 1)[0]
    if "." in first_segment and " " not in first_segment and "\t" not in first_segment:
        return "https://" + cleaned
    return cleaned
