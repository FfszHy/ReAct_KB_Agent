"""Tool permission management.

Loads ``config/permissions.yaml`` and enforces per-tool ``allow``/``ask``/``deny``
policies plus simple argument constraints (max_top_k, max_results, char limits,
schemes, etc.). ``ask`` tools require an interactive confirmation callback; in
non-interactive mode they are denied.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from pkb_agent.agent.errors import ToolPermissionDenied

if TYPE_CHECKING:
    from pkb_agent.tools.base import ToolContext


class Permission(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class ToolPermissionRule:
    permission: Permission
    constraints: dict[str, Any] | None = None


class PermissionManager:
    def __init__(
        self,
        rules: dict[str, ToolPermissionRule] | None = None,
        blocked_hosts: list[str] | None = None,
        confirm: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        self.rules: dict[str, ToolPermissionRule] = rules or {}
        self.blocked_hosts = {h.lower() for h in (blocked_hosts or [])}
        self.confirm = confirm

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, confirm: Callable | None = None) -> PermissionManager:
        path = Path(path)
        if not path.exists():
            return cls(confirm=confirm)
        raw = yaml.safe_load(path.read_text("utf-8")) or {}
        tools = raw.get("tools", {}) or {}
        rules: dict[str, ToolPermissionRule] = {}
        for name, cfg in tools.items():
            perm = Permission((cfg or {}).get("permission", "allow"))
            rules[name] = ToolPermissionRule(perm, (cfg or {}).get("constraints"))
        blocked = (raw.get("denylist") or {}).get("blocked_hosts", []) or []
        return cls(rules, blocked, confirm)

    # ------------------------------------------------------------------
    def rule_for(self, tool_name: str) -> ToolPermissionRule:
        return self.rules.get(tool_name, ToolPermissionRule(Permission.ALLOW))

    def check(self, tool_name: str, args: dict[str, Any], ctx: ToolContext) -> None:
        """Raise :class:`ToolPermissionDenied` if the call is not permitted."""
        rule = self.rule_for(tool_name)
        if rule.permission == Permission.DENY:
            raise ToolPermissionDenied(tool_name, "denied by policy")
        self._validate_constraints(tool_name, rule, args)
        if rule.permission == Permission.ASK:
            allowed = False
            if ctx.confirm_callback is not None:
                try:
                    allowed = bool(ctx.confirm_callback(tool_name, args))
                except Exception:
                    allowed = False
            if not allowed:
                raise ToolPermissionDenied(tool_name, "requires confirmation (not granted)")

    # ------------------------------------------------------------------
    def _validate_constraints(
        self, tool_name: str, rule: ToolPermissionRule, args: dict[str, Any]
    ) -> None:
        c = rule.constraints or {}
        if not c:
            return

        # Numeric ceilings.
        for arg_key, constraint_key in (
            ("top_k", "max_top_k"),
            ("max_results", "max_results"),
            ("limit", "max_results"),
            ("max_chars", "max_fetch_chars"),
        ):
            if constraint_key in c and arg_key in args and args[arg_key] is not None:
                try:
                    if int(args[arg_key]) > int(c[constraint_key]):
                        raise ToolPermissionDenied(
                            tool_name, f"{arg_key}={args[arg_key]} exceeds max {c[constraint_key]}"
                        )
                except (TypeError, ValueError):
                    raise ToolPermissionDenied(tool_name, f"invalid {arg_key} value") from None

        # Content length ceiling.
        if (
            "max_content_chars" in c
            and "content" in args
            and args["content"] is not None
            and len(str(args["content"])) > int(c["max_content_chars"])
        ):
            raise ToolPermissionDenied(
                tool_name,
                f"content length exceeds max {c['max_content_chars']} chars",
            )

        # Allowed schemes for URL-bearing tools.
        if "allowed_schemes" in c:
            allowed_schemes = {s.lower() for s in c["allowed_schemes"]}
            for url_key in ("url", "urls"):
                if args.get(url_key):
                    urls = args[url_key] if isinstance(args[url_key], list) else [args[url_key]]
                    for url in urls:
                        # Full URL safety (private hosts, etc.) is enforced by security.url_safety
                        # inside the tool; here we only gate the scheme.
                        scheme = str(url).split(":", 1)[0].lower()
                        if scheme not in allowed_schemes:
                            raise ToolPermissionDenied(
                                tool_name, f"scheme '{scheme}' not allowed"
                            )
