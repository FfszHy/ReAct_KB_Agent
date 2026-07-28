"""Memory policy: allowed kinds/scopes and content validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pkb_agent.security.secrets import redact_secrets

if TYPE_CHECKING:
    from pkb_agent.app.settings import Settings

ALLOWED_KINDS: set[str] = {"preference", "fact", "decision", "reference", "note"}
ALLOWED_SCOPES: set[str] = {"short", "long"}
DEFAULT_MAX_CONTENT_CHARS = 4000


@dataclass
class MemoryPolicyConfig:
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS
    allowed_kinds: set[str] = field(default_factory=lambda: set(ALLOWED_KINDS))
    allowed_scopes: set[str] = field(default_factory=lambda: set(ALLOWED_SCOPES))
    default_scope: str = "long"
    reject_secrets: bool = True


class MemoryPolicy:
    def __init__(self, config: MemoryPolicyConfig | None = None) -> None:
        self._config = config or MemoryPolicyConfig()
        if not self._config.allowed_scopes:
            raise ValueError("allowed_scopes must not be empty")
        if self._config.default_scope not in self._config.allowed_scopes:
            # Preserve the old ability to narrow ``allowed_scopes`` without
            # requiring every caller to restate a matching default.
            self._config = replace(
                self._config,
                default_scope=sorted(self._config.allowed_scopes)[0],
            )

    @classmethod
    def from_settings(cls, settings: Settings | None) -> MemoryPolicy:
        if settings is None:
            return cls(MemoryPolicyConfig())
        return cls(
            MemoryPolicyConfig(
                max_content_chars=settings.memory_max_content_chars,
                default_scope=settings.memory_default_scope,
                reject_secrets=settings.memory_reject_secrets,
            )
        )

    def validate(
        self,
        *,
        content: str,
        kind: str = "fact",
        scope: str = "long",
    ) -> tuple[bool, str]:
        if not isinstance(content, str) or not content.strip():
            return False, "content must be non-empty"
        if len(content) > self._config.max_content_chars:
            return False, f"content exceeds max length {self._config.max_content_chars}"
        if self._config.reject_secrets and redact_secrets(content) != content:
            return False, "content appears to contain a secret and cannot be persisted"
        if kind not in self._config.allowed_kinds:
            return False, f"kind '{kind}' not allowed; expected one of {sorted(self._config.allowed_kinds)}"
        if scope not in self._config.allowed_scopes:
            return False, f"scope '{scope}' not allowed; expected one of {sorted(self._config.allowed_scopes)}"
        return True, "ok"

    def default_scope(self) -> str:
        return self._config.default_scope


DEFAULT_POLICY = MemoryPolicy()
