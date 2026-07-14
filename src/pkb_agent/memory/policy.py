"""Memory policy: allowed kinds/scopes and content validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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


class MemoryPolicy:
    def __init__(self, config: MemoryPolicyConfig | None = None) -> None:
        self._config = config or MemoryPolicyConfig()

    @classmethod
    def from_settings(cls, settings: Settings) -> MemoryPolicy:
        return cls(MemoryPolicyConfig())

    def validate(
        self,
        *,
        content: str,
        kind: str = "fact",
        scope: str = "long",
    ) -> tuple[bool, str]:
        if not content or not content.strip():
            return False, "content must be non-empty"
        if len(content) > self._config.max_content_chars:
            return False, f"content exceeds max length {self._config.max_content_chars}"
        if kind not in self._config.allowed_kinds:
            return False, f"kind '{kind}' not allowed; expected one of {sorted(self._config.allowed_kinds)}"
        if scope not in self._config.allowed_scopes:
            return False, f"scope '{scope}' not allowed; expected one of {sorted(self._config.allowed_scopes)}"
        return True, "ok"

    def default_scope(self) -> str:
        return "long"


DEFAULT_POLICY = MemoryPolicy()
