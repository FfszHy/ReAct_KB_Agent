"""Manifest-backed prompt registry and lifecycle-aware prompt composition.

Prompt files are content, while ``config/prompts/manifest.yaml`` declares
where each prompt is allowed to participate in the runtime.  Keeping that
mapping here prevents prompt files from silently becoming dead documentation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from pkb_agent.agent.errors import ConfigError

if TYPE_CHECKING:
    from pkb_agent.app.settings import Settings


_VALID_PHASES = frozenset({"system", "pre_tool"})


@dataclass(frozen=True)
class PromptSpec:
    """One prompt declaration from the manifest."""

    id: str
    filename: str
    phase: str
    always: bool = False
    when_tool_available: str | None = None
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptReference:
    """Stable, non-content identifier suitable for traces."""

    id: str
    phase: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "phase": self.phase, "sha256": self.sha256}


@dataclass(frozen=True)
class PromptComposition:
    """Rendered prompt content and the source prompts that produced it."""

    content: str
    phase: str
    manifest_version: str
    references: tuple[PromptReference, ...] = ()

    def trace_context(self) -> dict[str, Any]:
        """Return metadata that is useful for auditing without storing prompt text."""
        return {
            "manifest_version": self.manifest_version,
            "phase": self.phase,
            "prompts": [reference.to_dict() for reference in self.references],
        }


class PromptRegistry:
    """Load and validate prompt files described by a YAML manifest."""

    def __init__(self, *, prompt_dir: Path, manifest_path: Path) -> None:
        self._prompt_dir = prompt_dir.resolve()
        self._manifest_path = manifest_path.resolve()
        self._manifest_version = "unversioned"
        self._specs: dict[str, PromptSpec] = {}
        self._contents: dict[str, str] = {}
        self._references: dict[str, PromptReference] = {}
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> PromptRegistry:
        from pkb_agent.app.settings import prompts_dir

        prompt_dir = prompts_dir()
        raw_path = (settings.prompts_manifest_path or "").strip()
        manifest_path = Path(raw_path) if raw_path else prompt_dir / "manifest.yaml"
        return cls(prompt_dir=prompt_dir, manifest_path=manifest_path)

    @property
    def manifest_version(self) -> str:
        return self._manifest_version

    def spec(self, prompt_id: str) -> PromptSpec:
        try:
            return self._specs[prompt_id]
        except KeyError as exc:
            raise ConfigError(f"prompt is not declared in manifest: {prompt_id}") from exc

    def content(self, prompt_id: str) -> str:
        self.spec(prompt_id)
        return self._contents[prompt_id]

    def reference(self, prompt_id: str) -> PromptReference:
        self.spec(prompt_id)
        return self._references[prompt_id]

    def specs_for_phase(self, phase: str) -> list[PromptSpec]:
        if phase not in _VALID_PHASES:
            raise ConfigError(f"unknown prompt phase: {phase}")
        return [spec for spec in self._specs.values() if spec.phase == phase]

    def _load(self) -> None:
        if not self._manifest_path.exists():
            raise ConfigError(f"prompt manifest not found: {self._manifest_path}")

        try:
            raw = yaml.safe_load(self._manifest_path.read_text("utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid prompt manifest: {self._manifest_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError("prompt manifest must be a YAML object")
        prompts = raw.get("prompts")
        if not isinstance(prompts, dict) or not prompts:
            raise ConfigError("prompt manifest must contain a non-empty 'prompts' mapping")

        self._manifest_version = str(raw.get("version", "unversioned"))
        for prompt_id, config in prompts.items():
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ConfigError("prompt manifest contains an invalid prompt id")
            if not isinstance(config, dict):
                raise ConfigError(f"prompt '{prompt_id}' configuration must be an object")

            spec = self._parse_spec(prompt_id, config)
            path = self._resolve_prompt_path(spec.filename)
            if not path.exists():
                raise ConfigError(f"prompt file not found for '{prompt_id}': {path}")
            content = path.read_text("utf-8").strip()
            if not content:
                raise ConfigError(f"prompt file is empty: {path}")

            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            self._specs[prompt_id] = spec
            self._contents[prompt_id] = content
            self._references[prompt_id] = PromptReference(
                id=prompt_id,
                phase=spec.phase,
                sha256=digest,
            )

    def _parse_spec(self, prompt_id: str, config: dict[str, Any]) -> PromptSpec:
        filename = config.get("file", f"{prompt_id}.md")
        phase = config.get("phase")
        if not isinstance(filename, str) or not filename:
            raise ConfigError(f"prompt '{prompt_id}' must define a non-empty file")
        if phase not in _VALID_PHASES:
            raise ConfigError(
                f"prompt '{prompt_id}' has invalid phase {phase!r}; "
                f"expected one of {sorted(_VALID_PHASES)}"
            )

        always = bool(config.get("always", False))
        when_tool_available = config.get("when_tool_available")
        if when_tool_available is not None and not isinstance(when_tool_available, str):
            raise ConfigError(f"prompt '{prompt_id}'.when_tool_available must be a string")

        raw_tools = config.get("tools", [])
        if not isinstance(raw_tools, list) or not all(isinstance(tool, str) for tool in raw_tools):
            raise ConfigError(f"prompt '{prompt_id}'.tools must be a list of strings")
        if phase == "pre_tool" and not raw_tools:
            raise ConfigError(f"pre_tool prompt '{prompt_id}' must target at least one tool")

        return PromptSpec(
            id=prompt_id,
            filename=filename,
            phase=phase,
            always=always,
            when_tool_available=when_tool_available,
            tools=tuple(raw_tools),
        )

    def _resolve_prompt_path(self, filename: str) -> Path:
        path = (self._prompt_dir / filename).resolve()
        try:
            path.relative_to(self._prompt_dir)
        except ValueError as exc:
            raise ConfigError(f"prompt file escapes prompts directory: {filename}") from exc
        return path


class PromptComposer:
    """Select registered prompts for a runtime lifecycle phase."""

    def __init__(self, registry: PromptRegistry) -> None:
        self._registry = registry

    def compose_system(self, available_tools: Collection[str]) -> PromptComposition:
        tool_names = set(available_tools)
        specs = [
            spec
            for spec in self._registry.specs_for_phase("system")
            if self._is_active_system_prompt(spec, tool_names)
        ]
        if not specs:
            raise ConfigError("prompt manifest selected no active system prompts")
        return self._compose("system", specs)

    def compose_pre_tool(self, tool_name: str) -> PromptComposition:
        specs = [
            spec
            for spec in self._registry.specs_for_phase("pre_tool")
            if tool_name in spec.tools
        ]
        return self._compose("pre_tool", specs)

    def _is_active_system_prompt(self, spec: PromptSpec, available_tools: set[str]) -> bool:
        if spec.always:
            return True
        if spec.when_tool_available:
            return spec.when_tool_available in available_tools
        return False

    def _compose(self, phase: str, specs: list[PromptSpec]) -> PromptComposition:
        content = "\n\n---\n\n".join(self._registry.content(spec.id) for spec in specs)
        references = tuple(self._registry.reference(spec.id) for spec in specs)
        return PromptComposition(
            content=content,
            phase=phase,
            manifest_version=self._registry.manifest_version,
            references=references,
        )
