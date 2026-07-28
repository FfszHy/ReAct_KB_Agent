"""Application settings.

Secrets and connection parameters come from environment variables (``.env``).
Non-secret tunables can additionally be supplied by ``config/default.yaml``;
the YAML source has *lower* priority than the environment, so env vars always
win. Field names match the flattened YAML keys (e.g. ``agent.max_steps`` ->
``agent_max_steps``) and the corresponding uppercase env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from pkb_agent.agent.errors import ConfigError


def project_root() -> Path:
    """Return the project root (four levels up from this file)."""
    return Path(__file__).resolve().parents[3]


def yaml_config_path() -> Path:
    env = os.getenv("PKB_CONFIG_PATH")
    if env:
        return Path(env)
    return project_root() / "config" / "default.yaml"


def permissions_config_path() -> Path:
    env = os.getenv("PKB_PERMISSIONS_PATH")
    if env:
        return Path(env)
    return project_root() / "config" / "permissions.yaml"


def prompts_dir() -> Path:
    return project_root() / "config" / "prompts"


class YamlConfigSource(PydanticBaseSettingsSource):
    """Loads ``config/default.yaml`` (flattened with ``_``) as a low-priority source."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        path = yaml_config_path()
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text("utf-8")) or {}
        return self._flatten(raw)

    def _flatten(self, raw: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in raw.items():
            full = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                out.update(self._flatten(value, full))
            else:
                out[full] = value
        return out

    def get_field_value(self, field, field_name: str):
        value = self._data.get(field_name)
        return value, field_name, value is not None

    def prepare_field_value(self, field_name, field, value, value_is_complex):
        return value

    def __call__(self) -> dict[str, Any]:
        return {k: self._data[k] for k in self.settings_cls.model_fields if k in self._data}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    # ``DEEPSEEK_TEMPERATURE`` remains supported as a high-priority override.
    # The normal YAML-facing setting is ``agent.temperature``.
    deepseek_temperature: float | None = None

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    # Supabase is the local knowledge-base control plane.  Do not silently
    # inherit an editor/shell proxy (often a stale localhost proxy) unless the
    # deployment explicitly opts in with ``SUPABASE_TRUST_ENV=true``.
    supabase_trust_env: bool = False

    # Embedding (OpenAI-compatible /embeddings endpoint)
    embedding_api_key: str = ""
    embedding_api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64
    embedding_timeout: int = 60

    # Web search
    web_search_provider: str = "tavily"
    web_search_api_key: str = ""
    web_fetch_timeout: int = 20
    web_max_results: int = 5
    web_fetch_max_chars: int = 8000
    # Web-page citation provenance and freshness policy.
    web_evidence_ttl_hours: int = 168
    web_trusted_domains: list[str] = Field(default_factory=list)

    # Agent
    agent_max_steps: int = 12
    agent_tool_result_max_chars: int = 6000
    agent_temperature: float = 0.2
    agent_answer_verification_max_retries: int = 2
    # DeepSeek JSON Output needs an explicit enough completion ceiling so a
    # valid answer object is not cut off mid-generation.
    agent_json_output_max_tokens: int = Field(default=4096, ge=1)

    # RAG
    rag_top_k: int = 6
    rag_rrf_k: int = 60
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_min_query_len: int = 2
    rag_vector_weight: float = 0.6
    rag_fts_weight: float = 0.4

    # Memory
    memory_max_results: int = 5
    memory_default_scope: str = "long"
    memory_max_content_chars: int = 4000
    memory_reject_secrets: bool = True

    # Prompts
    prompts_manifest_path: str = ""
    prompts_query_rewrite_enabled: bool = True
    prompts_query_rewrite_max_queries: int = 3

    # Dynamic permission overrides
    permissions_db_overrides_enabled: bool = True
    permissions_refresh_seconds: float = 0.0

    # Trace
    trace_enabled: bool = True
    trace_redact_secrets: bool = True

    # Visual workbench API
    api_allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    api_approval_timeout_seconds: int = Field(default=300, ge=10, le=3600)

    # Observability. Prices are configurable because they are part of the
    # provider/account contract. DeepSeek exposes cache-hit and cache-miss
    # token counts separately, so the workbench can price both accurately.
    observability_currency: str = "USD"
    # Legacy single input rate. When set, it remains the fallback for an
    # unclassified input or an API response without cache details.
    observability_input_token_cost_per_million: float | None = Field(default=None, ge=0)
    observability_cache_hit_input_token_cost_per_million: float = Field(default=0.0, ge=0)
    observability_cache_miss_input_token_cost_per_million: float = Field(default=0.0, ge=0)
    observability_output_token_cost_per_million: float = Field(default=0.0, ge=0)

    # ------------------------------------------------------------------
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Order = priority (first wins). Env overrides YAML.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSource(settings_cls),
            file_secret_settings,
        )

    # ------------------------------------------------------------------
    def require(self, *names: str) -> None:
        """Ensure the given settings are non-empty, else raise ConfigError."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise ConfigError(f"missing required config: {', '.join(missing)}")

    def require_llm(self) -> None:
        self.require("deepseek_api_key")

    def require_supabase(self) -> None:
        self.require("supabase_url", "supabase_service_role_key")

    def require_embedding(self) -> None:
        self.require("embedding_api_key", "embedding_api_base_url", "embedding_model")

    def require_web_search(self) -> None:
        self.require("web_search_api_key")

    @property
    def llm_temperature(self) -> float:
        """Return the legacy DeepSeek override or the configured agent default."""
        if self.deepseek_temperature is not None:
            return self.deepseek_temperature
        return self.agent_temperature


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None


def load_prompt(name: str) -> str:
    """Load a markdown prompt from ``config/prompts/<name>.md``."""
    path = prompts_dir() / f"{name}.md"
    if not path.exists():
        raise ConfigError(f"prompt file not found: {path}")
    return path.read_text("utf-8")


# Re-exported for convenience.
Field = Field
