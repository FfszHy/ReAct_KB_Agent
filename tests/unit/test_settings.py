from __future__ import annotations

from pkb_agent.app.settings import Settings


def test_yaml_agent_temperature_maps_to_llm_temperature(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "agent:\n  temperature: 0.37\nmemory:\n  default_scope: short\n  max_content_chars: 123\n"
        "permissions:\n  db_overrides_enabled: false\n  refresh_seconds: 12.5\n",
        "utf-8",
    )
    monkeypatch.setenv("PKB_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("DEEPSEEK_TEMPERATURE", raising=False)

    settings = Settings(_env_file=None)

    assert settings.agent_temperature == 0.37
    assert settings.llm_temperature == 0.37
    assert settings.memory_default_scope == "short"
    assert settings.memory_max_content_chars == 123
    assert settings.permissions_db_overrides_enabled is False
    assert settings.permissions_refresh_seconds == 12.5


def test_deepseek_temperature_environment_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_TEMPERATURE", "0.11")

    settings = Settings(_env_file=None, agent_temperature=0.37)

    assert settings.llm_temperature == 0.11
