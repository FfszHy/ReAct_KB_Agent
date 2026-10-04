"""Restricted runtimes assemble without credentials for unavailable tools."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pkb_agent.agent import runtime as runtime_module
from pkb_agent.agent.errors import ConfigError
from pkb_agent.agent.runtime import AgentRuntime
from pkb_agent.app.settings import Settings
from pkb_agent.web.search_provider import make_search_provider

KB_TOOLS = ["rag_search", "rag_read", "rag_list_documents"]


def _stub_network_clients(monkeypatch):
    clients = []
    for factory in (
        runtime_module.SupabaseClient,
        runtime_module.EmbeddingProvider,
        runtime_module.DeepSeekClient,
    ):
        client = SimpleNamespace(close=AsyncMock())
        clients.append(client)
        monkeypatch.setattr(factory, "from_settings", lambda settings, value=client: value)
    return clients


@pytest.mark.parametrize("allowed_tools", [KB_TOOLS, ["web_fetch"]])
async def test_disabled_search_is_not_assembled_or_required(monkeypatch, allowed_tools):
    clients = _stub_network_clients(monkeypatch)
    settings = Settings(_env_file=None, web_search_api_key="", web_search_provider="tavily")

    # Keep the real factory's credential validation as the regression trigger.
    with pytest.raises(ConfigError, match="api_key is required"):
        make_search_provider(settings)

    async with AgentRuntime.build(settings, allowed_tools=allowed_tools) as runtime:
        assert set(runtime.registry.names()) == set(allowed_tools)
        assert runtime.ctx.services.get("search_provider") is None
        assert runtime._assembled
        assert set(runtime._compose_system_prompt().available_tools) == set(allowed_tools)

    for client in clients:
        client.close.assert_awaited_once()


def test_enabled_search_still_validates_its_credentials(monkeypatch):
    _stub_network_clients(monkeypatch)
    settings = Settings(_env_file=None, web_search_api_key="", web_search_provider="tavily")

    with pytest.raises(ConfigError, match="api_key is required"):
        AgentRuntime.build(settings)


@pytest.mark.parametrize("allowed_tools", [[], ["unknown"]])
def test_invalid_allowlist_fails_before_network_clients_are_assembled(monkeypatch, allowed_tools):
    def unexpected_client(settings):
        pytest.fail("invalid allowlist must fail before client construction")

    monkeypatch.setattr(runtime_module.SupabaseClient, "from_settings", unexpected_client)

    with pytest.raises(ValueError, match="allowlist"):
        AgentRuntime.build(Settings(_env_file=None), allowed_tools=allowed_tools)
