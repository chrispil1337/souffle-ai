# SPDX-License-Identifier: GPL-3.0-or-later
from unittest.mock import patch

import pytest

from buzz.ai.providers import registry
from buzz.ai.providers.anthropic_provider import AnthropicProvider
from buzz.ai.providers.ollama_provider import OllamaProvider
from buzz.ai.providers.openai_provider import OpenAIProvider
from buzz.ai.providers.registry import (
    ProviderConfig,
    ProviderId,
    create_provider,
    load_provider_config,
)
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key


class FakeSettings:
    """Minimal stand-in for ``Settings`` backed by a dict."""

    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, default_value, value_type=None):
        return self.values.get(key, default_value)


@pytest.fixture
def keyring():
    store = {}
    with (
        patch.object(registry, "get_password", side_effect=lambda key: store.get(key, "")),
        patch.object(
            registry, "set_password", side_effect=lambda key, value: store.__setitem__(key, value)
        ),
        patch.object(registry, "delete_password", side_effect=lambda key: store.pop(key, None)),
    ):
        yield store


class TestLoadProviderConfig:
    def test_defaults_to_local_ollama(self, keyring):
        config = load_provider_config(FakeSettings())
        assert config.provider_id == ProviderId.OLLAMA
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.2
        assert config.model == ""

    def test_invalid_provider_falls_back_to_default(self, keyring):
        settings = FakeSettings({Settings.Key.AI_PROVIDER: "unknown"})
        assert load_provider_config(settings).provider_id == ProviderId.OLLAMA

    def test_anthropic_reads_key_from_keychain_and_effort(self, keyring):
        keyring[Key.ANTHROPIC_API_KEY] = "sk-ant-secret"
        settings = FakeSettings(
            {
                Settings.Key.AI_PROVIDER: "anthropic",
                Settings.Key.AI_ANTHROPIC_EFFORT: "high",
            }
        )
        config = load_provider_config(settings)
        assert config.provider_id == ProviderId.ANTHROPIC
        assert config.api_key == "sk-ant-secret"
        assert config.model == "claude-opus-5-5"
        assert config.effort == "high"

    def test_openai_uses_shared_openai_key_and_base_url(self, keyring):
        keyring[Key.OPENAI_API_KEY] = "sk-openai"
        settings = FakeSettings(
            {
                Settings.Key.AI_PROVIDER: "openai",
                Settings.Key.AI_MODEL_OPENAI: "my-model",
                Settings.Key.AI_OPENAI_BASE_URL: "https://proxy.example.com/v1",
            }
        )
        config = load_provider_config(settings)
        assert config.api_key == "sk-openai"
        assert config.model == "my-model"
        assert config.base_url == "https://proxy.example.com/v1"
        assert config.temperature is None

    def test_explicit_provider_overrides_selection(self, keyring):
        settings = FakeSettings({Settings.Key.AI_PROVIDER: "openai"})
        config = load_provider_config(settings, ProviderId.ANTHROPIC)
        assert config.provider_id == ProviderId.ANTHROPIC

    def test_temperature_and_timeout_parsing(self, keyring):
        settings = FakeSettings(
            {
                Settings.Key.AI_PROVIDER: "openai",
                Settings.Key.AI_TEMPERATURE: "0.5",
                Settings.Key.AI_REQUEST_TIMEOUT: "120",
            }
        )
        config = load_provider_config(settings)
        assert config.temperature == 0.5
        assert config.timeout == 120.0

    def test_invalid_temperature_is_ignored(self, keyring):
        settings = FakeSettings(
            {Settings.Key.AI_PROVIDER: "openai", Settings.Key.AI_TEMPERATURE: "warm"}
        )
        assert load_provider_config(settings).temperature is None


class TestApiKeys:
    def test_set_and_get(self, keyring):
        registry.set_api_key(ProviderId.ANTHROPIC, "  sk-ant  ")
        assert keyring[Key.ANTHROPIC_API_KEY] == "sk-ant"
        assert registry.get_api_key(ProviderId.ANTHROPIC) == "sk-ant"

    def test_empty_key_deletes_entry(self, keyring):
        keyring[Key.OLLAMA_API_KEY] = "old"
        registry.set_api_key(ProviderId.OLLAMA, "")
        assert Key.OLLAMA_API_KEY not in keyring

    def test_every_provider_has_its_own_keychain_entry(self):
        entries = set(registry.API_KEY_ENTRIES.values())
        assert len(entries) == len(ProviderId)


class TestCreateProvider:
    def test_openai(self):
        provider = create_provider(
            ProviderConfig(ProviderId.OPENAI, "m", api_key="k", base_url="", temperature=None)
        )
        assert isinstance(provider, OpenAIProvider)
        assert provider.base_url is None

    def test_anthropic(self):
        provider = create_provider(
            ProviderConfig(ProviderId.ANTHROPIC, "m", api_key="k", effort="low")
        )
        assert isinstance(provider, AnthropicProvider)
        assert provider.effort == "low"

    def test_ollama(self):
        provider = create_provider(
            ProviderConfig(ProviderId.OLLAMA, "m", base_url="http://localhost:11434")
        )
        assert isinstance(provider, OllamaProvider)
        assert provider.is_cloud is False


def test_sdk_loggers_do_not_log_requests():
    """At DEBUG level the SDKs would write transcript text to the log file."""
    import logging

    from buzz.ai.providers import SDK_LOGGERS, quiet_sdk_loggers

    logging.getLogger("openai").setLevel(logging.DEBUG)
    quiet_sdk_loggers()
    for name in SDK_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING
