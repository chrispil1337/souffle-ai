# SPDX-License-Identifier: GPL-3.0-or-later
"""Ollama Cloud: direct API access to https://ollama.com without local Ollama."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
import requests

from buzz.ai.providers import registry
from buzz.ai.providers.base import (
    AIAuthError,
    AIConfigurationError,
    AIConnectionError,
    AIResponseError,
    CompletionRequest,
)
from buzz.ai.providers.ollama_cloud_provider import OllamaCloudProvider
from buzz.ai.providers.openai_provider import strip_reasoning
from buzz.ai.providers.registry import ProviderConfig, ProviderId, create_provider, load_provider_config
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key
from buzz.widgets.ai.cloud_consent import has_consent


def attach(provider, handler):
    provider._client = openai.OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return provider


def chat_response(content):
    return {
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


def tags_response(status=200, payload=None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload if payload is not None else {
        "models": [{"name": "gemma4:31b"}, {"name": "deepseek-v4.1-flash"}, {"model": "qwen3:480b"}]
    }
    return response


class TestOllamaCloudProvider:
    def test_chat_goes_to_ollama_com_with_bearer_key(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=chat_response('{"segments": []}'))

        provider = attach(OllamaCloudProvider(api_key="ok-key"), handler)
        result = provider.complete(CompletionRequest("s", "u", "deepseek-v4.1-flash", json_mode=True))

        assert result == '{"segments": []}'
        assert seen["url"] == "https://ollama.com/v1/chat/completions"
        assert seen["auth"] == "Bearer ok-key"
        assert seen["body"]["model"] == "deepseek-v4.1-flash"
        assert seen["body"]["temperature"] == 0.2
        assert seen["body"]["response_format"] == {"type": "json_object"}

    def test_reasoning_is_removed_from_answer(self):
        def handler(request):
            return httpx.Response(200, json=chat_response('<think>{"denken": 1}</think>\n{"a": 1}'))

        provider = attach(OllamaCloudProvider(api_key="k"), handler)
        assert provider.complete(CompletionRequest("s", "u", "m")) == '{"a": 1}'

    def test_is_a_cloud_provider_and_needs_a_key(self):
        provider = OllamaCloudProvider(api_key="")
        assert provider.is_cloud is True
        with pytest.raises(AIConfigurationError):
            provider.complete(CompletionRequest("s", "u", "m"))
        with pytest.raises(AIConfigurationError):
            provider.list_models()

    def test_list_models_from_api_tags(self):
        with patch.object(requests, "get", return_value=tags_response()) as get:
            models = OllamaCloudProvider(api_key="k").list_models()
        assert models == ["deepseek-v4.1-flash", "gemma4:31b", "qwen3:480b"]
        assert get.call_args.args[0] == "https://ollama.com/api/tags"
        assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer k"}

    @pytest.mark.parametrize(
        "status,error", [(401, AIAuthError), (403, AIAuthError), (500, AIResponseError)]
    )
    def test_list_models_errors(self, status, error):
        with patch.object(requests, "get", return_value=tags_response(status)):
            with pytest.raises(error):
                OllamaCloudProvider(api_key="k").list_models()

    def test_list_models_offline(self):
        with patch.object(requests, "get", side_effect=requests.ConnectionError("offline")):
            with pytest.raises(AIConnectionError) as info:
                OllamaCloudProvider(api_key="k").list_models()
        assert "Ollama Cloud" in info.value.user_message

    def test_connection_test_checks_the_key(self):
        def rejected(request):
            return httpx.Response(401, json={"error": "unauthorized"})

        provider = attach(OllamaCloudProvider(api_key="falsch"), rejected)
        with patch.object(requests, "get", return_value=tags_response()):
            with pytest.raises(AIAuthError):
                provider.test_connection()

    def test_connection_test_tolerates_missing_models_endpoint(self):
        def not_found(request):
            return httpx.Response(404, json={"error": "not found"})

        provider = attach(OllamaCloudProvider(api_key="k"), not_found)
        with patch.object(requests, "get", return_value=tags_response()):
            assert "deepseek-v4.1-flash" in provider.test_connection()


class TestOllamaCloudConfiguration:
    @pytest.fixture
    def keyring(self):
        store = {}
        with (
            patch.object(registry, "get_password", side_effect=lambda key: store.get(key, "")),
            patch.object(registry, "set_password", side_effect=lambda k, v: store.__setitem__(k, v)),
            patch.object(registry, "delete_password", side_effect=lambda key: store.pop(key, None)),
        ):
            yield store

    def test_config_uses_own_keychain_entry_and_model(self, keyring):
        keyring[Key.OLLAMA_CLOUD_API_KEY] = "cloud-key"
        keyring[Key.OLLAMA_API_KEY] = "local-key"
        values = {
            Settings.Key.AI_PROVIDER: "ollama_cloud",
            Settings.Key.AI_MODEL_OLLAMA_CLOUD: "deepseek-v4.1-flash",
        }
        settings = SimpleNamespace(value=lambda key, default, value_type=None: values.get(key, default))

        config = load_provider_config(settings)

        assert config.provider_id == ProviderId.OLLAMA_CLOUD
        assert config.api_key == "cloud-key"
        assert config.model == "deepseek-v4.1-flash"
        assert config.temperature == 0.2
        provider = create_provider(config)
        assert isinstance(provider, OllamaCloudProvider)
        assert provider.base_url == "https://ollama.com/v1"

    def test_needs_consent_and_key(self, settings):
        settings.set_value(Settings.Key.AI_CLOUD_CONSENT, "")
        config = ProviderConfig(ProviderId.OLLAMA_CLOUD, "m", api_key="k")
        assert not has_consent(settings, config)
        assert ProviderId.OLLAMA_CLOUD.needs_api_key
        assert not ProviderId.OLLAMA.needs_api_key


def test_strip_reasoning():
    assert strip_reasoning("<think>\nüberlegen\n</think>\n\nAntwort") == "Antwort"
    assert strip_reasoning("<THINK>a</THINK>b<think>c</think>d") == "bd"
    assert strip_reasoning("ohne") == "ohne"
