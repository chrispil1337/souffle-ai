# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the AI provider layer. No network access: SDK clients are mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import httpx2
import openai
import pytest

from buzz.ai.providers import base
from buzz.ai.providers.anthropic_provider import (
    FALLBACK_BETA,
    MAX_OUTPUT_TOKENS,
    AnthropicProvider,
)
from buzz.ai.providers.base import (
    AIAuthError,
    AIConfigurationError,
    AIConnectionError,
    AIRateLimitError,
    AIRefusalError,
    AIResponseError,
    AITimeoutError,
    CompletionRequest,
    with_retry,
)
from buzz.ai.providers.ollama_provider import OllamaProvider, normalize_ollama_url
from buzz.ai.providers.openai_provider import OpenAIProvider

REQUEST = CompletionRequest(system="system prompt", user="transcript", model="test-model")


@pytest.fixture(autouse=True)
def no_retry_sleep():
    with patch.object(base.time, "sleep") as sleep:
        yield sleep


# --- with_retry ------------------------------------------------------------


class TestWithRetry:
    def test_returns_result_without_retry(self):
        sleep = MagicMock()
        assert with_retry(lambda: "ok", sleep=sleep) == "ok"
        sleep.assert_not_called()

    def test_retries_retryable_errors_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise AIConnectionError("offline")
            return "ok"

        sleep = MagicMock()
        assert with_retry(fn, max_attempts=3, sleep=sleep) == "ok"
        assert len(calls) == 3
        assert sleep.call_count == 2
        # exponential backoff: the second delay is longer than the first
        first, second = (c.args[0] for c in sleep.call_args_list)
        assert second > first

    def test_gives_up_after_max_attempts(self):
        fn = MagicMock(side_effect=AIRateLimitError("limit"))
        with pytest.raises(AIRateLimitError):
            with_retry(fn, max_attempts=3, sleep=MagicMock())
        assert fn.call_count == 3

    def test_does_not_retry_non_retryable_errors(self):
        fn = MagicMock(side_effect=AIAuthError("bad key"))
        with pytest.raises(AIAuthError):
            with_retry(fn, max_attempts=3, sleep=MagicMock())
        assert fn.call_count == 1

    def test_server_errors_are_retryable_client_errors_are_not(self):
        assert AIResponseError("x", retryable=True).retryable
        assert not AIResponseError("x").retryable

    def test_stops_retrying_when_canceled(self):
        fn = MagicMock(side_effect=AITimeoutError("slow"))
        with pytest.raises(AITimeoutError):
            with_retry(fn, max_attempts=5, sleep=MagicMock(), is_canceled=lambda: True)
        assert fn.call_count == 1


# --- helpers to build real SDK exceptions ------------------------------------


def openai_status_error(cls, status):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("error", response=response, body=None)


def openai_request():
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def anthropic_status_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls("error", response=response, body=None)


def anthropic_request():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def openai_completion(content="cleaned text", finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ]
    )


# --- OpenAI ------------------------------------------------------------------


class TestOpenAIProvider:
    def make(self, **kwargs):
        provider = OpenAIProvider(api_key="sk-test", **kwargs)
        provider._client = MagicMock()
        return provider

    def test_sends_system_and_user_messages(self):
        provider = self.make()
        provider._client.chat.completions.create.return_value = openai_completion()

        assert provider.complete(REQUEST) == "cleaned text"

        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "transcript"},
        ]
        assert "temperature" not in kwargs
        assert "response_format" not in kwargs

    def test_sends_temperature_and_json_mode_when_set(self):
        provider = self.make(temperature=0.2)
        provider._client.chat.completions.create.return_value = openai_completion("{}")

        provider.complete(CompletionRequest("s", "u", "m", json_mode=True))

        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.2
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_truncated_response_is_an_error(self):
        provider = self.make()
        provider._client.chat.completions.create.return_value = openai_completion(
            "half", finish_reason="length"
        )
        with pytest.raises(AIResponseError):
            provider.complete(REQUEST)

    def test_empty_response_is_an_error(self):
        provider = self.make()
        provider._client.chat.completions.create.return_value = openai_completion("  ")
        with pytest.raises(AIResponseError):
            provider.complete(REQUEST)

    def test_missing_model_is_a_configuration_error(self):
        provider = self.make()
        with pytest.raises(AIConfigurationError):
            provider.complete(CompletionRequest("s", "u", model=""))
        provider._client.chat.completions.create.assert_not_called()

    def test_missing_api_key_is_a_configuration_error(self):
        provider = OpenAIProvider(api_key="")
        with pytest.raises(AIConfigurationError):
            provider.complete(REQUEST)

    def test_client_is_created_without_sdk_retries(self):
        provider = OpenAIProvider(api_key="sk-test", base_url="https://example.com/v1")
        with patch("openai.OpenAI") as client_cls:
            _ = provider.client
        kwargs = client_cls.call_args.kwargs
        assert kwargs["max_retries"] == 0
        assert kwargs["base_url"] == "https://example.com/v1"

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (lambda: openai_status_error(openai.AuthenticationError, 401), AIAuthError),
            (lambda: openai_status_error(openai.PermissionDeniedError, 403), AIAuthError),
            (lambda: openai_status_error(openai.RateLimitError, 429), AIRateLimitError),
            (lambda: openai_status_error(openai.NotFoundError, 404), AIResponseError),
            (lambda: openai_status_error(openai.BadRequestError, 400), AIResponseError),
            (lambda: openai_status_error(openai.InternalServerError, 500), AIResponseError),
            (lambda: openai.APITimeoutError(request=openai_request()), AITimeoutError),
            (lambda: openai.APIConnectionError(request=openai_request()), AIConnectionError),
        ],
    )
    def test_error_mapping(self, exc, expected):
        provider = self.make()
        provider._client.chat.completions.create.side_effect = exc()
        with pytest.raises(expected) as info:
            provider._complete(REQUEST)
        assert info.value.user_message

    def test_server_error_is_retried(self):
        provider = self.make()
        provider._client.chat.completions.create.side_effect = [
            openai_status_error(openai.InternalServerError, 503),
            openai_completion("ok"),
        ]
        assert provider.complete(REQUEST) == "ok"
        assert provider._client.chat.completions.create.call_count == 2

    def test_auth_error_is_not_retried(self):
        provider = self.make()
        provider._client.chat.completions.create.side_effect = openai_status_error(
            openai.AuthenticationError, 401
        )
        with pytest.raises(AIAuthError):
            provider.complete(REQUEST)
        assert provider._client.chat.completions.create.call_count == 1

    def test_list_models_is_sorted(self):
        provider = self.make()
        provider._client.models.list.return_value = [
            SimpleNamespace(id="b-model"),
            SimpleNamespace(id="a-model"),
        ]
        assert provider.test_connection() == ["a-model", "b-model"]

    def test_list_models_maps_errors(self):
        provider = self.make()
        provider._client.models.list.side_effect = openai_status_error(
            openai.AuthenticationError, 401
        )
        with pytest.raises(AIAuthError):
            provider.list_models()


# --- Ollama ------------------------------------------------------------------


class TestOllamaProvider:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("", "http://localhost:11434/v1"),
            ("http://localhost:11434", "http://localhost:11434/v1"),
            ("http://localhost:11434/", "http://localhost:11434/v1"),
            ("http://localhost:11434/v1", "http://localhost:11434/v1"),
            ("192.168.1.20:11434", "http://192.168.1.20:11434/v1"),
            ("https://ollama.example.com", "https://ollama.example.com/v1"),
        ],
    )
    def test_normalize_url(self, url, expected):
        assert normalize_ollama_url(url) == expected

    def test_local_ollama_is_not_cloud(self):
        assert OllamaProvider("http://localhost:11434").is_cloud is False
        assert OllamaProvider("http://127.0.0.1:11434").is_cloud is False

    def test_remote_ollama_counts_as_cloud(self):
        assert OllamaProvider("http://192.168.1.20:11434").is_cloud is True

    def test_works_without_api_key(self):
        provider = OllamaProvider()
        assert provider.api_key == "ollama"
        assert provider.temperature == 0.2

    def test_connection_error_mentions_url(self):
        provider = OllamaProvider("http://localhost:11434")
        provider._client = MagicMock()
        provider._client.models.list.side_effect = openai.APIConnectionError(
            request=openai_request()
        )
        with pytest.raises(AIConnectionError) as info:
            provider.test_connection()
        assert "http://localhost:11434" in info.value.user_message


# --- Anthropic ---------------------------------------------------------------


def anthropic_message(blocks=None, stop_reason="end_turn"):
    if blocks is None:
        blocks = [SimpleNamespace(type="text", text="cleaned text")]
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, stop_details=None)


class TestAnthropicProvider:
    def make(self, effort="medium"):
        provider = AnthropicProvider(api_key="sk-ant-test", effort=effort)
        provider._client = MagicMock()
        return provider

    @staticmethod
    def set_message(stream_method, message):
        stream_method.return_value.__enter__.return_value.get_final_message.return_value = (
            message
        )

    def test_new_models_use_server_side_fallback(self):
        provider = self.make()
        stream = provider._client.beta.messages.stream
        self.set_message(stream, anthropic_message())

        result = provider.complete(CompletionRequest("system", "user", "claude-opus-5-5"))

        assert result == "cleaned text"
        kwargs = stream.call_args.kwargs
        assert kwargs["betas"] == [FALLBACK_BETA]
        assert kwargs["fallbacks"] == "default"
        assert kwargs["model"] == "claude-opus-5-5"
        assert kwargs["system"] == "system"
        assert kwargs["messages"] == [{"role": "user", "content": "user"}]
        assert kwargs["max_tokens"] == MAX_OUTPUT_TOKENS
        assert kwargs["output_config"] == {"effort": "medium"}
        # Sampling parameters are not accepted by the current API
        assert "temperature" not in kwargs
        provider._client.messages.stream.assert_not_called()

    def test_other_models_use_plain_stream(self):
        provider = self.make(effort="")
        stream = provider._client.messages.stream
        self.set_message(stream, anthropic_message())

        provider.complete(CompletionRequest("s", "u", "claude-haiku-4-5"))

        kwargs = stream.call_args.kwargs
        assert "fallbacks" not in kwargs
        assert "output_config" not in kwargs

    def test_invalid_effort_is_not_sent(self):
        assert AnthropicProvider(api_key="k", effort="extreme").effort is None

    def test_only_text_blocks_are_returned(self):
        provider = self.make()
        self.set_message(
            provider._client.beta.messages.stream,
            anthropic_message(
                [
                    SimpleNamespace(type="thinking", thinking=""),
                    SimpleNamespace(type="text", text="Teil 1 "),
                    SimpleNamespace(type="text", text="Teil 2"),
                ]
            ),
        )
        assert provider.complete(CompletionRequest("s", "u", "claude-opus-5-5")) == (
            "Teil 1 Teil 2"
        )

    def test_refusal_is_reported(self):
        provider = self.make()
        self.set_message(
            provider._client.beta.messages.stream,
            anthropic_message([], stop_reason="refusal"),
        )
        with pytest.raises(AIRefusalError):
            provider.complete(CompletionRequest("s", "u", "claude-opus-5-5"))

    def test_truncated_response_is_an_error(self):
        provider = self.make()
        self.set_message(
            provider._client.beta.messages.stream,
            anthropic_message(stop_reason="max_tokens"),
        )
        with pytest.raises(AIResponseError):
            provider.complete(CompletionRequest("s", "u", "claude-opus-5-5"))

    def test_missing_api_key_is_a_configuration_error(self):
        provider = AnthropicProvider(api_key="")
        with pytest.raises(AIConfigurationError):
            provider.list_models()

    def test_client_is_created_with_explicit_key_and_no_sdk_retries(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        with patch("anthropic.Anthropic") as client_cls:
            _ = provider.client
        kwargs = client_cls.call_args.kwargs
        assert kwargs["api_key"] == "sk-ant-test"
        assert kwargs["max_retries"] == 0

    def test_list_models_only_active_and_sorted(self):
        provider = self.make()
        provider._client.models.list.return_value = [
            SimpleNamespace(id="claude-sonnet-5-5"),
            SimpleNamespace(id="claude-opus-5-5"),
        ]
        assert provider.list_models() == ["claude-opus-5-5", "claude-sonnet-5-5"]
        assert provider._client.models.list.call_args.kwargs == {"lifecycle": ["active"]}

    @pytest.mark.parametrize(
        "exc,expected,retryable",
        [
            (lambda: anthropic_status_error(anthropic.AuthenticationError, 401), AIAuthError, False),
            (lambda: anthropic_status_error(anthropic.PermissionDeniedError, 403), AIAuthError, False),
            (lambda: anthropic_status_error(anthropic.RateLimitError, 429), AIRateLimitError, True),
            (lambda: anthropic_status_error(anthropic.NotFoundError, 404), AIResponseError, False),
            (lambda: anthropic_status_error(anthropic.BadRequestError, 400), AIResponseError, False),
            (lambda: anthropic_status_error(anthropic.APIStatusError, 529), AIResponseError, True),
            (lambda: anthropic.APITimeoutError(request=anthropic_request()), AITimeoutError, True),
            (lambda: anthropic.APIConnectionError(request=anthropic_request()), AIConnectionError, True),
        ],
    )
    def test_error_mapping(self, exc, expected, retryable):
        provider = self.make()
        provider._client.beta.messages.stream.side_effect = exc()
        with pytest.raises(expected) as info:
            provider._complete(CompletionRequest("s", "u", "claude-opus-5-5"))
        assert info.value.retryable is retryable
        assert info.value.user_message
