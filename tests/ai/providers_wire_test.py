# SPDX-License-Identifier: GPL-3.0-or-later
"""Wire-level tests: the real SDKs run against an in-process fake HTTP server.

These check what would actually be sent to the providers (headers, request
body) and that real streaming responses are parsed, without any network.
"""

import json

import anthropic
import httpx
import httpx2
import openai
import pytest

from buzz.ai.providers.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from buzz.ai.providers.base import AIAuthError, CompletionRequest
from buzz.ai.providers.ollama_provider import OllamaProvider
from buzz.ai.providers.openai_provider import OpenAIProvider


def sse(events):
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    ).encode()


def anthropic_stream_body(text, stop_reason="end_turn"):
    return sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-5-5",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


class TestAnthropicWire:
    def make(self, handler, effort="medium"):
        provider = AnthropicProvider(api_key="sk-ant-test", effort=effort)
        provider._client = anthropic.Anthropic(
            api_key="sk-ant-test",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )
        return provider

    def test_streamed_request_with_fallback(self):
        seen = {}

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen["url"] = str(request.url)
            seen["headers"] = request.headers
            seen["body"] = json.loads(request.content)
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=anthropic_stream_body("Bereinigter Text."),
            )

        provider = self.make(handler)
        result = provider.complete(
            CompletionRequest("Bereinige.", "äh also der Text", "claude-opus-5-5")
        )

        assert result == "Bereinigter Text."
        assert seen["url"].endswith("/v1/messages?beta=true") or seen["url"].endswith(
            "/v1/messages"
        )
        assert FALLBACK_BETA in seen["headers"].get("anthropic-beta", "")
        body = seen["body"]
        assert body["model"] == "claude-opus-5-5"
        assert body["stream"] is True
        assert body["fallbacks"] == "default"
        assert body["output_config"] == {"effort": "medium"}
        assert body["system"] == "Bereinige."
        assert body["messages"] == [{"role": "user", "content": "äh also der Text"}]
        assert "temperature" not in body

    def test_auth_error_from_server(self):
        def handler(request):
            return httpx2.Response(
                401,
                json={
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "invalid x-api-key"},
                },
            )

        provider = self.make(handler)
        with pytest.raises(AIAuthError):
            provider.complete(CompletionRequest("s", "u", "claude-opus-5-5"))

    def test_list_models(self):
        def handler(request):
            assert request.url.path == "/v1/models"
            return httpx2.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "claude-sonnet-5-5",
                            "type": "model",
                            "display_name": "Claude Sonnet 5.5",
                            "created_at": "2026-01-01T00:00:00Z",
                        },
                        {
                            "id": "claude-opus-5-5",
                            "type": "model",
                            "display_name": "Claude Opus 5.5",
                            "created_at": "2026-01-01T00:00:00Z",
                        },
                    ],
                    "has_more": False,
                    "first_id": "claude-sonnet-5-5",
                    "last_id": "claude-opus-5-5",
                },
            )

        provider = self.make(handler)
        assert provider.test_connection() == ["claude-opus-5-5", "claude-sonnet-5-5"]


def openai_chat_response(content):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class TestOpenAICompatibleWire:
    def attach(self, provider, handler):
        provider._client = openai.OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        return provider

    def test_openai_chat_completion(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=openai_chat_response("Sauberer Text."))

        provider = self.attach(OpenAIProvider(api_key="sk-test"), handler)
        result = provider.complete(CompletionRequest("s", "u", "gpt-5", json_mode=True))

        assert result == "Sauberer Text."
        assert seen["auth"] == "Bearer sk-test"
        assert seen["body"]["response_format"] == {"type": "json_object"}
        assert "temperature" not in seen["body"]

    def test_ollama_uses_local_v1_endpoint(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=openai_chat_response("Lokal bereinigt."))

        provider = self.attach(OllamaProvider("http://localhost:11434"), handler)
        result = provider.complete(CompletionRequest("s", "u", "qwen2.5:14b"))

        assert result == "Lokal bereinigt."
        assert seen["url"] == "http://localhost:11434/v1/chat/completions"
        assert seen["body"]["temperature"] == 0.2
