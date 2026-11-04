# SPDX-License-Identifier: GPL-3.0-or-later
"""Ollama provider: local models through Ollama's OpenAI-compatible API.

Ollama serves ``/v1/chat/completions`` and ``/v1/models``, so the OpenAI SDK is
reused. Nothing leaves the computer when Ollama runs locally.
"""

from typing import Optional

from buzz.ai.providers.openai_provider import OpenAIProvider
from buzz.locale import _

DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Low temperature keeps local models close to the transcript. Ollama's own
# default (0.8) is meant for creative chat.
DEFAULT_OLLAMA_TEMPERATURE = 0.2


def normalize_ollama_url(url: str) -> str:
    """Return the OpenAI-compatible base URL (``.../v1``) for an Ollama URL."""
    url = (url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


class OllamaProvider(OpenAIProvider):
    display_name = "Ollama"

    def __init__(
        self,
        url: str = DEFAULT_OLLAMA_URL,
        api_key: str = "",
        temperature: Optional[float] = DEFAULT_OLLAMA_TEMPERATURE,
        timeout: float = 600.0,
    ):
        # Ollama ignores the key, but the OpenAI SDK requires one. A real key
        # is only needed when Ollama runs behind an authenticating proxy.
        super().__init__(
            api_key=api_key or "ollama",
            base_url=normalize_ollama_url(url),
            temperature=temperature,
            timeout=timeout,
        )
        self.url = url or DEFAULT_OLLAMA_URL

    @property
    def is_cloud(self) -> bool:
        host = self.base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        return host not in ("localhost", "127.0.0.1", "::1", "[::1]")

    def connection_error_message(self) -> str:
        return _(
            "Could not connect to Ollama at {url}. Is Ollama installed and running?"
        ).format(url=self.url)
