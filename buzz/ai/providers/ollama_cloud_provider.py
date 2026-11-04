# SPDX-License-Identifier: GPL-3.0-or-later
"""Ollama Cloud: models hosted by Ollama, used directly through ollama.com.

No local Ollama installation is needed. Requests go to the OpenAI-compatible
endpoint ``https://ollama.com/v1`` with the API key from
https://ollama.com/settings/keys as ``Authorization: Bearer``. The list of
cloud models comes from ``https://ollama.com/api/tags``; requests use the
names from that list (without a ``-cloud`` suffix).

Transcripts are sent to Ollama's servers, so this is a cloud provider that
needs the user's consent, like OpenAI and Anthropic.
"""

import logging
from typing import List, Optional

from buzz.ai.providers.base import (
    AIAuthError,
    AIConnectionError,
    AIError,
    AIRateLimitError,
    AIResponseError,
    AITimeoutError,
)
from buzz.ai.providers.ollama_provider import DEFAULT_OLLAMA_TEMPERATURE
from buzz.ai.providers.openai_provider import DEFAULT_TIMEOUT_SECONDS, OpenAIProvider
from buzz.locale import _

logger = logging.getLogger(__name__)

OLLAMA_CLOUD_HOST = "https://ollama.com"
OLLAMA_CLOUD_API_BASE = f"{OLLAMA_CLOUD_HOST}/v1"
MODEL_LIST_TIMEOUT_SECONDS = 30.0


class OllamaCloudProvider(OpenAIProvider):
    display_name = "Ollama Cloud"
    is_cloud = True

    def __init__(
        self,
        api_key: str,
        temperature: Optional[float] = DEFAULT_OLLAMA_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        host: str = OLLAMA_CLOUD_HOST,
    ):
        super().__init__(
            api_key=api_key,
            base_url=f"{host.rstrip('/')}/v1",
            temperature=temperature,
            timeout=timeout,
        )
        self.host = host.rstrip("/")

    def list_models(self) -> List[str]:
        """Cloud models from ``/api/tags`` (documented for ollama.com)."""
        import requests

        api_key = self._require_api_key()
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=MODEL_LIST_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise AITimeoutError(
                _("{provider} did not answer in time.").format(provider=self.display_name),
                type(exc).__name__,
            ) from exc
        except requests.RequestException as exc:
            raise AIConnectionError(self.connection_error_message(), type(exc).__name__) from exc

        if response.status_code in (401, 403):
            raise AIAuthError(
                _("{provider} rejected the API key. Please check it in the AI settings.").format(
                    provider=self.display_name
                ),
                f"HTTP {response.status_code}",
            )
        if response.status_code == 429:
            raise AIRateLimitError(
                _("{provider} rate limit or quota reached. Please try again later.").format(
                    provider=self.display_name
                ),
                "HTTP 429",
            )
        if response.status_code >= 400:
            raise AIResponseError(
                _("{provider} reported an error (HTTP {status}).").format(
                    provider=self.display_name, status=response.status_code
                ),
                f"HTTP {response.status_code}",
                retryable=response.status_code >= 500,
            )
        try:
            models = response.json().get("models", [])
            names = {str(m.get("name") or m.get("model")) for m in models if isinstance(m, dict)}
        except (ValueError, AttributeError) as exc:
            raise AIResponseError(
                _("Unexpected error from {provider}.").format(provider=self.display_name),
                type(exc).__name__,
            ) from exc
        names.discard("None")
        return sorted(names)

    def test_connection(self) -> List[str]:
        """Listing models may work without a valid key, so also check the
        key with an authenticated request to the OpenAI-compatible API."""
        models = self.list_models()
        try:
            self.client.models.list()
        except AIError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to AIError
            import openai

            # Some endpoints are not offered by the cloud API; only a
            # rejected key counts as a failed test here
            if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
                raise self.map_error(exc) from exc
            logger.debug("Ollama Cloud /v1/models not available: %s", type(exc).__name__)
        return models

    def connection_error_message(self) -> str:
        return _("Could not connect to {provider}. Please check your internet connection.").format(
            provider=self.display_name
        )
