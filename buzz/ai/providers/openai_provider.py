# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenAI (and OpenAI-compatible) chat completions provider."""

import logging
import re
from typing import List, Optional

from buzz.ai.providers.base import (
    AIAuthError,
    AIConfigurationError,
    AIConnectionError,
    AIError,
    AIRateLimitError,
    AIResponseError,
    AITimeoutError,
    CompletionRequest,
    LLMProvider,
)
from buzz.locale import _

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 600.0

# Some reasoning models (e.g. DeepSeek) served through OpenAI-compatible APIs
# put their reasoning into the answer as <think>...</think>
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text or "").strip()


class OpenAIProvider(LLMProvider):
    display_name = "OpenAI"
    is_cloud = True

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key
        self.base_url = base_url or None
        # None means "use the model's default". Many current OpenAI models
        # only accept their default temperature, so it is not sent then.
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise AIConfigurationError(
                _("No API key for {provider}. Please enter it in the AI settings.").format(
                    provider=self.display_name
                )
            )
        return self.api_key

    @property
    def client(self):
        # Created on first use: importing openai takes about half a second.
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._require_api_key(),
                base_url=self.base_url,
                # Retries are handled uniformly by ``with_retry``
                max_retries=0,
                timeout=self.timeout,
            )
        return self._client

    def _complete(self, request: CompletionRequest) -> str:
        kwargs = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = self.client.chat.completions.create(**kwargs)
        except AIError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to AIError
            raise self.map_error(exc) from exc

        if not completion or not completion.choices:
            raise AIResponseError(_("The AI returned an empty response."))
        choice = completion.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise AIResponseError(
                _("The AI response was cut off because it was too long.")
            )
        return strip_reasoning(choice.message.content or "") if choice.message else ""

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            return sorted(model.id for model in models)
        except AIError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to AIError
            raise self.map_error(exc) from exc

    def connection_error_message(self) -> str:
        return _("Could not connect to {provider}. Please check your internet connection.").format(
            provider=self.display_name
        )

    def map_error(self, exc: Exception) -> AIError:
        """Translate an openai SDK exception into an ``AIError``."""
        import openai

        detail = f"{type(exc).__name__}: {getattr(exc, 'message', '') or exc}"
        logger.error("%s request failed: %s", self.display_name, type(exc).__name__)

        # APITimeoutError is a subclass of APIConnectionError, check it first
        if isinstance(exc, openai.APITimeoutError):
            return AITimeoutError(
                _("{provider} did not answer in time.").format(provider=self.display_name),
                detail,
            )
        if isinstance(exc, openai.APIConnectionError):
            return AIConnectionError(self.connection_error_message(), detail)
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return AIAuthError(
                _("{provider} rejected the API key. Please check it in the AI settings.").format(
                    provider=self.display_name
                ),
                detail,
            )
        if isinstance(exc, openai.RateLimitError):
            return AIRateLimitError(
                _("{provider} rate limit or quota reached. Please try again later.").format(
                    provider=self.display_name
                ),
                detail,
            )
        if isinstance(exc, openai.NotFoundError):
            return AIResponseError(
                _("The model was not found at {provider}. Please choose another model.").format(
                    provider=self.display_name
                ),
                detail,
            )
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", 0) or 0
            return AIResponseError(
                _("{provider} reported an error (HTTP {status}).").format(
                    provider=self.display_name, status=status
                ),
                detail,
                retryable=status >= 500,
            )
        return AIResponseError(
            _("Unexpected error from {provider}.").format(provider=self.display_name),
            detail,
        )
