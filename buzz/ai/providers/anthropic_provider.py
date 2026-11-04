# SPDX-License-Identifier: GPL-3.0-or-later
"""Anthropic (Claude) provider using the official ``anthropic`` SDK (2.x).

Notes on the current Claude API that shape this code:

- Sampling parameters (``temperature`` etc.) are no longer accepted, so none
  are sent. Answer quality is steered with ``output_config.effort``.
- Responses can be long (structured documents), so requests are streamed and
  collected with ``get_final_message()``, which avoids HTTP timeouts.
- For the newest models the server-side refusal fallback is enabled: if a
  safety classifier declines, the API re-runs the request on a suitable model.
"""

import logging
from typing import List, Optional

from buzz.ai.providers.base import (
    AIAuthError,
    AIConfigurationError,
    AIConnectionError,
    AIError,
    AIRateLimitError,
    AIRefusalError,
    AIResponseError,
    AITimeoutError,
    CompletionRequest,
    LLMProvider,
)
from buzz.locale import _

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
MAX_OUTPUT_TOKENS = 64000

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept ``fallbacks: "default"`` on the Claude API
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)


class AnthropicProvider(LLMProvider):
    display_name = "Anthropic"
    is_cloud = True

    def __init__(
        self,
        api_key: str,
        effort: Optional[str] = DEFAULT_EFFORT,
        timeout: float = 600.0,
    ):
        self.api_key = api_key
        # Empty effort means "do not send", for older models without effort
        self.effort = effort if effort in EFFORT_LEVELS else None
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                # Never fall back to credentials from the environment or a
                # CLI profile: the user chooses the key in the settings.
                raise AIConfigurationError(
                    _("No API key for {provider}. Please enter it in the AI settings.").format(
                        provider=self.display_name
                    )
                )
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                max_retries=0,
                timeout=self.timeout,
            )
        return self._client

    def _complete(self, request: CompletionRequest) -> str:
        kwargs = {
            "model": request.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        try:
            if request.model in FALLBACK_MODELS:
                stream = self.client.beta.messages.stream(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                stream = self.client.messages.stream(**kwargs)
            with stream as message_stream:
                message = message_stream.get_final_message()
        except AIError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to AIError
            raise self.map_error(exc) from exc

        if message.stop_reason == "refusal":
            raise AIRefusalError(
                _("The AI declined to process this transcript."),
                str(getattr(message, "stop_details", "") or ""),
            )
        if message.stop_reason == "max_tokens":
            raise AIResponseError(
                _("The AI response was cut off because it was too long.")
            )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list(lifecycle=["active"])
            return sorted(model.id for model in models)
        except AIError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to AIError
            raise self.map_error(exc) from exc

    def map_error(self, exc: Exception) -> AIError:
        """Translate an anthropic SDK exception into an ``AIError``."""
        import anthropic

        detail = f"{type(exc).__name__}: {getattr(exc, 'message', '') or exc}"
        logger.error("Anthropic request failed: %s", type(exc).__name__)

        # APITimeoutError is a subclass of APIConnectionError, check it first
        if isinstance(exc, anthropic.APITimeoutError):
            return AITimeoutError(_("Anthropic did not answer in time."), detail)
        if isinstance(exc, anthropic.APIConnectionError):
            return AIConnectionError(
                _("Could not connect to {provider}. Please check your internet connection.").format(
                    provider=self.display_name
                ),
                detail,
            )
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return AIAuthError(
                _("{provider} rejected the API key. Please check it in the AI settings.").format(
                    provider=self.display_name
                ),
                detail,
            )
        if isinstance(exc, anthropic.RateLimitError):
            return AIRateLimitError(
                _("{provider} rate limit or quota reached. Please try again later.").format(
                    provider=self.display_name
                ),
                detail,
            )
        if isinstance(exc, anthropic.NotFoundError):
            return AIResponseError(
                _("The model was not found at {provider}. Please choose another model.").format(
                    provider=self.display_name
                ),
                detail,
            )
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", 0) or 0
            return AIResponseError(
                _("{provider} reported an error (HTTP {status}).").format(
                    provider=self.display_name, status=status
                ),
                detail,
                # 529 = overloaded, 5xx = server errors
                retryable=status >= 500,
            )
        return AIResponseError(
            _("Unexpected error from {provider}.").format(provider=self.display_name),
            detail,
        )
