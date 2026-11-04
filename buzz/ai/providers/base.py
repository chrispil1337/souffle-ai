# SPDX-License-Identifier: GPL-3.0-or-later
"""Provider-neutral interface to large language models.

Every provider turns its SDK's exceptions into the ``AIError`` hierarchy below,
so the rest of the app (pipeline, UI) never has to know which SDK is in use.
Each error carries a ``user_message`` that can be shown in the UI as is.

Privacy: nothing in this module logs prompts, transcript text or API keys.
"""

import abc
import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, TypeVar

from buzz.locale import _

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 30.0


class AIError(Exception):
    """Base class of all AI errors. ``retryable`` errors are retried."""

    retryable = False

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail


class AIConfigurationError(AIError):
    """Missing or invalid local configuration (no API key, no model, ...)."""


class AIAuthError(AIError):
    """The provider rejected the credentials."""


class AIRateLimitError(AIError):
    retryable = True


class AIConnectionError(AIError):
    retryable = True


class AITimeoutError(AIError):
    retryable = True


class AIRefusalError(AIError):
    """The model declined to answer (safety classifier)."""


class AIResponseError(AIError):
    """The provider answered with an error or an unusable response."""

    def __init__(self, user_message: str, detail: str = "", retryable: bool = False):
        super().__init__(user_message, detail)
        self.retryable = retryable


@dataclass(frozen=True)
class CompletionRequest:
    system: str
    user: str
    model: str
    # Ask the provider for a JSON object, where it supports a JSON mode. The
    # prompt itself must still ask for JSON; callers parse defensively.
    json_mode: bool = False


def with_retry(
    fn: Callable[[], T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Optional[Callable[[float], None]] = None,
    is_canceled: Optional[Callable[[], bool]] = None,
) -> T:
    """Call ``fn``, retrying retryable ``AIError``s with exponential backoff."""
    sleep = sleep or time.sleep
    attempt = 1
    while True:
        try:
            return fn()
        except AIError as exc:
            canceled = is_canceled is not None and is_canceled()
            if not exc.retryable or attempt >= max_attempts or canceled:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), MAX_DELAY_SECONDS)
            delay += random.uniform(0, delay / 4)
            logger.warning(
                "AI request failed (%s), retry %d/%d in %.1fs",
                type(exc).__name__,
                attempt,
                max_attempts - 1,
                delay,
            )
            sleep(delay)
            attempt += 1


class LLMProvider(abc.ABC):
    """A configured connection to one LLM provider."""

    #: Human readable provider name, used in messages.
    display_name: str = ""
    #: True if requests leave this computer (needs user consent).
    is_cloud: bool = True

    def complete(
        self,
        request: CompletionRequest,
        is_canceled: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Run one completion with retries. Returns the response text."""
        if not request.model:
            raise AIConfigurationError(
                _("No AI model selected. Please choose a model in the AI settings.")
            )
        text = with_retry(lambda: self._complete(request), is_canceled=is_canceled)
        if not text or not text.strip():
            raise AIResponseError(_("The AI returned an empty response."))
        return text

    def test_connection(self) -> List[str]:
        """Check that the provider is reachable and the credentials work.

        Returns the available model IDs. Raises an ``AIError`` on failure.
        Listing models costs nothing, unlike a test completion.
        """
        return self.list_models()

    @abc.abstractmethod
    def _complete(self, request: CompletionRequest) -> str:
        """Run one completion without retries."""

    @abc.abstractmethod
    def list_models(self) -> List[str]:
        """Return the model IDs available to these credentials, sorted."""
