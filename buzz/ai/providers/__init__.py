# SPDX-License-Identifier: GPL-3.0-or-later
import logging

# Buzz logs at DEBUG level to its log file. At that level the SDKs and HTTP
# libraries log whole requests, i.e. transcript text. Keep them quiet so no
# transcript content (or credentials) ends up in the logs.
SDK_LOGGERS = ("openai", "anthropic", "httpx", "httpx2", "httpcore", "httpcore2")


def quiet_sdk_loggers() -> None:
    for name in SDK_LOGGERS:
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET or logger.level < logging.WARNING:
            logger.setLevel(logging.WARNING)


quiet_sdk_loggers()

from buzz.ai.providers.base import (  # noqa: E402
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
from buzz.ai.providers.registry import (  # noqa: E402
    ProviderConfig,
    ProviderId,
    create_provider,
    get_api_key,
    load_provider_config,
    set_api_key,
)

__all__ = [
    "AIAuthError",
    "AIConfigurationError",
    "AIConnectionError",
    "AIError",
    "AIRateLimitError",
    "AIRefusalError",
    "AIResponseError",
    "AITimeoutError",
    "CompletionRequest",
    "LLMProvider",
    "ProviderConfig",
    "ProviderId",
    "create_provider",
    "get_api_key",
    "load_provider_config",
    "set_api_key",
]
