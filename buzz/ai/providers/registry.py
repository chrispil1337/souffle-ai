# SPDX-License-Identifier: GPL-3.0-or-later
"""Provider selection and configuration.

Non-secret settings (provider, model, URLs) are stored with ``Settings``
(QSettings). API keys are stored only in the OS secret store (the macOS
Keychain on macOS) through ``buzz.store.keyring_store``.
"""

import enum
import logging
from dataclasses import dataclass
from typing import Optional

from buzz.ai.providers.anthropic_provider import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_EFFORT,
    AnthropicProvider,
)
from buzz.ai.providers.base import LLMProvider
from buzz.ai.providers.ollama_provider import (
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_OLLAMA_URL,
    OllamaProvider,
)
from buzz.ai.providers.openai_provider import DEFAULT_TIMEOUT_SECONDS, OpenAIProvider
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key, get_password, set_password, delete_password

logger = logging.getLogger(__name__)


class ProviderId(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

    @property
    def display_name(self) -> str:
        return {
            ProviderId.OPENAI: "OpenAI",
            ProviderId.ANTHROPIC: "Anthropic",
            ProviderId.OLLAMA: "Ollama",
        }[self]


DEFAULT_PROVIDER = ProviderId.OLLAMA  # local first: nothing leaves the computer

DEFAULT_MODELS = {
    ProviderId.OPENAI: "gpt-5",
    ProviderId.ANTHROPIC: DEFAULT_ANTHROPIC_MODEL,
    # Local models differ per machine; the user picks one from the list
    ProviderId.OLLAMA: "",
}

MODEL_SETTING_KEYS = {
    ProviderId.OPENAI: Settings.Key.AI_MODEL_OPENAI,
    ProviderId.ANTHROPIC: Settings.Key.AI_MODEL_ANTHROPIC,
    ProviderId.OLLAMA: Settings.Key.AI_MODEL_OLLAMA,
}

API_KEY_ENTRIES = {
    ProviderId.OPENAI: Key.OPENAI_API_KEY,
    ProviderId.ANTHROPIC: Key.ANTHROPIC_API_KEY,
    ProviderId.OLLAMA: Key.OLLAMA_API_KEY,
}


@dataclass
class ProviderConfig:
    provider_id: ProviderId
    model: str
    api_key: str = ""
    base_url: str = ""
    temperature: Optional[float] = None
    effort: str = ""
    timeout: float = DEFAULT_TIMEOUT_SECONDS


def get_api_key(provider_id: ProviderId) -> str:
    return get_password(API_KEY_ENTRIES[provider_id]) or ""


def set_api_key(provider_id: ProviderId, api_key: str) -> None:
    """Store (or, for an empty key, delete) the API key in the secret store."""
    entry = API_KEY_ENTRIES[provider_id]
    api_key = (api_key or "").strip()
    if api_key:
        set_password(entry, api_key)
    else:
        delete_password(entry)


def _parse_optional_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid AI temperature setting")
        return None


def get_selected_provider(settings: Settings) -> ProviderId:
    value = settings.value(Settings.Key.AI_PROVIDER, DEFAULT_PROVIDER.value)
    try:
        return ProviderId(value)
    except ValueError:
        return DEFAULT_PROVIDER


def get_model(settings: Settings, provider_id: ProviderId) -> str:
    return settings.value(MODEL_SETTING_KEYS[provider_id], DEFAULT_MODELS[provider_id])


def load_provider_config(
    settings: Settings, provider_id: Optional[ProviderId] = None
) -> ProviderConfig:
    """Read the configuration of a provider (default: the selected one).

    Reads the API key from the secret store. On Linux this can block briefly
    (D-Bus), so call it once before starting background work.
    """
    provider_id = provider_id or get_selected_provider(settings)
    temperature = _parse_optional_float(settings.value(Settings.Key.AI_TEMPERATURE, ""))
    timeout = _parse_optional_float(
        settings.value(Settings.Key.AI_REQUEST_TIMEOUT, "")
    ) or DEFAULT_TIMEOUT_SECONDS

    config = ProviderConfig(
        provider_id=provider_id,
        model=get_model(settings, provider_id),
        api_key=get_api_key(provider_id),
        temperature=temperature,
        timeout=timeout,
    )
    if provider_id == ProviderId.OPENAI:
        config.base_url = settings.value(Settings.Key.AI_OPENAI_BASE_URL, "")
    elif provider_id == ProviderId.OLLAMA:
        config.base_url = settings.value(Settings.Key.AI_OLLAMA_URL, DEFAULT_OLLAMA_URL)
        if config.temperature is None:
            config.temperature = DEFAULT_OLLAMA_TEMPERATURE
    elif provider_id == ProviderId.ANTHROPIC:
        config.effort = settings.value(Settings.Key.AI_ANTHROPIC_EFFORT, DEFAULT_EFFORT)
    return config


def create_provider(config: ProviderConfig) -> LLMProvider:
    if config.provider_id == ProviderId.OPENAI:
        return OpenAIProvider(
            api_key=config.api_key,
            base_url=config.base_url or None,
            temperature=config.temperature,
            timeout=config.timeout,
        )
    if config.provider_id == ProviderId.ANTHROPIC:
        return AnthropicProvider(
            api_key=config.api_key,
            effort=config.effort,
            timeout=config.timeout,
        )
    if config.provider_id == ProviderId.OLLAMA:
        return OllamaProvider(
            url=config.base_url or DEFAULT_OLLAMA_URL,
            api_key=config.api_key,
            temperature=config.temperature,
            timeout=config.timeout,
        )
    raise ValueError(f"Unknown provider: {config.provider_id}")
