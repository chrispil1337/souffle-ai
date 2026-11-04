# SPDX-License-Identifier: GPL-3.0-or-later
"""Ask before transcripts are sent to a cloud provider for the first time."""

from typing import Optional

from PyQt6.QtWidgets import QMessageBox, QWidget

from buzz.ai.providers.registry import ProviderConfig, ProviderId, create_provider
from buzz.locale import _
from buzz.settings.settings import Settings


def _consented(settings: Settings) -> set:
    value = settings.value(Settings.Key.AI_CLOUD_CONSENT, "")
    return {v for v in str(value or "").split(",") if v}


def requires_consent(config: ProviderConfig) -> bool:
    return create_provider(config).is_cloud


def has_consent(settings: Settings, config: ProviderConfig) -> bool:
    return not requires_consent(config) or config.provider_id.value in _consented(settings)


def store_consent(settings: Settings, provider_id: ProviderId) -> None:
    consented = _consented(settings) | {provider_id.value}
    settings.set_value(Settings.Key.AI_CLOUD_CONSENT, ",".join(sorted(consented)))


def revoke_consent(settings: Settings, provider_id: ProviderId) -> None:
    consented = _consented(settings) - {provider_id.value}
    settings.set_value(Settings.Key.AI_CLOUD_CONSENT, ",".join(sorted(consented)))


def ask_consent(
    parent: Optional[QWidget], settings: Settings, config: ProviderConfig
) -> bool:
    """Return True if sending is allowed (asking the user once per provider)."""
    if has_consent(settings, config):
        return True
    provider = config.provider_id.display_name
    answer = QMessageBox.question(
        parent,
        _("Send transcript to {provider}?").format(provider=provider),
        _(
            "For AI post-processing, the transcript text is sent to {provider} and "
            "processed on their servers. The audio recording is not sent.\n\n"
            "Only continue if this is allowed for the content of the conversation "
            "(e.g. personal data, confidential customer information).\n\n"
            "For processing that stays on this computer, choose Ollama in the AI "
            "settings.\n\nAllow sending transcripts to {provider}?"
        ).format(provider=provider),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if answer == QMessageBox.StandardButton.Yes:
        store_consent(settings, config.provider_id)
        return True
    return False
