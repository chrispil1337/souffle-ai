# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic AI post-processing after a transcription has been saved.

Off by default. When enabled in the AI settings, every newly completed
transcription is post-processed with the chosen profile. Runs are queued and
processed one at a time. A run is skipped (and logged) if no model is set, a
cloud provider has no API key, or the user has not agreed to send transcripts
to that provider yet; the automation never shows dialogs.
"""

import logging
from collections import deque
from typing import Callable, Deque, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from buzz.ai.controller import PostProcessingController, get_shared_controller
from buzz.ai.profiles import GENERAL_CLEANUP_ID
from buzz.ai.providers.registry import ProviderId, load_provider_config
from buzz.locale import _
from buzz.settings.settings import Settings
from buzz.widgets.ai.cloud_consent import has_consent

logger = logging.getLogger(__name__)


class AutoPostProcessor(QObject):
    # transcription id, title, version id
    run_started = pyqtSignal(str, str, int)
    run_finished = pyqtSignal(str, str, int)
    run_failed = pyqtSignal(str, str, str)  # transcription id, title, message
    # transcription id, reason shown to the user
    skipped = pyqtSignal(str, str)

    def __init__(
        self,
        transcription_service,
        settings: Optional[Settings] = None,
        controller_factory: Callable[..., PostProcessingController] = get_shared_controller,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.transcription_service = transcription_service
        self.settings = settings or Settings()
        self.controller_factory = controller_factory
        self._controller: Optional[PostProcessingController] = None
        self._queue: Deque[Tuple[str, str]] = deque()
        self._current: Optional[Tuple[str, str, int]] = None

    def is_enabled(self) -> bool:
        return bool(self.settings.value(Settings.Key.AI_AUTO_PROCESS, False))

    @property
    def controller(self) -> PostProcessingController:
        if self._controller is None:
            self._controller = self.controller_factory(self.transcription_service)
            self._controller.finished.connect(self._on_controller_finished)
            self._controller.failed.connect(self._on_controller_failed)
        return self._controller

    def pending_count(self) -> int:
        return len(self._queue) + (1 if self._current else 0)

    def skip_reason(self) -> Optional[str]:
        config = load_provider_config(self.settings)
        if not config.model:
            return _("No AI model selected. Please choose a model in the AI settings.")
        if config.provider_id != ProviderId.OLLAMA and not config.api_key:
            return _("No API key for {provider}. Please enter it in the AI settings.").format(
                provider=config.provider_id.display_name
            )
        if not has_consent(self.settings, config):
            return _(
                "Automatic post-processing needs your consent to send transcripts to "
                "{provider}. Run the AI post-processing once by hand to give it."
            ).format(provider=config.provider_id.display_name)
        return None

    def on_transcription_saved(self, transcription_id, title: str = "") -> bool:
        """Queue a newly saved transcription. Returns True if it was queued."""
        if not self.is_enabled():
            return False
        transcription_id = str(transcription_id)
        reason = self.skip_reason()
        if reason:
            logger.info("Automatic AI post-processing skipped: %s", reason)
            self.skipped.emit(transcription_id, reason)
            return False
        self._queue.append((transcription_id, title))
        self._start_next()
        return True

    def _start_next(self) -> None:
        while self._current is None and self._queue:
            transcription_id, title = self._queue.popleft()
            profile_id = self.settings.value(Settings.Key.AI_AUTO_PROFILE, GENERAL_CLEANUP_ID)
            try:
                version_id = self.controller.start(transcription_id, profile_id, title=title)
            except Exception as exc:  # noqa: BLE001 - logged, continue with the next
                message = getattr(exc, "user_message", "") or str(exc)
                logger.warning("Automatic AI post-processing could not start: %s", message)
                self.run_failed.emit(transcription_id, title, message)
                continue
            self._current = (transcription_id, title, version_id)
            self.run_started.emit(transcription_id, title, version_id)

    def _on_controller_finished(self, version_id: int) -> None:
        if self._current and self._current[2] == version_id:
            transcription_id, title, _version = self._current
            self._current = None
            self.run_finished.emit(transcription_id, title, version_id)
            self._start_next()

    def _on_controller_failed(self, version_id: int, message: str) -> None:
        if self._current and self._current[2] == version_id:
            transcription_id, title, _version = self._current
            self._current = None
            self.run_failed.emit(transcription_id, title, message)
            self._start_next()
