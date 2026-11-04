# SPDX-License-Identifier: GPL-3.0-or-later
"""Starts AI post-processing runs and stores their results.

Used by the AI window (manual runs) and by the automatic post-processing
after a transcription. Everything in here runs on the UI thread except the
pipeline itself, which runs as a background job. Database access therefore
always happens on the UI thread, as the Qt SQL connection requires.
"""

import logging
from typing import Dict, Optional
from uuid import UUID

from PyQt6.QtCore import QObject, QThreadPool, pyqtSignal, pyqtSlot

from buzz.ai.jobs import CallableJob, error_message, start_job
from buzz.ai.pipeline import PipelineCanceled, PipelineResult, PostProcessingPipeline, numbered_segments_from_records
from buzz.ai.profiles import ProfileStore
from buzz.ai.providers.base import AIConfigurationError
from buzz.ai.providers.registry import create_provider, load_provider_config
from buzz.db.entity.ai_version import AIVersion, AIVersionSegment, AIVersionStatus
from buzz.db.service.ai_version_service import AIVersionService
from buzz.locale import _
from buzz.settings.settings import Settings

logger = logging.getLogger(__name__)


class _JobRelay(QObject):
    """Lives on the UI thread and forwards one job's signals with its id."""

    def __init__(self, controller: "PostProcessingController", version_id: int):
        super().__init__(controller)
        self.controller = controller
        self.version_id = version_id

    @pyqtSlot(object)
    def on_succeeded(self, result):
        self.controller._on_job_succeeded(self.version_id, result)

    @pyqtSlot(object)
    def on_failed(self, exc):
        self.controller._on_job_failed(self.version_id, exc)

    @pyqtSlot(float, str)
    def on_progress(self, fraction, message):
        self.controller.progress.emit(self.version_id, fraction, message)


class PostProcessingController(QObject):
    started = pyqtSignal(int)  # version id
    progress = pyqtSignal(int, float, str)  # version id, fraction, message
    finished = pyqtSignal(int)  # version id
    failed = pyqtSignal(int, str)  # version id, user message

    def __init__(
        self,
        transcription_service,
        ai_service: Optional[AIVersionService] = None,
        settings: Optional[Settings] = None,
        profile_store: Optional[ProfileStore] = None,
        thread_pool: Optional[QThreadPool] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.transcription_service = transcription_service
        self.ai_service = ai_service or AIVersionService.from_transcription_service(
            transcription_service
        )
        self.settings = settings or Settings()
        self.profile_store = profile_store or ProfileStore()
        # None: every run gets its own daemon thread (see ``start_job``)
        self.thread_pool = thread_pool
        self._jobs: Dict[int, CallableJob] = {}
        self._relays: Dict[int, _JobRelay] = {}

    def is_running(self, version_id: Optional[int] = None) -> bool:
        if version_id is None:
            return bool(self._jobs)
        return version_id in self._jobs

    def running_version_for(self, transcription_id) -> Optional[int]:
        for version_id in self._jobs:
            version = self.ai_service.get_version(version_id)
            if version is not None and version.transcription_id == str(transcription_id):
                return version_id
        return None

    def start(self, transcription_id, profile_id: str, title: str = "") -> int:
        """Start a run and return the id of the new version.

        Raises ``AIConfigurationError`` right away if the provider is not
        configured, so the caller can tell the user what to fix.
        """
        transcription_uuid = (
            transcription_id if isinstance(transcription_id, UUID) else UUID(str(transcription_id))
        )
        profile = self.profile_store.get_or_default(profile_id)
        config = load_provider_config(self.settings)
        if not config.model:
            raise AIConfigurationError(
                _("No AI model selected. Please choose a model in the AI settings.")
            )
        provider = create_provider(config)

        records = self.transcription_service.get_transcription_segments(
            transcription_id=transcription_uuid
        )
        segments = numbered_segments_from_records(records)

        version_id = self.ai_service.create_version(
            AIVersion(
                transcription_id=str(transcription_uuid),
                profile_id=profile.id,
                profile_name=profile.name,
                provider=config.provider_id.value,
                model=config.model,
                status=AIVersionStatus.RUNNING.value,
            )
        )

        def run(job: CallableJob) -> PipelineResult:
            pipeline = PostProcessingPipeline(
                provider=provider,
                model=config.model,
                profile=profile,
                progress=job.report_progress,
                is_canceled=job.cancel_token.is_canceled,
            )
            return pipeline.run(segments, title=title)

        relay = _JobRelay(self, version_id)
        self._relays[version_id] = relay
        self._jobs[version_id] = start_job(
            run,
            on_success=relay.on_succeeded,
            on_failure=relay.on_failed,
            on_progress=relay.on_progress,
            thread_pool=self.thread_pool,
        )
        logger.info(
            "AI post-processing started (version %s, provider %s)",
            version_id,
            config.provider_id.value,
        )
        self.started.emit(version_id)
        return version_id

    def cancel(self, version_id: int) -> None:
        job = self._jobs.get(version_id)
        if job is not None:
            job.cancel()

    def cancel_all(self) -> None:
        for job in self._jobs.values():
            job.cancel()

    def _cleanup(self, version_id: int) -> None:
        self._jobs.pop(version_id, None)
        relay = self._relays.pop(version_id, None)
        if relay is not None:
            relay.deleteLater()

    def _on_job_succeeded(self, version_id: int, result: PipelineResult) -> None:
        self._cleanup(version_id)
        try:
            self.ai_service.save_result(
                version_id,
                [
                    AIVersionSegment(
                        ai_version_id=version_id,
                        original_text=s.original_text,
                        text=s.text,
                        segment_id=s.segment_id,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        speaker=s.speaker,
                        flag=s.flag,
                    )
                    for s in result.segments
                ],
                result.document_markdown,
                result.analysis,
                result.prompt_snapshot,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            logger.error("Saving AI result failed: %s", exc)
            message = _("The AI result could not be saved.")
            self.ai_service.set_status(version_id, AIVersionStatus.FAILED, message)
            self.failed.emit(version_id, message)
            return
        logger.info("AI post-processing finished (version %s)", version_id)
        self.finished.emit(version_id)

    def _on_job_failed(self, version_id: int, exc: Exception) -> None:
        self._cleanup(version_id)
        message = error_message(exc)
        status = (
            AIVersionStatus.CANCELED if isinstance(exc, PipelineCanceled) else AIVersionStatus.FAILED
        )
        try:
            self.ai_service.set_status(version_id, status, message)
        except Exception as db_exc:  # noqa: BLE001
            logger.error("Saving AI error status failed: %s", db_exc)
        self.failed.emit(version_id, message)


_shared_controller: Optional[PostProcessingController] = None


def get_shared_controller(transcription_service) -> PostProcessingController:
    """The app-wide controller, shared by all windows and the automation.

    Runs keep going when the AI window is closed. On first use, runs left over
    from a previous session (the app was closed while they ran) are marked as
    failed, as they can not resume.
    """
    global _shared_controller
    if _shared_controller is None:
        from PyQt6.QtCore import QCoreApplication

        ai_service = AIVersionService.from_transcription_service(transcription_service)
        ai_service.mark_interrupted_as_failed(_("Interrupted because the app was closed."))
        _shared_controller = PostProcessingController(
            transcription_service,
            ai_service=ai_service,
            parent=QCoreApplication.instance(),
        )
    return _shared_controller


def reset_shared_controller() -> None:
    """For tests."""
    global _shared_controller
    if _shared_controller is not None:
        _shared_controller.cancel_all()
    _shared_controller = None
