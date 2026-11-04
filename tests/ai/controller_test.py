# SPDX-License-Identifier: GPL-3.0-or-later
import threading
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QThreadPool

from buzz.ai import controller as controller_module
from buzz.ai.controller import PostProcessingController
from buzz.ai.profiles import MEETING_MINUTES_ID, ProfileStore
from buzz.ai.providers.base import AIAuthError, AIConfigurationError
from buzz.ai.providers.registry import ProviderConfig, ProviderId
from buzz.db.entity.ai_version import AIVersionStatus
from tests.ai.fakes import FakeProvider


@pytest.fixture
def provider_config():
    config = ProviderConfig(provider_id=ProviderId.OLLAMA, model="fake-model")
    with patch.object(controller_module, "load_provider_config", return_value=config):
        yield config


def make_controller(qtbot, transcription_service, ai_service, tmp_path, provider):
    controller = PostProcessingController(
        transcription_service,
        ai_service=ai_service,
        profile_store=ProfileStore(str(tmp_path / "profiles.json")),
        thread_pool=QThreadPool(),
    )
    patcher = patch.object(controller_module, "create_provider", return_value=provider)
    patcher.start()
    return controller, patcher


class TestPostProcessingController:
    def test_successful_run_is_stored(
        self, qtbot, transcription_service, ai_service, stored_transcription, provider_config, tmp_path
    ):
        controller, patcher = make_controller(
            qtbot, transcription_service, ai_service, tmp_path, FakeProvider()
        )
        try:
            progress = []
            controller.progress.connect(lambda vid, f, m: progress.append(f))
            with qtbot.waitSignal(controller.finished, timeout=10000) as finished:
                version_id = controller.start(stored_transcription, MEETING_MINUTES_ID, title="Baustelle")
                assert controller.is_running(version_id)
                assert ai_service.get_version(version_id).status == "running"
            assert finished.args == [version_id]
        finally:
            patcher.stop()

        version = ai_service.get_version(version_id)
        assert version.status == AIVersionStatus.COMPLETED.value
        assert version.provider == "ollama"
        assert version.model == "fake-model"
        assert version.profile_id == MEETING_MINUTES_ID
        assert "## Tasks" in version.document_markdown
        segments = ai_service.get_segments(version_id)
        assert [s.text for s in segments] == [
            "Das Dach ist undicht.",
            "Wir sollten den Dachdecker anrufen.",
        ]
        assert all(s.segment_id > 0 for s in segments)
        assert progress and progress[-1] == pytest.approx(1.0)
        assert not controller.is_running()

    def test_provider_error_marks_version_failed(
        self, qtbot, transcription_service, ai_service, stored_transcription, provider_config, tmp_path
    ):
        class FailingProvider(FakeProvider):
            def _complete(self, request):
                raise AIAuthError("API-Key abgelehnt.")

        controller, patcher = make_controller(
            qtbot, transcription_service, ai_service, tmp_path, FailingProvider()
        )
        try:
            with qtbot.waitSignal(controller.failed, timeout=10000) as failed:
                version_id = controller.start(stored_transcription, MEETING_MINUTES_ID)
        finally:
            patcher.stop()

        assert failed.args == [version_id, "API-Key abgelehnt."]
        version = ai_service.get_version(version_id)
        assert (version.status, version.error_message) == ("failed", "API-Key abgelehnt.")

    def test_cancel(
        self, qtbot, transcription_service, ai_service, stored_transcription, provider_config, tmp_path
    ):
        block = threading.Event()
        controller, patcher = make_controller(
            qtbot, transcription_service, ai_service, tmp_path, FakeProvider(block=block)
        )
        try:
            with qtbot.waitSignal(controller.failed, timeout=10000):
                version_id = controller.start(stored_transcription, MEETING_MINUTES_ID)
                controller.cancel(version_id)
                block.set()
        finally:
            patcher.stop()

        assert ai_service.get_version(version_id).status == AIVersionStatus.CANCELED.value

    def test_missing_model_fails_before_creating_a_version(
        self, transcription_service, ai_service, stored_transcription, tmp_path
    ):
        controller = PostProcessingController(
            transcription_service,
            ai_service=ai_service,
            profile_store=ProfileStore(str(tmp_path / "profiles.json")),
        )
        config = ProviderConfig(provider_id=ProviderId.OLLAMA, model="")
        with patch.object(controller_module, "load_provider_config", return_value=config):
            with pytest.raises(AIConfigurationError):
                controller.start(stored_transcription, MEETING_MINUTES_ID)
        assert ai_service.list_versions(stored_transcription) == []
