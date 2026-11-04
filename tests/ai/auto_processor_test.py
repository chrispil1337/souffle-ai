# SPDX-License-Identifier: GPL-3.0-or-later
import uuid
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QThreadPool

from buzz.ai import auto_processor as auto_module
from buzz.ai import controller as controller_module
from buzz.ai.auto_processor import AutoPostProcessor
from buzz.ai.controller import PostProcessingController
from buzz.ai.profiles import SITE_INSPECTION_ID, ProfileStore
from buzz.ai.providers.registry import ProviderConfig, ProviderId
from buzz.db.entity.transcription import Transcription
from buzz.settings.settings import Settings
from tests.ai.conftest import STORED_SEGMENTS
from tests.ai.fakes import FakeProvider

LOCAL = ProviderConfig(ProviderId.OLLAMA, "fake-model", base_url="http://localhost:11434")


@pytest.fixture
def auto_settings(settings):
    settings.set_value(Settings.Key.AI_AUTO_PROCESS, True)
    settings.set_value(Settings.Key.AI_AUTO_PROFILE, SITE_INSPECTION_ID)
    settings.set_value(Settings.Key.AI_CLOUD_CONSENT, "")
    yield settings
    settings.set_value(Settings.Key.AI_AUTO_PROCESS, False)


@pytest.fixture
def make_processor(transcription_service, ai_service, tmp_path):
    def make(settings, config=LOCAL):
        controller = PostProcessingController(
            transcription_service, ai_service=ai_service, settings=settings,
            profile_store=ProfileStore(str(tmp_path / "p.json")), thread_pool=QThreadPool(),
        )
        processor = AutoPostProcessor(
            transcription_service, settings, controller_factory=lambda service: controller
        )
        return processor

    with (
        patch.object(controller_module, "load_provider_config", return_value=LOCAL),
        patch.object(controller_module, "create_provider", return_value=FakeProvider()),
    ):
        yield make


def add_transcription(transcription_dao, transcription_service, name):
    transcription_id = uuid.uuid4()
    transcription_dao.insert(Transcription(id=str(transcription_id), status="in_progress", name=name))
    transcription_service.update_transcription_as_completed(transcription_id, STORED_SEGMENTS)
    return transcription_id


class TestAutoPostProcessor:
    def test_disabled_by_default(self, settings, make_processor, stored_transcription):
        settings.set_value(Settings.Key.AI_AUTO_PROCESS, False)
        processor = make_processor(settings)
        assert processor.on_transcription_saved(stored_transcription) is False
        assert processor.pending_count() == 0

    def test_runs_with_configured_profile(
        self, qtbot, auto_settings, make_processor, stored_transcription, ai_service
    ):
        processor = make_processor(auto_settings)
        with patch.object(auto_module, "load_provider_config", return_value=LOCAL):
            with qtbot.waitSignal(processor.run_finished, timeout=10000) as finished:
                assert processor.on_transcription_saved(stored_transcription, "Baustelle")
        transcription_id, title, version_id = finished.args
        assert (transcription_id, title) == (str(stored_transcription), "Baustelle")
        version = ai_service.get_version(version_id)
        assert version.status == "completed"
        assert version.profile_id == SITE_INSPECTION_ID

    def test_runs_one_after_another(
        self, qtbot, auto_settings, make_processor, stored_transcription,
        transcription_dao, transcription_service, ai_service
    ):
        second = add_transcription(transcription_dao, transcription_service, "Zweite")
        processor = make_processor(auto_settings)
        started = []
        processor.run_started.connect(lambda tid, title, vid: started.append(tid))
        finished = []
        processor.run_finished.connect(lambda tid, title, vid: finished.append(tid))

        with patch.object(auto_module, "load_provider_config", return_value=LOCAL):
            processor.on_transcription_saved(stored_transcription)
            processor.on_transcription_saved(second)
            assert started == [str(stored_transcription)]
            assert processor.pending_count() == 2
            qtbot.waitUntil(lambda: len(finished) == 2, timeout=10000)

        assert started == [str(stored_transcription), str(second)]
        assert processor.pending_count() == 0
        assert ai_service.latest_completed_version(second) is not None

    def test_skipped_without_model(self, qtbot, auto_settings, make_processor, stored_transcription):
        processor = make_processor(auto_settings)
        no_model = ProviderConfig(ProviderId.OLLAMA, "")
        with patch.object(auto_module, "load_provider_config", return_value=no_model):
            with qtbot.waitSignal(processor.skipped) as skipped:
                assert processor.on_transcription_saved(stored_transcription) is False
        assert "model" in skipped.args[1]

    def test_skipped_without_api_key(self, qtbot, auto_settings, make_processor, stored_transcription):
        processor = make_processor(auto_settings)
        no_key = ProviderConfig(ProviderId.ANTHROPIC, "claude-opus-5-5", api_key="")
        with patch.object(auto_module, "load_provider_config", return_value=no_key):
            with qtbot.waitSignal(processor.skipped) as skipped:
                processor.on_transcription_saved(stored_transcription)
        assert "API key" in skipped.args[1]

    def test_cloud_provider_needs_consent_given_before(
        self, qtbot, auto_settings, make_processor, stored_transcription
    ):
        processor = make_processor(auto_settings)
        cloud = ProviderConfig(ProviderId.ANTHROPIC, "claude-opus-5-5", api_key="k")
        with patch.object(auto_module, "load_provider_config", return_value=cloud):
            with qtbot.waitSignal(processor.skipped) as skipped:
                assert processor.on_transcription_saved(stored_transcription) is False
            assert "consent" in skipped.args[1]

            auto_settings.set_value(Settings.Key.AI_CLOUD_CONSENT, "anthropic")
            assert processor.skip_reason() is None
