# SPDX-License-Identifier: GPL-3.0-or-later
import json
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QMessageBox

from buzz.ai import controller as controller_module
from buzz.ai.controller import PostProcessingController
from buzz.ai.profiles import SITE_INSPECTION_ID, ProfileStore
from buzz.ai.providers.registry import ProviderConfig, ProviderId
from buzz.export.transcript_export import ExportKind
from buzz.settings.settings import Settings
from buzz.widgets.ai import ai_postprocessing_window as window_module
from buzz.widgets.ai import cloud_consent
from buzz.widgets.ai.ai_postprocessing_window import (
    COLUMN_AI,
    AIPostProcessingWindow,
)
from tests.ai.fakes import FakeProvider


@pytest.fixture
def ollama_config():
    config = ProviderConfig(provider_id=ProviderId.OLLAMA, model="fake-model",
                            base_url="http://localhost:11434")
    with (
        patch.object(controller_module, "load_provider_config", return_value=config),
        patch.object(window_module, "load_provider_config", return_value=config),
        patch.object(controller_module, "create_provider", return_value=FakeProvider()),
    ):
        yield config


@pytest.fixture
def window(qtbot, transcription_service, transcription_dao, ai_service, stored_transcription,
           settings, tmp_path, ollama_config):
    profiles = ProfileStore(str(tmp_path / "profiles.json"))
    controller = PostProcessingController(
        transcription_service, ai_service=ai_service, settings=settings,
        profile_store=profiles, thread_pool=QThreadPool(),
    )
    transcription = transcription_dao.find_by_id(str(stored_transcription))
    widget = AIPostProcessingWindow(
        transcription, transcription_service, controller=controller,
        settings=settings, profile_store=profiles,
    )
    qtbot.add_widget(widget)
    return widget


def run_post_processing(qtbot, window, profile_id=None):
    if profile_id:
        window.profile_combo_box.setCurrentIndex(window.profile_combo_box.findData(profile_id))
    with qtbot.waitSignal(window.controller.finished, timeout=10000):
        window.start_button.click()
    qtbot.waitUntil(lambda: window.start_button.isEnabled(), timeout=5000)


class TestAIPostProcessingWindow:
    def test_initial_state(self, window):
        assert window.version_combo_box.count() == 1
        assert window.version_combo_box.currentData() is None
        assert "original transcript stays unchanged" in window.message_label.text()
        assert "Anna: Äh also das Dach ist halt undicht." in window.original_text.toPlainText()
        assert window.provider_label.text() == "Ollama · fake-model"
        assert window.profile_combo_box.count() == 5
        assert not window.cancel_button.isEnabled()

    def test_run_shows_side_by_side_result(self, qtbot, window, settings):
        run_post_processing(qtbot, window, SITE_INSPECTION_ID)

        assert settings.value(Settings.Key.AI_LAST_PROFILE, "") == SITE_INSPECTION_ID
        assert window.current_version.status == "completed"
        assert window.current_version.profile_id == SITE_INSPECTION_ID
        table = window.segments_table
        assert table.rowCount() == 2
        assert table.item(0, 2).text() == "Äh also das Dach ist halt undicht."
        assert table.item(0, COLUMN_AI).text() == "Das Dach ist undicht."
        assert table.item(1, 1).text() == "Bernd"
        assert "## Tasks" in window.document_editor.toPlainText()
        assert "line-through" in window.diff_browser.toHtml()
        # the profile was selected and the version list shows the run
        assert "✓" in window.version_combo_box.currentText()
        assert not window.progress_bar.isVisible()

    def test_editing_a_segment_is_saved(self, qtbot, window, ai_service):
        run_post_processing(qtbot, window)
        window.segments_table.item(0, COLUMN_AI).setText("Das Dach ist sehr undicht.")

        version_id = window.current_version.id
        assert ai_service.get_segments(version_id)[0].text == "Das Dach ist sehr undicht."
        assert ai_service.get_version(version_id).manually_edited

    def test_original_column_is_read_only(self, qtbot, window):
        from PyQt6.QtCore import Qt

        run_post_processing(qtbot, window)
        assert not (window.segments_table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable)
        assert window.segments_table.item(0, COLUMN_AI).flags() & Qt.ItemFlag.ItemIsEditable

    def test_editing_the_document_is_saved(self, qtbot, window, ai_service):
        run_post_processing(qtbot, window)
        window.document_editor.setPlainText("# Von Hand\n\nNeu.")
        window.flush_document()

        version = ai_service.get_version(window.current_version.id)
        assert version.document_markdown == "# Von Hand\n\nNeu."
        assert version.manually_edited

    def test_preview_renders_markdown(self, qtbot, window):
        run_post_processing(qtbot, window)
        window.preview_button.setChecked(True)
        assert window.document_stack.currentIndex() == 1
        assert "Tasks" in window.document_preview.toPlainText()

    def test_versions_are_kept_and_selectable(self, qtbot, window):
        run_post_processing(qtbot, window)
        first = window.current_version.id
        run_post_processing(qtbot, window)
        assert window.version_combo_box.count() == 2
        assert window.current_version.id != first
        window.version_combo_box.setCurrentIndex(window.version_combo_box.findData(first))
        assert window.current_version.id == first

    def test_export(self, qtbot, window, tmp_path):
        run_post_processing(qtbot, window)
        path = tmp_path / "export.json"
        window.export(ExportKind.COMBINED_JSON, str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["original"]["segments"][0]["text"] == "Äh also das Dach ist halt undicht."
        assert data["ai_version"]["segments"][0]["text"] == "Das Dach ist undicht."

    def test_delete_version(self, qtbot, window, ai_service):
        run_post_processing(qtbot, window)
        version_id = window.current_version.id
        with patch.object(window_module.QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.Yes):
            window.delete_button.click()
        assert ai_service.get_version(version_id) is None
        assert window.version_combo_box.currentData() is None

    def test_failure_is_shown(self, qtbot, window):
        from buzz.ai.providers.base import AIConnectionError

        class Offline(FakeProvider):
            def _complete(self, request):
                raise AIConnectionError("Ollama läuft nicht.")

        with patch.object(controller_module, "create_provider", return_value=Offline()):
            with qtbot.waitSignal(window.controller.failed, timeout=10000):
                window.start_button.click()
        qtbot.waitUntil(lambda: window.start_button.isEnabled(), timeout=5000)
        assert "Ollama läuft nicht." in window.message_label.text()
        assert "failed" in window.version_combo_box.currentText()

    def test_missing_model_shows_hint_and_does_not_start(self, qtbot, window):
        config = ProviderConfig(provider_id=ProviderId.OLLAMA, model="")
        with (
            patch.object(window_module, "load_provider_config", return_value=config),
            patch.object(window_module.QMessageBox, "information") as information,
        ):
            window.start_button.click()
        information.assert_called_once()
        assert window.version_combo_box.currentData() is None


class TestCloudConsent:
    def test_local_ollama_needs_no_consent(self, settings):
        config = ProviderConfig(ProviderId.OLLAMA, "m", base_url="http://localhost:11434")
        assert cloud_consent.has_consent(settings, config)

    def test_cloud_provider_asks_once(self, settings):
        settings.set_value(Settings.Key.AI_CLOUD_CONSENT, "")
        config = ProviderConfig(ProviderId.ANTHROPIC, "claude-opus-5-5", api_key="k")
        assert not cloud_consent.has_consent(settings, config)

        with patch.object(cloud_consent.QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.No) as question:
            assert cloud_consent.ask_consent(None, settings, config) is False
        assert question.call_count == 1

        with patch.object(cloud_consent.QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.Yes):
            assert cloud_consent.ask_consent(None, settings, config) is True

        with patch.object(cloud_consent.QMessageBox, "question") as question:
            assert cloud_consent.ask_consent(None, settings, config) is True
        question.assert_not_called()

        cloud_consent.revoke_consent(settings, ProviderId.ANTHROPIC)
        assert not cloud_consent.has_consent(settings, config)

    def test_window_does_not_start_without_consent(self, qtbot, window, settings):
        settings.set_value(Settings.Key.AI_CLOUD_CONSENT, "")
        config = ProviderConfig(ProviderId.OPENAI, "gpt-5", api_key="k")
        with (
            patch.object(window_module, "load_provider_config", return_value=config),
            patch.object(cloud_consent.QMessageBox, "question",
                         return_value=QMessageBox.StandardButton.No),
            patch.object(window.controller, "start") as start,
        ):
            window.start_button.click()
        start.assert_not_called()
