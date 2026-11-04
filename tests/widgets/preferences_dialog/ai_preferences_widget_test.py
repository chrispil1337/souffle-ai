# SPDX-License-Identifier: GPL-3.0-or-later
from unittest.mock import MagicMock, patch

import pytest

from buzz.ai.providers import registry
from buzz.ai.providers.base import AIAuthError
from buzz.ai.providers.registry import ProviderId
from buzz.settings.settings import Settings
from buzz.store.keyring_store import Key
from buzz.widgets.preferences_dialog import ai_preferences_widget
from buzz.widgets.preferences_dialog.ai_preferences_widget import AIPreferencesWidget

AI_KEYS = [key for key in Settings.Key if key.value.startswith("ai/")]


@pytest.fixture
def ai_settings(settings):
    for key in AI_KEYS:
        settings.settings.remove(key.value)
    yield settings
    for key in AI_KEYS:
        settings.settings.remove(key.value)


@pytest.fixture
def keyring():
    store = {}
    with (
        patch.object(registry, "get_password", side_effect=lambda key: store.get(key, "")),
        patch.object(
            registry, "set_password", side_effect=lambda key, value: store.__setitem__(key, value)
        ),
        patch.object(registry, "delete_password", side_effect=lambda key: store.pop(key, None)),
    ):
        yield store


@pytest.fixture
def fake_provider():
    provider = MagicMock()
    provider.list_models.return_value = ["model-a", "model-b"]
    provider.test_connection.return_value = ["model-a", "model-b"]
    with patch.object(ai_preferences_widget, "create_provider", return_value=provider):
        yield provider


def make_widget(qtbot, settings):
    widget = AIPreferencesWidget(settings=settings)
    qtbot.add_widget(widget)
    return widget


class TestAIPreferencesWidget:
    def test_defaults_to_local_ollama(self, qtbot, ai_settings, keyring):
        widget = make_widget(qtbot, ai_settings)
        assert widget.provider_id == ProviderId.OLLAMA
        assert widget.url_line_edit.text() == "http://localhost:11434"
        assert not widget.effort_combo_box.isVisibleTo(widget)
        assert "do not leave" in widget.privacy_label.text()

    def test_switching_provider_saves_selection_and_shows_fields(
        self, qtbot, ai_settings, keyring
    ):
        keyring[Key.ANTHROPIC_API_KEY] = "sk-ant-stored"
        widget = make_widget(qtbot, ai_settings)

        widget.provider_combo_box.setCurrentIndex(
            widget.provider_combo_box.findData(ProviderId.ANTHROPIC)
        )

        assert ai_settings.value(Settings.Key.AI_PROVIDER, "") == "anthropic"
        assert widget.api_key_line_edit.text() == "sk-ant-stored"
        assert widget.model_combo_box.currentText() == "claude-opus-5-5"
        assert widget.effort_combo_box.isVisibleTo(widget)
        assert not widget.url_line_edit.isVisibleTo(widget)
        assert not widget.temperature_line_edit.isVisibleTo(widget)
        assert "consent" in widget.privacy_label.text()

    def test_api_key_is_saved_to_keychain_on_editing_finished(
        self, qtbot, ai_settings, keyring
    ):
        ai_settings.set_value(Settings.Key.AI_PROVIDER, "openai")
        widget = make_widget(qtbot, ai_settings)

        widget.api_key_line_edit.setText("  sk-new  ")
        widget.api_key_line_edit.editingFinished.emit()

        assert keyring[Key.OPENAI_API_KEY] == "sk-new"
        # Never stored in plain settings
        assert all(
            "sk-new" not in str(ai_settings.settings.value(k))
            for k in ai_settings.settings.allKeys()
        )

    def test_model_and_url_are_saved_per_provider(self, qtbot, ai_settings, keyring):
        widget = make_widget(qtbot, ai_settings)

        widget.model_combo_box.setEditText("qwen2.5:14b")
        widget.url_line_edit.setText("http://192.168.1.5:11434")

        assert ai_settings.value(Settings.Key.AI_MODEL_OLLAMA, "") == "qwen2.5:14b"
        assert ai_settings.value(Settings.Key.AI_OLLAMA_URL, "") == "http://192.168.1.5:11434"
        assert ai_settings.value(Settings.Key.AI_MODEL_OPENAI, "") == ""

    def test_invalid_temperature_is_not_saved(self, qtbot, ai_settings, keyring):
        widget = make_widget(qtbot, ai_settings)

        widget.temperature_line_edit.setText("heiß")
        assert ai_settings.value(Settings.Key.AI_TEMPERATURE, "") == ""
        assert widget.status_label.text()

        widget.temperature_line_edit.setText("0,3")
        assert ai_settings.value(Settings.Key.AI_TEMPERATURE, "") == "0.3"

    def test_effort_is_saved(self, qtbot, ai_settings, keyring):
        ai_settings.set_value(Settings.Key.AI_PROVIDER, "anthropic")
        widget = make_widget(qtbot, ai_settings)

        widget.effort_combo_box.setCurrentIndex(widget.effort_combo_box.findData("high"))

        assert ai_settings.value(Settings.Key.AI_ANTHROPIC_EFFORT, "") == "high"

    def test_test_connection_success(self, qtbot, ai_settings, keyring, fake_provider):
        widget = make_widget(qtbot, ai_settings)
        widget.model_combo_box.setEditText("model-b")

        widget.test_button.click()
        qtbot.waitUntil(lambda: widget.test_job is None, timeout=5000)

        assert widget.test_button.isEnabled()
        assert widget.status_label.text() == "Connection successful."
        assert widget.model_combo_box.count() == 2
        assert widget.model_combo_box.currentText() == "model-b"

    def test_test_connection_reports_missing_model(
        self, qtbot, ai_settings, keyring, fake_provider
    ):
        widget = make_widget(qtbot, ai_settings)
        widget.model_combo_box.setEditText("does-not-exist")

        widget.test_button.click()
        qtbot.waitUntil(lambda: widget.test_job is None, timeout=5000)

        assert "does-not-exist" in widget.status_label.text()

    def test_test_connection_failure_shows_user_message(
        self, qtbot, ai_settings, keyring, fake_provider
    ):
        fake_provider.test_connection.side_effect = AIAuthError("Key rejected.")
        widget = make_widget(qtbot, ai_settings)

        widget.test_button.click()
        qtbot.waitUntil(lambda: widget.test_job is None, timeout=5000)

        assert widget.status_label.text() == "Key rejected."
        assert widget.test_button.isEnabled()

    def test_load_models_fills_list_and_selects_first_when_empty(
        self, qtbot, ai_settings, keyring, fake_provider
    ):
        widget = make_widget(qtbot, ai_settings)
        assert widget.model_combo_box.currentText() == ""

        widget.load_models_button.click()
        qtbot.waitUntil(lambda: widget.models_job is None, timeout=5000)

        assert [widget.model_combo_box.itemText(i) for i in range(2)] == [
            "model-a",
            "model-b",
        ]
        assert widget.model_combo_box.currentText() == "model-a"
        assert ai_settings.value(Settings.Key.AI_MODEL_OLLAMA, "") == "model-a"


class TestAIPreferencesWidgetOllamaCloud:
    def test_ollama_cloud_fields(self, qtbot, ai_settings, keyring):
        widget = make_widget(qtbot, ai_settings)
        widget.provider_combo_box.setCurrentIndex(
            widget.provider_combo_box.findData(ProviderId.OLLAMA_CLOUD)
        )

        assert ai_settings.value(Settings.Key.AI_PROVIDER, "") == "ollama_cloud"
        assert widget.provider_combo_box.currentText() == "Ollama Cloud"
        assert not widget.url_line_edit.isVisibleTo(widget)
        assert not widget.effort_combo_box.isVisibleTo(widget)
        assert widget.temperature_line_edit.isVisibleTo(widget)
        assert widget.api_key_label.text() == "API key"
        assert "consent" in widget.privacy_label.text()

        widget.api_key_line_edit.setText("cloud-key")
        widget.api_key_line_edit.editingFinished.emit()
        widget.model_combo_box.setEditText("deepseek-v4.1-flash")

        assert keyring[Key.OLLAMA_CLOUD_API_KEY] == "cloud-key"
        assert Key.OLLAMA_API_KEY not in keyring
        assert ai_settings.value(Settings.Key.AI_MODEL_OLLAMA_CLOUD, "") == "deepseek-v4.1-flash"
        assert ai_settings.value(Settings.Key.AI_MODEL_OLLAMA, "") == ""
