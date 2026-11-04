# SPDX-License-Identifier: GPL-3.0-or-later
"""Preferences tab for the AI post-processing (provider, model, API key)."""

import logging
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from buzz.ai.jobs import CallableJob, error_message, start_job
from buzz.ai.providers.anthropic_provider import EFFORT_LEVELS
from buzz.ai.providers.ollama_provider import DEFAULT_OLLAMA_URL
from buzz.ai.providers.registry import (
    MODEL_SETTING_KEYS,
    ProviderId,
    create_provider,
    get_api_key,
    get_model,
    get_selected_provider,
    load_provider_config,
    set_api_key,
)
from buzz.locale import _
from buzz.settings.settings import Settings
from buzz.widgets.line_edit import LineEdit
from buzz.widgets.password_line_edit import PasswordLineEdit

logger = logging.getLogger(__name__)


def provider_label(provider_id: ProviderId) -> str:
    if provider_id == ProviderId.OLLAMA:
        return _("Ollama (local)")
    return provider_id.display_name


class AIPreferencesWidget(QWidget):
    """Settings are saved immediately, like the other preference tabs.

    API keys are written to the OS secret store when editing finishes, not
    on every keystroke (each write can show a Keychain prompt on macOS).
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, settings: Optional[Settings] = None):
        super().__init__(parent)
        self.settings = settings or Settings()
        self.test_job: Optional[CallableJob] = None
        self.models_job: Optional[CallableJob] = None
        self._loading = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.provider_combo_box = QComboBox(self)
        for provider_id in (ProviderId.OLLAMA, ProviderId.OPENAI, ProviderId.ANTHROPIC):
            self.provider_combo_box.addItem(provider_label(provider_id), provider_id)
        self.provider_combo_box.currentIndexChanged.connect(self.on_provider_changed)
        form.addRow(_("AI provider"), self.provider_combo_box)

        self.api_key_label = QLabel(_("API key"), self)
        self.api_key_line_edit = PasswordLineEdit(parent=self)
        self.api_key_line_edit.setMinimumWidth(250)
        self.api_key_line_edit.editingFinished.connect(self.on_api_key_editing_finished)
        form.addRow(self.api_key_label, self.api_key_line_edit)

        self.url_label = QLabel(_("Server URL"), self)
        self.url_line_edit = LineEdit("", self)
        self.url_line_edit.textChanged.connect(self.on_url_changed)
        form.addRow(self.url_label, self.url_line_edit)

        self.model_combo_box = QComboBox(self)
        self.model_combo_box.setEditable(True)
        self.model_combo_box.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.model_combo_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo_box.editTextChanged.connect(self.on_model_changed)
        self.load_models_button = QPushButton(_("Load models"), self)
        self.load_models_button.clicked.connect(self.on_load_models_clicked)
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(self.model_combo_box, 1)
        model_row.addWidget(self.load_models_button)
        form.addRow(_("Model"), model_row)

        self.effort_label = QLabel(_("Effort"), self)
        self.effort_combo_box = QComboBox(self)
        self.effort_combo_box.addItem(_("Model default"), "")
        for level in EFFORT_LEVELS:
            self.effort_combo_box.addItem(level, level)
        self.effort_combo_box.setToolTip(
            _("How thoroughly Claude works. Higher effort is slower and more expensive.")
        )
        self.effort_combo_box.currentIndexChanged.connect(self.on_effort_changed)
        form.addRow(self.effort_label, self.effort_combo_box)

        self.temperature_label = QLabel(_("Temperature"), self)
        self.temperature_line_edit = LineEdit("", self)
        self.temperature_line_edit.setPlaceholderText(_("Model default"))
        self.temperature_line_edit.setToolTip(
            _("Lower values stay closer to the transcript. Leave empty for the model default.")
        )
        self.temperature_line_edit.textChanged.connect(self.on_temperature_changed)
        form.addRow(self.temperature_label, self.temperature_line_edit)

        self.test_button = QPushButton(_("Test connection"), self)
        self.test_button.clicked.connect(self.on_test_connection_clicked)
        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        test_row = QHBoxLayout()
        test_row.setContentsMargins(0, 0, 0, 0)
        test_row.addWidget(self.test_button)
        test_row.addWidget(self.status_label, 1)
        form.addRow("", test_row)

        self.privacy_label = QLabel(self)
        self.privacy_label.setWordWrap(True)
        self.privacy_label.setStyleSheet("color: gray;")
        layout.addWidget(self.privacy_label)

        # Extra sections (e.g. automation) are added by later code to this layout
        self.extra_layout = QVBoxLayout()
        layout.addLayout(self.extra_layout)
        layout.addStretch(1)

        selected = get_selected_provider(self.settings)
        index = self.provider_combo_box.findData(selected)
        self.provider_combo_box.blockSignals(True)
        self.provider_combo_box.setCurrentIndex(max(index, 0))
        self.provider_combo_box.blockSignals(False)
        self.load_provider_fields()

    # --- helpers -------------------------------------------------------------

    @property
    def provider_id(self) -> ProviderId:
        return self.provider_combo_box.currentData()

    def load_provider_fields(self) -> None:
        """Show the stored values of the selected provider."""
        self._loading = True
        try:
            provider_id = self.provider_id
            is_ollama = provider_id == ProviderId.OLLAMA
            is_anthropic = provider_id == ProviderId.ANTHROPIC

            self.api_key_label.setText(_("API key (optional)") if is_ollama else _("API key"))
            self.api_key_line_edit.setText(get_api_key(provider_id))
            self.api_key_line_edit.setPlaceholderText(
                {
                    ProviderId.OPENAI: "sk-...",
                    ProviderId.ANTHROPIC: "sk-ant-...",
                    ProviderId.OLLAMA: _("Only needed behind a proxy"),
                }[provider_id]
            )

            self.url_label.setVisible(not is_anthropic)
            self.url_line_edit.setVisible(not is_anthropic)
            if is_ollama:
                self.url_label.setText(_("Ollama URL"))
                self.url_line_edit.setPlaceholderText(DEFAULT_OLLAMA_URL)
                self.url_line_edit.setText(
                    self.settings.value(Settings.Key.AI_OLLAMA_URL, DEFAULT_OLLAMA_URL)
                )
            elif provider_id == ProviderId.OPENAI:
                self.url_label.setText(_("Base URL (optional)"))
                self.url_line_edit.setPlaceholderText("https://api.openai.com/v1")
                self.url_line_edit.setText(
                    self.settings.value(Settings.Key.AI_OPENAI_BASE_URL, "")
                )

            self.model_combo_box.clear()
            self.model_combo_box.setEditText(get_model(self.settings, provider_id))

            self.effort_label.setVisible(is_anthropic)
            self.effort_combo_box.setVisible(is_anthropic)
            effort = load_provider_config(self.settings, provider_id).effort if is_anthropic else ""
            self.effort_combo_box.setCurrentIndex(max(self.effort_combo_box.findData(effort), 0))

            self.temperature_label.setVisible(not is_anthropic)
            self.temperature_line_edit.setVisible(not is_anthropic)
            self.temperature_line_edit.setText(
                str(self.settings.value(Settings.Key.AI_TEMPERATURE, ""))
            )

            self.status_label.setText("")
            if is_ollama:
                self.privacy_label.setText(
                    _(
                        "Ollama runs AI models on your own computer: transcripts do not "
                        "leave this computer."
                    )
                )
            else:
                self.privacy_label.setText(
                    _(
                        "With {provider}, transcripts are sent to the provider's servers "
                        "for processing. You will be asked for consent before the first "
                        "transfer. API keys are stored in the system keychain."
                    ).format(provider=provider_id.display_name)
                )
        finally:
            self._loading = False

    def set_models(self, models: List[str]) -> None:
        current = self.model_combo_box.currentText()
        self._loading = True
        try:
            self.model_combo_box.clear()
            self.model_combo_box.addItems(models)
            self.model_combo_box.setEditText(current)
        finally:
            self._loading = False
        if not current and models:
            self.model_combo_box.setEditText(models[0])

    def save_api_key(self) -> bool:
        try:
            set_api_key(self.provider_id, self.api_key_line_edit.text())
            return True
        except Exception as exc:  # noqa: BLE001 - keychain errors are shown
            logger.warning("Unable to save API key: %s", type(exc).__name__)
            self.status_label.setText(_("Unable to save the API key to the keychain."))
            return False

    def _create_provider(self):
        # Read settings and the keychain here, on the UI thread
        return create_provider(load_provider_config(self.settings, self.provider_id))

    # --- slots ---------------------------------------------------------------

    def on_provider_changed(self, _index: int) -> None:
        self.settings.set_value(Settings.Key.AI_PROVIDER, self.provider_id.value)
        self.load_provider_fields()
        self.settings_changed.emit()

    def on_api_key_editing_finished(self) -> None:
        if self.api_key_line_edit.text().strip() == get_api_key(self.provider_id):
            return
        if self.save_api_key():
            self.settings_changed.emit()

    def on_url_changed(self, text: str) -> None:
        if self._loading:
            return
        key = (
            Settings.Key.AI_OLLAMA_URL
            if self.provider_id == ProviderId.OLLAMA
            else Settings.Key.AI_OPENAI_BASE_URL
        )
        self.settings.set_value(key, text.strip())
        self.settings_changed.emit()

    def on_model_changed(self, text: str) -> None:
        if self._loading:
            return
        self.settings.set_value(MODEL_SETTING_KEYS[self.provider_id], text.strip())
        self.settings_changed.emit()

    def on_effort_changed(self, _index: int) -> None:
        if self._loading:
            return
        self.settings.set_value(
            Settings.Key.AI_ANTHROPIC_EFFORT, self.effort_combo_box.currentData()
        )
        self.settings_changed.emit()

    def on_temperature_changed(self, text: str) -> None:
        if self._loading:
            return
        text = text.strip().replace(",", ".")
        if text:
            try:
                value = float(text)
            except ValueError:
                self.status_label.setText(_("Temperature must be a number, e.g. 0.2"))
                return
            if not 0.0 <= value <= 2.0:
                self.status_label.setText(_("Temperature must be between 0 and 2."))
                return
        self.status_label.setText("")
        self.settings.set_value(Settings.Key.AI_TEMPERATURE, text)
        self.settings_changed.emit()

    def on_load_models_clicked(self) -> None:
        self.save_api_key()
        self.load_models_button.setEnabled(False)
        self.status_label.setText(_("Loading models…"))
        provider = self._create_provider()
        self.models_job = start_job(
            lambda job: provider.list_models(),
            on_success=self.on_models_loaded,
            on_failure=self.on_models_failed,
        )

    def on_models_loaded(self, models: List[str]) -> None:
        self.load_models_button.setEnabled(True)
        self.models_job = None
        self.set_models(models)
        if models:
            self.status_label.setText(_("{count} models found.").format(count=len(models)))
        else:
            self.status_label.setText(_("No models found."))

    def on_models_failed(self, exc: Exception) -> None:
        self.load_models_button.setEnabled(True)
        self.models_job = None
        self.status_label.setText(error_message(exc))

    def on_test_connection_clicked(self) -> None:
        self.save_api_key()
        self.test_button.setEnabled(False)
        self.status_label.setText(_("Testing connection…"))
        provider = self._create_provider()
        self.test_job = start_job(
            lambda job: provider.test_connection(),
            on_success=self.on_test_succeeded,
            on_failure=self.on_test_failed,
        )

    def on_test_succeeded(self, models: List[str]) -> None:
        self.test_button.setEnabled(True)
        self.test_job = None
        self.set_models(models)
        model = self.model_combo_box.currentText().strip()
        if model and models and model not in models:
            self.status_label.setText(
                _("Connected, but the model “{model}” is not available.").format(model=model)
            )
        else:
            self.status_label.setText(_("Connection successful."))

    def on_test_failed(self, exc: Exception) -> None:
        self.test_button.setEnabled(True)
        self.test_job = None
        self.status_label.setText(error_message(exc))
