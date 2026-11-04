# SPDX-License-Identifier: GPL-3.0-or-later
"""Window for AI post-processing of one transcription.

Shows the original next to the AI version, highlights changes and lets the
user edit the AI version. The original transcript is only read, never
changed. Runs continue in the background when the window is closed.
"""

import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from buzz.ai.chunking import format_timestamp
from buzz.ai.controller import PostProcessingController, get_shared_controller
from buzz.ai.diff import segments_diff_html
from buzz.ai.jobs import error_message
from buzz.ai.profiles import GENERAL_CLEANUP_ID, ProfileStore
from buzz.ai.providers.base import AIError
from buzz.ai.providers.registry import load_provider_config
from buzz.db.entity.ai_version import AISegmentFlag, AIVersion, AIVersionStatus
from buzz.export.transcript_export import (
    ExportKind,
    build_export,
    default_export_path,
    transcription_title,
    write_export,
)
from buzz.locale import _
from buzz.settings.settings import Settings
from buzz.widgets.ai.cloud_consent import ask_consent

logger = logging.getLogger(__name__)

DOCUMENT_SAVE_DELAY_MS = 800
FLAGGED_COLOR = QColor(255, 236, 179)

COLUMN_TIME, COLUMN_SPEAKER, COLUMN_ORIGINAL, COLUMN_AI = range(4)


def status_label(status: str) -> str:
    return {
        AIVersionStatus.PENDING.value: _("waiting"),
        AIVersionStatus.RUNNING.value: _("running…"),
        AIVersionStatus.COMPLETED.value: "✓",
        AIVersionStatus.FAILED.value: _("failed"),
        AIVersionStatus.CANCELED.value: _("canceled"),
    }.get(status, status)


def version_label(version: AIVersion) -> str:
    try:
        created = datetime.fromisoformat(version.time_created).strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError):
        created = version.time_created or ""
    edited = " ✎" if version.manually_edited else ""
    return f"{created} · {version.profile_name} · {version.model} · {status_label(version.status)}{edited}"


def flag_tooltip(flag: str) -> str:
    if flag == AISegmentFlag.MISSING.value:
        return _("The AI did not return this segment. The original text was kept.")
    if flag == AISegmentFlag.SUSPICIOUS.value:
        return _("The AI result looked unreliable. The original text was kept.")
    return ""


class AIPostProcessingWindow(QWidget):
    def __init__(
        self,
        transcription,
        transcription_service,
        controller: Optional[PostProcessingController] = None,
        settings: Optional[Settings] = None,
        profile_store: Optional[ProfileStore] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.transcription = transcription
        self.transcription_service = transcription_service
        self.controller = controller or get_shared_controller(transcription_service)
        self.ai_service = self.controller.ai_service
        self.settings = settings or self.controller.settings
        self.profile_store = profile_store or self.controller.profile_store

        self.current_version: Optional[AIVersion] = None
        self.ai_segments = []
        self._loading = False
        self._document_dirty = False

        self.setWindowTitle(
            _("AI post-processing – {name}").format(name=transcription_title(transcription))
        )
        self.resize(1200, 800)

        layout = QVBoxLayout(self)

        # --- run controls ------------------------------------------------------
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel(_("Profile:"), self))
        self.profile_combo_box = QComboBox(self)
        self.profile_combo_box.setMinimumWidth(240)
        run_row.addWidget(self.profile_combo_box)
        self.provider_label = QLabel(self)
        self.provider_label.setStyleSheet("color: gray;")
        run_row.addWidget(self.provider_label)
        run_row.addStretch(1)
        self.start_button = QPushButton(_("Post-process with AI"), self)
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.on_start_clicked)
        run_row.addWidget(self.start_button)
        self.cancel_button = QPushButton(_("Cancel"), self)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        run_row.addWidget(self.cancel_button)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1000)
        self.progress_label = QLabel(self)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.progress_label, 2)
        layout.addLayout(progress_row)

        # --- versions ------------------------------------------------------------
        version_row = QHBoxLayout()
        version_row.addWidget(QLabel(_("Version:"), self))
        self.version_combo_box = QComboBox(self)
        self.version_combo_box.currentIndexChanged.connect(self.on_version_selected)
        version_row.addWidget(self.version_combo_box, 1)
        self.export_button = QToolButton(self)
        self.export_button.setText(_("Export"))
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_menu = QMenu(self)
        for kind, label in (
            (ExportKind.AI_MARKDOWN, _("Markdown – AI version")),
            (ExportKind.COMBINED_MARKDOWN, _("Markdown – AI version and original")),
            (ExportKind.COMBINED_JSON, _("JSON – AI version and original")),
            (ExportKind.ORIGINAL_MARKDOWN, _("Markdown – original only")),
        ):
            action = QAction(label, self)
            action.setData(kind)
            self.export_menu.addAction(action)
        self.export_menu.triggered.connect(self.on_export_triggered)
        self.export_button.setMenu(self.export_menu)
        version_row.addWidget(self.export_button)
        self.delete_button = QPushButton(_("Delete version"), self)
        self.delete_button.clicked.connect(self.on_delete_clicked)
        version_row.addWidget(self.delete_button)
        layout.addLayout(version_row)

        self.message_label = QLabel(self)
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.message_label)

        # --- content -------------------------------------------------------------
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        self.segments_table = QTableWidget(0, 4, self)
        self.segments_table.setHorizontalHeaderLabels(
            [_("Time"), _("Speaker"), _("Original"), _("AI version (editable)")]
        )
        header = self.segments_table.horizontalHeader()
        header.setSectionResizeMode(COLUMN_TIME, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMN_SPEAKER, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMN_ORIGINAL, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COLUMN_AI, QHeaderView.ResizeMode.Stretch)
        self.segments_table.setWordWrap(True)
        self.segments_table.verticalHeader().setVisible(False)
        self.segments_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.segments_table.itemChanged.connect(self.on_segment_item_changed)
        self.tabs.addTab(self.segments_table, _("Segments"))

        document_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        original_panel = QWidget(self)
        original_layout = QVBoxLayout(original_panel)
        original_layout.setContentsMargins(0, 0, 0, 0)
        original_layout.addWidget(QLabel(_("Original transcript"), self))
        self.original_text = QPlainTextEdit(self)
        self.original_text.setReadOnly(True)
        original_layout.addWidget(self.original_text)
        document_splitter.addWidget(original_panel)

        document_panel = QWidget(self)
        document_layout = QVBoxLayout(document_panel)
        document_layout.setContentsMargins(0, 0, 0, 0)
        document_header = QHBoxLayout()
        document_header.addWidget(QLabel(_("AI document (Markdown, editable)"), self))
        document_header.addStretch(1)
        self.preview_button = QPushButton(_("Preview"), self)
        self.preview_button.setCheckable(True)
        self.preview_button.toggled.connect(self.on_preview_toggled)
        document_header.addWidget(self.preview_button)
        document_layout.addLayout(document_header)
        self.document_stack = QStackedWidget(self)
        self.document_editor = QPlainTextEdit(self)
        self.document_editor.textChanged.connect(self.on_document_changed)
        self.document_preview = QTextBrowser(self)
        self.document_preview.setOpenExternalLinks(False)
        self.document_stack.addWidget(self.document_editor)
        self.document_stack.addWidget(self.document_preview)
        document_layout.addWidget(self.document_stack)
        document_splitter.addWidget(document_panel)
        document_splitter.setSizes([500, 700])
        self.tabs.addTab(document_splitter, _("Document"))

        self.diff_browser = QTextBrowser(self)
        self.tabs.addTab(self.diff_browser, _("Changes"))

        self.document_save_timer = QTimer(self)
        self.document_save_timer.setSingleShot(True)
        self.document_save_timer.setInterval(DOCUMENT_SAVE_DELAY_MS)
        self.document_save_timer.timeout.connect(self.save_document)

        self.controller.started.connect(self.on_run_started)
        self.controller.progress.connect(self.on_run_progress)
        self.controller.finished.connect(self.on_run_finished)
        self.controller.failed.connect(self.on_run_failed)

        self.load_profiles()
        self.load_original()
        self.update_provider_label()
        self.reload_versions()
        self.update_run_state()

    # --- data loading ------------------------------------------------------------

    @property
    def transcription_uuid(self) -> UUID:
        return UUID(str(self.transcription.id))

    def load_profiles(self) -> None:
        last = self.settings.value(Settings.Key.AI_LAST_PROFILE, GENERAL_CLEANUP_ID)
        self.profile_combo_box.clear()
        for profile in self.profile_store.profiles():
            self.profile_combo_box.addItem(profile.name, profile.id)
            if profile.description:
                self.profile_combo_box.setItemData(
                    self.profile_combo_box.count() - 1,
                    profile.description,
                    Qt.ItemDataRole.ToolTipRole,
                )
        index = self.profile_combo_box.findData(last)
        self.profile_combo_box.setCurrentIndex(max(index, 0))

    def load_original(self) -> None:
        segments = self.transcription_service.get_transcription_segments(
            transcription_id=self.transcription_uuid
        )
        self.original_segments = sorted(segments, key=lambda s: (s.start_time, s.id))
        lines = []
        for segment in self.original_segments:
            speaker = f" {segment.speaker}:" if segment.speaker else ""
            lines.append(f"({format_timestamp(segment.start_time)}){speaker} {segment.text.strip()}")
        self.original_text.setPlainText("\n".join(lines))

    def update_provider_label(self) -> None:
        config = load_provider_config(self.settings)
        model = config.model or _("no model selected")
        self.provider_label.setText(f"{config.provider_id.display_name} · {model}")

    def reload_versions(self, select_id: Optional[int] = None) -> None:
        if select_id is None and self.current_version is not None:
            select_id = self.current_version.id
        versions = self.ai_service.list_versions(self.transcription_uuid)
        self._loading = True
        try:
            self.version_combo_box.clear()
            for version in versions:
                self.version_combo_box.addItem(version_label(version), version.id)
            if not versions:
                self.version_combo_box.addItem(_("No AI version yet"), None)
            index = self.version_combo_box.findData(select_id) if select_id is not None else 0
        finally:
            self._loading = False
        self.version_combo_box.setCurrentIndex(max(index, 0))
        self.on_version_selected(self.version_combo_box.currentIndex())

    def on_version_selected(self, _index: int) -> None:
        if self._loading:
            return
        self.flush_document()
        version_id = self.version_combo_box.currentData()
        self.current_version = self.ai_service.get_version(version_id) if version_id else None
        self.ai_segments = self.ai_service.get_segments(version_id) if version_id else []
        self.show_version()

    def show_version(self) -> None:
        version = self.current_version
        completed = version is not None and version.status == AIVersionStatus.COMPLETED.value
        self.export_button.setEnabled(True)
        for action in self.export_menu.actions():
            kind = action.data()
            action.setEnabled(completed or not kind.needs_ai_version)
        self.delete_button.setEnabled(version is not None and not self.controller.is_running(version.id))

        if version is None:
            self.message_label.setText(
                _("Choose a profile and click “Post-process with AI”. The original transcript stays unchanged.")
            )
        elif version.status == AIVersionStatus.FAILED.value:
            self.message_label.setText("⚠ " + (version.error_message or _("failed")))
        elif version.status == AIVersionStatus.CANCELED.value:
            self.message_label.setText(_("This run was canceled."))
        elif completed:
            try:
                warnings = json.loads(version.analysis_json or "{}").get("warnings", [])
            except ValueError:
                warnings = []
            self.message_label.setText("\n".join("⚠ " + w for w in warnings))
        else:
            self.message_label.setText(_("AI post-processing is running…"))

        self._loading = True
        try:
            self.segments_table.setRowCount(len(self.ai_segments))
            for row, segment in enumerate(self.ai_segments):
                items = [
                    QTableWidgetItem(format_timestamp(segment.start_time)),
                    QTableWidgetItem(segment.speaker),
                    QTableWidgetItem(segment.original_text),
                    QTableWidgetItem(segment.text),
                ]
                for column, item in enumerate(items):
                    if column != COLUMN_AI:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if segment.flag:
                        item.setBackground(FLAGGED_COLOR)
                        item.setToolTip(flag_tooltip(segment.flag))
                    self.segments_table.setItem(row, column, item)
            self.segments_table.resizeRowsToContents()

            self.document_editor.setPlainText(version.document_markdown if completed else "")
            self.document_editor.setReadOnly(not completed)
            self._document_dirty = False
            self.update_preview()
            self.update_diff()
        finally:
            self._loading = False

    def update_preview(self) -> None:
        if self.preview_button.isChecked():
            self.document_preview.setMarkdown(self.document_editor.toPlainText())

    def update_diff(self) -> None:
        rows = [
            (format_timestamp(s.start_time), s.speaker, s.original_text, s.text)
            for s in self.ai_segments
        ]
        if rows:
            self.diff_browser.setHtml(segments_diff_html(rows))
        else:
            self.diff_browser.setPlainText(_("No AI version yet"))

    # --- editing -----------------------------------------------------------------

    def on_segment_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != COLUMN_AI or self.current_version is None:
            return
        segment = self.ai_segments[item.row()]
        text = item.text()
        if text == segment.text:
            return
        try:
            self.ai_service.update_segment_text(self.current_version.id, segment.id, text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Saving AI segment failed: %s", exc)
            QMessageBox.warning(self, _("Error"), _("The change could not be saved."))
            return
        segment.text = text
        self.current_version.manually_edited = True
        self.update_diff()

    def on_document_changed(self) -> None:
        if self._loading or self.current_version is None:
            return
        self._document_dirty = True
        self.document_save_timer.start()

    def save_document(self) -> None:
        if not self._document_dirty or self.current_version is None:
            return
        text = self.document_editor.toPlainText()
        try:
            self.ai_service.update_document(self.current_version.id, text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Saving AI document failed: %s", exc)
            QMessageBox.warning(self, _("Error"), _("The change could not be saved."))
            return
        self.current_version.document_markdown = text
        self.current_version.manually_edited = True
        self._document_dirty = False

    def flush_document(self) -> None:
        self.document_save_timer.stop()
        self.save_document()

    def on_preview_toggled(self, checked: bool) -> None:
        self.flush_document()
        self.update_preview()
        self.document_stack.setCurrentIndex(1 if checked else 0)

    # --- runs --------------------------------------------------------------------

    def running_version_id(self) -> Optional[int]:
        return self.controller.running_version_for(self.transcription_uuid)

    def update_run_state(self, fraction: float = 0.0, message: str = "") -> None:
        running = self.running_version_id() is not None
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.profile_combo_box.setEnabled(not running)
        self.progress_bar.setVisible(running)
        self.progress_label.setVisible(running)
        if running:
            self.progress_bar.setValue(int(fraction * 1000))
            self.progress_label.setText(message)

    def _is_mine(self, version_id: int) -> bool:
        version = self.ai_service.get_version(version_id)
        return version is not None and version.transcription_id == str(self.transcription_uuid)

    def on_start_clicked(self) -> None:
        self.flush_document()
        self.update_provider_label()
        config = load_provider_config(self.settings)
        if not config.model:
            QMessageBox.information(
                self,
                _("AI post-processing"),
                _("No AI model selected. Please choose a model in the AI settings."),
            )
            return
        if not ask_consent(self, self.settings, config):
            return
        profile_id = self.profile_combo_box.currentData()
        self.settings.set_value(Settings.Key.AI_LAST_PROFILE, profile_id)
        try:
            self.controller.start(
                self.transcription_uuid,
                profile_id,
                title=transcription_title(self.transcription),
            )
        except AIError as exc:
            QMessageBox.warning(self, _("AI post-processing"), error_message(exc))
        except Exception as exc:  # noqa: BLE001
            logger.error("Starting AI post-processing failed: %s", exc)
            QMessageBox.warning(self, _("AI post-processing"), error_message(exc))

    def on_cancel_clicked(self) -> None:
        version_id = self.running_version_id()
        if version_id is not None:
            self.controller.cancel(version_id)
            self.progress_label.setText(_("Canceling…"))
            self.cancel_button.setEnabled(False)

    def on_run_started(self, version_id: int) -> None:
        if self._is_mine(version_id):
            self.reload_versions(select_id=version_id)
            self.update_run_state(0.0, _("Starting AI post-processing"))

    def on_run_progress(self, version_id: int, fraction: float, message: str) -> None:
        if self.running_version_id() == version_id:
            self.update_run_state(fraction, message)

    def on_run_finished(self, version_id: int) -> None:
        if self._is_mine(version_id):
            self.reload_versions(select_id=version_id)
            self.update_run_state()

    def on_run_failed(self, version_id: int, message: str) -> None:
        if self._is_mine(version_id):
            self.reload_versions(select_id=version_id)
            self.update_run_state()

    # --- versions and export -------------------------------------------------------

    def on_delete_clicked(self) -> None:
        if self.current_version is None:
            return
        answer = QMessageBox.question(
            self,
            _("Delete version"),
            _("Delete this AI version? The original transcript is not affected."),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.document_save_timer.stop()
        self._document_dirty = False
        self.ai_service.delete_version(self.current_version.id)
        self.current_version = None
        self.reload_versions()

    def export(self, kind: ExportKind, path: str) -> None:
        self.flush_document()
        content = build_export(
            kind,
            self.transcription,
            self.original_segments,
            self.current_version,
            self.ai_segments,
        )
        write_export(path, content)

    def on_export_triggered(self, action: QAction) -> None:
        kind: ExportKind = action.data()
        suggested = default_export_path(self.transcription, kind)
        file_filter = _("JSON files (*.json)") if kind.extension == "json" else _("Markdown files (*.md)")
        path, _selected = QFileDialog.getSaveFileName(self, _("Export"), suggested, file_filter)
        if not path:
            return
        try:
            self.export(kind, path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Export failed: %s", exc)
            QMessageBox.warning(self, _("Export"), _("The file could not be written."))

    def closeEvent(self, event) -> None:
        self.flush_document()
        super().closeEvent(event)
