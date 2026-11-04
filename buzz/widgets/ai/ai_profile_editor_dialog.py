# SPDX-License-Identifier: GPL-3.0-or-later
"""Dialog to edit the AI post-processing profiles (prompt templates)."""

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from buzz.ai.pipeline import BASE_RULES
from buzz.ai.profiles import Profile, ProfileStore
from buzz.locale import _


class AIProfileEditorDialog(QDialog):
    def __init__(self, profile_store: Optional[ProfileStore] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.store = profile_store or ProfileStore()
        self.current: Optional[Profile] = None
        self._loading = False
        self._dirty = False

        self.setWindowTitle(_("AI profiles"))
        self.resize(950, 650)
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        layout.addWidget(splitter, 1)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.profile_list = QListWidget(self)
        self.profile_list.currentItemChanged.connect(self.on_current_item_changed)
        left_layout.addWidget(self.profile_list)
        list_buttons = QHBoxLayout()
        self.new_button = QPushButton(_("New"), self)
        self.new_button.clicked.connect(self.on_new_clicked)
        self.duplicate_button = QPushButton(_("Duplicate"), self)
        self.duplicate_button.clicked.connect(self.on_duplicate_clicked)
        self.delete_button = QPushButton(_("Delete"), self)
        self.delete_button.clicked.connect(self.on_delete_clicked)
        for button in (self.new_button, self.duplicate_button, self.delete_button):
            list_buttons.addWidget(button)
        left_layout.addLayout(list_buttons)
        splitter.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.name_edit = QLineEdit(self)
        self.description_edit = QLineEdit(self)
        self.cleanup_edit = QPlainTextEdit(self)
        self.cleanup_edit.setPlaceholderText(
            _("Notes for the text cleanup, e.g. technical terms or names to watch out for.")
        )
        self.document_edit = QPlainTextEdit(self)
        self.document_edit.setPlaceholderText(
            _("What the document should focus on, e.g. defects, quantities, agreements.")
        )
        self.sections_edit = QPlainTextEdit(self)
        self.sections_edit.setPlaceholderText(_("One section title per line"))
        self.sections_edit.setMaximumHeight(140)
        form.addRow(_("Name"), self.name_edit)
        form.addRow(_("Description"), self.description_edit)
        form.addRow(_("Cleanup instructions"), self.cleanup_edit)
        form.addRow(_("Document instructions"), self.document_edit)
        form.addRow(_("Additional sections"), self.sections_edit)
        right_layout.addLayout(form)
        for widget in (self.name_edit, self.description_edit):
            widget.textChanged.connect(self.on_changed)
        for widget in (self.cleanup_edit, self.document_edit, self.sections_edit):
            widget.textChanged.connect(self.on_changed)

        rules = QLabel(
            _("These base rules are always added and can not be changed:")
            + "\n" + BASE_RULES.split("\n", 2)[-1],
            self,
        )
        rules.setWordWrap(True)
        rules.setStyleSheet("color: gray;")
        right_layout.addWidget(rules)

        edit_buttons = QHBoxLayout()
        edit_buttons.addStretch(1)
        self.reset_button = QPushButton(_("Reset to default"), self)
        self.reset_button.clicked.connect(self.on_reset_clicked)
        self.save_button = QPushButton(_("Save"), self)
        self.save_button.clicked.connect(lambda: self.save_current())
        edit_buttons.addWidget(self.reset_button)
        edit_buttons.addWidget(self.save_button)
        right_layout.addLayout(edit_buttons)
        splitter.addWidget(right)
        splitter.setSizes([280, 670])

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close_box.button(QDialogButtonBox.StandardButton.Close).setText(_("Close"))
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)

        self.reload()

    # --- list ------------------------------------------------------------------

    def reload(self, select_id: Optional[str] = None) -> None:
        select_id = select_id or (self.current.id if self.current else None)
        self._loading = True
        self.profile_list.clear()
        for profile in self.store.profiles():
            label = profile.name
            if profile.builtin and self.store.is_modified(profile.id):
                label += " ✎"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, profile.id)
            self.profile_list.addItem(item)
        self._loading = False
        row = 0
        for index in range(self.profile_list.count()):
            if self.profile_list.item(index).data(Qt.ItemDataRole.UserRole) == select_id:
                row = index
        self.profile_list.setCurrentRow(row)
        self.on_current_item_changed(self.profile_list.currentItem(), None)

    def on_current_item_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        if self._loading:
            return
        if self._dirty and self.current is not None:
            # Keep the user's new selection: save without reloading the list
            self.save_current(reload=False)
        profile_id = current.data(Qt.ItemDataRole.UserRole) if current else None
        self.current = self.store.get(profile_id) if profile_id else None
        self.show_profile()

    def show_profile(self) -> None:
        profile = self.current
        self._loading = True
        self.name_edit.setText(profile.name if profile else "")
        self.description_edit.setText(profile.description if profile else "")
        self.cleanup_edit.setPlainText(profile.cleanup_prompt if profile else "")
        self.document_edit.setPlainText(profile.document_prompt if profile else "")
        self.sections_edit.setPlainText("\n".join(profile.sections) if profile else "")
        self._loading = False
        self._dirty = False
        builtin = bool(profile and profile.builtin)
        self.delete_button.setEnabled(bool(profile) and not builtin)
        self.reset_button.setEnabled(builtin and self.store.is_modified(profile.id))
        self.save_button.setEnabled(False)

    def on_changed(self) -> None:
        if self._loading:
            return
        self._dirty = True
        self.save_button.setEnabled(True)

    # --- actions -----------------------------------------------------------------

    def edited_profile(self) -> Optional[Profile]:
        if self.current is None:
            return None
        return self.current.copy(
            name=self.name_edit.text().strip(),
            description=self.description_edit.text().strip(),
            cleanup_prompt=self.cleanup_edit.toPlainText().strip(),
            document_prompt=self.document_edit.toPlainText().strip(),
            sections=[
                line.strip().lstrip("-•").strip()
                for line in self.sections_edit.toPlainText().splitlines()
                if line.strip()
            ],
        )

    def save_current(self, reload: bool = True) -> bool:
        profile = self.edited_profile()
        if profile is None:
            return False
        if not profile.name:
            QMessageBox.warning(self, _("AI profiles"), _("Please enter a name for the profile."))
            return False
        self.current = self.store.save_profile(profile)
        self._dirty = False
        if reload:
            self.reload(select_id=self.current.id)
        return True

    def on_new_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, _("New profile"), _("Name"))
        if ok and name.strip():
            if self._dirty:
                self.save_current()
            profile = self.store.create(name.strip())
            self.current = None
            self.reload(select_id=profile.id)

    def on_duplicate_clicked(self) -> None:
        if self.current is None:
            return
        if self._dirty:
            self.save_current()
        profile = self.store.duplicate(self.current.id)
        self.current = None
        self.reload(select_id=profile.id)

    def on_delete_clicked(self) -> None:
        if self.current is None or self.current.builtin:
            return
        answer = QMessageBox.question(
            self, _("Delete"), _("Delete the profile “{name}”?").format(name=self.current.name)
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.store.delete(self.current.id)
            self.current = None
            self._dirty = False
            self.reload()

    def on_reset_clicked(self) -> None:
        if self.current is None or not self.current.builtin:
            return
        answer = QMessageBox.question(
            self,
            _("Reset to default"),
            _("Undo all changes to the profile “{name}”?").format(name=self.current.name),
        )
        if answer == QMessageBox.StandardButton.Yes:
            profile_id = self.current.id
            self.store.reset(profile_id)
            self._dirty = False
            self.current = None
            self.reload(select_id=profile_id)

    def reject(self) -> None:
        if self._dirty:
            self.save_current()
        super().reject()
