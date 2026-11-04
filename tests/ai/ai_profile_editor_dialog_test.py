# SPDX-License-Identifier: GPL-3.0-or-later
from unittest.mock import patch

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from buzz.ai.profiles import GENERAL_CLEANUP_ID, SITE_INSPECTION_ID, ProfileStore
from buzz.widgets.ai import ai_profile_editor_dialog as dialog_module
from buzz.widgets.ai.ai_profile_editor_dialog import AIProfileEditorDialog


@pytest.fixture
def store(tmp_path):
    return ProfileStore(str(tmp_path / "profiles.json"))


@pytest.fixture
def dialog(qtbot, store):
    widget = AIProfileEditorDialog(store)
    qtbot.add_widget(widget)
    return widget


def select(dialog, profile_id):
    for row in range(dialog.profile_list.count()):
        if dialog.profile_list.item(row).data(Qt.ItemDataRole.UserRole) == profile_id:
            dialog.profile_list.setCurrentRow(row)
            return
    raise AssertionError(profile_id)


class TestAIProfileEditorDialog:
    def test_lists_profiles_and_shows_first(self, dialog):
        assert dialog.profile_list.count() == 5
        assert dialog.current.id == GENERAL_CLEANUP_ID
        assert dialog.name_edit.text() == dialog.current.name
        assert not dialog.delete_button.isEnabled()
        assert not dialog.save_button.isEnabled()

    def test_edit_builtin_and_reset(self, dialog, store):
        select(dialog, SITE_INSPECTION_ID)
        dialog.document_edit.setPlainText("Nur Mängel.")
        dialog.sections_edit.setPlainText("- Mängel\n\n• Maße\n")
        assert dialog.save_button.isEnabled()
        dialog.save_button.click()

        saved = ProfileStore(store.path).get(SITE_INSPECTION_ID)
        assert saved.document_prompt == "Nur Mängel."
        assert saved.sections == ["Mängel", "Maße"]
        assert dialog.reset_button.isEnabled()
        assert dialog.profile_list.currentItem().text().endswith("✎")

        with patch.object(dialog_module.QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.Yes):
            dialog.reset_button.click()
        assert not ProfileStore(store.path).is_modified(SITE_INSPECTION_ID)
        assert dialog.document_edit.toPlainText() != "Nur Mängel."

    def test_switching_profile_saves_and_keeps_new_selection(self, dialog, store):
        dialog.description_edit.setText("Geändert")
        select(dialog, SITE_INSPECTION_ID)
        assert dialog.current.id == SITE_INSPECTION_ID
        assert ProfileStore(store.path).get(GENERAL_CLEANUP_ID).description == "Geändert"

    def test_new_duplicate_delete(self, dialog, store):
        with patch.object(dialog_module.QInputDialog, "getText", return_value=("Eigenes", True)):
            dialog.new_button.click()
        assert dialog.current.name == "Eigenes"
        assert dialog.delete_button.isEnabled()

        dialog.duplicate_button.click()
        assert dialog.current.name == "Eigenes (copy)"
        assert dialog.profile_list.count() == 7

        with patch.object(dialog_module.QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.Yes):
            dialog.delete_button.click()
        assert dialog.profile_list.count() == 6
        assert [p.name for p in ProfileStore(store.path).profiles()][-1] == "Eigenes"

    def test_empty_name_is_rejected(self, dialog, store):
        dialog.name_edit.setText("")
        with patch.object(dialog_module.QMessageBox, "warning") as warning:
            assert dialog.save_current() is False
        warning.assert_called_once()

    def test_closing_saves_pending_changes(self, dialog, store):
        dialog.cleanup_edit.setPlainText("Fachbegriffe: Estrich")
        dialog.reject()
        assert ProfileStore(store.path).get(GENERAL_CLEANUP_ID).cleanup_prompt == "Fachbegriffe: Estrich"
