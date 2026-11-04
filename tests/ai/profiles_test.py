# SPDX-License-Identifier: GPL-3.0-or-later
import json

import pytest

from buzz.ai.profiles import (
    GENERAL_CLEANUP_ID,
    MEETING_MINUTES_ID,
    SITE_INSPECTION_ID,
    ProfileStore,
    builtin_profiles,
)


@pytest.fixture
def store(tmp_path):
    return ProfileStore(str(tmp_path / "ai_profiles.json"))


class TestProfileStore:
    def test_five_builtin_profiles(self, store):
        ids = [p.id for p in store.profiles()]
        assert ids == [p.id for p in builtin_profiles()]
        assert len(ids) == 5
        assert all(p.builtin for p in store.profiles())
        assert all(p.document_prompt for p in store.profiles())

    def test_edit_builtin_persists_and_can_be_reset(self, store, tmp_path):
        profile = store.get(SITE_INSPECTION_ID)
        profile.document_prompt = "Nur Mängel auflisten."
        store.save_profile(profile)

        reloaded = ProfileStore(store.path)
        assert reloaded.get(SITE_INSPECTION_ID).document_prompt == "Nur Mängel auflisten."
        assert reloaded.is_modified(SITE_INSPECTION_ID)

        reloaded.reset(SITE_INSPECTION_ID)
        again = ProfileStore(store.path)
        assert not again.is_modified(SITE_INSPECTION_ID)
        assert again.get(SITE_INSPECTION_ID).document_prompt != "Nur Mängel auflisten."

    def test_saving_unchanged_builtin_stores_nothing(self, store):
        store.save_profile(store.get(GENERAL_CLEANUP_ID))
        assert not store.is_modified(GENERAL_CLEANUP_ID)

    def test_duplicate_create_and_delete_custom(self, store):
        copy = store.duplicate(MEETING_MINUTES_ID, name="Mein Protokoll")
        assert not copy.builtin
        assert copy.id != MEETING_MINUTES_ID
        assert copy.sections == store.get(MEETING_MINUTES_ID).sections

        created = store.create("Leer")
        reloaded = ProfileStore(store.path)
        names = [p.name for p in reloaded.profiles()]
        assert "Mein Protokoll" in names and "Leer" in names

        reloaded.delete(copy.id)
        assert reloaded.get(copy.id) is None
        assert reloaded.get(created.id) is not None

    def test_builtin_can_not_be_deleted(self, store):
        with pytest.raises(ValueError):
            store.delete(GENERAL_CLEANUP_ID)

    def test_profile_needs_a_name(self, store):
        profile = store.create("Name")
        profile.name = "  "
        with pytest.raises(ValueError):
            store.save_profile(profile)

    def test_returned_profiles_are_copies(self, store):
        store.get(GENERAL_CLEANUP_ID).sections.append("Hack")
        assert "Hack" not in store.get(GENERAL_CLEANUP_ID).sections

    def test_broken_file_falls_back_to_defaults_and_is_kept(self, tmp_path):
        path = tmp_path / "ai_profiles.json"
        path.write_text("{ kaputt", encoding="utf-8")
        store = ProfileStore(str(path))
        assert len(store.profiles()) == 5
        assert (tmp_path / "ai_profiles.json.broken").exists()

    def test_file_format(self, store):
        store.create("Mein Profil")
        data = json.loads(open(store.path, encoding="utf-8").read())
        assert data["version"] == 1
        assert data["profiles"][0]["name"] == "Mein Profil"
        assert "builtin" not in data["profiles"][0]

    def test_unknown_profile_falls_back_to_general(self, store):
        assert store.get_or_default("gibt-es-nicht").id == GENERAL_CLEANUP_ID
