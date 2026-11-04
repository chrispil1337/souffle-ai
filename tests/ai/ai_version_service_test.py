# SPDX-License-Identifier: GPL-3.0-or-later
import json
import sqlite3

from PyQt6.QtSql import QSqlQuery

from buzz.db.entity.ai_version import AIVersion, AIVersionSegment, AIVersionStatus
from buzz.db.helpers import run_sqlite_migrations


def make_version(transcription_id, **kwargs):
    return AIVersion(transcription_id=str(transcription_id), profile_id="meeting_minutes", **kwargs)


class TestAIVersionService:
    def test_create_and_list_newest_first(self, ai_service, stored_transcription):
        first = ai_service.create_version(make_version(stored_transcription))
        second = ai_service.create_version(make_version(stored_transcription, model="m2"))

        versions = ai_service.list_versions(stored_transcription)
        assert [v.id for v in versions] == [second, first]
        assert versions[0].model == "m2"
        assert versions[0].status == "pending"
        assert versions[0].manually_edited is False
        assert versions[0].time_created

    def test_save_result_and_read_back(self, ai_service, stored_transcription):
        version_id = ai_service.create_version(make_version(stored_transcription))
        ai_service.save_result(
            version_id,
            [
                AIVersionSegment(0, "äh Original", "Original", start_time=0, end_time=1000, speaker="Anna"),
                AIVersionSegment(0, "zwei", "Zwei", flag="missing"),
            ],
            "# Dokument",
            {"summary": "Kurz", "tasks": []},
            '{"p": 1}',
        )

        version = ai_service.get_version(version_id)
        assert version.status == AIVersionStatus.COMPLETED.value
        assert version.document_markdown == "# Dokument"
        assert json.loads(version.analysis_json) == {"summary": "Kurz", "tasks": []}
        segments = ai_service.get_segments(version_id)
        assert [(s.position, s.text, s.original_text, s.flag) for s in segments] == [
            (0, "Original", "äh Original", ""),
            (1, "Zwei", "zwei", "missing"),
        ]
        assert ai_service.latest_completed_version(stored_transcription).id == version_id

    def test_save_result_again_replaces_segments(self, ai_service, stored_transcription):
        version_id = ai_service.create_version(make_version(stored_transcription))
        ai_service.save_result(version_id, [AIVersionSegment(0, "a", "A")], "", {}, "")
        ai_service.save_result(version_id, [AIVersionSegment(0, "b", "B")], "", {}, "")
        assert [s.text for s in ai_service.get_segments(version_id)] == ["B"]

    def test_manual_edits_mark_version(self, ai_service, stored_transcription):
        version_id = ai_service.create_version(make_version(stored_transcription))
        ai_service.save_result(version_id, [AIVersionSegment(0, "a", "A")], "# Alt", {}, "")
        segment = ai_service.get_segments(version_id)[0]

        ai_service.update_segment_text(version_id, segment.id, "Von Hand")
        assert ai_service.get_segments(version_id)[0].text == "Von Hand"
        assert ai_service.get_segments(version_id)[0].original_text == "a"
        assert ai_service.get_version(version_id).manually_edited is True

        ai_service.update_document(version_id, "# Neu")
        assert ai_service.get_version(version_id).document_markdown == "# Neu"

    def test_original_transcript_is_untouched(
        self, ai_service, stored_transcription, transcription_service
    ):
        before = [
            (s.id, s.text) for s in transcription_service.get_transcription_segments(stored_transcription)
        ]
        version_id = ai_service.create_version(make_version(stored_transcription))
        ai_service.save_result(
            version_id, [AIVersionSegment(0, t, "BEREINIGT") for _, t in before], "", {}, ""
        )
        after = [
            (s.id, s.text) for s in transcription_service.get_transcription_segments(stored_transcription)
        ]
        assert after == before

    def test_status_and_delete(self, ai_service, stored_transcription):
        version_id = ai_service.create_version(make_version(stored_transcription))
        ai_service.set_status(version_id, AIVersionStatus.FAILED, "Netzwerkfehler")
        version = ai_service.get_version(version_id)
        assert (version.status, version.error_message) == ("failed", "Netzwerkfehler")
        assert ai_service.latest_completed_version(stored_transcription) is None

        ai_service.save_result(version_id, [AIVersionSegment(0, "a", "A")], "", {}, "")
        ai_service.delete_version(version_id)
        assert ai_service.get_version(version_id) is None
        assert ai_service.get_segments(version_id) == []

    def test_interrupted_runs_are_marked_failed(self, ai_service, stored_transcription):
        running = ai_service.create_version(
            make_version(stored_transcription, status=AIVersionStatus.RUNNING.value)
        )
        done = ai_service.create_version(
            make_version(stored_transcription, status=AIVersionStatus.COMPLETED.value)
        )
        ai_service.mark_interrupted_as_failed("Abgebrochen")
        assert ai_service.get_version(running).status == "failed"
        assert ai_service.get_version(done).status == "completed"

    def test_deleting_transcription_deletes_versions(self, ai_service, stored_transcription, db):
        version_id = ai_service.create_version(make_version(stored_transcription))
        ai_service.save_result(version_id, [AIVersionSegment(0, "a", "A")], "", {}, "")

        query = QSqlQuery(db)
        query.prepare("DELETE FROM transcription WHERE id = :id")
        query.bindValue(":id", str(stored_transcription))
        assert query.exec()

        assert ai_service.get_version(version_id) is None
        assert ai_service.get_segments(version_id) == []


def test_migration_adds_ai_tables_to_existing_database(tmp_path):
    """An existing Buzz database gets the AI tables; its data is kept."""
    path = tmp_path / "buzz.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE transcription (
            id TEXT PRIMARY KEY, error_message TEXT, export_formats TEXT, file TEXT,
            output_folder TEXT, progress DOUBLE PRECISION DEFAULT 0.0, language TEXT,
            model_type TEXT, source TEXT, status TEXT, task TEXT, time_ended TIMESTAMP,
            time_queued TIMESTAMP NOT NULL, time_started TIMESTAMP, url TEXT,
            whisper_model_size TEXT, hugging_face_model_id TEXT,
            word_level_timings BOOLEAN DEFAULT FALSE, extract_speech BOOLEAN DEFAULT FALSE,
            name TEXT, notes TEXT, metadata TEXT DEFAULT '[]'
        );
        CREATE TABLE transcription_segment (
            id INTEGER PRIMARY KEY, end_time INT DEFAULT 0, start_time INT DEFAULT 0,
            text TEXT NOT NULL, translation TEXT DEFAULT '', transcription_id TEXT,
            speaker TEXT DEFAULT '', metadata TEXT DEFAULT '[]',
            FOREIGN KEY (transcription_id) REFERENCES transcription(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_transcription_id ON transcription_segment(transcription_id);
        INSERT INTO transcription (id, time_queued) VALUES ('t1', '2026-01-01');
        INSERT INTO transcription_segment (id, text, transcription_id)
            VALUES (1, 'Originaltext bleibt', 't1');
        """
    )

    run_sqlite_migrations(connection)

    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"ai_version", "ai_version_segment"} <= tables
    assert connection.execute("SELECT text FROM transcription_segment WHERE id = 1").fetchone() == (
        "Originaltext bleibt",
    )
    connection.close()
