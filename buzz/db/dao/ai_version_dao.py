# SPDX-License-Identifier: GPL-3.0-or-later
from datetime import datetime
from typing import List, Optional

from PyQt6.QtSql import QSqlDatabase

from buzz.db.dao.dao import DAO
from buzz.db.entity.ai_version import AIVersion, AIVersionSegment


def _now() -> str:
    return datetime.now().isoformat()


def _insert_returning_id(dao: DAO, record) -> int:
    query = dao._create_query()
    fields = [f for f in record.__dict__.keys() if f not in dao.ignore_fields]
    query.prepare(
        f"""
        INSERT INTO {dao.table} ({", ".join(fields)})
        VALUES ({", ".join(f":{f}" for f in fields)})
        """
    )
    for field in fields:
        query.bindValue(f":{field}", getattr(record, field))
    if not query.exec():
        raise Exception(query.lastError().text())
    return int(query.lastInsertId())


class AIVersionDAO(DAO[AIVersion]):
    entity = AIVersion
    ignore_fields = ["id"]

    def __init__(self, db: QSqlDatabase):
        super().__init__("ai_version", db)

    def to_entity(self, record):
        entity = super().to_entity(record)
        # SQLite returns booleans as integers and NULL text as None
        entity.manually_edited = bool(entity.manually_edited)
        for name in (
            "profile_id",
            "profile_name",
            "provider",
            "model",
            "prompt_snapshot",
            "status",
            "error_message",
            "document_markdown",
        ):
            if getattr(entity, name) is None:
                setattr(entity, name, "")
        if not entity.analysis_json:
            entity.analysis_json = "{}"
        return entity

    def create(self, version: AIVersion) -> int:
        version.time_created = version.time_created or _now()
        version.time_updated = version.time_updated or version.time_created
        version.id = _insert_returning_id(self, version)
        return version.id

    def update(self, version: AIVersion) -> None:
        version.time_updated = _now()
        query = self._create_query()
        query.prepare(
            """
            UPDATE ai_version SET
                profile_id = :profile_id,
                profile_name = :profile_name,
                provider = :provider,
                model = :model,
                prompt_snapshot = :prompt_snapshot,
                status = :status,
                error_message = :error_message,
                document_markdown = :document_markdown,
                analysis_json = :analysis_json,
                manually_edited = :manually_edited,
                time_updated = :time_updated
            WHERE id = :id
            """
        )
        for field in (
            "profile_id",
            "profile_name",
            "provider",
            "model",
            "prompt_snapshot",
            "status",
            "error_message",
            "document_markdown",
            "analysis_json",
            "manually_edited",
            "time_updated",
            "id",
        ):
            query.bindValue(f":{field}", getattr(version, field))
        if not query.exec():
            raise Exception(query.lastError().text())

    def get(self, version_id: int) -> Optional[AIVersion]:
        return self.find_by_id(version_id)

    def list_for_transcription(self, transcription_id: str) -> List[AIVersion]:
        query = self._create_query()
        query.prepare(
            """
            SELECT * FROM ai_version
            WHERE transcription_id = :transcription_id
            ORDER BY time_created DESC, id DESC
            """
        )
        query.bindValue(":transcription_id", str(transcription_id))
        return self._execute_all(query)

    def delete(self, version_id: int) -> None:
        query = self._create_query()
        query.prepare("DELETE FROM ai_version WHERE id = :id")
        query.bindValue(":id", version_id)
        if not query.exec():
            raise Exception(query.lastError().text())

    def mark_interrupted_as_failed(self, message: str) -> None:
        """Runs that were in progress when the app closed can not resume."""
        query = self._create_query()
        query.prepare(
            """
            UPDATE ai_version SET status = 'failed', error_message = :message
            WHERE status IN ('pending', 'running')
            """
        )
        query.bindValue(":message", message)
        if not query.exec():
            raise Exception(query.lastError().text())


class AIVersionSegmentDAO(DAO[AIVersionSegment]):
    entity = AIVersionSegment
    ignore_fields = ["id"]

    def __init__(self, db: QSqlDatabase):
        super().__init__("ai_version_segment", db)

    def to_entity(self, record):
        entity = super().to_entity(record)
        entity.speaker = entity.speaker or ""
        entity.flag = entity.flag or ""
        return entity

    def insert_segment(self, segment: AIVersionSegment) -> int:
        segment.id = _insert_returning_id(self, segment)
        return segment.id

    def get_segments(self, ai_version_id: int) -> List[AIVersionSegment]:
        query = self._create_query()
        query.prepare(
            """
            SELECT * FROM ai_version_segment
            WHERE ai_version_id = :ai_version_id
            ORDER BY position, id
            """
        )
        query.bindValue(":ai_version_id", ai_version_id)
        return self._execute_all(query)

    def delete_segments(self, ai_version_id: int) -> None:
        query = self._create_query()
        query.prepare("DELETE FROM ai_version_segment WHERE ai_version_id = :id")
        query.bindValue(":id", ai_version_id)
        if not query.exec():
            raise Exception(query.lastError().text())

    def update_text(self, segment_id: int, text: str) -> None:
        query = self._create_query()
        query.prepare("UPDATE ai_version_segment SET text = :text WHERE id = :id")
        query.bindValue(":text", text)
        query.bindValue(":id", segment_id)
        if not query.exec():
            raise Exception(query.lastError().text())
