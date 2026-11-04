# SPDX-License-Identifier: GPL-3.0-or-later
"""Storage of AI post-processing versions.

Kept separate from ``TranscriptionService`` so the original transcript code
paths stay untouched. Uses the same (main-thread bound) database connection.
"""

import json
from typing import List, Optional

from PyQt6.QtSql import QSqlDatabase

from buzz.db.dao.ai_version_dao import AIVersionDAO, AIVersionSegmentDAO
from buzz.db.dao.dao import transaction
from buzz.db.entity.ai_version import AIVersion, AIVersionSegment, AIVersionStatus


class AIVersionService:
    def __init__(self, db: QSqlDatabase):
        self.db = db
        self.version_dao = AIVersionDAO(db)
        self.segment_dao = AIVersionSegmentDAO(db)

    @classmethod
    def from_transcription_service(cls, transcription_service) -> "AIVersionService":
        return cls(transcription_service.transcription_dao.db)

    def create_version(self, version: AIVersion) -> int:
        return self.version_dao.create(version)

    def get_version(self, version_id: int) -> Optional[AIVersion]:
        return self.version_dao.get(version_id)

    def list_versions(self, transcription_id) -> List[AIVersion]:
        return self.version_dao.list_for_transcription(str(transcription_id))

    def latest_completed_version(self, transcription_id) -> Optional[AIVersion]:
        for version in self.list_versions(transcription_id):
            if version.status == AIVersionStatus.COMPLETED.value:
                return version
        return None

    def get_segments(self, version_id: int) -> List[AIVersionSegment]:
        return self.segment_dao.get_segments(version_id)

    def set_status(
        self, version_id: int, status: AIVersionStatus, error_message: str = ""
    ) -> None:
        version = self.version_dao.get(version_id)
        if version is None:
            return
        version.status = status.value
        version.error_message = error_message
        self.version_dao.update(version)

    def save_result(
        self,
        version_id: int,
        segments: List[AIVersionSegment],
        document_markdown: str,
        analysis: dict,
        prompt_snapshot: str,
    ) -> None:
        """Store a finished run in one transaction."""
        with transaction(self.db):
            version = self.version_dao.get(version_id)
            if version is None:
                raise ValueError(f"AI version {version_id} not found")
            self.segment_dao.delete_segments(version_id)
            for position, segment in enumerate(segments):
                segment.ai_version_id = version_id
                segment.position = position
                self.segment_dao.insert_segment(segment)
            version.document_markdown = document_markdown
            version.analysis_json = json.dumps(analysis, ensure_ascii=False)
            version.prompt_snapshot = prompt_snapshot
            version.status = AIVersionStatus.COMPLETED.value
            version.error_message = ""
            self.version_dao.update(version)

    def update_segment_text(self, version_id: int, segment_id: int, text: str) -> None:
        """Manual edit of one AI segment."""
        with transaction(self.db):
            self.segment_dao.update_text(segment_id, text)
            self._mark_edited(version_id)

    def update_document(self, version_id: int, document_markdown: str) -> None:
        """Manual edit of the AI document."""
        version = self.version_dao.get(version_id)
        if version is None:
            return
        version.document_markdown = document_markdown
        version.manually_edited = True
        self.version_dao.update(version)

    def delete_version(self, version_id: int) -> None:
        with transaction(self.db):
            self.segment_dao.delete_segments(version_id)
            self.version_dao.delete(version_id)

    def mark_interrupted_as_failed(self, message: str) -> None:
        self.version_dao.mark_interrupted_as_failed(message)

    def _mark_edited(self, version_id: int) -> None:
        version = self.version_dao.get(version_id)
        if version is not None and not version.manually_edited:
            version.manually_edited = True
            self.version_dao.update(version)
