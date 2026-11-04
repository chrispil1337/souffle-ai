# SPDX-License-Identifier: GPL-3.0-or-later
import uuid

import pytest

from buzz.db.entity.transcription import Transcription
from buzz.db.service.ai_version_service import AIVersionService
from buzz.transcriber.transcriber import Segment

STORED_SEGMENTS = [
    Segment(start=0, end=4000, text="Äh also das Dach ist halt undicht.", speaker="Anna"),
    Segment(start=4000, end=9000, text="Ähm wir sollten den Dachdecker anrufen.", speaker="Bernd"),
]


@pytest.fixture
def ai_service(db) -> AIVersionService:
    return AIVersionService(db)


@pytest.fixture
def stored_transcription(transcription_dao, transcription_service):
    """A completed transcription with two segments in the test database."""
    transcription_id = uuid.uuid4()
    transcription_dao.insert(
        Transcription(
            id=str(transcription_id),
            file="/tmp/baustelle.wav",
            status="in_progress",
            time_queued="2026-11-04T10:00:00",
            task="transcribe",
            model_type="Whisper",
            name="Baustelle",
        )
    )
    transcription_service.update_transcription_as_completed(transcription_id, STORED_SEGMENTS)
    return transcription_id
