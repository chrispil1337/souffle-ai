# SPDX-License-Identifier: GPL-3.0-or-later
import enum
from dataclasses import dataclass
from typing import Optional

from buzz.db.entity.entity import Entity


class AIVersionStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class AIVersion(Entity):
    """One AI post-processing run of a transcription."""

    transcription_id: str
    profile_id: str = ""
    profile_name: str = ""
    provider: str = ""
    model: str = ""
    prompt_snapshot: str = ""
    status: str = AIVersionStatus.PENDING.value
    error_message: str = ""
    document_markdown: str = ""
    analysis_json: str = "{}"
    manually_edited: bool = False
    time_created: Optional[str] = None
    time_updated: Optional[str] = None
    id: int = -1


class AISegmentFlag(enum.Enum):
    NONE = ""
    # The AI did not return this segment; the original text was kept
    MISSING = "missing"
    # The AI result looked wrong (e.g. much shorter); the original was kept
    SUSPICIOUS = "suspicious"


@dataclass
class AIVersionSegment(Entity):
    """The AI-cleaned text of one original segment."""

    ai_version_id: int
    original_text: str
    text: str
    position: int = 0
    segment_id: int = -1
    start_time: int = 0
    end_time: int = 0
    speaker: str = ""
    flag: str = AISegmentFlag.NONE.value
    id: int = -1
