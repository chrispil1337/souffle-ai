# SPDX-License-Identifier: GPL-3.0-or-later
"""Split transcripts into requests and parse the AI answers defensively.

Segments are numbered ``[1]``, ``[2]``, ... across the whole transcript. The
numbers let the AI refer back to its sources and let us match every cleaned
text to its original segment. A segment is never split across requests.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Roughly 3–4 characters per token for German/English text. Requests stay well
# below the context window of small local models as well.
DEFAULT_CLEANUP_CHUNK_CHARS = 8000
DEFAULT_DOCUMENT_CHUNK_CHARS = 60000

# A cleaned segment shorter than this share of the original is suspicious
# (unless the original is short, e.g. "Äh, ja, genau.")
MIN_LENGTH_RATIO = 0.3
MIN_LENGTH_CHECK_CHARS = 25
MAX_LENGTH_RATIO = 2.0
MAX_LENGTH_SLACK_CHARS = 40


@dataclass
class NumberedSegment:
    number: int  # 1-based, unique within the transcript
    segment_id: int
    start_time: int  # ms
    end_time: int  # ms
    speaker: str
    text: str


def format_timestamp(ms: int) -> str:
    seconds = max(0, int(ms) // 1000)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_segment_line(segment: NumberedSegment) -> str:
    speaker = f" {segment.speaker}:" if segment.speaker else ""
    return f"[{segment.number}] ({format_timestamp(segment.start_time)}){speaker} {segment.text}"


def format_segments(segments: Sequence[NumberedSegment]) -> str:
    return "\n".join(format_segment_line(s) for s in segments)


def chunk_segments(
    segments: Sequence[NumberedSegment], max_chars: int
) -> List[List[NumberedSegment]]:
    """Group consecutive segments so each group's text stays below ``max_chars``.

    A single segment longer than ``max_chars`` forms its own group.
    """
    chunks: List[List[NumberedSegment]] = []
    current: List[NumberedSegment] = []
    size = 0
    for segment in segments:
        line_size = len(format_segment_line(segment)) + 1
        if current and size + line_size > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(segment)
        size += line_size
    if current:
        chunks.append(current)
    return chunks


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str):
    """Parse the first JSON object or array in an AI answer.

    Handles code fences and text before/after the JSON. Raises ``ValueError``.
    """
    if text is None:
        raise ValueError("empty response")
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in response")
    start = min(starts)
    closing = "}" if cleaned[start] == "{" else "]"
    end = cleaned.rfind(closing)
    if end <= start:
        raise ValueError("no complete JSON found in response")
    return json.loads(cleaned[start : end + 1])


_NUMBERED_LINE_RE = re.compile(r"\[(\d+)\]\s*")


def parse_numbered_text(text: str) -> Dict[int, str]:
    """Fallback parser for answers in the ``[1] text`` format."""
    parts = _NUMBERED_LINE_RE.split(text or "")
    result: Dict[int, str] = {}
    for i in range(1, len(parts) - 1, 2):
        number = int(parts[i])
        content = parts[i + 1].strip()
        # Drop a repeated "(00:12) Speaker:" prefix if the AI echoed it
        content = re.sub(r"^\(\d{1,2}:\d{2}(?::\d{2})?\)\s*", "", content)
        result.setdefault(number, content)
    return result


def parse_cleanup_response(text: str) -> Dict[int, str]:
    """Return ``{segment number: cleaned text}`` from an AI cleanup answer."""
    try:
        data = extract_json(text)
    except ValueError:
        return parse_numbered_text(text)

    items = data.get("segments") if isinstance(data, dict) else data
    result: Dict[int, str] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            number = item.get("n", item.get("number", item.get("id")))
            content = item.get("text")
            try:
                number = int(number)
            except (TypeError, ValueError):
                continue
            if isinstance(content, str):
                result.setdefault(number, content.strip())
    elif isinstance(data, dict):
        # {"1": "text", "2": "text"}
        for key, value in data.items():
            if str(key).isdigit() and isinstance(value, str):
                result.setdefault(int(key), value.strip())
    return result


def strip_speaker_prefix(text: str, speaker: str) -> str:
    """Remove a ``Speaker:`` prefix the AI may have repeated."""
    if speaker and text.startswith(f"{speaker}:"):
        return text[len(speaker) + 1 :].lstrip()
    return text


def check_cleaned_text(original: str, cleaned: Optional[str]) -> Optional[str]:
    """Return a problem description if ``cleaned`` should not be trusted."""
    if cleaned is None:
        return "missing"
    original_len = len(original.strip())
    cleaned_len = len(cleaned.strip())
    if original_len >= MIN_LENGTH_CHECK_CHARS and cleaned_len < original_len * MIN_LENGTH_RATIO:
        return "suspicious"
    if cleaned_len > original_len * MAX_LENGTH_RATIO + MAX_LENGTH_SLACK_CHARS:
        return "suspicious"
    return None
