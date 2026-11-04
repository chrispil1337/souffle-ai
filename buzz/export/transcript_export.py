# SPDX-License-Identifier: GPL-3.0-or-later
"""Markdown and JSON export of transcripts and their AI versions.

The original transcript and the AI version are always kept apart in the
output, so a reader can tell what was said from what the AI wrote.
"""

import enum
import json
import os
from datetime import datetime
from typing import List, Optional, Sequence

from buzz.__version__ import VERSION
from buzz.ai.chunking import format_timestamp
from buzz.ai.pipeline import render_transcript_paragraphs
from buzz.locale import _

JSON_SCHEMA_VERSION = 1


class ExportKind(enum.Enum):
    ORIGINAL_MARKDOWN = "original_markdown"
    AI_MARKDOWN = "ai_markdown"
    COMBINED_MARKDOWN = "combined_markdown"
    COMBINED_JSON = "combined_json"
    ORIGINAL_JSON = "original_json"

    @property
    def extension(self) -> str:
        return "json" if self in (ExportKind.COMBINED_JSON, ExportKind.ORIGINAL_JSON) else "md"

    @property
    def needs_ai_version(self) -> bool:
        return self in (ExportKind.AI_MARKDOWN, ExportKind.COMBINED_MARKDOWN, ExportKind.COMBINED_JSON)


def transcription_title(transcription) -> str:
    if transcription.name:
        return transcription.name
    if transcription.file:
        return os.path.splitext(os.path.basename(transcription.file))[0]
    return _("Transcript")


def _format_date(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return str(value)


def _metadata_lines(transcription) -> List[str]:
    lines = []
    if transcription.file:
        lines.append(f"- {_('File')}: {os.path.basename(transcription.file)}")
    date = _format_date(transcription.time_ended or transcription.time_queued)
    if date:
        lines.append(f"- {_('Date')}: {date}")
    model = transcription.whisper_model_size or transcription.hugging_face_model_id
    if model:
        lines.append(f"- {_('Transcription model')}: {model}")
    if transcription.language:
        lines.append(f"- {_('Language')}: {transcription.language}")
    return lines


def _ai_metadata_lines(version) -> List[str]:
    lines = [
        f"- {_('Profile')}: {version.profile_name or version.profile_id}",
        f"- {_('AI model')}: {version.provider} / {version.model}",
        f"- {_('Created')}: {_format_date(version.time_created)}",
    ]
    if version.manually_edited:
        lines.append(f"- {_('Edited manually')}: {_('yes')}")
    return lines


def _without_title(markdown: str) -> str:
    lines = (markdown or "").strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _demote_headings(markdown: str) -> str:
    """Shift headings one level down so the document fits below a section."""
    out = []
    for line in markdown.splitlines():
        out.append("#" + line if line.startswith("#") else line)
    return "\n".join(out)


def original_markdown(transcription, segments: Sequence) -> str:
    lines = [f"# {transcription_title(transcription)}", ""]
    lines += _metadata_lines(transcription) + [""]
    lines += [f"## {_('Original transcript')}", ""]
    lines += render_transcript_paragraphs(segments)
    return "\n".join(lines).rstrip() + "\n"


def ai_markdown(transcription, version) -> str:
    document = (version.document_markdown or "").strip()
    lines = [document, "", "---", "", f"_{_('Created with AI post-processing; please check before use.')}_", ""]
    lines += _metadata_lines(transcription) + _ai_metadata_lines(version)
    return "\n".join(lines).rstrip() + "\n"


def combined_markdown(transcription, segments: Sequence, version) -> str:
    lines = [f"# {transcription_title(transcription)}", ""]
    lines += _metadata_lines(transcription) + [""]
    lines += [f"## {_('AI version')}", ""]
    lines += _ai_metadata_lines(version) + [""]
    lines += [_demote_headings(_without_title(version.document_markdown)), ""]
    lines += [f"## {_('Original transcript')}", ""]
    lines += render_transcript_paragraphs(segments)
    return "\n".join(lines).rstrip() + "\n"


def _segment_dict(segment) -> dict:
    return {
        "start_ms": int(segment.start_time or 0),
        "end_ms": int(segment.end_time or 0),
        "start": format_timestamp(segment.start_time or 0),
        "speaker": (getattr(segment, "speaker", "") or ""),
        "text": segment.text or "",
    }


def transcript_json(
    transcription,
    segments: Sequence,
    version=None,
    ai_segments: Optional[Sequence] = None,
) -> dict:
    data = {
        "schema_version": JSON_SCHEMA_VERSION,
        "generator": f"Soufflé AI (Buzz {VERSION})",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "transcription": {
            "id": transcription.id,
            "name": transcription_title(transcription),
            "file": transcription.file,
            "language": transcription.language,
            "model": transcription.whisper_model_size or transcription.hugging_face_model_id,
            "created": transcription.time_queued,
            "completed": transcription.time_ended,
        },
        "original": {"segments": [_segment_dict(s) for s in segments]},
        "ai_version": None,
    }
    if version is not None:
        try:
            analysis = json.loads(version.analysis_json or "{}")
        except ValueError:
            analysis = {}
        ai_segment_dicts = []
        for segment in ai_segments or []:
            entry = _segment_dict(segment)
            entry["original_text"] = segment.original_text
            entry["flag"] = segment.flag or ""
            ai_segment_dicts.append(entry)
        data["ai_version"] = {
            "id": version.id,
            "profile_id": version.profile_id,
            "profile_name": version.profile_name,
            "provider": version.provider,
            "model": version.model,
            "created": version.time_created,
            "updated": version.time_updated,
            "manually_edited": bool(version.manually_edited),
            "segments": ai_segment_dicts,
            "analysis": analysis,
            "document_markdown": version.document_markdown,
        }
    return data


def build_export(
    kind: ExportKind,
    transcription,
    segments: Sequence,
    version=None,
    ai_segments: Optional[Sequence] = None,
) -> str:
    if kind.needs_ai_version and version is None:
        raise ValueError("This export needs an AI version")
    if kind == ExportKind.ORIGINAL_MARKDOWN:
        return original_markdown(transcription, segments)
    if kind == ExportKind.AI_MARKDOWN:
        return ai_markdown(transcription, version)
    if kind == ExportKind.COMBINED_MARKDOWN:
        return combined_markdown(transcription, segments, version)
    if kind == ExportKind.ORIGINAL_JSON:
        return json.dumps(transcript_json(transcription, segments), ensure_ascii=False, indent=2) + "\n"
    if kind == ExportKind.COMBINED_JSON:
        return (
            json.dumps(
                transcript_json(transcription, segments, version, ai_segments),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    raise ValueError(kind)


def default_export_path(transcription, kind: ExportKind, directory: Optional[str] = None) -> str:
    suffix = {
        ExportKind.ORIGINAL_MARKDOWN: "",
        ExportKind.ORIGINAL_JSON: "",
        ExportKind.AI_MARKDOWN: " (AI)",
        ExportKind.COMBINED_MARKDOWN: " (AI + original)",
        ExportKind.COMBINED_JSON: " (AI + original)",
    }[kind]
    base = transcription_title(transcription)
    base = "".join(c for c in base if c not in '<>:"/\\|?*').strip() or "transcript"
    if directory is None:
        directory = os.path.dirname(transcription.file) if transcription.file else os.path.expanduser("~")
    return os.path.join(directory, f"{base}{suffix}.{kind.extension}")


def write_export(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
