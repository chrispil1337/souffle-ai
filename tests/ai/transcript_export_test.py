# SPDX-License-Identifier: GPL-3.0-or-later
import json
import os
from types import SimpleNamespace

import pytest

from buzz.db.entity.ai_version import AIVersion, AIVersionSegment
from buzz.export.transcript_export import (
    ExportKind,
    build_export,
    default_export_path,
    write_export,
)


@pytest.fixture
def transcription():
    return SimpleNamespace(
        id="1234", name=None, file="/data/Baustelle Müller.wav", language="de",
        whisper_model_size="large-v3-turbo", hugging_face_model_id=None,
        time_queued="2026-11-04T10:00:00", time_ended="2026-11-04T10:05:00",
    )


@pytest.fixture
def segments():
    return [
        SimpleNamespace(start_time=0, end_time=4000, speaker="Anna", text="Äh das Dach ist undicht."),
        SimpleNamespace(start_time=4000, end_time=8000, speaker="Anna", text="Wirklich."),
        SimpleNamespace(start_time=8000, end_time=9000, speaker="", text="Ohne Sprecher."),
    ]


@pytest.fixture
def version():
    return AIVersion(
        id=7, transcription_id="1234", profile_id="site_inspection",
        profile_name="Baustellenbesichtigung", provider="anthropic", model="claude-opus-5-5",
        status="completed", document_markdown="# Besichtigung\n\n## Summary\n\nDach undicht.",
        analysis_json=json.dumps({"summary": "Dach undicht.", "tasks": []}),
        manually_edited=True, time_created="2026-11-04T10:06:00",
    )


@pytest.fixture
def ai_segments():
    return [AIVersionSegment(7, "Äh das Dach ist undicht.", "Das Dach ist undicht.",
                             start_time=0, end_time=4000, speaker="Anna")]


def test_original_markdown(transcription, segments):
    md = build_export(ExportKind.ORIGINAL_MARKDOWN, transcription, segments)
    assert md.startswith("# Baustelle Müller\n")
    assert "- File: Baustelle Müller.wav" in md
    assert "- Date: 04.11.2026 10:05" in md
    assert "- Transcription model: large-v3-turbo" in md
    assert "**Anna** (00:00): Äh das Dach ist undicht. Wirklich." in md
    assert "(00:08) Ohne Sprecher." in md


def test_ai_markdown_has_provenance(transcription, version):
    md = build_export(ExportKind.AI_MARKDOWN, transcription, [], version)
    assert md.startswith("# Besichtigung")
    assert "please check before use" in md
    assert "- AI model: anthropic / claude-opus-5-5" in md
    assert "- Edited manually: yes" in md


def test_combined_markdown_keeps_both_apart(transcription, segments, version):
    md = build_export(ExportKind.COMBINED_MARKDOWN, transcription, segments, version)
    assert md.startswith("# Baustelle Müller")
    ai_part, original_part = md.split("## Original transcript")
    assert "## AI version" in ai_part
    assert "### Summary" in ai_part  # document headings are demoted
    assert "# Besichtigung" not in ai_part
    assert "Äh das Dach ist undicht." in original_part


def test_combined_json(transcription, segments, version, ai_segments):
    data = json.loads(build_export(ExportKind.COMBINED_JSON, transcription, segments, version, ai_segments))
    assert data["schema_version"] == 1
    assert data["transcription"]["name"] == "Baustelle Müller"
    assert data["original"]["segments"][0] == {
        "start_ms": 0, "end_ms": 4000, "start": "00:00", "speaker": "Anna",
        "text": "Äh das Dach ist undicht.",
    }
    ai = data["ai_version"]
    assert ai["model"] == "claude-opus-5-5"
    assert ai["manually_edited"] is True
    assert ai["segments"][0]["text"] == "Das Dach ist undicht."
    assert ai["segments"][0]["original_text"] == "Äh das Dach ist undicht."
    assert ai["analysis"]["summary"] == "Dach undicht."


def test_original_json_has_no_ai_version(transcription, segments):
    data = json.loads(build_export(ExportKind.ORIGINAL_JSON, transcription, segments))
    assert data["ai_version"] is None
    assert len(data["original"]["segments"]) == 3


def test_ai_exports_need_a_version(transcription, segments):
    with pytest.raises(ValueError):
        build_export(ExportKind.AI_MARKDOWN, transcription, segments)


def test_default_path_and_write(transcription, tmp_path):
    assert default_export_path(transcription, ExportKind.COMBINED_JSON) == os.path.join(
        "/data", "Baustelle Müller (AI + original).json"
    )
    path = default_export_path(transcription, ExportKind.AI_MARKDOWN, str(tmp_path))
    write_export(path, "Inhalt äöü\n")
    assert open(path, encoding="utf-8").read() == "Inhalt äöü\n"
