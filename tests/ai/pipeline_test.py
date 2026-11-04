# SPDX-License-Identifier: GPL-3.0-or-later
import json
from types import SimpleNamespace

import pytest

from buzz.ai.chunking import NumberedSegment
from buzz.ai.pipeline import (
    BASE_RULES,
    PipelineCanceled,
    PostProcessingPipeline,
    document_system_prompt,
    normalize_analysis,
    numbered_segments_from_records,
    render_markdown,
)
from buzz.ai.profiles import QUOTE_PREPARATION_ID, ProfileStore
from buzz.ai.providers.base import AIResponseError
from tests.ai.fakes import FakeProvider

ORIGINALS = [
    ("Anna", "Äh also das Dach ist halt undicht."),
    ("Bernd", "Ähm wir sollten den Dachdecker anrufen."),
    ("Anna", ""),
    ("Anna", "Also gut, dann machen wir das so."),
]


def segments():
    return [
        NumberedSegment(i, 100 + i, i * 10_000, i * 10_000 + 5_000, speaker, text)
        for i, (speaker, text) in enumerate(ORIGINALS, start=1)
    ]


@pytest.fixture
def profiles(tmp_path):
    return ProfileStore(str(tmp_path / "profiles.json"))


def run(provider, profile, **kwargs):
    progress = []
    pipeline = PostProcessingPipeline(
        provider, "fake-model", profile, progress=lambda f, m: progress.append(f), **kwargs
    )
    return pipeline.run(segments(), title="Test"), progress


class TestPipeline:
    def test_cleans_segments_and_keeps_links(self, profiles):
        result, _ = run(FakeProvider(), profiles.get_or_default(""))

        texts = [s.text for s in result.segments]
        assert texts == [
            "Das Dach ist undicht.",
            "Wir sollten den Dachdecker anrufen.",
            "",
            "Gut, dann machen wir das so.",
        ]
        assert [s.original_text for s in result.segments] == [t for _, t in ORIGINALS]
        assert [s.segment_id for s in result.segments] == [101, 102, 103, 104]
        assert [s.speaker for s in result.segments] == ["Anna", "Bernd", "Anna", "Anna"]
        assert all(s.flag == "" for s in result.segments)

    def test_empty_segments_are_not_sent(self, profiles):
        provider = FakeProvider()
        run(provider, profiles.get_or_default(""))
        cleanup_request = provider.requests[0]
        assert "[3]" not in cleanup_request.user
        assert cleanup_request.json_mode is True
        assert BASE_RULES in cleanup_request.system

    def test_missing_segment_keeps_original(self, profiles):
        result, _ = run(FakeProvider(drop_numbers={2}), profiles.get_or_default(""))
        second = result.segments[1]
        assert second.text == ORIGINALS[1][1]
        assert second.flag == "missing"
        assert any("unchanged" in w for w in result.warnings)

    def test_suspicious_cleanup_keeps_original(self, profiles):
        result, _ = run(FakeProvider(cleanup=lambda t: "x"), profiles.get_or_default(""))
        assert result.segments[0].flag == "suspicious"
        assert result.segments[0].text == ORIGINALS[0][1]
        assert result.segments[1].text == ORIGINALS[1][1]

    def test_document_items_are_checked_against_sources(self, profiles):
        result, _ = run(FakeProvider(), profiles.get_or_default(""))
        tasks = result.analysis["tasks"]
        assert tasks[0] == {
            "text": "Dachdecker anrufen",
            "sources": [1],
            "verified": True,
            "owner": "Anna",
            "due": "",
        }
        assert tasks[1]["verified"] is False
        assert tasks[1]["sources"] == []
        assert any("no source" in w for w in result.warnings)
        assert "⚠" in result.document_markdown

    def test_markdown_document(self, profiles):
        result, _ = run(FakeProvider(), profiles.get_or_default(""))
        md = result.document_markdown
        assert md.startswith("# Besprechung")
        assert "## Summary" in md
        assert "- [ ] Dachdecker anrufen — Responsible: Anna _(Source: 00:10)_" in md
        assert "## Cleaned transcript" in md
        assert "**Bernd** (00:20): Wir sollten den Dachdecker anrufen." in md
        # consecutive segments of one speaker form one paragraph
        assert "**Anna** (00:40): Gut, dann machen wir das so." in md

    def test_progress_reaches_one(self, profiles):
        _, progress = run(FakeProvider(), profiles.get_or_default(""))
        assert progress[0] == 0.0
        assert progress[-1] == pytest.approx(1.0)
        assert progress == sorted(progress)

    def test_long_transcripts_are_chunked_and_merged(self, profiles):
        provider = FakeProvider()
        result, progress = run(
            provider,
            profiles.get_or_default(""),
            cleanup_chunk_chars=60,
            document_chunk_chars=80,
        )
        kinds = [
            "cleanup" if "Bereinige" in r.system else "merge" if "Teilergebnisse" in r.system else "doc"
            for r in provider.requests
        ]
        assert kinds.count("cleanup") > 1
        assert kinds.count("doc") > 1
        assert kinds[-1] == "merge"
        assert result.analysis["summary"]
        assert progress[-1] == pytest.approx(1.0)

    def test_profile_sections_are_requested(self, profiles):
        profile = profiles.get(QUOTE_PREPARATION_ID)
        prompt = document_system_prompt(profile)
        for title in profile.sections:
            assert title in prompt
        provider = FakeProvider()
        run(provider, profile)
        assert profile.document_prompt in provider.requests[-1].system

    def test_prompt_snapshot_is_recorded(self, profiles):
        result, _ = run(FakeProvider(), profiles.get_or_default(""))
        snapshot = json.loads(result.prompt_snapshot)
        assert BASE_RULES in snapshot["cleanup_system_prompt"]
        assert snapshot["profile_id"] == "general_cleanup"

    def test_invalid_json_is_retried_once_then_fails(self, profiles):
        class BrokenDocument(FakeProvider):
            def _complete(self, request):
                if "strukturiertes Dokument" in request.system:
                    self.requests.append(request)
                    return "Das kann ich leider nicht."
                return super()._complete(request)

        provider = BrokenDocument()
        with pytest.raises(AIResponseError):
            run(provider, profiles.get_or_default(""))
        assert sum("strukturiertes Dokument" in r.system for r in provider.requests) == 2

    def test_cancel(self, profiles):
        pipeline = PostProcessingPipeline(
            FakeProvider(), "m", profiles.get_or_default(""), is_canceled=lambda: True
        )
        with pytest.raises(PipelineCanceled):
            pipeline.run(segments())

    def test_empty_transcript(self, profiles):
        pipeline = PostProcessingPipeline(FakeProvider(), "m", profiles.get_or_default(""))
        with pytest.raises(AIResponseError):
            pipeline.run([NumberedSegment(1, 1, 0, 0, "", "")])


class TestHelpers:
    def test_numbered_segments_from_records_sorts_by_time(self):
        records = [
            SimpleNamespace(id=2, start_time=5000, end_time=6000, speaker=None, text=" B "),
            SimpleNamespace(id=1, start_time=0, end_time=1000, speaker="Anna", text="A"),
        ]
        numbered = numbered_segments_from_records(records)
        assert [(n.number, n.segment_id, n.text, n.speaker) for n in numbered] == [
            (1, 1, "A", "Anna"),
            (2, 2, "B", ""),
        ]

    def test_normalize_analysis_handles_garbage(self):
        analysis = normalize_analysis(
            {
                "summary": ["Satz eins.", "Satz zwei."],
                "topics": "Ein Thema",
                "tasks": ["Nur Text", {"text": ""}, {"text": "Mit Quelle", "sources": ["2", 2, "x"]}],
                "sections": [{"title": "A", "content": "B", "sources": 1}],
                "decisions": "kein Array",
            },
            valid_numbers={1, 2},
        )
        assert analysis["summary"] == "Satz eins.\nSatz zwei."
        assert analysis["topics"] == ["Ein Thema"]
        assert [t["text"] for t in analysis["tasks"]] == ["Nur Text", "Mit Quelle"]
        assert analysis["tasks"][1]["sources"] == [2]
        assert analysis["tasks"][0]["verified"] is False
        assert analysis["sections"] == [{"title": "A", "content": "B", "sources": [1]}]
        assert analysis["decisions"] == []
        assert normalize_analysis("Unsinn", set())["tasks"] == []

    def test_render_markdown_lists_none_mentioned(self):
        md = render_markdown(normalize_analysis({}, set()), [], title="Leer")
        assert md.startswith("# Leer")
        assert md.count("None mentioned.") == 3
