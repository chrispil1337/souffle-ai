# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

from buzz.ai.chunking import (
    NumberedSegment,
    check_cleaned_text,
    chunk_segments,
    extract_json,
    format_segment_line,
    format_timestamp,
    parse_cleanup_response,
    strip_speaker_prefix,
)


def seg(number, text="Hallo", speaker="", start=0):
    return NumberedSegment(number, number * 10, start, start + 1000, speaker, text)


class TestFormatting:
    @pytest.mark.parametrize(
        "ms,expected", [(0, "00:00"), (65_000, "01:05"), (3_725_000, "1:02:05"), (-5, "00:00")]
    )
    def test_timestamp(self, ms, expected):
        assert format_timestamp(ms) == expected

    def test_line_with_and_without_speaker(self):
        assert format_segment_line(seg(3, "Text", "Anna", 65_000)) == "[3] (01:05) Anna: Text"
        assert format_segment_line(seg(4, "Text")) == "[4] (00:00) Text"


class TestChunking:
    def test_keeps_segments_whole_and_in_order(self):
        segments = [seg(i, "x" * 40) for i in range(1, 11)]
        chunks = chunk_segments(segments, max_chars=150)
        assert [s.number for chunk in chunks for s in chunk] == list(range(1, 11))
        assert all(len(chunk) >= 1 for chunk in chunks)
        assert len(chunks) > 1

    def test_oversized_segment_gets_own_chunk(self):
        segments = [seg(1, "a"), seg(2, "b" * 500), seg(3, "c")]
        chunks = chunk_segments(segments, max_chars=100)
        assert [[s.number for s in c] for c in chunks] == [[1], [2], [3]]

    def test_empty(self):
        assert chunk_segments([], 100) == []


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_text_around(self):
        assert extract_json('Gerne! {"a": [1, 2]} Viel Erfolg.') == {"a": [1, 2]}

    def test_invalid(self):
        with pytest.raises(ValueError):
            extract_json("keine Daten")


class TestParseCleanupResponse:
    def test_json_segments(self):
        answer = '{"segments": [{"n": 1, "text": " Eins "}, {"n": "2", "text": "Zwei"}]}'
        assert parse_cleanup_response(answer) == {1: "Eins", 2: "Zwei"}

    def test_json_list_and_bad_items(self):
        answer = '[{"n": 1, "text": "Eins"}, {"n": "x", "text": "?"}, "Unsinn", {"n": 3}]'
        assert parse_cleanup_response(answer) == {1: "Eins"}

    def test_json_mapping(self):
        assert parse_cleanup_response('{"1": "Eins", "2": "Zwei"}') == {1: "Eins", 2: "Zwei"}

    def test_numbered_text_fallback(self):
        answer = "[1] (00:01) Eins\n[2] Zwei\nmit Zeilenumbruch"
        assert parse_cleanup_response(answer) == {1: "Eins", 2: "Zwei\nmit Zeilenumbruch"}

    def test_first_answer_for_a_number_wins(self):
        assert parse_cleanup_response('[{"n":1,"text":"A"},{"n":1,"text":"B"}]') == {1: "A"}


class TestChecks:
    def test_missing(self):
        assert check_cleaned_text("Text", None) == "missing"

    def test_short_original_may_become_empty(self):
        assert check_cleaned_text("Äh, ja.", "") is None
        assert check_cleaned_text("Ähm, also, ja, genau.", "Genau.") is None

    def test_long_original_much_shorter_is_suspicious(self):
        original = "Wir müssen das Dach im Obergeschoss bis Freitag komplett abdichten lassen."
        assert check_cleaned_text(original, "Dach.") == "suspicious"

    def test_much_longer_is_suspicious(self):
        assert check_cleaned_text("Kurz.", "Kurz. " + "Erfunden " * 20) == "suspicious"

    def test_normal_cleanup_is_fine(self):
        assert check_cleaned_text("Äh also das ist halt so", "Das ist so.") is None

    def test_strip_speaker_prefix(self):
        assert strip_speaker_prefix("Anna: Hallo", "Anna") == "Hallo"
        assert strip_speaker_prefix("Anna: Hallo", "") == "Anna: Hallo"
