# SPDX-License-Identifier: GPL-3.0-or-later
from buzz.ai.diff import change_ratio, segments_diff_html, word_diff, word_diff_html


def test_word_diff_marks_deletions_and_insertions():
    pieces = word_diff("Äh das ist halt gut", "Das ist gut")
    assert ("delete", "Äh das") in pieces or any(op == "delete" for op, _ in pieces)
    assert any(op == "insert" and "Das" in text for op, text in pieces)
    # joining equal + insert pieces gives the new text
    assert "".join(t for op, t in pieces if op != "delete") == "Das ist gut"
    assert "".join(t for op, t in pieces if op != "insert") == "Äh das ist halt gut"


def test_identical_texts():
    assert word_diff("gleich", "gleich") == [("equal", "gleich")]
    assert change_ratio("gleich", "gleich") == 0.0
    assert change_ratio("", "") == 0.0
    assert change_ratio("a b c", "x y z") > 0.9


def test_html_is_escaped():
    html = word_diff_html("<b>alt</b>", "<i>neu</i>")
    assert "<b>" not in html and "&lt;b&gt;" in html
    assert "line-through" in html


def test_segments_page():
    page = segments_diff_html([("00:01", "Anna", "äh ja", "Ja"), ("00:05", "", "Gleich", "Gleich")])
    assert page.startswith("<html>")
    assert "00:01 Anna" in page
    assert "color:gray;" in page  # unchanged row is grey
