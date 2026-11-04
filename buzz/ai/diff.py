# SPDX-License-Identifier: GPL-3.0-or-later
"""Word level comparison of original and AI text, rendered as HTML."""

import difflib
import html
import re
from typing import List, Sequence, Tuple

_TOKEN_RE = re.compile(r"\s+|[^\s]+")

DELETED_STYLE = "background-color:#f8d7da;color:#842029;text-decoration:line-through;"
INSERTED_STYLE = "background-color:#d1e7dd;color:#0f5132;"


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def word_diff(original: str, changed: str) -> List[Tuple[str, str]]:
    """Return ``(op, text)`` pieces with op in ``equal``, ``delete``, ``insert``."""
    a, b = tokenize(original), tokenize(changed)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    pieces: List[Tuple[str, str]] = []
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == "equal":
            pieces.append(("equal", "".join(a[a1:a2])))
        else:
            if a1 != a2:
                pieces.append(("delete", "".join(a[a1:a2])))
            if b1 != b2:
                pieces.append(("insert", "".join(b[b1:b2])))
    return pieces


def word_diff_html(original: str, changed: str) -> str:
    out = []
    for op, text in word_diff(original, changed):
        escaped = html.escape(text)
        if op == "delete":
            out.append(f'<span style="{DELETED_STYLE}">{escaped}</span>')
        elif op == "insert":
            out.append(f'<span style="{INSERTED_STYLE}">{escaped}</span>')
        else:
            out.append(escaped)
    return "".join(out)


def change_ratio(original: str, changed: str) -> float:
    """Share of changed words (0 = identical, 1 = completely different)."""
    a, b = (original or "").split(), (changed or "").split()
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def segments_diff_html(rows: Sequence[Tuple[str, str, str, str]]) -> str:
    """Render ``(timestamp, speaker, original, changed)`` rows as an HTML page."""
    parts = ['<html><body style="font-family:sans-serif;">']
    for timestamp, speaker, original, changed in rows:
        label = html.escape(f"{timestamp} {speaker}".strip())
        if original == changed:
            body = html.escape(original)
            style = "color:gray;"
        else:
            body = word_diff_html(original, changed)
            style = ""
        parts.append(
            f'<p style="margin:4px 0;{style}"><b style="color:gray;">{label}</b>&nbsp; {body}</p>'
        )
    parts.append("</body></html>")
    return "".join(parts)
