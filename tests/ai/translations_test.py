# SPDX-License-Identifier: GPL-3.0-or-later
"""All user-facing strings of the AI features have a German translation."""

import ast
import glob
import os
import re

import polib
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AI_SOURCES = [
    "buzz/ai/*.py",
    "buzz/ai/providers/*.py",
    "buzz/export/*.py",
    "buzz/widgets/ai/*.py",
    "buzz/widgets/preferences_dialog/ai_preferences_widget.py",
]
PLACEHOLDER = re.compile(r"\{[a-z_]*\}")


def ai_msgids():
    msgids = set()
    for pattern in AI_SOURCES:
        for path in glob.glob(os.path.join(ROOT, pattern)):
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    msgids.add(node.args[0].value)
    return msgids


@pytest.fixture(scope="module")
def german():
    po = polib.pofile(os.path.join(ROOT, "buzz/locale/de_DE/LC_MESSAGES/buzz.po"))
    return {entry.msgid: entry.msgstr for entry in po}


def test_ai_strings_are_translated_to_german(german):
    msgids = ai_msgids()
    assert len(msgids) > 100
    missing = sorted(m for m in msgids if not german.get(m))
    assert missing == []


def test_placeholders_are_kept(german):
    for msgid in ai_msgids():
        msgstr = german.get(msgid, "")
        assert sorted(PLACEHOLDER.findall(msgid)) == sorted(PLACEHOLDER.findall(msgstr)), msgid
