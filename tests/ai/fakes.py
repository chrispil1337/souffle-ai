# SPDX-License-Identifier: GPL-3.0-or-later
"""Test doubles for the AI pipeline."""

import json
import re
import threading
from typing import Callable, List, Optional

from buzz.ai.providers.base import CompletionRequest, LLMProvider

FILLERS = re.compile(r"\b(äh|ähm|also|halt)\b,?\s*", re.IGNORECASE)
LINE = re.compile(r"^\[(\d+)\] \([^)]*\)(?: ([^:]+):)? (.*)$")


def fake_cleanup(text: str) -> str:
    cleaned = FILLERS.sub("", text).strip()
    return cleaned[:1].upper() + cleaned[1:]


def parse_lines(user: str):
    for line in user.splitlines():
        match = LINE.match(line)
        if match:
            yield int(match.group(1)), match.group(2) or "", match.group(3)


def default_document(numbers: List[int]) -> dict:
    return {
        "title": "Besprechung",
        "summary": "Es wurde über das Dach gesprochen.",
        "topics": ["Dach"],
        "sections": [],
        "conversation": [
            {"heading": "Dach", "content": "Das Dach ist undicht.", "sources": numbers[:1]}
        ],
        "tasks": [
            {"text": "Dachdecker anrufen", "owner": "Anna", "due": "", "sources": numbers[:1]},
            {"text": "Erfundene Aufgabe", "owner": "", "due": "", "sources": [999]},
        ],
        "decisions": [{"text": "Reparatur beauftragen", "sources": numbers[-1:]}],
        "open_questions": [],
    }


class FakeProvider(LLMProvider):
    """Answers cleanup and document requests like a well-behaved model."""

    display_name = "Fake"
    is_cloud = False

    def __init__(
        self,
        cleanup: Callable[[str], str] = fake_cleanup,
        document: Optional[Callable[[List[int]], dict]] = None,
        drop_numbers=(),
        block: Optional[threading.Event] = None,
    ):
        self.cleanup = cleanup
        self.document = document or default_document
        self.drop_numbers = set(drop_numbers)
        self.requests: List[CompletionRequest] = []
        self.block = block

    def _complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        if self.block is not None:
            self.block.wait(5)
        if "Bereinige jedes nummerierte Segment" in request.system:
            segments = [
                {"n": n, "text": self.cleanup(text)}
                for n, _speaker, text in parse_lines(request.user)
                if n not in self.drop_numbers
            ]
            return "```json\n" + json.dumps({"segments": segments}, ensure_ascii=False) + "\n```"
        if "Teilergebnisse" in request.system:
            partials = json.loads(request.user)
            merged = dict(partials[0])
            for key in ("tasks", "decisions", "open_questions"):
                merged[key] = [item for p in partials for item in p.get(key, [])]
            return json.dumps(merged, ensure_ascii=False)
        numbers = [n for n, _speaker, _text in parse_lines(request.user)]
        return "Hier ist das Dokument:\n" + json.dumps(self.document(numbers), ensure_ascii=False)

    def list_models(self):
        return ["fake-model"]
