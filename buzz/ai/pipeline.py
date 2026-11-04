# SPDX-License-Identifier: GPL-3.0-or-later
"""The AI post-processing pipeline.

Step 1 – cleanup: every segment is cleaned on its own (spelling, grammar,
filler words). The numbering keeps the link to the original segment and its
timestamps. Results that look wrong are replaced by the original text.

Step 2 – document: the cleaned transcript is turned into a structured
document (summary, topics, profile sections, structured conversation, tasks,
decisions, open questions). Every task, decision and open question must name
the segments it is based on; the references are checked against the
transcript so unsupported items are visible.

The pipeline runs on a worker thread. It never touches the database or Qt
widgets and never logs transcript text.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from buzz.ai.chunking import (
    DEFAULT_CLEANUP_CHUNK_CHARS,
    DEFAULT_DOCUMENT_CHUNK_CHARS,
    NumberedSegment,
    check_cleaned_text,
    chunk_segments,
    extract_json,
    format_segments,
    format_timestamp,
    parse_cleanup_response,
    strip_speaker_prefix,
)
from buzz.ai.profiles import Profile
from buzz.ai.providers.base import AIError, AIResponseError, CompletionRequest, LLMProvider
from buzz.locale import _

logger = logging.getLogger(__name__)

BASE_RULES = """Du bist ein sorgfältiger Redakteur für Gesprächstranskripte.
Diese Grundregeln gelten immer und haben Vorrang vor allen anderen Hinweisen:
- Erfinde keine Informationen. Verwende ausschließlich, was im Transkript steht.
- Verändere Aussagen nicht inhaltlich: keine neuen Fakten, keine Wertungen, keine Interpretationen.
- Übernimm Zahlen, Maße, Mengen, Beträge, Namen, Termine und Fachbegriffe exakt.
- Behalte die Zuordnung von Aussagen zu Sprechern bei.
- Rate nicht bei unverständlichen oder unklaren Stellen, sondern kennzeichne sie mit [unverständlich].
- Antworte in der Sprache des Transkripts."""

CLEANUP_TASK = """Aufgabe: Bereinige jedes nummerierte Segment einzeln.
- Korrigiere Rechtschreibung, Grammatik und Zeichensetzung.
- Entferne Füllwörter (z. B. äh, ähm, also, halt, sozusagen), Versprecher und unnötige Wiederholungen.
- Formuliere Umgangssprache verständlich, ohne den Sinn zu verändern.
- Fasse Segmente nicht zusammen und teile sie nicht auf. Jedes Segment behält seine Nummer.
- Wiederhole weder Zeitmarke noch Sprechername im Text.
- Ein Segment, das nur aus Füllwörtern besteht, darf leer werden ("").
Antworte ausschließlich mit JSON in diesem Format:
{"segments": [{"n": 1, "text": "bereinigter Text"}]}"""

DOCUMENT_TASK = """Aufgabe: Erstelle aus dem nummerierten, bereinigten Transkript ein strukturiertes Dokument.
- Strukturiere den Gesprächsverlauf nach Themen bzw. Abschnitten.
- Jeder Eintrag in "tasks", "decisions" und "open_questions" muss in "sources" die Nummern der Segmente angeben, auf denen er beruht.
- Nimm nur auf, was im Transkript tatsächlich vorkommt. Lasse Listen leer, wenn es dazu nichts gibt.
- Verantwortliche ("owner") und Fristen ("due") nur angeben, wenn sie genannt wurden, sonst "".
Antworte ausschließlich mit JSON in diesem Format:
{
  "title": "kurzer, sachlicher Titel",
  "summary": "Zusammenfassung in 3 bis 8 Sätzen",
  "topics": ["Thema"],
  "sections": [{"title": "Abschnittstitel", "content": "Text in Markdown", "sources": [1, 2]}],
  "conversation": [{"heading": "Thema", "content": "strukturierte Wiedergabe in Markdown", "sources": [1, 2]}],
  "tasks": [{"text": "Aufgabe", "owner": "", "due": "", "sources": [3]}],
  "decisions": [{"text": "Entscheidung", "sources": [5]}],
  "open_questions": [{"text": "offene Frage", "sources": [7]}]
}"""

MERGE_TASK = """Aufgabe: Du erhältst Teilergebnisse für aufeinanderfolgende Abschnitte desselben Gesprächs (als JSON-Liste).
Führe sie zu einem einzigen Dokument im selben JSON-Format zusammen.
- Entferne Doppelungen, erfinde nichts hinzu.
- Behalte alle Segmentnummern in "sources" bei.
- Schreibe eine gemeinsame Zusammenfassung für das ganze Gespräch.
Antworte ausschließlich mit JSON im selben Format wie die Teilergebnisse."""

ANALYSIS_LIST_KEYS = ("tasks", "decisions", "open_questions")


class PipelineCanceled(AIError):
    def __init__(self):
        super().__init__(_("AI post-processing was canceled."))


@dataclass
class CleanedSegment:
    number: int
    segment_id: int
    start_time: int
    end_time: int
    speaker: str
    original_text: str
    text: str
    flag: str = ""  # "", "missing" or "suspicious"


@dataclass
class PipelineResult:
    segments: List[CleanedSegment]
    analysis: dict
    document_markdown: str
    prompt_snapshot: str
    warnings: List[str] = field(default_factory=list)


def cleanup_system_prompt(profile: Profile) -> str:
    parts = [BASE_RULES, CLEANUP_TASK]
    if profile.cleanup_prompt.strip():
        parts.append("Hinweise zu diesem Gespräch:\n" + profile.cleanup_prompt.strip())
    return "\n\n".join(parts)


def document_system_prompt(profile: Profile) -> str:
    parts = [BASE_RULES, DOCUMENT_TASK]
    if profile.document_prompt.strip():
        parts.append("Schwerpunkt dieses Dokuments:\n" + profile.document_prompt.strip())
    if profile.sections:
        listed = "\n".join(f"- {title}" for title in profile.sections)
        parts.append(
            'Erstelle in "sections" genau diese Abschnitte in dieser Reihenfolge. '
            'Gibt es zu einem Abschnitt nichts im Transkript, schreibe als Inhalt '
            '"Keine Angaben im Gespräch.":\n' + listed
        )
    else:
        parts.append('Lasse "sections" leer.')
    return "\n\n".join(parts)


def merge_system_prompt(profile: Profile) -> str:
    return "\n\n".join([BASE_RULES, MERGE_TASK, document_system_prompt(profile)])


def numbered_segments_from_records(records: Sequence) -> List[NumberedSegment]:
    """Build numbered segments from ``TranscriptionSegment`` rows (or similar)."""
    ordered = sorted(records, key=lambda r: (int(r.start_time or 0), int(getattr(r, "id", 0) or 0)))
    return [
        NumberedSegment(
            number=index,
            segment_id=int(getattr(record, "id", -1) or -1),
            start_time=int(record.start_time or 0),
            end_time=int(record.end_time or 0),
            speaker=(getattr(record, "speaker", "") or "").strip(),
            text=(record.text or "").strip(),
        )
        for index, record in enumerate(ordered, start=1)
    ]


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(v) for v in value if _as_text(v))
    return str(value).strip()


def _as_sources(value, valid_numbers: set) -> List[int]:
    if not isinstance(value, (list, tuple)):
        value = [value] if value not in (None, "") else []
    sources = []
    for item in value:
        try:
            number = int(str(item).strip().strip("[]"))
        except (TypeError, ValueError):
            continue
        if number in valid_numbers and number not in sources:
            sources.append(number)
    return sources


def normalize_analysis(data, valid_numbers: set) -> dict:
    """Bring an AI document answer into a fixed, safe shape.

    Items of tasks/decisions/open_questions without a valid source reference
    get ``"verified": False`` so the UI and document can flag them.
    """
    if not isinstance(data, dict):
        data = {}

    def as_list(value) -> list:
        return value if isinstance(value, list) else []

    def items(key, extra_fields=()):
        result = []
        for item in as_list(data.get(key)):
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                continue
            text = _as_text(item.get("text"))
            if not text:
                continue
            sources = _as_sources(item.get("sources"), valid_numbers)
            entry = {"text": text, "sources": sources, "verified": bool(sources)}
            for name in extra_fields:
                entry[name] = _as_text(item.get(name))
            result.append(entry)
        return result

    def blocks(key, title_key):
        result = []
        for item in as_list(data.get(key)):
            if not isinstance(item, dict):
                continue
            title = _as_text(item.get(title_key) or item.get("title"))
            content = _as_text(item.get("content"))
            if not title and not content:
                continue
            result.append(
                {
                    title_key: title,
                    "content": content,
                    "sources": _as_sources(item.get("sources"), valid_numbers),
                }
            )
        return result

    topics = data.get("topics") or []
    if isinstance(topics, str):
        topics = [topics]
    return {
        "title": _as_text(data.get("title")),
        "summary": _as_text(data.get("summary")),
        "topics": [_as_text(t) for t in topics if _as_text(t)],
        "sections": blocks("sections", "title"),
        "conversation": blocks("conversation", "heading"),
        "tasks": items("tasks", ("owner", "due")),
        "decisions": items("decisions"),
        "open_questions": items("open_questions"),
    }


def render_markdown(
    analysis: dict,
    segments: Sequence[CleanedSegment],
    title: str = "",
) -> str:
    """Render the structured document as Markdown."""
    times: Dict[int, int] = {s.number: s.start_time for s in segments}

    def source_note(sources: Sequence[int], verified: bool = True) -> str:
        if not verified:
            return " ⚠ " + _("(no source in the transcript)")
        if not sources:
            return ""
        stamps = ", ".join(format_timestamp(times[n]) for n in sources if n in times)
        return f" _({_('Source')}: {stamps})_"

    lines: List[str] = [f"# {analysis.get('title') or title or _('AI post-processing')}", ""]

    if analysis.get("summary"):
        lines += [f"## {_('Summary')}", "", analysis["summary"], ""]
    if analysis.get("topics"):
        lines += [f"## {_('Topics')}", ""] + [f"- {t}" for t in analysis["topics"]] + [""]

    for section in analysis.get("sections", []):
        lines += [f"## {section['title']}", "", section["content"] or "–", ""]

    if analysis.get("conversation"):
        lines += [f"## {_('Course of the conversation')}", ""]
        for block in analysis["conversation"]:
            if block.get("heading"):
                lines += [f"### {block['heading']}", ""]
            lines += [block.get("content", "") + source_note(block.get("sources", [])), ""]

    def item_list(heading: str, key: str, checkbox: bool = False):
        entries = analysis.get(key) or []
        if not entries:
            return [f"## {heading}", "", _("None mentioned."), ""]
        out = [f"## {heading}", ""]
        for entry in entries:
            details = []
            if entry.get("owner"):
                details.append(f"{_('Responsible')}: {entry['owner']}")
            if entry.get("due"):
                details.append(f"{_('Due')}: {entry['due']}")
            suffix = f" — {' · '.join(details)}" if details else ""
            prefix = "- [ ] " if checkbox else "- "
            out.append(
                prefix + entry["text"] + suffix
                + source_note(entry.get("sources", []), entry.get("verified", True))
            )
        return out + [""]

    lines += item_list(_("Tasks"), "tasks", checkbox=True)
    lines += item_list(_("Decisions"), "decisions")
    lines += item_list(_("Open questions"), "open_questions")

    lines += [f"## {_('Cleaned transcript')}", ""]
    paragraph: List[str] = []
    current_speaker: Optional[str] = None
    for segment in segments:
        if not segment.text.strip():
            continue
        if segment.speaker != current_speaker and paragraph:
            lines += [" ".join(paragraph), ""]
            paragraph = []
        if not paragraph:
            stamp = format_timestamp(segment.start_time)
            head = f"**{segment.speaker}** ({stamp}):" if segment.speaker else f"({stamp})"
            paragraph.append(head)
        paragraph.append(segment.text.strip())
        current_speaker = segment.speaker
    if paragraph:
        lines += [" ".join(paragraph), ""]

    return "\n".join(lines).rstrip() + "\n"


class PostProcessingPipeline:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        profile: Profile,
        progress: Optional[Callable[[float, str], None]] = None,
        is_canceled: Optional[Callable[[], bool]] = None,
        cleanup_chunk_chars: int = DEFAULT_CLEANUP_CHUNK_CHARS,
        document_chunk_chars: int = DEFAULT_DOCUMENT_CHUNK_CHARS,
    ):
        self.provider = provider
        self.model = model
        self.profile = profile
        self.progress = progress or (lambda fraction, message: None)
        self.is_canceled = is_canceled or (lambda: False)
        self.cleanup_chunk_chars = cleanup_chunk_chars
        self.document_chunk_chars = document_chunk_chars
        self._steps_done = 0
        self._steps_total = 1

    # --- helpers -------------------------------------------------------------

    def _check_canceled(self) -> None:
        if self.is_canceled():
            raise PipelineCanceled()

    def _step(self, message: str) -> None:
        self._steps_done += 1
        self.progress(self._steps_done / self._steps_total, message)

    def _ask(self, system: str, user: str, json_mode: bool = True) -> str:
        self._check_canceled()
        return self.provider.complete(
            CompletionRequest(system=system, user=user, model=self.model, json_mode=json_mode),
            is_canceled=self.is_canceled,
        )

    def _ask_json(self, system: str, user: str):
        """Ask for JSON; on unparseable output ask once more, then fail."""
        text = self._ask(system, user)
        try:
            return extract_json(text)
        except ValueError:
            logger.warning("AI answer was not valid JSON, asking again")
        text = self._ask(
            system,
            user + "\n\nWichtig: Antworte ausschließlich mit gültigem JSON, ohne weiteren Text.",
        )
        try:
            return extract_json(text)
        except ValueError as exc:
            raise AIResponseError(
                _("The AI answer could not be read. Please try again or choose another model.")
            ) from exc

    # --- steps ---------------------------------------------------------------

    def cleanup(self, segments: Sequence[NumberedSegment]) -> List[CleanedSegment]:
        system = cleanup_system_prompt(self.profile)
        to_send = [s for s in segments if s.text]
        cleaned: Dict[int, str] = {}
        chunks = chunk_segments(to_send, self.cleanup_chunk_chars)
        for index, chunk in enumerate(chunks, start=1):
            answer = self._ask(system, format_segments(chunk))
            cleaned.update(parse_cleanup_response(answer))
            self._step(_("Cleaning up text ({current}/{total})").format(current=index, total=len(chunks)))

        result = []
        for segment in segments:
            if not segment.text:
                result.append(
                    CleanedSegment(
                        segment.number, segment.segment_id, segment.start_time,
                        segment.end_time, segment.speaker, segment.text, "",
                    )
                )
                continue
            text = cleaned.get(segment.number)
            if text is not None:
                text = strip_speaker_prefix(text, segment.speaker)
            problem = check_cleaned_text(segment.text, text)
            result.append(
                CleanedSegment(
                    number=segment.number,
                    segment_id=segment.segment_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    speaker=segment.speaker,
                    original_text=segment.text,
                    # Never lose content: keep the original when in doubt
                    text=segment.text if problem else text,
                    flag=problem or "",
                )
            )
        return result

    def document(self, segments: Sequence[CleanedSegment]) -> dict:
        valid_numbers = {s.number for s in segments}
        numbered = [
            NumberedSegment(s.number, s.segment_id, s.start_time, s.end_time, s.speaker, s.text)
            for s in segments
            if s.text.strip()
        ]
        system = document_system_prompt(self.profile)
        chunks = chunk_segments(numbered, self.document_chunk_chars)
        if len(chunks) <= 1:
            data = self._ask_json(system, format_segments(numbered))
            self._step(_("Creating document"))
            return normalize_analysis(data, valid_numbers)

        partials = []
        for index, chunk in enumerate(chunks, start=1):
            data = self._ask_json(system, format_segments(chunk))
            partials.append(normalize_analysis(data, valid_numbers))
            self._step(_("Creating document ({current}/{total})").format(current=index, total=len(chunks)))
        merged = self._ask_json(
            merge_system_prompt(self.profile), json.dumps(partials, ensure_ascii=False)
        )
        self._step(_("Merging document"))
        return normalize_analysis(merged, valid_numbers)

    def run(self, segments: Sequence[NumberedSegment], title: str = "") -> PipelineResult:
        if not any(s.text for s in segments):
            raise AIResponseError(_("The transcript is empty."))

        text_segments = [s for s in segments if s.text]
        cleanup_steps = len(chunk_segments(text_segments, self.cleanup_chunk_chars))
        document_chunks = len(chunk_segments(text_segments, self.document_chunk_chars))
        document_steps = 1 if document_chunks <= 1 else document_chunks + 1
        self._steps_total = cleanup_steps + document_steps
        self._steps_done = 0
        self.progress(0.0, _("Starting AI post-processing"))

        cleaned = self.cleanup(segments)
        analysis = self.document(cleaned)

        warnings = []
        kept = sum(1 for s in cleaned if s.flag)
        if kept:
            warnings.append(
                _("{count} segments were kept unchanged because the AI result looked unreliable.").format(
                    count=kept
                )
            )
        unverified = sum(
            1 for key in ANALYSIS_LIST_KEYS for item in analysis[key] if not item["verified"]
        )
        if unverified:
            warnings.append(
                _("{count} items have no source in the transcript. Please check them.").format(
                    count=unverified
                )
            )
        analysis["warnings"] = warnings

        snapshot = json.dumps(
            {
                "profile_id": self.profile.id,
                "cleanup_system_prompt": cleanup_system_prompt(self.profile),
                "document_system_prompt": document_system_prompt(self.profile),
            },
            ensure_ascii=False,
        )
        return PipelineResult(
            segments=cleaned,
            analysis=analysis,
            document_markdown=render_markdown(analysis, cleaned, title),
            prompt_snapshot=snapshot,
            warnings=warnings,
        )
