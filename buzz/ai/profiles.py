# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-processing profiles (prompt templates).

Five profiles ship with the app. Users can edit them, duplicate them and add
their own; changes are stored locally in ``ai_profiles.json`` in the app data
folder. Built-in profiles can always be reset to their defaults.

The rules that keep the AI faithful to the transcript (no invented facts,
keep speakers, ...) are not part of a profile: the pipeline always adds them,
so editing a profile can not switch them off.
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

from buzz.locale import _

logger = logging.getLogger(__name__)

PROFILES_FILE_NAME = "ai_profiles.json"
FILE_VERSION = 1


@dataclass
class Profile:
    id: str
    name: str
    description: str = ""
    # Extra instructions for the segment-by-segment cleanup
    cleanup_prompt: str = ""
    # Instructions for the structured document (focus, style)
    document_prompt: str = ""
    # Additional document sections, in this order, besides the standard
    # summary / tasks / decisions / open questions
    sections: List[str] = field(default_factory=list)
    builtin: bool = False

    def copy(self, **changes) -> "Profile":
        """Deep enough copy: the sections list is not shared."""
        copied = replace(self, **changes)
        copied.sections = list(copied.sections)
        return copied

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("builtin", None)
        return data

    @classmethod
    def from_dict(cls, data: dict, builtin: bool = False) -> "Profile":
        sections = data.get("sections") or []
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or data["id"]),
            description=str(data.get("description") or ""),
            cleanup_prompt=str(data.get("cleanup_prompt") or ""),
            document_prompt=str(data.get("document_prompt") or ""),
            sections=[str(s) for s in sections if str(s).strip()],
            builtin=builtin,
        )


GENERAL_CLEANUP_ID = "general_cleanup"
CUSTOMER_CONVERSATION_ID = "customer_conversation"
MEETING_MINUTES_ID = "meeting_minutes"
SITE_INSPECTION_ID = "site_inspection"
QUOTE_PREPARATION_ID = "quote_preparation"


def builtin_profiles() -> List[Profile]:
    """The profiles shipped with the app (fresh copies)."""
    return [
        Profile(
            id=GENERAL_CLEANUP_ID,
            name=_("General transcript cleanup"),
            description=_("Readable transcript with a short summary."),
            cleanup_prompt=(
                "Bereinige das Transkript zu gut lesbarem Text. Behalte den "
                "Gesprächscharakter bei."
            ),
            document_prompt=(
                "Erstelle ein übersichtliches Dokument des Gesprächs mit einer "
                "kurzen Zusammenfassung und den wichtigsten Themen."
            ),
            sections=[],
            builtin=True,
        ),
        Profile(
            id=CUSTOMER_CONVERSATION_ID,
            name=_("Customer conversation"),
            description=_("Customer wishes, agreements and next steps."),
            cleanup_prompt=(
                "Es handelt sich um ein Gespräch mit einem Kunden. Achte besonders "
                "auf korrekte Namen, Zahlen, Beträge, Termine und Produktbezeichnungen."
            ),
            document_prompt=(
                "Fasse das Kundengespräch für die interne Nachbereitung zusammen. "
                "Halte fest, was der Kunde wünscht, welche Anforderungen, Bedenken "
                "und Einwände genannt wurden, was vereinbart wurde und was als "
                "Nächstes zu tun ist."
            ),
            sections=[
                "Anliegen und Wünsche des Kunden",
                "Anforderungen und Rahmenbedingungen",
                "Bedenken und Einwände",
                "Vereinbarungen",
            ],
            builtin=True,
        ),
        Profile(
            id=MEETING_MINUTES_ID,
            name=_("Meeting minutes"),
            description=_("Minutes with agenda items, decisions and tasks."),
            cleanup_prompt=(
                "Es handelt sich um ein Meeting mit mehreren Teilnehmern. Achte auf "
                "korrekte Namen und Fachbegriffe."
            ),
            document_prompt=(
                "Erstelle ein Ergebnisprotokoll des Meetings. Gliedere den Verlauf "
                "nach Tagesordnungspunkten bzw. Themen. Ordne Aussagen den "
                "Sprechern zu, soweit sie bekannt sind. Aufgaben enthalten, wenn "
                "genannt, die verantwortliche Person und die Frist."
            ),
            sections=["Teilnehmer", "Besprochene Themen"],
            builtin=True,
        ),
        Profile(
            id=SITE_INSPECTION_ID,
            name=_("Construction site inspection"),
            description=_("Findings, defects, measurements and responsibilities on site."),
            cleanup_prompt=(
                "Es handelt sich um eine Baustellenbesichtigung. Achte besonders auf "
                "Maße, Mengen, Einheiten (m, m², m³, cm, Stk.), Raum- und "
                "Bauteilbezeichnungen, Gewerke und Materialien. Schreibe Zahlen "
                "und Einheiten einheitlich (z. B. „2,40 m“)."
            ),
            document_prompt=(
                "Erstelle ein Besichtigungsprotokoll. Ordne Feststellungen nach Ort "
                "(Raum, Geschoss, Bauteil). Halte Mängel und Schäden, genannte Maße "
                "und Mengen, betroffene Gewerke, Zuständigkeiten sowie erwähnte "
                "Fotos, Nachweise oder Unterlagen fest. Übernimm Maße und Mengen "
                "exakt so, wie sie genannt wurden."
            ),
            sections=[
                "Ort und Anlass",
                "Feststellungen nach Bereich",
                "Mängel und Schäden",
                "Maße und Mengen",
                "Gewerke und Zuständigkeiten",
                "Fotos, Nachweise und Unterlagen",
            ],
            builtin=True,
        ),
        Profile(
            id=QUOTE_PREPARATION_ID,
            name=_("Quote preparation (construction)"),
            description=_("Work items, quantities and open points for a quote."),
            cleanup_prompt=(
                "Es handelt sich um ein Gespräch zur Vorbereitung eines Angebots im "
                "Bauwesen. Achte besonders auf Leistungen, Mengen, Einheiten, "
                "Materialien, Qualitäten, Termine und Preise."
            ),
            document_prompt=(
                "Bereite die Informationen für die Erstellung eines Angebots auf. "
                "Liste die besprochenen Leistungen als Positionen mit Menge und "
                "Einheit, sofern genannt. Erfinde keine Mengen, Preise oder "
                "Positionen. Fehlende Angaben, die für ein Angebot nötig sind, "
                "gehören zu den offenen Fragen (Rückfragen an den Kunden)."
            ),
            sections=[
                "Projekt und Ansprechpartner",
                "Leistungspositionen (Menge, Einheit, Beschreibung)",
                "Materialien und Qualitäten",
                "Termine und Ausführungszeitraum",
                "Preise und Budget",
                "Annahmen und Ausschlüsse",
            ],
            builtin=True,
        ),
    ]


def default_profiles_path() -> str:
    from platformdirs import user_data_dir

    from buzz.settings.settings import APP_NAME

    return os.path.join(user_data_dir(APP_NAME), PROFILES_FILE_NAME)


class ProfileStore:
    """Loads and saves profiles. All data stays on this computer."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_profiles_path()
        self._defaults: Dict[str, Profile] = {p.id: p for p in builtin_profiles()}
        self._overrides: Dict[str, Profile] = {}
        self._custom: List[Profile] = []
        self.load()

    # --- persistence -----------------------------------------------------------

    def load(self) -> None:
        self._overrides = {}
        self._custom = []
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("profiles", []):
                profile = Profile.from_dict(item)
                if profile.id in self._defaults:
                    profile.builtin = True
                    self._overrides[profile.id] = profile
                elif not any(p.id == profile.id for p in self._custom):
                    self._custom.append(profile)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("Could not read AI profiles, using defaults: %s", exc)
            self._backup_broken_file()
            self._overrides = {}
            self._custom = []

    def _backup_broken_file(self) -> None:
        try:
            os.replace(self.path, self.path + ".broken")
        except OSError:
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {
            "version": FILE_VERSION,
            "profiles": [p.to_dict() for p in self._overrides.values()]
            + [p.to_dict() for p in self._custom],
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    # --- queries ---------------------------------------------------------------

    def profiles(self) -> List[Profile]:
        """Built-in profiles first (with user changes), then custom ones."""
        result = [self._overrides.get(pid, p).copy() for pid, p in self._defaults.items()]
        return result + [p.copy() for p in self._custom]

    def get(self, profile_id: str) -> Optional[Profile]:
        for profile in self.profiles():
            if profile.id == profile_id:
                return profile
        return None

    def get_or_default(self, profile_id: str) -> Profile:
        return self.get(profile_id) or self.get(GENERAL_CLEANUP_ID)

    def is_modified(self, profile_id: str) -> bool:
        return profile_id in self._overrides

    # --- changes ---------------------------------------------------------------

    def save_profile(self, profile: Profile) -> Profile:
        if not profile.name.strip():
            raise ValueError("A profile needs a name")
        profile = profile.copy()
        if profile.id in self._defaults:
            profile.builtin = True
            if profile == self._defaults[profile.id]:
                self._overrides.pop(profile.id, None)
            else:
                self._overrides[profile.id] = profile
        else:
            profile.builtin = False
            for index, existing in enumerate(self._custom):
                if existing.id == profile.id:
                    self._custom[index] = profile
                    break
            else:
                self._custom.append(profile)
        self._save()
        return profile.copy()

    def duplicate(self, profile_id: str, name: Optional[str] = None) -> Profile:
        source = self.get(profile_id)
        if source is None:
            raise KeyError(profile_id)
        copy = source.copy(
            id=f"custom_{uuid.uuid4().hex[:8]}",
            name=name or _("{name} (copy)").format(name=source.name),
            builtin=False,
        )
        return self.save_profile(copy)

    def create(self, name: str) -> Profile:
        return self.save_profile(
            Profile(id=f"custom_{uuid.uuid4().hex[:8]}", name=name, builtin=False)
        )

    def reset(self, profile_id: str) -> Profile:
        """Restore a built-in profile to its shipped defaults."""
        if profile_id not in self._defaults:
            raise KeyError(profile_id)
        self._overrides.pop(profile_id, None)
        self._save()
        return self._defaults[profile_id].copy()

    def delete(self, profile_id: str) -> None:
        """Delete a custom profile. Built-in profiles can only be reset."""
        if profile_id in self._defaults:
            raise ValueError("Built-in profiles can not be deleted")
        self._custom = [p for p in self._custom if p.id != profile_id]
        self._save()
