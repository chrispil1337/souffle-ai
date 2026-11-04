# Soufflé AI – Kurzanleitung

Soufflé AI ist ein Fork von [Buzz](https://github.com/chidiwilliams/buzz). Er nimmt Gespräche lokal auf, transkribiert sie mit Whisper (z. B. Large V3 Turbo) und bearbeitet sie anschließend mit KI nach. Die ausführliche Beschreibung (Englisch) steht in [docs/docs/usage/8_ai_post_processing.md](docs/docs/usage/8_ai_post_processing.md).

## Einrichten

1. **Einstellungen → KI** öffnen.
2. Anbieter wählen:
   - **Ollama (lokal)**: Alles bleibt auf dem Computer. Ollama von [ollama.com](https://ollama.com) installieren und ein Modell laden, z. B. `ollama pull qwen2.5:14b`.
   - **OpenAI** oder **Anthropic**: API-Schlüssel eintragen. Er wird im macOS-Schlüsselbund gespeichert.
3. **Modelle laden**, ein Modell wählen und **Verbindung testen**.

## Transkript nachbearbeiten

1. Transkript öffnen und **Mit KI nachbearbeiten** klicken.
2. Profil wählen (Allgemein, Kundengespräch, Meeting-Protokoll, Baustellenbesichtigung, Angebotsvorbereitung) und starten.
3. Ergebnis prüfen:
   - **Segmente**: Original und KI-Text nebeneinander, die KI-Spalte ist bearbeitbar.
   - **Dokument**: Zusammenfassung, Aufgaben, Entscheidungen, offene Fragen (Markdown, bearbeitbar).
   - **Änderungen**: Was die KI verändert hat.
4. **Exportieren** als Markdown oder JSON.

Das Original bleibt immer unverändert. Jeder Durchlauf wird als eigene Version gespeichert.

## Sicherheitsnetze

- Die KI darf nichts erfinden. Diese Grundregeln sind fest eingebaut.
- Fehlt ein Segment in der KI-Antwort oder wirkt es falsch, bleibt der Originaltext stehen und die Zeile wird gelb markiert.
- Aufgaben, Entscheidungen und offene Fragen müssen auf Stellen im Transkript verweisen. Einträge ohne Beleg bekommen ein ⚠.

## Weitere Funktionen

- **Profile bearbeiten…**: eigene Vorlagen anlegen oder die mitgelieferten anpassen (zurücksetzbar).
- **Automatik** (Einstellungen → KI): neue Transkriptionen automatisch nachbearbeiten.
- **Live-Aufnahme speichern** (Aufnahmefenster): Die Aufnahme wird als WAV gespeichert und danach als Datei transkribiert.

## Entwicklung

```bash
git submodule update --init --recursive
uv sync
uv run buzz
```

Tests der KI-Funktionen:

```bash
uv run pytest tests/ai
```

## Lizenz

GPL-3.0-or-later, siehe [LICENSE](LICENSE). Der ursprüngliche MIT-Hinweis von Buzz liegt in [LICENSE-MIT-Buzz](LICENSE-MIT-Buzz), Details in [NOTICE](NOTICE).
