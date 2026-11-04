---
title: AI post-processing
---

Soufflé AI (this fork of Buzz) can clean up, structure and summarize finished transcripts with a large language model. The original transcript is never changed: every AI run is stored as a separate **version** next to it.

## Providers

Choose the provider in **Preferences → AI**:

| Provider | Where the text is processed | API key |
|---|---|---|
| **Ollama (local)** – default | On your computer ([ollama.com](https://ollama.com)) | not needed |
| **OpenAI** | OpenAI servers (or any OpenAI-compatible server via *Base URL*) | required, shared with the OpenAI key used for Whisper API |
| **Anthropic** | Anthropic servers (Claude models) | required |

API keys are stored in the system's secret store (on macOS the Keychain), the same way Buzz stores the OpenAI key, never in plain settings. Use **Load models** to fetch the available models and **Test connection** to check URL and key.

Before a transcript is sent to a cloud provider for the first time, you are asked for consent (once per provider). Only the transcript text is sent, never the audio.

Settings per provider:

- **Model** – any model offered by the provider. For local German transcripts, models with good German skills work best.
- **Effort** (Anthropic) – how thoroughly Claude works; higher is slower and more expensive.
- **Temperature** (OpenAI, Ollama) – lower values stay closer to the transcript. Empty means model default (Ollama: 0.2).

## Post-processing a transcript

Open a transcript and click **Post-process with AI** in the viewer toolbar. Choose a profile and start. The AI:

1. Cleans up every segment: spelling, grammar, punctuation, filler words, repetitions, colloquial wording. Timestamps and speakers are kept.
2. Writes a structured document: summary, topics, profile specific sections, structured course of the conversation, tasks (with responsible person and due date when mentioned), decisions and open questions.

Safeguards against invented content:

- Fixed base rules are always part of the prompt: no new facts, no changed statements, exact numbers and names, unclear passages marked as *[unverständlich]*.
- If the AI drops a segment or returns a result that looks wrong (for example much shorter), the original text is kept and the segment is highlighted.
- Every task, decision and open question must cite the segments it is based on. Entries without a valid source are marked with ⚠ in the document.

The window shows:

- **Segments** – original and AI text side by side. The AI column can be edited.
- **Document** – the original transcript next to the AI document (Markdown, editable, with preview).
- **Changes** – word level comparison of original and AI text.

All runs are kept in the version list with date, profile and model. Runs continue in the background when the window is closed.

## Profiles

Five profiles are included:

1. General transcript cleanup
2. Customer conversation
3. Meeting minutes
4. Construction site inspection
5. Quote preparation (construction)

Use **Edit profiles…** to change instructions and document sections, to duplicate profiles or to create your own. Built-in profiles can be reset to their defaults. Profiles are stored in `ai_profiles.json` in the app data folder.

## Export

- In the AI window: Markdown of the AI document, Markdown of AI version and original, JSON of AI version and original, Markdown of the original.
- In the viewer's **Export** menu: *Markdown – Text* and *JSON – Text* for the original, and the combined exports of the latest AI version.

The JSON export (`schema_version` 1) contains the original segments and the AI version with cleaned segments, analysis and document.

## Automation

In **Preferences → AI → Automation** you can enable **Post-process new transcriptions automatically** and choose the profile. New transcriptions are then processed one after another in the background. With a cloud provider, automation only runs after you have given consent once by post-processing a transcript by hand.

## Saving live recordings

In the live recording window, enable **Save recording and transcribe it afterwards**. The audio is saved as a WAV file (app data folder, `recordings`) and, when you stop, transcribed as a normal file with the model and language chosen in the recording window. The new transcript can then be post-processed like any other.

## Privacy

- All transcripts, AI versions and profiles are stored locally (SQLite database in the app data folder).
- With Ollama on the same computer nothing leaves the computer.
- Telemetry is disabled by default in this fork (`BUZZ_ENABLE_TELEMETRY` opts in).
- The app does not log transcript text or API keys.
