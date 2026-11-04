CREATE TABLE transcription (
    id TEXT PRIMARY KEY,
    error_message TEXT,
    export_formats TEXT,
    file TEXT,
    output_folder TEXT,
    progress DOUBLE PRECISION DEFAULT 0.0,
    language TEXT,
    model_type TEXT,
    source TEXT,
    status TEXT,
    task TEXT,
    time_ended TIMESTAMP,
    time_queued TIMESTAMP NOT NULL,
    time_started TIMESTAMP,
    url TEXT,
    whisper_model_size TEXT,
    hugging_face_model_id TEXT,
    word_level_timings BOOLEAN DEFAULT FALSE,
    extract_speech BOOLEAN DEFAULT FALSE,
    name TEXT,
    notes TEXT,
    metadata TEXT DEFAULT '[]'
);

CREATE TABLE transcription_segment (
    id INTEGER PRIMARY KEY,
    end_time INT DEFAULT 0,
    start_time INT DEFAULT 0,
    text TEXT NOT NULL,
    translation TEXT DEFAULT '',
    transcription_id TEXT,
    speaker TEXT DEFAULT '',
    metadata TEXT DEFAULT '[]',
    FOREIGN KEY (transcription_id) REFERENCES transcription(id) ON DELETE CASCADE
);
CREATE INDEX idx_transcription_id ON transcription_segment(transcription_id);

-- AI post-processing (Soufflé AI). The original transcript tables above are
-- never modified by the AI features; every AI run is stored as a version.
CREATE TABLE ai_version (
    id INTEGER PRIMARY KEY,
    transcription_id TEXT NOT NULL,
    profile_id TEXT DEFAULT '',
    profile_name TEXT DEFAULT '',
    provider TEXT DEFAULT '',
    model TEXT DEFAULT '',
    prompt_snapshot TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    error_message TEXT DEFAULT '',
    document_markdown TEXT DEFAULT '',
    analysis_json TEXT DEFAULT '{}',
    manually_edited BOOLEAN DEFAULT FALSE,
    time_created TIMESTAMP,
    time_updated TIMESTAMP,
    FOREIGN KEY (transcription_id) REFERENCES transcription(id) ON DELETE CASCADE
);
CREATE INDEX idx_ai_version_transcription_id ON ai_version(transcription_id);

CREATE TABLE ai_version_segment (
    id INTEGER PRIMARY KEY,
    ai_version_id INTEGER NOT NULL,
    position INT DEFAULT 0,
    segment_id INT DEFAULT -1,
    start_time INT DEFAULT 0,
    end_time INT DEFAULT 0,
    speaker TEXT DEFAULT '',
    original_text TEXT NOT NULL,
    text TEXT NOT NULL,
    flag TEXT DEFAULT '',
    FOREIGN KEY (ai_version_id) REFERENCES ai_version(id) ON DELETE CASCADE
);
CREATE INDEX idx_ai_version_segment_version_id ON ai_version_segment(ai_version_id);
