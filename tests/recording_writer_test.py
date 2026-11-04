# SPDX-License-Identifier: GPL-3.0-or-later
import os
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile

from buzz.model_loader import ModelType, TranscriptionModel, WhisperModelSize
from buzz.recording_writer import WavRecorder, new_recording_path
from buzz.transcriber.recording_transcriber import RecordingTranscriber
from buzz.transcriber.transcriber import Task, TranscriptionOptions


class TestWavRecorder:
    def test_writes_all_blocks(self, tmp_path):
        path = str(tmp_path / "rec.wav")
        recorder = WavRecorder(path, sample_rate=16000)
        recorder.start()
        for _ in range(10):
            recorder.write(np.full(1600, 0.25, dtype=np.float32))
        assert recorder.stop() == path

        data, sample_rate = soundfile.read(path, dtype="float32")
        assert sample_rate == 16000
        assert len(data) == 16000
        assert abs(float(data.mean()) - 0.25) < 0.001
        assert recorder.duration_seconds == 1.0

    def test_blocks_are_copied(self, tmp_path):
        path = str(tmp_path / "rec.wav")
        recorder = WavRecorder(path, sample_rate=8000)
        recorder.start()
        block = np.zeros(800, dtype=np.float32)
        recorder.write(block)
        block[:] = 1.0  # the audio callback reuses its buffer
        recorder.stop()
        data, _ = soundfile.read(path, dtype="float32")
        assert float(np.abs(data).max()) == 0.0

    def test_empty_recording_is_removed(self, tmp_path):
        path = str(tmp_path / "rec.wav")
        recorder = WavRecorder(path, sample_rate=16000)
        recorder.start()
        assert recorder.stop() is None
        assert not os.path.exists(path)

    def test_stop_without_start(self, tmp_path):
        assert WavRecorder(str(tmp_path / "x.wav"), 16000).stop() is None

    def test_new_recording_path_is_unique(self, tmp_path):
        first = new_recording_path(str(tmp_path))
        open(first, "w").close()
        second = new_recording_path(str(tmp_path))
        assert first != second
        assert first.endswith(".wav") and second.endswith(".wav")


def make_transcriber(audio_output_path):
    options = TranscriptionOptions(
        task=Task.TRANSCRIBE,
        model=TranscriptionModel(model_type=ModelType.WHISPER, whisper_model_size=WhisperModelSize.TINY),
    )
    with patch("buzz.transcriber.recording_transcriber.Settings") as MockSettings:
        MockSettings.return_value.value.return_value = 0
        return RecordingTranscriber(
            transcription_options=options,
            input_device_index=None,
            sample_rate=16000,
            model_path="tiny",
            sounddevice=MagicMock(),
            audio_output_path=audio_output_path,
        )


class TestRecordingTranscriberSavesAudio:
    def test_callback_audio_goes_to_file_and_signal(self, tmp_path):
        path = str(tmp_path / "live.wav")
        transcriber = make_transcriber(path)
        saved = []
        transcriber.audio_saved.connect(saved.append)

        transcriber._start_audio_recorder()
        for _ in range(5):
            transcriber.stream_callback(np.full((1600, 1), 0.1, dtype=np.float32), 1600, None, None)
        transcriber._stop_audio_recorder()

        assert saved == [path]
        data, _ = soundfile.read(path, dtype="float32")
        assert len(data) == 8000
        assert transcriber.audio_recorder is None

    def test_full_live_queue_does_not_drop_saved_audio(self, tmp_path):
        path = str(tmp_path / "live.wav")
        transcriber = make_transcriber(path)
        transcriber.max_queue_size = 0  # live queue drops everything
        transcriber._start_audio_recorder()
        transcriber.stream_callback(np.ones((1600, 1), dtype=np.float32), 1600, None, None)
        transcriber._stop_audio_recorder()
        data, _ = soundfile.read(path, dtype="float32")
        assert len(data) == 1600

    def test_no_file_without_path(self):
        transcriber = make_transcriber(None)
        saved = []
        transcriber.audio_saved.connect(saved.append)
        transcriber._start_audio_recorder()
        transcriber.stream_callback(np.ones((160, 1), dtype=np.float32), 160, None, None)
        transcriber._stop_audio_recorder()
        assert saved == []
        assert transcriber.audio_recorder is None
