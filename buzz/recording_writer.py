# SPDX-License-Identifier: GPL-3.0-or-later
"""Write a live recording to a WAV file while it is being transcribed.

The audio callback must never block, so it only hands copies of the blocks to
a queue; a writer thread stores them. Unlike the live transcription queue,
no audio is dropped when transcription falls behind.
"""

import logging
import os
import queue
import threading
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def default_recordings_folder() -> str:
    from platformdirs import user_data_dir

    from buzz.settings.settings import APP_NAME

    return os.path.join(user_data_dir(APP_NAME), "recordings")


def new_recording_path(folder: Optional[str] = None) -> str:
    folder = folder or default_recordings_folder()
    os.makedirs(folder, exist_ok=True)
    name = datetime.now().strftime("Recording %Y-%m-%d %H-%M-%S")
    path = os.path.join(folder, f"{name}.wav")
    counter = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{name} ({counter}).wav")
        counter += 1
    return path


class WavRecorder:
    def __init__(self, path: str, sample_rate: int, channels: int = 1):
        self.path = path
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.frames_written = 0
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._file = None
        self.error: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return self.frames_written / self.sample_rate if self.sample_rate else 0.0

    def start(self) -> None:
        import soundfile

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = soundfile.SoundFile(
            self.path,
            mode="w",
            samplerate=self.sample_rate,
            channels=self.channels,
            subtype="PCM_16",
        )
        self._thread = threading.Thread(target=self._run, name="wav-recorder", daemon=True)
        self._thread.start()

    def write(self, chunk: np.ndarray) -> None:
        """Called from the audio callback: never blocks."""
        if self._thread is not None:
            self._queue.put_nowait(np.array(chunk, dtype=np.float32, copy=True))

    def _run(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            try:
                self._file.write(chunk)
                self.frames_written += len(chunk)
            except Exception as exc:  # noqa: BLE001 - reported via stop()
                self.error = str(exc)
                logger.error("Writing recording failed: %s", exc)

    def stop(self) -> Optional[str]:
        """Finish the file. Returns its path, or None if nothing was recorded."""
        if self._thread is None:
            return None
        self._queue.put(None)
        self._thread.join()
        self._thread = None
        self._file.close()
        if self.frames_written == 0 or self.error:
            try:
                os.remove(self.path)
            except OSError:
                pass
            return None
        logger.info("Recording saved (%.1f s)", self.duration_seconds)
        return self.path
