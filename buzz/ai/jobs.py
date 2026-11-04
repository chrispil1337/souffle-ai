# SPDX-License-Identifier: GPL-3.0-or-later
"""Run AI work off the UI thread.

Network calls to LLM providers can take minutes, so they run on a
``QThreadPool``. Results come back through Qt signals, which are delivered on
the thread of the receiving object (the UI thread). Jobs never touch the
database or widgets themselves.
"""

import logging
import threading
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

logger = logging.getLogger(__name__)


class CancelToken:
    """Thread-safe cancel flag handed to long running work."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_canceled(self) -> bool:
        return self._event.is_set()


class JobSignals(QObject):
    succeeded = pyqtSignal(object)
    # The exception object, usually an ``AIError`` with ``user_message``
    failed = pyqtSignal(object)
    # (fraction 0..1, message)
    progress = pyqtSignal(float, str)


class CallableJob(QRunnable):
    """Run ``fn(job)`` on a thread pool and report the result via signals.

    ``fn`` receives the job so it can report progress
    (``job.report_progress``) and check ``job.cancel_token``.
    """

    def __init__(self, fn: Callable[["CallableJob"], Any]):
        super().__init__()
        self.fn = fn
        self.signals = JobSignals()
        self.cancel_token = CancelToken()
        self.setAutoDelete(False)

    def report_progress(self, fraction: float, message: str = "") -> None:
        self.signals.progress.emit(max(0.0, min(1.0, fraction)), message)

    def cancel(self) -> None:
        self.cancel_token.cancel()

    def run(self) -> None:
        try:
            result = self.fn(self)
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            logger.error("AI job failed: %s", type(exc).__name__)
            self.signals.failed.emit(exc)
            return
        self.signals.succeeded.emit(result)


def start_job(
    fn: Callable[[CallableJob], Any],
    on_success: Optional[Callable[[Any], None]] = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
    thread_pool: Optional[QThreadPool] = None,
) -> CallableJob:
    """Create, wire and start a job. Keep the returned job referenced until
    it has finished, otherwise its signals object may be garbage collected.

    Without ``thread_pool`` the job runs on its own daemon thread."""
    job = CallableJob(fn)
    if on_success is not None:
        job.signals.succeeded.connect(on_success)
    if on_failure is not None:
        job.signals.failed.connect(on_failure)
    if on_progress is not None:
        job.signals.progress.connect(on_progress)
    if thread_pool is not None:
        thread_pool.start(job)
    else:
        # A daemon thread instead of the global QThreadPool: quitting the app
        # must not wait for a slow AI request (up to the request timeout).
        # Signals are delivered to receivers on the UI thread as usual.
        threading.Thread(target=job.run, name="ai-job", daemon=True).start()
    return job


def error_message(exc: Exception) -> str:
    """Text to show the user for an exception from AI work."""
    message = getattr(exc, "user_message", "")
    if message:
        return message
    return str(exc) or type(exc).__name__
