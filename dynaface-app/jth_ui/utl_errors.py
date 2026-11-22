"""
What the app does with an exception nothing caught.

PyQt6 ends the app, through qFatal() and so abort(), when an exception leaves a
slot, an event handler or a QThread's run(), unless sys.excepthook has been
replaced. Unsaved work is lost, and nothing is logged. install() replaces it: each
such exception is logged with its traceback, the user is told once for each
different error, in a message box on the GUI thread, and the app goes on, so its
work can still be saved.
"""

import logging
import sys
import threading
import traceback
from types import TracebackType
from typing import Callable, Dict, Optional, Type

from PyQt6.QtCore import QObject, Qt, pyqtSignal

logger = logging.getLogger(__name__)

# How many times one error is logged with its traceback before its repeats go
# unlogged: an error in a timer or a paint event comes back many times a second.
LOGGED_REPEATS = 20

# The longest an error's description in the message box may be.
_SUMMARY_LENGTH = 300


def summarize(exc_type: Type[BaseException], exc_value: Optional[BaseException]) -> str:
    """An exception's type and message on one line, as a traceback ends, cut to
    _SUMMARY_LENGTH characters."""
    text = "".join(traceback.format_exception_only(exc_type, exc_value))
    summary = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(summary) > _SUMMARY_LENGTH:
        summary = summary[: _SUMMARY_LENGTH - 3] + "..."
    return summary


class _Reporter(QObject):
    """Carries an error's description to the GUI thread."""

    error = pyqtSignal(str)


class ExceptionHook:
    """
    A sys.excepthook that logs an exception and has notify(summary) tell the user
    about it, on the GUI thread, after the code that raised it has unwound: the
    hook can run on any thread, inside a slot, a paint event or a QThread. notify
    runs once for each different error (by its summary()), and not while another
    report is showing. Create it on the GUI thread.
    """

    def __init__(self, notify: Callable[[str], None]) -> None:
        self._notify = notify
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._showing = False
        self._reporter = _Reporter()
        # Queued, so the report waits for the event loop even when the error was
        # on the GUI thread, and never runs inside the code that raised.
        self._reporter.error.connect(self._show, Qt.ConnectionType.QueuedConnection)

    def __call__(
        self,
        exc_type: Type[BaseException],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        try:
            summary = summarize(exc_type, exc_value)
            with self._lock:
                count = self._counts[summary] = self._counts.get(summary, 0) + 1
            if count <= LOGGED_REPEATS:
                logger.error("Unexpected error", exc_info=(exc_type, exc_value, exc_tb))
            elif count == LOGGED_REPEATS + 1:
                logger.error(f"{summary} again; its repeats won't be logged")
            if count == 1:
                self._report(summary)
        except Exception:  # noqa: BLE001 - a hook that raises ends the app
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _report(self, summary: str) -> None:
        self._reporter.error.emit(summary)

    def _show(self, summary: str) -> None:
        # A report shows a modal message box, whose event loop delivers the next
        # queued report: drop it rather than stack a box on the box.
        if self._showing:
            return
        self._showing = True
        try:
            self._notify(summary)
        except Exception:  # noqa: BLE001 - reporting an error must not raise one
            logger.error("Unable to report an unexpected error", exc_info=True)
        finally:
            self._showing = False


def install(notify: Callable[[str], None]) -> ExceptionHook:
    """Replace sys.excepthook with an ExceptionHook calling notify, and return it.
    Call on the GUI thread."""
    hook = ExceptionHook(notify)
    sys.excepthook = hook
    return hook
