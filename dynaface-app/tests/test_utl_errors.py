"""
Tests for jth_ui.utl_errors: an exception nothing caught is logged, reported once
for each different error on the GUI thread, and the app goes on, where PyQt6
would end it.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from jth_ui import utl_errors

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _raised(error):
    """sys.exc_info() for error, raised."""
    try:
        raise error
    except BaseException:
        return sys.exc_info()


class ExceptionHookTest(unittest.TestCase):
    def setUp(self):
        self.notify = mock.Mock()
        self.hook = utl_errors.ExceptionHook(self.notify)

    def call(self, *infos):
        """Runs the hook on each exception info, and returns the summaries it
        reported and the records it logged."""
        with mock.patch.object(self.hook, "_report") as report, self.assertLogs(
            utl_errors.logger, "ERROR"
        ) as logs:
            for info in infos:
                self.hook(*info)
        return [c.args[0] for c in report.call_args_list], logs.records

    def test_logs_an_error_with_its_traceback_and_reports_it_once(self):
        info = _raised(ValueError("bad value"))
        reported, records = self.call(info, info)
        self.assertEqual(reported, ["ValueError: bad value"])
        self.assertEqual(len(records), 2)
        self.assertIs(records[0].exc_info[1], info[1])

    def test_reports_each_different_error(self):
        reported, _ = self.call(
            _raised(ValueError("one")), _raised(ValueError("two")), _raised(KeyError(3))
        )
        self.assertEqual(
            reported, ["ValueError: one", "ValueError: two", "KeyError: 3"]
        )

    def test_stops_logging_an_error_that_keeps_coming_back(self):
        # An error in a timer or a paint event comes back many times a second.
        info = _raised(ValueError("again"))
        _, records = self.call(*[info] * (utl_errors.LOGGED_REPEATS + 5))
        self.assertEqual(len(records), utl_errors.LOGGED_REPEATS + 1)
        self.assertIsNone(records[-1].exc_info)
        self.assertIn("won't be logged", records[-1].getMessage())

    def test_leaves_a_keyboard_interrupt_to_python(self):
        info = _raised(KeyboardInterrupt())
        with mock.patch.object(sys, "__excepthook__") as default, mock.patch.object(
            self.hook, "_report"
        ) as report:
            self.hook(*info)
        default.assert_called_once_with(*info)
        report.assert_not_called()

    def test_drops_a_report_that_arrives_while_one_shows(self):
        # A modal message box runs an event loop, which delivers the next report.
        shown = []

        def notify(summary):
            shown.append(summary)
            self.hook._show("second")

        self.hook._notify = notify
        self.hook._show("first")
        self.hook._show("third")
        self.assertEqual(shown, ["first", "third"])

    def test_logs_a_report_that_fails(self):
        self.notify.side_effect = RuntimeError("no window")
        with self.assertLogs(utl_errors.logger, "ERROR") as logs:
            self.hook._show("first")
        self.assertIn("Unable to report", logs.records[0].getMessage())
        self.notify.side_effect = None
        self.hook._show("second")
        self.notify.assert_called_with("second")

    def test_summarizes_on_one_line_cut_to_length(self):
        self.assertEqual(
            utl_errors.summarize(ValueError, ValueError("a\nb")), "ValueError: a b"
        )
        summary = utl_errors.summarize(ValueError, ValueError("x" * 1000))
        self.assertEqual(len(summary), utl_errors._SUMMARY_LENGTH)
        self.assertTrue(summary.endswith("..."))

    def test_install_replaces_sys_excepthook(self):
        with mock.patch.object(sys, "excepthook", sys.excepthook):
            hook = utl_errors.install(self.notify)
            self.assertIs(sys.excepthook, hook)


# A Qt app whose slot and QThread raise. Without the hook, PyQt6 aborts it at the
# first one; with it, the app runs on, and prints what it reported and whether each
# report ran on the GUI thread.
_APP = textwrap.dedent("""
    import sys
    import threading

    from PyQt6.QtCore import QCoreApplication, QThread, QTimer

    from jth_ui import utl_errors

    app = QCoreApplication(sys.argv)
    reports = []


    def notify(summary):
        on_gui = threading.current_thread() is threading.main_thread()
        reports.append((summary, on_gui))


    utl_errors.install(notify)


    def fail():
        raise ValueError("raised in a slot")


    class Worker(QThread):
        def run(self):
            raise RuntimeError("raised in a thread")


    def finish():
        worker.wait()
        app.processEvents()
        print(sorted(reports))
        app.quit()


    worker = Worker()
    QTimer.singleShot(0, fail)
    QTimer.singleShot(0, fail)
    QTimer.singleShot(0, worker.start)
    QTimer.singleShot(500, finish)
    app.exec()
    """)


class AppKeepsRunningTest(unittest.TestCase):
    def test_an_error_in_a_slot_or_a_thread_is_reported_and_the_app_goes_on(self):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        result = subprocess.run(
            [sys.executable, "-c", _APP],
            cwd=_APP_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            str(
                [
                    ("RuntimeError: raised in a thread", True),
                    ("ValueError: raised in a slot", True),
                ]
            ),
        )
        # Logged with the traceback, each time.
        self.assertEqual(result.stderr.count("ValueError: raised in a slot"), 2)
        self.assertIn("RuntimeError: raised in a thread", result.stderr)


if __name__ == "__main__":
    unittest.main()
