"""
Tests for what happens to a file the user asks to open while the AI models load,
or after they fail: AppDynaface._on_models_failed() keeps a queued document,
DynafaceWindow.background_timer() passes a queued file on to open_file(), and
open_file() opens a document without the models and says why a photo or video
can't open.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

from dynaface_window import DynafaceWindow
from jth_ui.window_jth import MainWindowJTH


def _import_app():
    """dynaface_app, imported without what it does first when the app starts
    (moving old settings, deleting old logs, setting up logging), all of which
    happens in the user's own folders."""
    with mock.patch(
        "jth_ui.app_jth.migrate_legacy_settings", return_value=[]
    ), mock.patch("jth_ui.utl_log.delete_old_logs"), mock.patch(
        "jth_ui.utl_log.setup_logging"
    ):
        return importlib.import_module("dynaface_app")


def _window(request, ready=False, failed=False):
    """A stand-in for the window and the app, as background_timer() sees them."""
    app = SimpleNamespace(
        file_open_request=request, models_ready=ready, models_failed=failed
    )
    return SimpleNamespace(
        app=app,
        _tab_widget=mock.Mock(count=mock.Mock(return_value=1)),
        _background_queue=[],
        open_file=mock.Mock(),
    )


class BackgroundTimerTest(unittest.TestCase):
    def test_waits_while_the_models_load(self):
        window = _window("/docs/patient.dyfc")
        DynafaceWindow.background_timer(window)
        window.open_file.assert_not_called()
        self.assertEqual(window.app.file_open_request, "/docs/patient.dyfc")

    def test_opens_a_queued_file_once_the_models_are_ready(self):
        window = _window("/docs/face.jpg", ready=True)
        DynafaceWindow.background_timer(window)
        window.open_file.assert_called_once_with("/docs/face.jpg")
        self.assertIsNone(window.app.file_open_request)

    def test_passes_a_queued_file_on_once_the_models_have_failed(self):
        # A file macOS asked to open (from Finder, or the Dock icon) was held
        # forever once the models failed; open_file() opens a document then,
        # and says why a photo or video can't open.
        for name in ("/docs/patient.dyfc", "/docs/face.jpg"):
            with self.subTest(name):
                window = _window(name, failed=True)
                DynafaceWindow.background_timer(window)
                window.open_file.assert_called_once_with(name)
                self.assertIsNone(window.app.file_open_request)


class ModelsFailedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_class = _import_app().AppDynaface

    def fail(self, request, loading="cpu", chosen="cpu"):
        """Runs _on_models_failed() for a load on the device loading, while
        Settings has chosen the device chosen, with request queued."""
        app = SimpleNamespace(
            file_open_request=request,
            models_failed=False,
            _loading_device=loading,
            _models_device=chosen,
            _stop_model_thread=mock.Mock(),
            _start_model_loader=mock.Mock(),
            main_window=None,
        )
        with self.assertLogs("dynaface_app", "ERROR"):
            self.app_class._on_models_failed(app, "simulated failure")
        return app

    def test_keeps_a_queued_document(self):
        # A document opens without the models, so it still opens once they
        # have failed; it used to be dropped.
        for name in ("/docs/patient.dyfc", "/docs/PATIENT.DYFC"):
            with self.subTest(name):
                app = self.fail(name)
                self.assertTrue(app.models_failed)
                self.assertEqual(app.file_open_request, name)

    def test_drops_a_queued_photo_or_video(self):
        # The failure message says why it can't open.
        for name in ("/docs/face.jpg", "/docs/clip.mov", None):
            with self.subTest(name):
                app = self.fail(name)
                self.assertTrue(app.models_failed)
                self.assertIsNone(app.file_open_request)

    def test_loads_on_the_device_settings_chose_meanwhile(self):
        app = self.fail("/docs/face.jpg", loading="mps", chosen="cpu")
        app._start_model_loader.assert_called_once_with()
        self.assertFalse(app.models_failed)
        self.assertEqual(app.file_open_request, "/docs/face.jpg")


class OpenFileTest(unittest.TestCase):
    def open(self, name, ready=False, failed=False):
        """Runs DynafaceWindow.open_file() on a window that isn't shown."""
        window = DynafaceWindow.__new__(DynafaceWindow)
        window.app = SimpleNamespace(
            models_ready=ready,
            models_failed=failed,
            file_open_request=None,
            MODELS_FAILED_MESSAGE="The models failed.",
        )
        window.display_message_box = mock.Mock()
        window.show_analyze_video = mock.Mock()
        window.update_recent_files = mock.Mock()
        with mock.patch.object(MainWindowJTH, "open_file"):
            DynafaceWindow.open_file(window, name)
        return window

    def test_opens_a_document_once_the_models_have_failed(self):
        window = self.open("/docs/patient.dyfc", failed=True)
        window.show_analyze_video.assert_called_once_with("/docs/patient.dyfc")
        window.display_message_box.assert_not_called()

    def test_says_why_a_photo_or_video_cannot_open_once_the_models_have_failed(self):
        for name in ("/docs/face.jpg", "/docs/clip.mp4"):
            with self.subTest(name):
                window = self.open(name, failed=True)
                window.show_analyze_video.assert_not_called()
                window.display_message_box.assert_called_once_with("The models failed.")

    def test_queues_a_file_while_the_models_load(self):
        window = self.open("/docs/face.jpg")
        window.show_analyze_video.assert_not_called()
        self.assertEqual(window.app.file_open_request, "/docs/face.jpg")


if __name__ == "__main__":
    unittest.main()
