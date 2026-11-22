"""
Tests for what AnalyzeVideoTab.save_document() tells the user when saving a
document fails, by how it failed (see utl_save.save_file()), and that the
document stays unsaved then; and that once a save succeeds, a later Save writes
the file it saved.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import errno
import unittest
from types import SimpleNamespace
from unittest import mock

import dlg_modal
import tab_analyze_video
import utl_save

_DOCUMENT = "/Volumes/Share/Patient 07.dyfc"
# The file the tab was opened from, or last saved.
_OPENED = "/Volumes/Share/Patient 07 (first visit).dyfc"
_KEPT = "/Volumes/Share/.dynaface-b94443eb88b6.tmp"


def _tab():
    """A stand-in for the tab, as save_document() sees it."""
    return SimpleNamespace(
        wait_load_complete=lambda: True,
        _chosen_measures=lambda: [],
        _frames=[],
        _frame_numbers=None,
        _frame_begin=0,
        _frame_end=0,
        frame_rate=30,
        _face=SimpleNamespace(pose=None, lateral=False),
        _window=SimpleNamespace(
            display_message_box=mock.Mock(), set_tab_title=mock.Mock()
        ),
        unsaved_changes=True,
        filename=_OPENED,
    )


def _missing(kept):
    error = utl_save.MissingFileError(
        errno.ENOENT, "No such file or directory", _DOCUMENT
    )
    error.kept = kept
    return error


class SaveMessageTest(unittest.TestCase):
    def save(self, error, platform="darwin"):
        """Runs save_document() on a tab whose save ends with error, on
        platform, and returns the tab."""
        tab = _tab()
        with mock.patch.object(
            dlg_modal, "display_please_wait", return_value=error
        ), mock.patch.object(
            tab_analyze_video, "sys", SimpleNamespace(platform=platform)
        ):
            tab_analyze_video.AnalyzeVideoTab.save_document(tab, _DOCUMENT)
        return tab

    def assert_told(self, tab, message):
        tab._window.display_message_box.assert_called_once_with(message)
        self.assertTrue(tab.unsaved_changes)
        # Save still writes the file it wrote before.
        self.assertEqual(tab.filename, _OPENED)
        tab._window.set_tab_title.assert_not_called()

    def test_names_the_copy_kept_and_how_to_open_it(self):
        # Dynaface opens no file of the staged file's name, and Finder hides it.
        told = (
            f"Unable to save {_DOCUMENT}: No such file or directory. The file may "
            "now be missing, so save the document again, or under another name. "
            f"A copy of the new version was kept in {_KEPT}."
        )
        opening = (
            ' To open it, rename it to end in .dyfc, for example "Patient 07 '
            '(recovered).dyfc".'
        )
        hidden = " Finder hides it until you press Command-Shift-Period."
        for platform, message in (
            ("darwin", told + hidden + opening),
            ("win32", told + opening),
        ):
            with self.subTest(platform):
                self.assert_told(self.save(_missing(_KEPT), platform), message)

    def test_says_the_file_may_hold_the_new_version_when_nothing_was_kept(self):
        # A replace reported failing that was done leaves no staged file.
        self.assert_told(
            self.save(_missing(None)),
            f"Unable to save {_DOCUMENT}: No such file or directory. The file may "
            "now be missing, or may hold the new version already, so save the "
            "document again, or under another name.",
        )

    def test_says_the_file_may_be_incomplete_when_writing_over_it_failed(self):
        error = utl_save.IncompleteSaveError(errno.EIO, "Input/output error")
        self.assert_told(
            self.save(error),
            f"Unable to save {_DOCUMENT}: Input/output error. The file may now be "
            "incomplete, so save the document again, or under another name.",
        )

    def test_says_the_file_was_not_changed_otherwise(self):
        error = OSError(errno.ENOSPC, "No space left on device")
        self.assert_told(
            self.save(error),
            f"Unable to save {_DOCUMENT}: No space left on device. The file "
            "wasn't changed.",
        )

    def test_marks_the_document_saved_once_it_is(self):
        tab = self.save(None)
        tab._window.display_message_box.assert_not_called()
        self.assertFalse(tab.unsaved_changes)

    def test_saves_later_to_the_file_it_saved(self):
        # After Save As, Save wrote over the file the tab was opened from.
        tab = self.save(None)
        self.assertEqual(tab.filename, _DOCUMENT)
        tab._window.set_tab_title.assert_called_once_with(
            tab, "Analyze: Patient 07.dyfc"
        )
        tab.save_document = mock.Mock()
        tab_analyze_video.AnalyzeVideoTab.on_save(tab)
        tab.save_document.assert_called_once_with(_DOCUMENT)


if __name__ == "__main__":
    unittest.main()
