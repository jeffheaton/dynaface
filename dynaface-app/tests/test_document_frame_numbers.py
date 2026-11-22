"""
Tests for two things a .dyfc document carries for its frames:

- Each frame's number in the video (DynafaceDocument.frame_numbers): the import
  skips frames without a face, so the numbers keep a reopened document's CSV
  and Evaluation numbers and times right. A document saved without them, or
  with numbers that aren't one rising whole number per frame, numbers its
  frames by position, as before.
- The head pose of a frame saved by Dynaface 2.0.4 or earlier, a placeholder:
  it restores as unknown (NaN), also once add_missing_views() gives a profile
  document's frames their view, and stays unknown when saved again.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import json
import math
import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
from dynaface import facial
from dynaface.const import Pose

import dlg_modal
import dynaface_document
from dynaface_document import DynafaceDocument
from tab_analyze_video import AnalyzeVideoTab


def _state(headpose=(0, 0, 0)):
    """A frame state as Dynaface 2.0.4 saved it: six entries and no view, with
    its [0, 0, 0] placeholder for the head pose; pupils 24 px apart, at 63 mm."""
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    landmarks = [(32, 32)] * 96 + [(20, 30), (44, 30)]
    return [image, list(headpose), landmarks, 24.0, 63.0 / 24.0, None]


def _document(lateral=False, frames=3):
    doc = DynafaceDocument()
    doc.frames = [_state() for _ in range(frames)]
    doc.fps = 30
    doc.lateral = lateral
    doc.pose = Pose.LATERAL if lateral else Pose.FRONTAL
    if lateral:
        doc.lateral_landmarks = np.arange(12, dtype=np.int32).reshape(6, 2)
        doc.sagittal_x = np.arange(5, dtype=np.int32)
        doc.sagittal_y = np.arange(5, 10, dtype=np.int32)
    return doc


class _InAFolder(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "test.dyfc")

    def load(self):
        doc = DynafaceDocument()
        doc.load(self.path)
        return doc

    def saved_json(self):
        with zipfile.ZipFile(self.path) as zf:
            return json.loads(zf.read("document.json"))


class FrameNumbersTest(_InAFolder):
    def test_survive_saving(self):
        doc = _document()
        doc.frame_numbers = [0, 3, 4]
        doc.save(self.path)
        self.assertEqual(self.load().frame_numbers, [0, 3, 4])

    def test_a_document_saved_without_them_numbers_by_position(self):
        _document().save(self.path)
        self.assertNotIn("frame_numbers", self.saved_json()["body"])
        self.assertIsNone(self.load().frame_numbers)

    def test_numbers_that_do_not_fit_the_frames_are_left_out(self):
        for numbers in (
            [0, 3],  # not one per frame
            [0, 3, 4, 5],
            [0, 4, 3],  # not rising
            [0, 3, 3],
            [-1, 3, 4],
            [0, 3, 2**31],
            [0, 3, 4.0],
            [0, 3, True],
            [0, 3, "4"],
            "0,3,4",
            {"0": 0},
            None,
        ):
            with self.subTest(numbers=numbers):
                self.assertIsNone(dynaface_document._frame_numbers(numbers, 3))
        doc = _document()
        doc.frame_numbers = [0, 3]
        doc.save(self.path)
        self.assertIsNone(self.load().frame_numbers)

    def tab(self, numbers):
        """A stand-in for the tab, as save_document() and load_document() see it,
        holding three frames with these numbers."""
        return SimpleNamespace(
            wait_load_complete=lambda: True,
            _chosen_measures=lambda: [],
            _frames=[_state() for _ in range(3)],
            _frame_numbers=numbers,
            _frame_begin=0,
            _frame_end=3,
            frame_rate=30,
            _face=SimpleNamespace(pose=Pose.FRONTAL, lateral=False),
            _window=SimpleNamespace(
                display_message_box=mock.Mock(), set_tab_title=mock.Mock()
            ),
            unsaved_changes=True,
        )

    def run_and_wait(self, window, f, message):
        """display_please_wait(), without the dialog: runs f and returns its
        error, or None."""
        try:
            f()
        except Exception as e:  # noqa: BLE001 - the dialog returns any error
            return e
        return None

    def save_tab(self, tab):
        with mock.patch.object(
            dlg_modal, "display_please_wait", side_effect=self.run_and_wait
        ):
            AnalyzeVideoTab.save_document(tab, self.path)
        tab._window.display_message_box.assert_not_called()

    def test_the_tab_saves_a_trimmed_clips_numbers(self):
        tab = self.tab([0, 3, 4])
        tab._frame_begin = 1
        self.save_tab(tab)
        doc = self.load()
        self.assertEqual(len(doc.frames), 2)
        self.assertEqual(doc.frame_numbers, [3, 4])

    def test_the_tab_saves_none_it_does_not_know(self):
        self.save_tab(self.tab(None))
        self.assertIsNone(self.load().frame_numbers)

    def test_the_tab_opens_them_with_the_document(self):
        doc = _document()
        doc.frame_numbers = [2, 5, 9]
        doc.save(self.path)
        tab = SimpleNamespace()
        app = SimpleNamespace(tilt_threshold=-1)
        with mock.patch(
            "tab_analyze_video.QApplication", **{"instance.return_value": app}
        ), mock.patch.object(
            dlg_modal, "display_please_wait", side_effect=self.run_and_wait
        ):
            self.assertTrue(AnalyzeVideoTab.load_document(tab, self.path))
        self.assertEqual(tab._frame_numbers, [2, 5, 9])
        self.assertEqual(len(tab._frames), 3)


class OldHeadPoseTest(_InAFolder):
    def restore(self, state):
        face = facial.AnalyzeFace([])
        face.load_state(state)
        return face

    def assert_unknown_pose(self, face):
        for name in ("yaw", "pitch", "roll"):
            self.assertTrue(math.isnan(getattr(face, name)), name)

    def test_a_profile_documents_frames_get_an_unknown_pose_with_their_view(self):
        doc = _document(lateral=True)
        doc.add_missing_views()
        for state in doc.frames:
            self.assertEqual(len(state), 7)
            self.assertTrue(all(math.isnan(v) for v in state[1]))
            face = self.restore(state)
            self.assertTrue(face.lateral)
            self.assert_unknown_pose(face)

    def test_frames_with_a_view_keep_their_pose(self):
        doc = _document(lateral=True, frames=1)
        doc.frames[0] = doc.frames[0] + [{"pose": "frontal", "flipped": False}]
        doc.frames[0][1] = [1.5, -2.0, 0.25]
        doc.add_missing_views()
        self.assertEqual(doc.frames[0][1], [1.5, -2.0, 0.25])

    def test_a_frontal_documents_frames_restore_an_unknown_pose_as_they_are(self):
        doc = _document()
        doc.add_missing_views()
        self.assertEqual([len(state) for state in doc.frames], [6, 6, 6])
        self.assert_unknown_pose(self.restore(doc.frames[0]))

    def test_the_unknown_pose_survives_saving(self):
        # A reopened 2.0.4 frame, saved again by this version: dump_state() now
        # writes a view and the NaN pose, which the document must load back.
        face = self.restore(_state())
        doc = _document(frames=0)
        doc.frames = [face.dump_state()]
        doc.save(self.path)
        state = self.load().frames[0]
        self.assertIsInstance(state[6], dict)
        self.assert_unknown_pose(self.restore(state))


if __name__ == "__main__":
    unittest.main()
