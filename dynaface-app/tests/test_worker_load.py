"""
Tests for the video import (WorkerLoad) and the numbers the tab reports from it:

- Every frame is measured with the pupillary distance in force when the import
  started, so saving Settings meanwhile doesn't give one video two scales; the
  Settings tab says the import keeps it.
- Each kept frame carries its number in the video, so the frames without a face
  that the import skips don't shift the later frames' numbers and times in the
  CSV and the Evaluation, while the chart counts kept frames, as the slider does.

The detector is mocked, so these run without the AI models.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import csv
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from dynaface import facial
from dynaface.const import Pose
from dynaface.measures import AnalyzePosition

import tab_eval
import worker_threads
from tab_analyze_video import AnalyzeVideoTab
from tab_settings import SettingsTab

# A frontal face for the mocked detector: 98 landmarks, the pupils level and
# 200 px apart, as dynaface-lib-python's test_persist.py loads one.
LANDMARKS = [(512, 512)] * 96 + [(400, 500), (600, 500)]
POSE = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
FACE = 128  # the gray of a frame with a face
NO_FACE = 0  # the gray of a frame without one


class FakeStream:
    """A video of gray frames, read as cv2.VideoCapture.read() reads one.
    on_read(n), if given, runs as the n-th frame (counting from 1) is read."""

    def __init__(self, grays, on_read=None):
        self._grays = list(grays)
        self._on_read = on_read
        self._reads = 0

    def read(self):
        if self._reads >= len(self._grays):
            return False, None
        gray = self._grays[self._reads]
        self._reads += 1
        if self._on_read is not None:
            self._on_read(self._reads)
        return True, np.full((1024, 1024, 3), gray, dtype=np.uint8)


def _find_landmarks(face, img):
    """The detector, mocked: no face on a NO_FACE frame (whichever way the import
    turns it), else LANDMARKS and POSE."""
    if int(img[0, 0, 0]) == NO_FACE:
        return [], np.array([0, 0, 0])
    return list(LANDMARKS), POSE.copy()


def run_import(grays, on_read=None):
    """Runs a WorkerLoad over these frames on this thread, without smoothing, and
    returns its target, a stand-in for the tab holding the states it kept and
    their numbers."""
    states, numbers = [], []
    target = SimpleNamespace(
        frame_count=len(grays),
        base_rotation=None,
        force_pose=Pose.FRONTAL,
        video_stream=FakeStream(grays, on_read),
        loading=False,
        frames_read=0,
        load_failure=None,
        states=states,
        numbers=numbers,
        add_frame=lambda state, number: (states.append(state), numbers.append(number)),
    )
    worker = worker_threads.WorkerLoad(target)
    app = SimpleNamespace(data_smoothing=1, tilt_threshold=-1)
    with mock.patch(
        "worker_threads.QApplication", **{"instance.return_value": app}
    ), mock.patch.object(facial.AnalyzeFace, "_find_landmarks", _find_landmarks):
        worker.run()
    if target.load_failure is not None:
        raise target.load_failure
    return target


def _pupil_distance(state):
    """The pupillary distance a state was measured with, as load_state() finds it."""
    return round(state[3] * state[4], 4)


class _KeepsTheSetting(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, facial.AnalyzeFace, "pd", facial.AnalyzeFace.pd)
        facial.AnalyzeFace.pd = 63


class PupilDistanceTest(_KeepsTheSetting):
    def test_a_settings_save_during_an_import_leaves_it_one_scale(self):
        def save_settings(reads):
            if reads == 3:
                # Settings > Save, as load_dynaface_settings() applies it.
                facial.AnalyzeFace.pd = 50

        target = run_import([FACE] * 5, on_read=save_settings)
        self.assertEqual([_pupil_distance(s) for s in target.states], [63.0] * 5)

        # The next import measures with the new one.
        target = run_import([FACE] * 2)
        self.assertEqual([_pupil_distance(s) for s in target.states], [50.0] * 2)


class SettingsNoteTest(_KeepsTheSetting):
    def save(self, pd_text, importing):
        """Runs SettingsTab.action_save() on a stand-in for the tab, as if a video
        were importing or not."""
        window = SimpleNamespace(
            app=SimpleNamespace(would_reload_models=lambda accelerator: False),
            display_message_box=mock.Mock(),
            close_current_tab=mock.Mock(),
        )
        tab = SimpleNamespace(
            _window=window,
            _text_pd=SimpleNamespace(text=lambda: pd_text),
            _chk_accelerator=SimpleNamespace(isChecked=lambda: True),
            check_values=lambda: True,
            detection_changed=lambda: False,
            _video_importing=lambda: importing,
            save_values=mock.Mock(),
        )
        SettingsTab.action_save(tab)
        tab.save_values.assert_called_once_with()
        window.close_current_tab.assert_called_once_with()
        return window.display_message_box

    def test_says_an_import_keeps_its_pupillary_distance(self):
        message = self.save("50", importing=True)
        message.assert_called_once()
        self.assertIn("keeps the pupillary distance", message.call_args.args[0])

    def test_quiet_without_an_import_or_a_change(self):
        self.save("50", importing=False).assert_not_called()
        self.save("63", importing=True).assert_not_called()


class FrameNumberTest(unittest.TestCase):
    # The video's frames 1 and 2 (counting from 0) have no face.
    VIDEO = [FACE, NO_FACE, NO_FACE, FACE, FACE]

    @classmethod
    def setUpClass(cls):
        cls.imported = run_import(cls.VIDEO)

    def test_kept_frames_carry_their_numbers_in_the_video(self):
        self.assertEqual(self.imported.numbers, [0, 3, 4])
        self.assertEqual(self.imported.frames_read, 5)

    def tab(self, numbers=True):
        """A stand-in for the tab after the import, as collect_data() and the
        Evaluation see it."""
        tab = SimpleNamespace(
            _frames=list(self.imported.states),
            _frame_numbers=list(self.imported.numbers) if numbers else None,
            _frame_begin=0,
            _frame_end=len(self.imported.states),
            frame_rate=30,
            _chosen_measures=lambda: [AnalyzePosition()],
        )
        tab.numbered_states = lambda: AnalyzeVideoTab.numbered_states(tab)
        return tab

    def test_csv_rows_keep_the_videos_numbers_and_times(self):
        out = io.StringIO()
        AnalyzeVideoTab.collect_data(self.tab()).write_csv(out)
        rows = list(csv.reader(io.StringIO(out.getvalue())))
        self.assertEqual(rows[0][:2], ["frame", "time"])
        # Before, these read 0, 1, 2 and 0.00, 0.03, 0.07 s.
        self.assertEqual(
            [row[:2] for row in rows[1:]],
            [["0", "0.00"], ["3", "0.10"], ["4", "0.13"]],
        )

    def test_a_trimmed_clip_keeps_its_numbers(self):
        tab = self.tab()
        tab._frame_begin = 1
        self.assertEqual(AnalyzeVideoTab.collect_data(tab).frames, [3, 4])

    def test_the_chart_counts_kept_frames_as_the_slider_does(self):
        series = AnalyzeVideoTab.collect_data(self.tab(), by_position=True)
        self.assertEqual(series.frames, [0, 1, 2])

    def test_frames_without_numbers_are_numbered_by_position(self):
        # A photo, or a document saved before the numbers were kept.
        series = AnalyzeVideoTab.collect_data(self.tab(numbers=False))
        self.assertEqual(series.frames, [0, 1, 2])

    def test_numbers_agree_with_the_states_while_frames_arrive(self):
        # add_frame() adds a frame's number before its state, so a reader can
        # find one number more than states: it takes as many as there are states.
        tab = self.tab()
        tab._frame_numbers.append(5)
        states, numbers, stop = AnalyzeVideoTab.numbered_states(tab)
        self.assertEqual((len(states), numbers, stop), (3, [0, 3, 4], 3))

    def test_the_evaluation_numbers_frames_as_in_the_video(self):
        evaluation = SimpleNamespace(_window=SimpleNamespace(_background_queue=[]))
        with mock.patch.object(
            tab_eval.FrameSeries,
            "from_states",
            wraps=tab_eval.FrameSeries.from_states,
        ) as from_states:
            tab_eval.TabEval.exec_eval(evaluation, self.tab())
        self.assertEqual(from_states.call_args.kwargs["frame_numbers"], [0, 3, 4])
        self.assertEqual(len(evaluation._window._background_queue), 1)


if __name__ == "__main__":
    unittest.main()
