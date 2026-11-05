"""
FrameSeries: a video's outputs frame by frame, and the CSV, peaks and evaluation
built from them. The .NET port's FrameSeriesTests checks the same.
"""

import io
import math
import unittest
from unittest import mock

import numpy as np

from dynaface import facial, measures
from dynaface.const import Pose
from dynaface.video import FrameSeries, Moment

NAN = float("nan")


def series_of(rows, measure_list, fps=0.0):
    series = FrameSeries(measure_list, fps)
    for frame, results in rows:
        series.add(frame, results)
    return series


def same(a, b):
    """Equal, counting NaN (or a missing value) as equal to NaN."""
    a = NAN if a is None else float(a)
    b = NAN if b is None else float(b)
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a))


def load_face(spread):
    """A gray frame with made-up landmarks in place of the models; the pupils are
    230 + spread pixels apart, so each spread measures differently."""
    right, left = (530, 320), (760 + spread, 330)
    landmarks = [
        (right[0] - 80 + (i * 37) % 380, right[1] - 120 + (i * 53) % 400)
        for i in range(98)
    ]
    landmarks[96], landmarks[97] = right, left
    face = facial.AnalyzeFace(measures=measures.all_measures())
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    found = (landmarks, np.array([8.0, 1.0, -2.0, 0.0, 0.0, 0.0]))
    with mock.patch.object(facial.AnalyzeFace, "_find_landmarks", return_value=found):
        face.load_image(frame, crop=True, force_pose=Pose.FRONTAL)
    return face


class TestFrameSeries(unittest.TestCase):
    def test_columns_are_the_enabled_outputs(self):
        eyes = measures.AnalyzeEyeArea()
        eyes.set_item_enabled("eye.diff", False)
        mouth = measures.AnalyzeMouthLength()
        mouth.enabled = False
        series = FrameSeries([eyes, mouth])
        self.assertEqual(series.names, ["eye.left", "eye.right", "eye.ratio"])
        self.assertEqual(
            series.names, facial.AnalyzeFace([eyes, mouth]).get_all_items()
        )

    def test_frames_without_a_face_are_nan(self):
        series = series_of([(0, {"fai": 1.5}), (1, None)], [measures.AnalyzeFAI()])
        self.assertEqual(len(series), 2)
        self.assertEqual(series.values["fai"][0], 1.5)
        self.assertTrue(math.isnan(series.values["fai"][1]))

    def test_csv(self):
        series = series_of(
            [
                (0, {"fai": 1.5, "ml": 50.0}),
                (1, None),
                (2, {"fai": NAN, "ml": 52.123456}),
                (3, {"fai": -0.00001, "ml": float("inf")}),
            ],
            [measures.AnalyzeFAI(), measures.AnalyzeMouthLength()],
            fps=30,
        )
        out = io.StringIO()
        series.write_csv(out)
        # Values to at most 4 decimal places; no face, NaN and infinity are empty.
        self.assertEqual(
            out.getvalue(),
            "frame,time,fai,ml\n0,0.00,1.5,50\n1,0.03,,\n2,0.07,,52.1235\n3,0.10,0,\n",
        )

    def test_csv_without_a_frame_rate(self):
        series = series_of([(4, {"fai": 2.0})], [measures.AnalyzeFAI()])
        out = io.StringIO()
        series.write_csv(out)
        self.assertEqual(out.getvalue(), "frame,time,fai\n4,,2\n")
        self.assertTrue(math.isnan(series.rows()[0]["time"]))

    def test_rows(self):
        series = series_of([(3, {"fai": 1.0})], [measures.AnalyzeFAI()], fps=10)
        self.assertEqual(series.rows(), [{"frame": 3, "time": 0.3, "fai": 1.0}])

    def test_peak(self):
        series = series_of(
            [
                (10, {"eye.left": 5.0, "eye.right": 5.0}),
                (11, {"eye.left": 8.0, "eye.right": NAN}),
                (12, {"eye.left": 6.0, "eye.right": 4.0}),
                (13, {"eye.left": 2.0, "eye.right": 1.0}),
            ],
            [measures.AnalyzeEyeArea()],
        )
        # Frame 11 has no right eye, so it can't count for both; 10 and 12 tie,
        # and the first wins.
        self.assertEqual(series.peak("eye.left", "eye.right"), 10)
        self.assertEqual(series.peak("eye.left"), 11)
        with self.assertRaises(KeyError):
            series.peak("dental_area")
        with self.assertRaises(ValueError):
            series.peak()
        self.assertIsNone(FrameSeries([measures.AnalyzeEyeArea()]).peak("eye.left"))

    def test_evaluation(self):
        both = [measures.AnalyzeDentalArea(), measures.AnalyzeEyeArea()]
        series = series_of(
            [
                (0, {"dental_area": 100.0, "eye.left": 60.0, "eye.right": 50.0}),
                (1, {"dental_area": 400.0, "eye.left": 30.0, "eye.right": 40.0}),
                (2, {"dental_area": 50.0, "eye.left": 70.0, "eye.right": 65.0}),
            ],
            both,
            fps=10,
        )
        evaluation = series.evaluate()
        assert evaluation is not None
        smile, eyes = evaluation.widest_smile, evaluation.widest_eyes
        self.assertEqual((smile.frame, smile.dental_area), (1, 400.0))
        self.assertAlmostEqual(smile.time, 0.1)
        self.assertEqual((smile.eye_left, smile.eye_right), (30.0, 40.0))
        self.assertAlmostEqual(smile.eye_left_over_right, 0.75)
        self.assertAlmostEqual(smile.eye_right_over_left, 40 / 30)
        self.assertEqual(
            (eyes.frame, eyes.dental_area, eyes.eye_total), (2, 50.0, 135.0)
        )

        # Without the outputs, or without a single value, there's nothing to compare.
        dental_only = [measures.AnalyzeDentalArea()]
        self.assertIsNone(
            series_of([(0, {"dental_area": 1.0})], dental_only).evaluate()
        )
        self.assertIsNone(series_of([(0, None)], both).evaluate())

    def test_ratio_of_a_closed_eye(self):
        closed = Moment(frame=0, time=NAN, dental_area=1.0, eye_left=5.0, eye_right=0.0)
        self.assertTrue(math.isnan(closed.eye_left_over_right))
        self.assertEqual(closed.eye_right_over_left, 0.0)


class TestFromStates(unittest.TestCase):
    """Measuring saved states again, as the apps do for a loaded video."""

    @classmethod
    def setUpClass(cls):
        faces = [load_face(spread) for spread in (0, 40, 80)]
        cls.states = [face.dump_state() for face in faces]
        cls.live = [face.analyze() for face in faces]

    def test_same_values_as_the_live_faces(self):
        series = FrameSeries.from_states(self.states, fps=30)
        self.assertEqual(series.frames, [0, 1, 2])
        self.assertEqual(series.names, facial.AnalyzeFace().get_all_items())
        for row, results in enumerate(self.live):
            for name in series.names:
                self.assertTrue(
                    same(series.values[name][row], results.get(name)), f"{name}, {row}"
                )
        # The frames differ, so the series isn't one state repeated.
        self.assertNotAlmostEqual(series.values["ml"][0], series.values["ml"][2])

    def test_ranges_keep_each_frame_number(self):
        self.assertEqual(FrameSeries.from_states(self.states, start=1).frames, [1, 2])
        self.assertEqual(FrameSeries.from_states(self.states, stop=2).frames, [0, 1])
        self.assertEqual(FrameSeries.from_states(self.states, step=2).frames, [0, 2])

    def test_chosen_measures(self):
        series = FrameSeries.from_states(
            self.states, [measures.AnalyzeDentalArea(), measures.AnalyzeEyeArea()]
        )
        evaluation = series.evaluate()
        assert evaluation is not None
        self.assertEqual(evaluation.widest_smile.frame, series.peak("dental_area"))
        self.assertTrue(
            same(
                evaluation.widest_eyes.dental_area,
                self.live[evaluation.widest_eyes.frame]["dental_area"],
            )
        )


if __name__ == "__main__":
    unittest.main()
