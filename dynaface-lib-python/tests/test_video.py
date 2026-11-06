"""
FrameSeries: a video's outputs frame by frame, and the CSV, peaks and evaluation
built from them; and VideoAnalyzer and LandmarkSmoother, which import a video.
The .NET port's FrameSeriesTests and VideoAnalyzerTests check the same.
"""

import io
import math
import unittest
from unittest import mock

import numpy as np

from dynaface import facial, measures
from dynaface.const import Pose
from dynaface.video import FrameSeries, LandmarkSmoother, Moment, VideoAnalyzer

NAN = float("nan")
FRAME = np.full((720, 1280, 3), 128, dtype=np.uint8)
NO_FACE = ([], np.array([0, 0, 0]))


def detected(spread):
    """What the models would find in a made-up frame: 98 landmarks and a head pose.
    The pupils are 230 + spread pixels apart, so each spread measures differently."""
    right, left = (530, 320), (760 + spread, 330)
    landmarks = [
        (right[0] - 80 + (i * 37) % 380, right[1] - 120 + (i * 53) % 400)
        for i in range(98)
    ]
    landmarks[96], landmarks[97] = right, left
    return landmarks, np.array([8.0, 1.0, -2.0, 0.0, 0.0, 0.0])


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
    """A gray frame analyzed with made-up detections in place of the models."""
    face = facial.AnalyzeFace(measures=measures.all_measures())
    found = detected(spread)
    with mock.patch.object(facial.AnalyzeFace, "_find_landmarks", return_value=found):
        face.load_image(FRAME, crop=True, force_pose=Pose.FRONTAL)
    return face


def import_video(detections, **options):
    """Run a VideoAnalyzer over gray frames, one per detection (NO_FACE for none)."""
    analyzer = VideoAnalyzer(force_pose=Pose.FRONTAL, **options)
    with mock.patch.object(
        facial.AnalyzeFace, "_find_landmarks", side_effect=detections
    ):
        for _ in detections:
            analyzer.add(FRAME)
    return analyzer


class TestForcedProfile(unittest.TestCase):
    def test_frame_without_a_face_is_skipped(self):
        # Treated as a profile, a frame without a face used to be traced anyway,
        # which failed and stopped the import.
        analyzer = VideoAnalyzer(force_pose=Pose.LATERAL)
        with mock.patch.object(
            facial.AnalyzeFace, "_find_landmarks", return_value=NO_FACE
        ), mock.patch("dynaface.lateral.analyze_lateral") as traced:
            self.assertFalse(analyzer.add(FRAME))
        traced.assert_not_called()
        self.assertEqual(analyzer.states, [])


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


class TestLandmarkSmoother(unittest.TestCase):
    def test_averages_the_last_frames(self):
        smoother = LandmarkSmoother(2)
        self.assertEqual(smoother.add([(10, 20), (30, 41)]), [(10, 20), (30, 41)])
        # Averages are truncated to whole pixels: 11.5 is 11, 40.5 is 40.
        self.assertEqual(smoother.add([(13, 25), (31, 40)]), [(11, 22), (30, 40)])
        # Only the last 2 frames count.
        self.assertEqual(smoother.add([(20, 30), (30, 40)]), [(16, 27), (30, 40)])
        smoother.reset()
        self.assertEqual(smoother.add([(0, 0), (1, 1)]), [(0, 0), (1, 1)])

    def test_size_one_is_off(self):
        smoother = LandmarkSmoother(1)
        smoother.add([(10, 20)])
        self.assertEqual(smoother.add([(13, 25)]), [(13, 25)])
        with self.assertRaises(ValueError):
            LandmarkSmoother(0)


class TestVideoAnalyzer(unittest.TestCase):
    """Importing a video, with made-up detections in place of the models."""

    def test_frames_without_a_face_are_skipped_but_counted(self):
        analyzer = import_video([detected(0), NO_FACE, detected(40)])
        self.assertEqual(analyzer.frames, [0, 2])
        self.assertEqual(len(analyzer.states), 2)
        self.assertIs(analyzer.thumbnail, analyzer.states[0][0])
        # The series keeps the frames' numbers, so their times are right too.
        series = analyzer.series([measures.AnalyzeFAI()], fps=30)
        self.assertEqual(series.frames, [0, 2])
        self.assertAlmostEqual(series.time(2), 2 / 30)

    def test_frames_that_couldnt_be_read(self):
        analyzer = VideoAnalyzer(force_pose=Pose.FRONTAL)
        with mock.patch.object(
            facial.AnalyzeFace, "_find_landmarks", side_effect=[detected(0)] * 2
        ):
            analyzer.add(FRAME)
            analyzer.skip()
            analyzer.add(FRAME)
        self.assertEqual(analyzer.frames, [0, 2])

    def test_only_the_first_frames(self):
        analyzer = import_video([detected(0)] * 3, max_frames=2)
        self.assertEqual(analyzer.frames, [0, 1])
        self.assertTrue(analyzer.full)
        self.assertFalse(analyzer.add(FRAME))
        self.assertIsNone(VideoAnalyzer().thumbnail)

    def test_without_smoothing_each_frame_is_as_analyzed(self):
        analyzer = import_video([detected(0), detected(80)])
        for state, spread in zip(analyzer.states, (0, 80)):
            face = load_face(spread)
            self.assertEqual(state[2], face.landmarks)
            self.assertEqual(state[4], face.pix2mm)

    def test_smoothing(self):
        analyzer = import_video([detected(0), detected(80)], smoothing=2)
        first, second = load_face(0), load_face(80)
        smoother = LandmarkSmoother(2)
        smoother.add(first.landmarks)
        averaged = smoother.add(second.landmarks)
        self.assertEqual(analyzer.states[0][2], first.landmarks)
        self.assertEqual(analyzer.states[1][2], averaged)
        self.assertNotEqual(averaged, second.landmarks)
        # The saved scale is the averaged landmarks' own.
        pd, pix2mm = facial.util_calc_pd(facial.util_get_pupils(averaged))
        self.assertEqual(analyzer.states[1][3:5], [pd, pix2mm])

    def test_frame_numbers_must_match_the_states(self):
        analyzer = import_video([detected(0), detected(40)])
        with self.assertRaises(ValueError):
            FrameSeries.from_states(analyzer.states, frame_numbers=[0])


if __name__ == "__main__":
    unittest.main()
