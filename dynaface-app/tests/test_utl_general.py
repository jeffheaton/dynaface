"""Tests for utl_general: assert_standard_python, which checks what a document
can hold, and the view and chart helpers of the analysis tab.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import math
import unittest

import numpy as np
from dynaface.const import Pose
from dynaface.measures import AnalyzeDentalArea, AnalyzeEyeArea
from dynaface.video import FrameSeries

from utl_general import POSE_LABELS, assert_standard_python, named_series, view_label


class _NoToList(np.ndarray):
    def tolist(self):
        raise AssertionError("a numeric array was walked element by element")


class AssertStandardPythonTest(unittest.TestCase):
    def test_accepts_numeric_arrays_without_walking_them(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8).view(_NoToList)
        assert_standard_python({"frames": [[image, np.arange(3.0).view(_NoToList)]]})

    def test_checks_the_elements_of_object_arrays(self):
        values = np.empty(2, dtype=object)
        values[0], values[1] = 1.5, object()
        with self.assertRaises(TypeError):
            assert_standard_python({"frames": [values]})

    def test_accepts_object_arrays_of_standard_values(self):
        values = np.empty(2, dtype=object)
        values[0], values[1] = 1.5, (1, "x")
        assert_standard_python(values)

    def test_rejects_other_objects(self):
        with self.assertRaises(TypeError):
            assert_standard_python({"frames": [[object()]]})


class ViewLabelTest(unittest.TestCase):
    def test_names_each_view(self):
        self.assertEqual(view_label(Pose.FRONTAL, False), "frontal")
        self.assertEqual(view_label(Pose.LATERAL, False), "lateral")
        self.assertEqual(
            view_label(Pose.QUARTER, True), "three-quarter, far cheek left"
        )
        self.assertEqual(
            view_label(Pose.QUARTER, False), "three-quarter, far cheek right"
        )

    def test_far_left_only_matters_on_a_quarter_face(self):
        self.assertEqual(view_label(Pose.FRONTAL, True), "frontal")
        self.assertEqual(view_label(Pose.LATERAL, True), "lateral")

    def test_an_unknown_pose_is_frontal(self):
        self.assertEqual(view_label(None, False), "frontal")
        self.assertEqual(view_label(Pose.DETECT, True), "frontal")

    def test_builds_on_the_pose_labels(self):
        # One place names the poses for the user: the labels the dialogs read.
        self.assertEqual(
            POSE_LABELS,
            {
                Pose.FRONTAL: "frontal",
                Pose.LATERAL: "lateral",
                Pose.QUARTER: "three-quarter",
            },
        )
        for pose, name in POSE_LABELS.items():
            self.assertTrue(view_label(pose, False).startswith(name))


class NamedSeriesTest(unittest.TestCase):
    def _series(self):
        return FrameSeries([AnalyzeDentalArea(), AnalyzeEyeArea()], fps=10)

    def test_keeps_the_outputs_with_a_finite_value_in_order(self):
        series = self._series()
        series.add(0, {"dental_area": math.nan, "eye.left": 1.5})
        series.add(1, {"dental_area": 2.0, "eye.left": math.nan})
        series.add(2, None)
        self.assertEqual(named_series(series), ["dental_area", "eye.left"])

    def test_drops_an_output_that_is_nan_on_every_frame(self):
        series = self._series()
        series.add(0, {"dental_area": 3.0})
        series.add(1, {"dental_area": 4.0, "eye.right": math.inf})
        self.assertEqual(named_series(series), ["dental_area"])

    def test_an_empty_series_names_nothing(self):
        self.assertEqual(named_series(self._series()), [])

    def test_accepts_any_object_with_names_and_values(self):
        class Plain:
            names = ["a", "b", "c"]
            values = {"a": [None, "x"], "b": [float("nan"), 0.0], "c": []}

        self.assertEqual(named_series(Plain()), ["b"])


if __name__ == "__main__":
    unittest.main()
