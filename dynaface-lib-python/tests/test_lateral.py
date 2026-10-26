"""Tests for the side-profile analysis in dynaface.lateral."""

import json
import os
import sys
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

from dynaface import facial, lateral, models
from dynaface.lateral import LateralSearchMode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROFILE = "./tests_data/img2-1024-right-lateral.jpg"
MISSING = [-1.0, -1.0]


class TestLateral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())

    def test_debug_plots(self):
        # With DEBUG on, analyze_lateral() also marks the outline's extrema,
        # corners and quarter lines on its chart. That changes the drawing, but
        # not the landmarks it finds.
        plain = facial.load_face_image(PROFILE, measures=[])
        with mock.patch.object(lateral, "DEBUG", True):
            debug = facial.load_face_image(PROFILE, measures=[])

        self.assertTrue(debug.lateral)
        np.testing.assert_array_equal(debug.lateral_landmarks, plain.lateral_landmarks)
        self.assertFalse(np.array_equal(debug.render_img, plain.render_img))

    def test_background_removal_needs_models(self):
        with mock.patch.object(models, "onnx_model", None):
            with self.assertRaises(ValueError):
                lateral._process_image(np.zeros((8, 8, 3), dtype=np.uint8))


class TestSharedProfiles(unittest.TestCase):
    """
    The synthetic outlines in tests_data/lateral_profiles.json, which
    dynaface-lib-dotnet's LateralProfileTests also checks, so both ports must find
    the same landmarks. scripts/make_lateral_profiles.py writes the file.
    """

    def test_profiles(self):
        with open("./tests_data/lateral_profiles.json", encoding="utf-8") as f:
            data = json.load(f)
        for case in data["cases"]:
            with self.subTest(case["name"]):
                outline = data["outlines"][case["outline"]]
                xs = np.array(outline["sagittal_x"])
                rows = np.array(outline["sagittal_y"])
                if case["search_only"]:
                    landmarks = lateral._find_lateral_landmarks(
                        xs,
                        rows,
                        np.array(case["max_indices"], dtype=int),
                        np.array(case["min_indices"], dtype=int),
                        np.array(case["corner_idxs"], dtype=int),
                        case["shift_x"],
                        case["frontal"],
                    )
                else:
                    found_max, found_min, corners, landmarks = (
                        lateral._find_profile_landmarks(
                            xs, rows, case["shift_x"], case["frontal"]
                        )
                    )
                    self.assertEqual(found_max.tolist(), case["max_indices"])
                    self.assertEqual(found_min.tolist(), case["min_indices"])
                    self.assertEqual(corners.tolist(), case["corner_idxs"])
                self.assertEqual(landmarks.tolist(), case["landmarks"])


# A small outline: x by row, rows 10 to 19.
XS = np.array([5, 3, 1, 0, 2, 4, 6, 4, 2, 3], dtype=float)
ROWS = np.arange(10, 20)


def with_nan(xs, start, stop):
    out = np.array(xs, dtype=float)
    out[start:stop] = np.nan
    return out


class TestSearchHelpers(unittest.TestCase):
    """Paths the full search never takes, such as values that aren't numbers."""

    def test_find_lateral_landmark(self):
        def find(mode, y_forward=None, y=14.5):
            point = lateral._find_lateral_landmark(
                XS,
                ROWS,
                np.array([6]),
                np.array([3]),
                np.array([2, 8]),
                y,
                mode,
                y_forward,
            )
            return point.tolist()

        self.assertEqual(find(LateralSearchMode.MAX), [6.0, 16.0])
        self.assertEqual(find(LateralSearchMode.CORNER, y_forward=True), [2.0, 18.0])
        self.assertEqual(find(LateralSearchMode.CORNER, y_forward=False), [1.0, 12.0])
        self.assertEqual(find(LateralSearchMode.MIN, y_forward=True, y=15), MISSING)
        self.assertEqual(find(None), MISSING)  # not a search mode

    def test_corner_landmark_in_range(self):
        def find(xs=XS, rows=ROWS, corners=(), lo=None, hi=None, **frontal):
            return lateral._find_corner_landmark_in_range(
                xs, rows, np.array(corners, dtype=int), lo, hi, **frontal
            ).tolist()

        self.assertEqual(find(xs=np.array([]), rows=np.array([])), MISSING)
        self.assertEqual(find(rows=ROWS[:5]), MISSING)  # lengths differ
        self.assertEqual(find(), MISSING)  # neither a range nor frontal landmarks
        frontal = np.array([[0, 12], [0, 17]])
        self.assertEqual(
            find(landmarks_frontal=frontal, frontal_lo_idx=0, frontal_hi_idx=2),
            MISSING,  # landmark 2 doesn't exist
        )
        self.assertEqual(find(lo=7, hi=3), [6.0, 16.0])  # bounds in either order
        nan_window = with_nan(XS, 2, 8)
        self.assertEqual(find(xs=nan_window, lo=3, hi=6), MISSING)
        # With no usable x in the window, the first corner in it is chosen.
        x, y = find(xs=nan_window, corners=[5], lo=3, hi=6)
        self.assertTrue(np.isnan(x))
        self.assertEqual(y, 15.0)
        self.assertEqual(find(xs=nan_window, corners=[9], lo=3, hi=6), MISSING)

    def test_landmark_in_range(self):
        def find(xs=XS, rows=ROWS, lo=None, hi=None, pick=LateralSearchMode.MIN, **kw):
            return lateral._find_lateral_landmark_in_range(
                xs, rows, lo, hi, pick=pick, **kw
            ).tolist()

        self.assertEqual(find(xs=np.array([]), rows=np.array([])), MISSING)
        self.assertEqual(find(), MISSING)  # neither a range nor frontal landmarks
        frontal = np.array([[0, 12], [0, 17]])
        self.assertEqual(
            find(landmarks_frontal=frontal, frontal_lo_idx=0, frontal_hi_idx=5),
            MISSING,  # landmark 5 doesn't exist
        )
        self.assertEqual(find(lo=8, hi=1), [0.0, 13.0])
        self.assertEqual(find(lo=4, hi=8, pick=LateralSearchMode.MAX), [6.0, 16.0])
        self.assertEqual(find(lo=0, hi=9, pick=LateralSearchMode.CORNER), MISSING)
        self.assertEqual(find(xs=with_nan(XS, 0, 10), lo=0, hi=9), MISSING)

    def test_landmark_minmax(self):
        def find(xs=XS, rows=ROWS, lo=None, hi=None, pick=LateralSearchMode.MAX):
            return lateral._find_lateral_landmark_minmax(
                xs, rows, lo, hi, pick=pick
            ).tolist()

        self.assertEqual(find(xs=np.array([]), rows=np.array([])), MISSING)
        self.assertEqual(find(), [6.0, 16.0])  # the whole outline
        self.assertEqual(find(lo=8, hi=1, pick=LateralSearchMode.MIN), [0.0, 13.0])
        self.assertEqual(find(pick=LateralSearchMode.NEAREST), MISSING)
        self.assertEqual(find(xs=with_nan(XS, 0, 10)), MISSING)

    def test_corner_finder_edges(self):
        empty = np.array([], dtype=np.int64)
        self.assertEqual(lateral._exclude_near(empty, [3]).tolist(), [])
        self.assertEqual(lateral._exclude_near(np.array([3, 9]), []).tolist(), [3, 9])
        self.assertEqual(lateral._nms_keep_best(empty, np.array([])).tolist(), [])
        self.assertEqual(lateral._nms_keep_best(np.array([4]), np.array([1.0])), [4])
        self.assertEqual(lateral._angle_change(np.ones(5), 0).tolist(), [0.0] * 5)
        self.assertEqual(lateral._angle_change(np.ones(5), 3).tolist(), [0.0] * 5)

        # A face whose corners are all near extrema finds none.
        face = 40 + 30 * np.sin(np.arange(200) / 15.0)
        corners = lateral._find_monotonic_corners(face, scales=[9])
        self.assertGreater(len(corners), 0)
        excluded = lateral._find_monotonic_corners(
            face, scales=[9], exclude_extrema=range(200)
        )
        self.assertEqual(excluded.tolist(), [])
        # When no scale finds a candidate, there are no corners.
        no_peaks = (np.array([], dtype=np.int64), {})
        with mock.patch.object(lateral, "find_peaks", return_value=no_peaks):
            self.assertEqual(lateral._find_monotonic_corners(face).tolist(), [])

    def test_pogonion_not_found(self):
        # With no usable x across the chin, the pogonion isn't found, so the
        # mentolabial point has nothing to overlap.
        with open("./tests_data/lateral_profiles.json", encoding="utf-8") as f:
            data = json.load(f)
        case = next(c for c in data["cases"] if c["name"] == "face")
        xs = np.array(data["outlines"]["face"]["sagittal_x"], dtype=float)
        rows = np.array(data["outlines"]["face"]["sagittal_y"])
        chin = (rows >= 380) & (rows <= 410)  # landmarks 14 and 16 are at 380 and 410
        xs[chin] = np.nan
        landmarks = lateral._find_lateral_landmarks(
            xs,
            rows,
            np.array(case["max_indices"]),
            np.array(case["min_indices"]),
            np.array(case["corner_idxs"]),
            case["shift_x"],
            case["frontal"],
        ).tolist()
        self.assertEqual(landmarks[5], [case["shift_x"] - 1, -1])
        self.assertEqual(landmarks[:4], case["landmarks"][:4])

    def test_outline_extraction_skips_empty_rows(self):
        mask = np.full((3, 4), 255, dtype=np.uint8)
        mask[0, 2] = 0
        mask[2, 1] = 0  # row 1 has no outline
        xs, rows = lateral._extract_sagittal_profile(mask)
        self.assertEqual(xs.tolist(), [2, 1])
        self.assertEqual(rows.tolist(), [0, 2])

    def test_chart_skips_what_is_missing(self):
        fig, ax = plt.subplots()
        lateral._plot_monotonic_corners(ax, XS, ROWS, np.array([], dtype=np.int64))
        landmarks = np.array([[3, 12]] + [[-1, -1]] * 5)
        lateral._plot_lateral_landmarks(ax, landmarks, shift_x=1)
        self.assertEqual(len(ax.collections), 1)  # only the one found landmark
        plt.close(fig)
