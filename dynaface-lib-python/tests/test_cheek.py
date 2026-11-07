"""
Tests of dynaface.cheek without the models: the pupil frame, the frontal
constructions, the calibration tables, the pose cues and the quarter
classification, on hand-placed points and on the landmarks recorded in
tests_data/cheek_landmarks.json by scripts/make_cheek_fixtures.py.
"""

import json
import math
import os
import sys
import unittest
import warnings
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import cheek, cheek_outline  # noqa: E402
from dynaface.const import Pose  # noqa: E402

TESTS_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests_data"
)
LANDMARKS_JSON = os.path.join(TESTS_DATA, "cheek_landmarks.json")
RENDERS_JSON = os.path.join(TESTS_DATA, "cheek", "cheek_renders.json")

# The positive sweep whose raw cues are the two calibration tables' provenance,
# in order of turn. The eye-width ratio ties at 0 and 5 degrees and the nose
# offset falls back at 65, so the monotonicity checks are non-strict and stop at
# 60.
SWEEP = (
    "yaw0",
    "yaw5",
    "yaw10",
    "yaw15",
    "yaw20",
    "yaw25",
    "yaw30",
    "yaw35",
    "yaw40",
    "yaw42",
    "yaw45",
    "yaw48",
    "yaw50",
    "yaw55",
    "yaw60",
)
# The negative turns: the tables were measured on the positive sweep, so these
# are the real check of the landmark yaw.
MIRRORED = ("yaw-15", "yaw-30", "yaw-45", "yaw-50")

PUPIL_R = (380.0, 480.0)
PUPIL_L = (640.0, 480.0)
IPD = 260.0


@lru_cache(maxsize=None)
def fixture() -> Dict[str, Any]:
    with open(LANDMARKS_JSON, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


@lru_cache(maxsize=None)
def renders() -> Dict[str, Any]:
    with open(RENDERS_JSON, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


def entry(name: str) -> Dict[str, Any]:
    e: Dict[str, Any] = fixture()["images"][name]
    return e


def raw_landmarks(name: str) -> List[List[float]]:
    lm: List[List[float]] = entry(name)["raw"]["landmarks"]
    return lm


def raw_yaw(name: str) -> float:
    return float(entry(name)["raw"]["yaw"])


def true_yaw(name: str) -> float:
    """The render's true turn from its name: yaw-45 -> -45, yaw45_hollow3 -> 45."""
    return float(name[3:].split("_")[0])


def stick_face() -> np.ndarray:
    """
    A symmetric 98-point face: pupils at (380, 480) and (640, 480), lateral
    canthi at (340, 480) and (680, 480), a half-ellipse contour (semi-axes 200
    across and 300 down from the pupil line) as points 0-32 with the chin (16)
    at (510, 780), the nose tip (54) and the inner-lip midpoints (90, 94) on the
    midline, and every other landmark parked at the center.
    """
    cx, cy, a, b = 510.0, 480.0, 200.0, 300.0
    pts = np.full((98, 2), (cx, cy + 100.0), dtype=np.float64)
    for i in range(17):
        phi = math.pi + (math.pi / 2.0) * i / 16.0
        pts[i] = (cx + a * math.cos(phi), cy - b * math.sin(phi))
    for i in range(16):
        pts[32 - i] = (2.0 * cx - pts[i, 0], pts[i, 1])
    pts[cheek.LM_CANTHUS_R] = (340.0, 480.0)
    pts[cheek.LM_CANTHUS_L] = (680.0, 480.0)
    pts[96] = PUPIL_R
    pts[97] = PUPIL_L
    pts[cheek.LM_NOSE_TIP] = (cx, 620.0)
    pts[cheek.LM_ALA_R] = (480.0, 640.0)
    pts[cheek.LM_ALA_L] = (540.0, 640.0)
    pts[cheek.LM_STOMION[0]] = (cx, 700.0)
    pts[cheek.LM_STOMION[1]] = (cx, 700.0)
    return pts


def rotated(points: Any, degrees: float, center: Any = (512.0, 512.0)) -> np.ndarray:
    """Every point rotated about center, as a photo rolled by that angle."""
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    ctr = np.asarray(center, dtype=float)
    return (np.asarray(points, dtype=float) - ctr) @ rot.T + ctr


def with_landmark_yaw(landmarks: Any, target: float) -> np.ndarray:
    """
    The landmarks with the nose tip (54) slid along the pupil axis so that
    landmark_yaw() reads target: the inverse of NOSE_OFFSET_CALIBRATION gives
    the offset in canthus spans, and the tip is placed at it.
    """
    pts = np.array(landmarks, dtype=np.float64)
    frame = cheek.pupil_frame(pts)
    assert frame is not None
    offsets = [offset for offset, _ in cheek.NOSE_OFFSET_CALIBRATION]
    trues = [true for _, true in cheek.NOSE_OFFSET_CALIBRATION]
    wanted = math.copysign(float(np.interp(abs(target), trues, offsets)), target)
    canthus_r = frame.offset(pts[cheek.LM_CANTHUS_R])
    canthus_l = frame.offset(pts[cheek.LM_CANTHUS_L])
    span = canthus_l - canthus_r
    tip_offset = (canthus_r + canthus_l) / 2.0 + wanted * span
    pts[cheek.LM_NOSE_TIP] += frame.across * (
        tip_offset - frame.offset(pts[cheek.LM_NOSE_TIP])
    )
    return pts


class TestFrame(unittest.TestCase):
    """Frame and pupil_frame() on hand-placed points."""

    def test_level_frame(self):
        pts = stick_face()
        frame = cheek.pupil_frame(pts)
        self.assertIsNotNone(frame)
        np.testing.assert_allclose(frame.origin, (510.0, 480.0))
        np.testing.assert_allclose(frame.across, (1.0, 0.0))
        np.testing.assert_allclose(frame.down, (0.0, 1.0))
        self.assertAlmostEqual(frame.ipd, IPD)
        self.assertAlmostEqual(frame.level((700.0, 580.0)), 100.0)
        self.assertAlmostEqual(frame.offset((700.0, 580.0)), 190.0)
        self.assertAlmostEqual(frame.offset((400.0, 580.0)), -110.0)
        np.testing.assert_allclose(frame.point(190.0, 100.0), (700.0, 580.0))

    def test_rolled_frame(self):
        """A 30 degree roll: the frame follows the pupils, the numbers don't."""
        pts = rotated(stick_face(), 30.0, center=(510.0, 480.0))
        frame = cheek.pupil_frame(pts)
        self.assertIsNotNone(frame)
        np.testing.assert_allclose(frame.origin, (510.0, 480.0), atol=1e-9)
        np.testing.assert_allclose(
            frame.across, (math.cos(math.radians(30)), math.sin(math.radians(30)))
        )
        self.assertAlmostEqual(frame.across @ frame.down, 0.0)
        self.assertAlmostEqual(frame.ipd, IPD)
        rolled_point = rotated((700.0, 580.0), 30.0, center=(510.0, 480.0))
        self.assertAlmostEqual(frame.level(rolled_point), 100.0)
        self.assertAlmostEqual(frame.offset(rolled_point), 190.0)
        np.testing.assert_allclose(frame.point(190.0, 100.0), rolled_point)

    def test_coincident_pupils_give_none(self):
        pts = stick_face()
        pts[97] = pts[96]
        self.assertIsNone(cheek.pupil_frame(pts))
        self.assertIsNone(cheek.frontal_constructions(pts))

    def test_fewer_than_98_landmarks_give_none(self):
        pts = stick_face()[:97]
        self.assertIsNone(cheek.pupil_frame(pts))
        self.assertIsNone(cheek.frontal_constructions(pts))
        self.assertIsNone(cheek.pupil_frame([]))
        self.assertIsNone(cheek.frontal_constructions([]))

    def test_non_finite_pupils_give_none(self):
        pts = stick_face()
        pts[96] = (math.nan, 480.0)
        self.assertIsNone(cheek.pupil_frame(pts))
        pts[96] = (math.inf, 480.0)
        self.assertIsNone(cheek.pupil_frame(pts))

    def test_non_finite_landmark_gives_none(self):
        """
        A NaN or inf anywhere, a chin or a brow point, means no frame and no
        point, as cheek_outline's own check works: None, not NaN points behind a
        good frame, and no numpy warning on the way. The cues read NaN, the
        classification says frontal, and the far side falls back to the yaw.
        """
        good = cheek.pupil_frame(stick_face())
        for index, value in (
            (cheek.LM_CHIN, math.nan),
            (cheek.LM_CANTHUS_L, math.inf),
            (33, -math.inf),
        ):
            pts = stick_face()
            pts[index, 1] = value
            with self.subTest(index=index, value=value):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assertIsNone(cheek.pupil_frame(pts))
                    self.assertIsNone(cheek.frontal_constructions(pts))
                    self.assertTrue(math.isnan(cheek.nose_offset(pts, good)))
                    self.assertTrue(math.isnan(cheek.landmark_yaw(pts, good)))
                    self.assertTrue(math.isnan(cheek.nose_margin(pts, good)))
                    self.assertEqual(cheek.classify_quarter(pts, 28.0), (False, False))
                    self.assertTrue(cheek.far_left_of(pts, yaw_reported=28.0))
                    self.assertFalse(cheek.far_left_of(pts))
                    for pose in (Pose.FRONTAL, Pose.QUARTER):
                        points = cheek.locate_all(pts, pose, True, None, None, None)
                        self.assertEqual(points.shape, (10, 2))
                        self.assertTrue(np.isnan(points).all())
                self.assertEqual(
                    [str(w.message) for w in caught if w.category is RuntimeWarning],
                    [],
                )

    def test_accepts_integer_tuples(self):
        """AnalyzeFace.landmarks is a list of integer tuples."""
        pts = [(int(x), int(y)) for x, y in stick_face()]
        frame = cheek.pupil_frame(pts)
        self.assertIsNotNone(frame)
        self.assertAlmostEqual(frame.ipd, IPD)


class TestPolylines(unittest.TestCase):
    """cross_polyline() and contour_below()."""

    def test_cross_polyline_interpolates(self):
        points = [(0.0, 0.0), (0.0, 10.0), (0.0, 20.0)]
        point, status = cheek.cross_polyline(points, [0.0, 10.0, 20.0], 5.0)
        self.assertEqual(status, 0)
        np.testing.assert_allclose(point, (0.0, 5.0))
        point, status = cheek.cross_polyline(points, [0.0, 10.0, 20.0], 17.5)
        self.assertEqual(status, 0)
        np.testing.assert_allclose(point, (0.0, 17.5))

    def test_cross_polyline_clips_to_nearer_end(self):
        points = [(0.0, 0.0), (0.0, 10.0), (0.0, 20.0)]
        point, status = cheek.cross_polyline(points, [0.0, 10.0, 20.0], -5.0)
        self.assertEqual(status, -1)
        np.testing.assert_allclose(point, (0.0, 0.0))
        point, status = cheek.cross_polyline(points, [0.0, 10.0, 20.0], 25.0)
        self.assertEqual(status, 1)
        np.testing.assert_allclose(point, (0.0, 20.0))

    def test_cross_polyline_skips_flat_segment(self):
        points = [(0.0, 0.0), (0.0, 10.0), (0.0, 20.0)]
        point, status = cheek.cross_polyline(points, [5.0, 5.0, 10.0], 5.0)
        self.assertEqual(status, 0)
        np.testing.assert_allclose(point, (0.0, 10.0))
        # A polyline that is flat everywhere clips rather than dividing by zero.
        point, status = cheek.cross_polyline(points, [5.0, 5.0, 5.0], 5.0)
        self.assertEqual(status, -1)
        np.testing.assert_allclose(point, (0.0, 0.0))

    def test_contour_below_takes_the_lower_crossing(self):
        """A contour zigzagging across the vertical: the lowest crossing wins."""
        points = np.array([(0.0, 0.0), (20.0, 10.0), (0.0, 20.0), (20.0, 30.0)])
        offsets = points[:, 0]
        levels = points[:, 1]
        np.testing.assert_allclose(
            cheek.contour_below(points, offsets, levels, 10.0), (10.0, 25.0)
        )
        # Reversed order: the same lowest crossing, so it isn't "the last one".
        np.testing.assert_allclose(
            cheek.contour_below(points[::-1], offsets[::-1], levels[::-1], 10.0),
            (10.0, 25.0),
        )

    def test_contour_below_falls_back_to_nearest_vertex(self):
        points = np.array([(0.0, 0.0), (20.0, 10.0), (0.0, 20.0), (20.0, 30.0)])
        offsets = points[:, 0]
        levels = points[:, 1]
        np.testing.assert_allclose(
            cheek.contour_below(points, offsets, levels, 50.0), (20.0, 10.0)
        )
        np.testing.assert_allclose(
            cheek.contour_below(points, offsets, levels, -3.0), (0.0, 0.0)
        )

    def test_contour_below_interpolates(self):
        points = np.array([(0.0, 0.0), (10.0, 10.0), (10.0, 30.0)])
        offsets = points[:, 0]
        levels = points[:, 1]
        np.testing.assert_allclose(
            cheek.contour_below(points, offsets, levels, 5.0), (5.0, 5.0)
        )


class TestStickFace(unittest.TestCase):
    """The frontal constructions on the symmetric stick face."""

    def setUp(self):
        self.pts = stick_face()
        self.fc = cheek.frontal_constructions(self.pts)
        self.assertIsNotNone(self.fc)
        self.frame = self.fc.frame

    def test_zygion_level_is_exactly_the_ipd_fraction(self):
        self.assertAlmostEqual(self.fc.zygion_level, cheek.ZYGION_LEVEL_IPD * IPD)
        self.assertAlmostEqual(self.frame.level(self.fc.zygion_r), 78.0)
        self.assertAlmostEqual(self.frame.level(self.fc.zygion_l), 78.0)

    def test_zygion_crosses_unclipped(self):
        self.assertEqual(self.fc.zygion_r_clip, cheek.CLIP_NONE)
        self.assertEqual(self.fc.zygion_l_clip, cheek.CLIP_NONE)
        # The level 78 lies between contour points 2 (58.5) and 3 (87.2).
        self.assertLess(self.pts[2, 1], self.fc.zygion_r[1])
        self.assertLess(self.fc.zygion_r[1], self.pts[3, 1])
        self.assertLess(self.pts[2, 0], self.fc.zygion_r[0])
        self.assertLess(self.fc.zygion_r[0], self.pts[3, 0])

    def test_sides_mirror(self):
        self.assertAlmostEqual(self.fc.zygion_r_px, self.fc.zygion_l_px, places=6)
        self.assertGreater(self.fc.zygion_r_px, 0.0)
        self.assertAlmostEqual(self.fc.zygion_r[0], 1020.0 - self.fc.zygion_l[0], 6)
        self.assertAlmostEqual(self.fc.zygion_r[1], self.fc.zygion_l[1], places=6)
        self.assertAlmostEqual(self.fc.cheek_ref_r_px, self.fc.cheek_ref_l_px, 6)
        self.assertAlmostEqual(self.fc.foot_r[1], self.fc.foot_l[1], places=6)
        self.assertAlmostEqual(self.fc.foot_r[0], 1020.0 - self.fc.foot_l[0], 6)
        self.assertAlmostEqual(
            self.fc.submalar_ref_r[0], 1020.0 - self.fc.submalar_ref_l[0], 6
        )
        self.assertAlmostEqual(
            self.fc.submalar_ref_r[1], self.fc.submalar_ref_l[1], places=6
        )

    def test_canthus_vertical_ends(self):
        np.testing.assert_allclose(self.fc.canthus_r, (340.0, 480.0))
        np.testing.assert_allclose(self.fc.canthus_l, (680.0, 480.0))
        # The feet sit exactly on the verticals, on the contour below the pupils.
        self.assertAlmostEqual(self.fc.foot_r[0], 340.0, places=9)
        self.assertAlmostEqual(self.fc.foot_l[0], 680.0, places=9)
        # On the half-ellipse at x = 340: y = 480 + 300 * sqrt(1 - 0.85^2) = 638,
        # which the polyline's chord approximates from above (a chord lies inside).
        self.assertGreater(self.fc.foot_r[1], 620.0)
        self.assertLess(self.fc.foot_r[1], 640.0)

    def test_exact_thirds_and_halves(self):
        drop = self.frame.level(self.fc.foot_r) - self.frame.level(self.fc.canthus_r)
        self.assertAlmostEqual(self.fc.cheek_ref_r_px, drop / 3.0, places=9)
        self.assertAlmostEqual(self.fc.cheek_ref_r[0], 340.0, places=9)
        self.assertAlmostEqual(self.fc.cheek_ref_r[1], 480.0 + drop / 3.0, places=9)
        self.assertAlmostEqual(self.fc.stomion_level, 220.0)
        self.assertAlmostEqual(self.fc.submalar_ref_r[0], 340.0, places=9)
        self.assertAlmostEqual(
            self.frame.level(self.fc.submalar_ref_r),
            (self.frame.level(self.fc.cheek_ref_r) + 220.0) / 2.0,
            places=9,
        )

    def test_clip_top_when_the_contour_tops_are_below_the_level(self):
        pts = stick_face()
        pts[list(cheek.RIGHT_CONTOUR_TOP) + list(cheek.LEFT_CONTOUR_TOP), 1] += 100.0
        fc = cheek.frontal_constructions(pts)
        self.assertEqual(fc.zygion_r_clip, cheek.CLIP_TOP)
        self.assertEqual(fc.zygion_l_clip, cheek.CLIP_TOP)
        np.testing.assert_allclose(fc.zygion_r, pts[0])
        np.testing.assert_allclose(fc.zygion_l, pts[32])

    def test_clip_bottom_when_point_4_is_above_the_level(self):
        pts = stick_face()
        pts[list(cheek.RIGHT_CONTOUR_TOP) + list(cheek.LEFT_CONTOUR_TOP), 1] -= 100.0
        fc = cheek.frontal_constructions(pts)
        self.assertEqual(fc.zygion_r_clip, cheek.CLIP_BOTTOM)
        self.assertEqual(fc.zygion_l_clip, cheek.CLIP_BOTTOM)
        np.testing.assert_allclose(fc.zygion_r, pts[4])
        np.testing.assert_allclose(fc.zygion_l, pts[28])

    def test_zygion_level_parameter(self):
        point, clip = cheek.zygion(self.pts, self.frame, left=True, level_ipd=0.2)
        self.assertEqual(clip, cheek.CLIP_NONE)
        self.assertAlmostEqual(self.frame.level(point), 52.0)
        # Nearer the pupil line the half-ellipse is wider: further out on each side.
        self.assertGreater(point[0], self.fc.zygion_l[0])
        point, clip = cheek.zygion(self.pts, self.frame, left=False, level_ipd=0.2)
        self.assertEqual(clip, cheek.CLIP_NONE)
        self.assertAlmostEqual(self.frame.level(point), 52.0)
        self.assertLess(point[0], self.fc.zygion_r[0])

    def test_pose_cues_on_the_midline_nose(self):
        self.assertAlmostEqual(cheek.nose_offset(self.pts, self.frame), 0.0)
        self.assertAlmostEqual(cheek.landmark_yaw(self.pts, self.frame), 0.0)
        self.assertAlmostEqual(cheek.nose_margin(self.pts, self.frame), 0.5)
        self.assertFalse(cheek.far_left_of(self.pts))

    def test_pose_cues_with_the_nose_moved(self):
        """A tip 0.25 of the canthus span toward image right."""
        pts = stick_face()
        pts[cheek.LM_NOSE_TIP, 0] += 0.25 * 340.0
        frame = cheek.pupil_frame(pts)
        self.assertAlmostEqual(cheek.nose_offset(pts, frame), 0.25)
        self.assertTrue(cheek.far_left_of(pts))
        # The contour spans 310 to 710: the tip at 595 keeps 115 / 400 of the width.
        self.assertAlmostEqual(cheek.nose_margin(pts, frame), 115.0 / 400.0)
        pts[cheek.LM_NOSE_TIP, 0] = 730.0
        self.assertAlmostEqual(cheek.nose_margin(pts, frame), -20.0 / 400.0)
        pts[cheek.LM_NOSE_TIP, 0] = 510.0 - 0.25 * 340.0
        self.assertAlmostEqual(cheek.nose_offset(pts, frame), -0.25)
        self.assertFalse(cheek.far_left_of(pts))


class TestImg4(unittest.TestCase):
    """
    The pipeline landmarks of img4 (yaw 1.1) reproduce the frontal reference
    readings in millimeters through the fixture's pix2mm: level 18.9 mm, zy.r 65.5,
    zy.l 65.1, zy.w 130.6, no clip, the crossings on segments 1-2 (t 0.78) and
    31-30 (t 0.83), reference depths 28.8 on both sides, submalar 48.3.
    Measured: 18.90, 65.53, 65.06, 130.59, 28.83 and 28.83, 48.28. The left
    depth is 28.83 and not the prototype's 28.59 because contour_below() takes
    the interpolated crossing (segment 22-21, t 0.12, level 361 px, the same
    level as the right side's vertex 10) rather than the nearest vertex (22).
    """

    @classmethod
    def setUpClass(cls):
        pipeline = entry("img4")["pipeline"]
        cls.pts = np.array(pipeline["landmarks"], dtype=np.float64)
        cls.mm = float(pipeline["pix2mm"])
        cls.fc = cheek.frontal_constructions(cls.pts)
        assert cls.fc is not None

    def test_zygion_width(self):
        fc = self.fc
        self.assertEqual(fc.zygion_r_clip, cheek.CLIP_NONE)
        self.assertEqual(fc.zygion_l_clip, cheek.CLIP_NONE)
        self.assertAlmostEqual(fc.zygion_level * self.mm, 18.9, delta=0.3)
        self.assertAlmostEqual(fc.zygion_r_px * self.mm, 65.5, delta=0.3)
        self.assertAlmostEqual(fc.zygion_l_px * self.mm, 65.1, delta=0.3)
        self.assertAlmostEqual(
            (fc.zygion_r_px + fc.zygion_l_px) * self.mm, 130.6, delta=0.3
        )

    def test_zygion_crossings(self):
        fc = self.fc
        frame = fc.frame
        # Segment 1-2 at t 0.78 on the right, 31-30 at t 0.83 on the left.
        t_r = (fc.zygion_level - frame.level(self.pts[1])) / (
            frame.level(self.pts[2]) - frame.level(self.pts[1])
        )
        t_l = (fc.zygion_level - frame.level(self.pts[31])) / (
            frame.level(self.pts[30]) - frame.level(self.pts[31])
        )
        self.assertAlmostEqual(t_r, 0.78, delta=0.01)
        self.assertAlmostEqual(t_l, 0.83, delta=0.01)
        np.testing.assert_allclose(
            fc.zygion_r, self.pts[1] + t_r * (self.pts[2] - self.pts[1])
        )
        np.testing.assert_allclose(
            fc.zygion_l, self.pts[31] + t_l * (self.pts[30] - self.pts[31])
        )
        # The contour tops sit 2.7 and 3.1 mm below the pupil line: a lid level
        # could not be used.
        self.assertAlmostEqual(frame.level(self.pts[0]) * self.mm, 2.7, delta=0.1)
        self.assertAlmostEqual(frame.level(self.pts[32]) * self.mm, 3.1, delta=0.1)

    def test_reference_points(self):
        fc = self.fc
        frame = fc.frame
        self.assertAlmostEqual(fc.cheek_ref_r_px * self.mm, 28.8, delta=0.3)
        self.assertAlmostEqual(fc.cheek_ref_l_px * self.mm, 28.8, delta=0.3)
        for canthus, submalar in (
            (fc.canthus_r, fc.submalar_ref_r),
            (fc.canthus_l, fc.submalar_ref_l),
        ):
            depth = (frame.level(submalar) - frame.level(canthus)) * self.mm
            self.assertAlmostEqual(depth, 48.3, delta=0.3)
        # The reference points lie on their verticals.
        for canthus, cheek_ref, submalar in (
            (fc.canthus_r, fc.cheek_ref_r, fc.submalar_ref_r),
            (fc.canthus_l, fc.cheek_ref_l, fc.submalar_ref_l),
        ):
            self.assertAlmostEqual(frame.offset(cheek_ref), frame.offset(canthus), 9)
            self.assertAlmostEqual(frame.offset(submalar), frame.offset(canthus), 9)

    def test_pose_cues(self):
        frame = self.fc.frame
        self.assertAlmostEqual(cheek.nose_offset(self.pts, frame), 0.0, delta=0.02)
        self.assertLess(abs(cheek.landmark_yaw(self.pts, frame)), 2.0)
        self.assertAlmostEqual(cheek.nose_margin(self.pts, frame), 0.49, delta=0.02)


class TestRollInvariance(unittest.TestCase):
    """
    A 15 degree roll of every landmark about the image center changes no
    level, offset or px distance beyond 1e-6, and moves every point with the
    roll.
    """

    def test_frontal_constructions(self):
        pts = np.array(entry("img4")["pipeline"]["landmarks"], dtype=np.float64)
        fc = cheek.frontal_constructions(pts)
        fr = cheek.frontal_constructions(rotated(pts, 15.0))
        for name in (
            "zygion_level",
            "zygion_r_px",
            "zygion_l_px",
            "cheek_ref_r_px",
            "cheek_ref_l_px",
            "stomion_level",
        ):
            self.assertAlmostEqual(getattr(fc, name), getattr(fr, name), delta=1e-6)
        self.assertEqual(fc.zygion_r_clip, fr.zygion_r_clip)
        self.assertEqual(fc.zygion_l_clip, fr.zygion_l_clip)
        for name in (
            "zygion_r",
            "zygion_l",
            "canthus_r",
            "canthus_l",
            "foot_r",
            "foot_l",
            "cheek_ref_r",
            "cheek_ref_l",
            "submalar_ref_r",
            "submalar_ref_l",
        ):
            np.testing.assert_allclose(
                getattr(fr, name), rotated(getattr(fc, name), 15.0), atol=1e-6
            )
        self.assertAlmostEqual(fr.frame.ipd, fc.frame.ipd, delta=1e-6)

    def test_pose_cues(self):
        for name in ("img4", "yaw45", "yaw-45"):
            pts = np.array(raw_landmarks(name), dtype=np.float64)
            frame = cheek.pupil_frame(pts)
            rolled = rotated(pts, 15.0, center=(400.0, 400.0))
            frame_r = cheek.pupil_frame(rolled)
            self.assertAlmostEqual(
                cheek.nose_offset(pts, frame), cheek.nose_offset(rolled, frame_r), 9
            )
            self.assertAlmostEqual(
                cheek.landmark_yaw(pts, frame), cheek.landmark_yaw(rolled, frame_r), 9
            )
            self.assertAlmostEqual(
                cheek.nose_margin(pts, frame), cheek.nose_margin(rolled, frame_r), 9
            )
            self.assertEqual(
                cheek.classify_quarter(pts, raw_yaw(name)),
                cheek.classify_quarter(rolled, raw_yaw(name)),
            )


class TestCalibrationTables(unittest.TestCase):
    """The tables in cheek.py and the two functions that read them."""

    def test_tables_are_increasing(self):
        for table in (cheek.YAW_CALIBRATION, cheek.NOSE_OFFSET_CALIBRATION):
            xs = [x for x, _ in table]
            ys = [y for _, y in table]
            self.assertEqual(xs, sorted(xs))
            self.assertEqual(len(set(xs)), len(xs))
            self.assertEqual(ys, sorted(ys))
            self.assertEqual(table[0][1], 0.0)
        # The yaw table's first row is the level head's own reading, which the
        # offset puts below it, so the cue column less the offset starts above
        # 0; the nose offset needs no anchor.
        self.assertEqual(cheek.YAW_CALIBRATION[0], (1.5, 0.0))
        self.assertLess(cheek.YAW_RAW_OFFSET, cheek.YAW_CALIBRATION[0][0])
        self.assertGreater(cheek.YAW_RAW_OFFSET, 0.0)
        self.assertEqual(cheek.NOSE_OFFSET_CALIBRATION[0], (0.0, 0.0))

    def test_table_rows(self):
        """The re-measured rows: 45, 65, the extra rows at 42 and 48, 60 last."""
        self.assertIn((28.28, 45.0), cheek.YAW_CALIBRATION)
        self.assertIn((44.31, 65.0), cheek.YAW_CALIBRATION)
        self.assertIn((25.77, 42.0), cheek.YAW_CALIBRATION)
        self.assertIn((30.54, 48.0), cheek.YAW_CALIBRATION)
        self.assertEqual(cheek.YAW_CALIBRATION[-1], (44.31, 65.0))
        self.assertIn((0.4449, 45.0), cheek.NOSE_OFFSET_CALIBRATION)
        self.assertIn((0.4181, 42.0), cheek.NOSE_OFFSET_CALIBRATION)
        self.assertIn((0.514, 48.0), cheek.NOSE_OFFSET_CALIBRATION)
        self.assertEqual(cheek.NOSE_OFFSET_CALIBRATION[-1], (0.7696, 60.0))

    def test_calibrate_yaw_reads_the_table(self):
        offset = cheek.YAW_RAW_OFFSET
        for reported, true in cheek.YAW_CALIBRATION:
            self.assertAlmostEqual(cheek.calibrate_yaw(reported), true, places=9)
            # The mirrored reading: as far below the offset as this is above it.
            self.assertAlmostEqual(
                cheek.calibrate_yaw(2.0 * offset - reported), -true, places=9
            )
        self.assertAlmostEqual(cheek.calibrate_yaw(28.28), 45.0)
        self.assertAlmostEqual(cheek.calibrate_yaw(44.31), 65.0)
        self.assertAlmostEqual(cheek.calibrate_yaw(1.5), 0.0)
        # Between rows: linear.
        self.assertAlmostEqual(cheek.calibrate_yaw((28.28 + 30.54) / 2), 46.5)

    def test_calibrate_yaw_beyond_the_table(self):
        offset = cheek.YAW_RAW_OFFSET
        self.assertAlmostEqual(
            cheek.calibrate_yaw(50.0), 65.0 + (50.0 - 44.31) * cheek.YAW_SLOPE_BEYOND
        )
        # A reading of -50 is 50 + offset below the offset, and the table's
        # last row sits 44.31 - offset above it.
        self.assertAlmostEqual(
            cheek.calibrate_yaw(-50.0),
            -(65.0 + (50.0 + 2.0 * offset - 44.31) * cheek.YAW_SLOPE_BEYOND),
        )
        # The offset itself is a level head; a reading of 0 is a turn toward
        # the subject's right, read on the table's first segment (the level
        # head's 1.5 to the 5 degree render's 3.04, both less the offset).
        self.assertEqual(cheek.calibrate_yaw(offset), 0.0)
        self.assertAlmostEqual(
            cheek.calibrate_yaw(0.0),
            -5.0 * (offset - (1.5 - offset)) / (3.04 - 1.5),
        )
        self.assertAlmostEqual(cheek.calibrate_yaw(0.0), -3.12, delta=0.01)

    def test_calibrate_yaw_is_odd_about_the_offset_and_monotonic(self):
        offset = cheek.YAW_RAW_OFFSET
        for reported in np.linspace(0.0, 60.0, 121):
            self.assertAlmostEqual(
                cheek.calibrate_yaw(offset - reported),
                -cheek.calibrate_yaw(offset + reported),
            )
        values = [cheek.calibrate_yaw(r) for r in np.linspace(-60.0, 60.0, 1201)]
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))
        self.assertLess(cheek.calibrate_yaw(0.0), 0.0)
        self.assertEqual(cheek.calibrate_yaw(1.5), 0.0)  # the level head's row
        self.assertGreater(cheek.calibrate_yaw(2.0), 0.0)
        # Never -0.0: a level head has no sign.
        self.assertEqual(math.copysign(1.0, cheek.calibrate_yaw(1.08)), 1.0)

    def test_nan_passes_through(self):
        self.assertTrue(math.isnan(cheek.calibrate_yaw(math.nan)))
        pts = stick_face()
        frame = cheek.pupil_frame(pts)
        pts[cheek.LM_NOSE_TIP] = (math.nan, math.nan)
        self.assertTrue(math.isnan(cheek.nose_offset(pts, frame)))
        self.assertTrue(math.isnan(cheek.landmark_yaw(pts, frame)))
        self.assertTrue(math.isnan(cheek.nose_margin(pts, frame)))
        self.assertEqual(cheek.classify_quarter(pts, 28.28), (False, False))
        self.assertFalse(cheek.far_left_of(pts))

    def test_landmark_yaw_reads_the_table(self):
        """Nose offsets placed at each table row, on the stick face."""
        base = stick_face()
        for offset, true in cheek.NOSE_OFFSET_CALIBRATION:
            for sign in (1.0, -1.0):
                pts = base.copy()
                pts[cheek.LM_NOSE_TIP, 0] += sign * offset * 340.0
                frame = cheek.pupil_frame(pts)
                self.assertAlmostEqual(cheek.nose_offset(pts, frame), sign * offset)
                self.assertAlmostEqual(
                    cheek.landmark_yaw(pts, frame), sign * true, places=9
                )
        pts = base.copy()
        pts[cheek.LM_NOSE_TIP, 0] += 0.4449 * 340.0
        self.assertAlmostEqual(cheek.landmark_yaw(pts, cheek.pupil_frame(pts)), 45.0)

    def test_landmark_yaw_beyond_the_table(self):
        pts = stick_face()
        pts[cheek.LM_NOSE_TIP, 0] -= 0.9 * 340.0
        frame = cheek.pupil_frame(pts)
        self.assertAlmostEqual(
            cheek.landmark_yaw(pts, frame),
            -(60.0 + (0.9 - 0.7696) * cheek.NOSE_OFFSET_SLOPE_BEYOND),
        )

    def test_landmark_yaw_is_monotonic(self):
        base = stick_face()
        values = []
        for offset in np.linspace(0.0, 1.0, 201):
            pts = base.copy()
            pts[cheek.LM_NOSE_TIP, 0] += offset * 340.0
            values.append(cheek.landmark_yaw(pts, cheek.pupil_frame(pts)))
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))


class TestRenderSweep(unittest.TestCase):
    """
    The pose cues on the synthetic-head renders' raw landmarks. The tables were
    measured on the positive sweep, so it reproduces them within 0.05 degrees
    (the rounding of the fixture's cues); the mirrored turns and the hollow and
    bump variants are the check. Measured: the landmark yaw errs by 2.26
    degrees on the level head (yaw0: SPIGA puts its nose 0.021 spans off
    center, which the table's (0, 0) row does not absorb; the model's raw 1.50
    there is the yaw table's first row, so calibrate_yaw gives 0.00) and by at
    most 0.005 from 5 to 60; on the negative turns by 1.41 (-15), -0.86 (-30),
    -0.06 (-45) and 0.50 (-50) degrees, where the model's calibrated yaw, read
    less YAW_RAW_OFFSET (the mean of the level head's raw yaw and the mirrored
    pairs' half-sums, 1.23), errs by 0.67, -1.41, 0.23 and 0.15 (6.36, 2.48,
    3.14 and 2.70 low without the offset); the mirrored 45 degree pair reads
    45.00 and -45.06, a gap of 0.06 degrees, where the model's calibrated pair
    reads 45.00 and -44.77, a gap of 0.23 degrees, pinned at +/- 0.05.
    yaw45_bump3, the render with a 3 mm malar bump, reads 42.45 (-2.55); yaw65
    reads 57.97 because the nose offset falls back past 60.
    """

    def cue(self, name):
        pts = raw_landmarks(name)
        frame = cheek.pupil_frame(pts)
        self.assertIsNotNone(frame)
        return pts, frame

    def test_nose_offset_reproduces_the_fixture_cue(self):
        """The provenance: nose_offset() is what the fixture script measured."""
        for name, e in fixture()["images"].items():
            pts, frame = self.cue(name)
            self.assertAlmostEqual(
                cheek.nose_offset(pts, frame), e["cues"]["nose_offset"], delta=5e-5
            )

    def test_landmark_yaw_against_the_true_turn(self):
        """
        The positive sweep is the nose-offset table's provenance, so it must
        reproduce within 0.05 degrees (measured 0.005; a corrupted row would
        read as its own error); the mirrored turns and the variants are the
        check, within 1.5.
        """
        for name in SWEEP[1:]:
            pts, frame = self.cue(name)
            self.assertAlmostEqual(
                cheek.landmark_yaw(pts, frame), true_yaw(name), delta=0.05, msg=name
            )
        for name in MIRRORED + (
            "yaw45_hollow1.5",
            "yaw45_hollow3",
            "yaw45_hollow4.5",
            "yaw50_hollow3",
        ):
            pts, frame = self.cue(name)
            self.assertAlmostEqual(
                cheek.landmark_yaw(pts, frame), true_yaw(name), delta=1.5, msg=name
            )
        pts, frame = self.cue("yaw0")
        self.assertAlmostEqual(cheek.landmark_yaw(pts, frame), 2.26, delta=0.05)
        pts, frame = self.cue("yaw45_bump3")
        self.assertAlmostEqual(cheek.landmark_yaw(pts, frame), 42.45, delta=0.05)

    def test_calibrate_yaw_against_the_true_turn(self):
        """
        The positive sweep, the level head included, is the yaw table's
        provenance, so it must reproduce within 0.05 degrees (measured 0.000);
        the mirrored turns, which the table never saw, and the variants are the
        check, within 1.5 (measured 0.67, -1.41, 0.23 and 0.15 on the mirrored
        turns: the offset's doing, since without it they read 2.5 to 6.4 low).
        """
        for name in SWEEP + ("yaw65",):
            self.assertAlmostEqual(
                cheek.calibrate_yaw(raw_yaw(name)),
                true_yaw(name),
                delta=0.05,
                msg=name,
            )
        for name in MIRRORED + (
            "yaw45_hollow1.5",
            "yaw45_hollow3",
            "yaw45_hollow4.5",
            "yaw50_hollow3",
            "yaw45_bump3",
        ):
            self.assertAlmostEqual(
                cheek.calibrate_yaw(raw_yaw(name)), true_yaw(name), delta=1.5, msg=name
            )
        self.assertAlmostEqual(
            cheek.calibrate_yaw(raw_yaw("yaw-30")), -31.41, delta=0.05
        )

    def test_yaw_offset_is_fitted_to_the_level_head_and_the_mirrored_pairs(self):
        """
        YAW_RAW_OFFSET's provenance: the mean of the level head's raw yaw and
        the half-sums of the four mirrored pairs' raw yaws, each an estimate of
        the constant the model reads over the truth (1.50, 1.36, 0.67, 1.33 and
        1.31).
        """
        estimates = [raw_yaw("yaw0")]
        for turn in (15, 30, 45, 50):
            estimates.append((raw_yaw(f"yaw{turn}") + raw_yaw(f"yaw-{turn}")) / 2.0)
        self.assertAlmostEqual(estimates[0], 1.5, delta=0.005)
        for estimate in estimates[1:]:
            self.assertGreater(estimate, 0.5)
            self.assertLess(estimate, 1.5)
        self.assertAlmostEqual(
            cheek.YAW_RAW_OFFSET, float(np.mean(estimates)), delta=0.005
        )

    def test_mirrored_45_pair(self):
        pts_p, frame_p = self.cue("yaw45")
        pts_n, frame_n = self.cue("yaw-45")
        yaw_p = cheek.landmark_yaw(pts_p, frame_p)
        yaw_n = cheek.landmark_yaw(pts_n, frame_n)
        self.assertGreater(yaw_p, 0.0)
        self.assertLess(yaw_n, 0.0)
        self.assertAlmostEqual(abs(yaw_p) - abs(yaw_n), -0.06, delta=0.05)
        self.assertLess(abs(abs(yaw_p) - abs(yaw_n)), 1.0)
        model_p = cheek.calibrate_yaw(raw_yaw("yaw45"))
        model_n = cheek.calibrate_yaw(raw_yaw("yaw-45"))
        self.assertAlmostEqual(abs(model_p) - abs(model_n), 0.23, delta=0.05)
        self.assertLess(abs(abs(model_p) - abs(model_n)), 1.0)
        # The nose offsets themselves: 0.4449 and -0.4463, within 0.02.
        self.assertLess(
            abs(
                abs(cheek.nose_offset(pts_p, frame_p))
                - abs(cheek.nose_offset(pts_n, frame_n))
            ),
            0.02,
        )

    def test_landmark_and_model_yaws_agree(self):
        """
        cheek.yaw and cheek.yaw.model within YAW_AGREE_DEG, the rule every yaw
        gate applies, on every fixture entry but four. yaw65 is excluded
        because the nose offset falls back past 60 degrees (57.97 against the
        model's 65.0, a gap of 7.0), the profiles img2 and img3 because their
        nose is past the far contour (margins -0.15 and -0.13), where both
        tables extrapolate: img2 reads 73.8 against 69.8 (a gap of 4.0, by
        chance rather than by calibration) and img3 -76.1 against -79.5 (3.4),
        and yaw45_pitch10 because the nose-offset cue reads chin up as more
        turn (49.55 against the model's 43.00, a gap of 6.5; TestPitchedRenders
        measures it). Measured on the rest: the widest gap is yaw45_pitch5's
        3.76 (the same pitch effect at half the angle), then yaw45_bump3's 2.56
        (the bump moves the landmarks' nose offset, not the model), img1's 2.52
        and the level head's 2.26 (the landmarks' own bias there, which the
        model's offset no longer happens to match); the mirrored turns, which
        read 4.95, 3.34, 3.20 and 2.20 before the offset, read 0.74, 0.54, 0.29
        and 0.35.
        """
        excluded = {"yaw65", "img2", "img3", "yaw45_pitch10"}
        gaps = {}
        for name in fixture()["images"]:
            pts, frame = self.cue(name)
            gaps[name] = abs(
                cheek.landmark_yaw(pts, frame) - cheek.calibrate_yaw(raw_yaw(name))
            )
            if name not in excluded:
                self.assertLessEqual(gaps[name], cheek.YAW_AGREE_DEG, name)
        self.assertAlmostEqual(gaps["yaw-15"], 0.74, delta=0.05)
        self.assertAlmostEqual(gaps["yaw-45"], 0.29, delta=0.05)
        widest = max(gap for name, gap in gaps.items() if name not in excluded)
        self.assertAlmostEqual(widest, gaps["yaw45_pitch5"], delta=1e-9)
        self.assertAlmostEqual(widest, 3.76, delta=0.05)
        self.assertAlmostEqual(gaps["yaw45_bump3"], 2.56, delta=0.05)
        unpitched = max(
            gap
            for name, gap in gaps.items()
            if name not in excluded and "_pitch" not in name
        )
        self.assertAlmostEqual(unpitched, gaps["yaw45_bump3"], delta=1e-9)
        # The one that disagrees, so the exclusion is not a convenience.
        self.assertGreater(gaps["yaw65"], cheek.YAW_AGREE_DEG)

    def test_mirrored_45s_agree_with_margin(self):
        """
        Both 45 degree renders keep at least 1.5 degrees of the agreement
        budget (YAW_AGREE_DEG) in hand, so a real face's own asymmetry has room
        before apex.ok fails on the agreement: measured 0.00 and 0.29 (3.20 on
        the mirrored turn before the offset, 1.8 in hand).
        """
        for name in ("yaw45", "yaw-45"):
            pts, frame = self.cue(name)
            gap = abs(
                cheek.landmark_yaw(pts, frame) - cheek.calibrate_yaw(raw_yaw(name))
            )
            self.assertLessEqual(gap, cheek.YAW_AGREE_DEG - 1.5, name)

    def test_cues_are_monotonic_over_0_to_60(self):
        """Non-strict: the eye-width ratio ties at 0 and 5 degrees."""
        offsets = []
        reported = []
        ratios = []
        for name in SWEEP:
            pts, frame = self.cue(name)
            offsets.append(abs(cheek.nose_offset(pts, frame)))
            reported.append(raw_yaw(name))
            ratios.append(entry(name)["cues"]["eye_width_ratio"])
        self.assertTrue(all(b >= a for a, b in zip(offsets, offsets[1:])))
        self.assertTrue(all(b >= a for a, b in zip(reported, reported[1:])))
        self.assertTrue(all(b <= a for a, b in zip(ratios, ratios[1:])))
        self.assertEqual(ratios[0], ratios[1])

    def test_nose_margin_on_the_renders(self):
        """0.47 face-on, 0.05 at 45, 0 at 50 and negative past it."""
        expected = {
            "yaw0": 0.47,
            "yaw45": 0.05,
            "yaw-45": 0.05,
            "yaw50": 0.00,
            "yaw55": -0.06,
            "yaw60": -0.09,
            "img2": -0.15,
            "img3": -0.13,
        }
        for name, margin in expected.items():
            pts, frame = self.cue(name)
            self.assertAlmostEqual(
                cheek.nose_margin(pts, frame), margin, delta=0.01, msg=name
            )

    def test_far_side(self):
        for name in ("yaw45", "yaw48", "yaw50", "img2"):
            self.assertTrue(cheek.far_left_of(raw_landmarks(name)), name)
        for name in ("yaw-45", "yaw-50", "img3", "quarter_sketch"):
            self.assertFalse(cheek.far_left_of(raw_landmarks(name)), name)

    def test_far_side_falls_back_to_the_yaw(self):
        short = raw_landmarks("yaw45")[:50]
        self.assertTrue(cheek.far_left_of(short, 28.0))
        self.assertFalse(cheek.far_left_of(short, -28.0))
        self.assertFalse(cheek.far_left_of(short))
        self.assertFalse(cheek.far_left_of(short, math.nan))


class TestSensitivity(unittest.TestCase):
    """
    Guards on the 0 and 10 degree renders, at the renders' own 0.4 mm/px: the
    zygion moves under 1.5 mm per degree of yaw and the reference points under
    0.3. Measured: zy.r +1.15, zy.l -0.87, cr.r +0.06, cr.l -0.05 mm/degree.
    SPIGA's contour reads the symmetric head at yaw 0 as zy.r 75.2 against zy.l
    67.8 mm, the detector's known contour bias.
    """

    def test_guards(self):
        mm = float(renders()["mm_per_px"])
        at_0 = cheek.frontal_constructions(raw_landmarks("yaw0"))
        at_10 = cheek.frontal_constructions(raw_landmarks("yaw10"))
        for fc in (at_0, at_10):
            self.assertEqual(fc.zygion_r_clip, cheek.CLIP_NONE)
            self.assertEqual(fc.zygion_l_clip, cheek.CLIP_NONE)
        for name, limit in (
            ("zygion_r_px", 1.5),
            ("zygion_l_px", 1.5),
            ("cheek_ref_r_px", 0.3),
            ("cheek_ref_l_px", 0.3),
        ):
            per_degree = (getattr(at_10, name) - getattr(at_0, name)) * mm / 10.0
            self.assertLess(abs(per_degree), limit, name)
        # The turn widens the near side and narrows the far side.
        self.assertGreater(at_10.zygion_r_px, at_0.zygion_r_px)
        self.assertLess(at_10.zygion_l_px, at_0.zygion_l_px)
        self.assertAlmostEqual(at_0.zygion_r_px * mm, 75.2, delta=0.1)
        self.assertAlmostEqual(at_0.zygion_l_px * mm, 67.8, delta=0.1)


class TestPitchedRenders(unittest.TestCase):
    """
    The level head pitched by -5, +5, +10 and +25 degrees (chin up positive)
    and the 45 degree turn pitched by -5, +5 and +10, on the renders' raw
    landmarks at 0.4 mm/px: the definitions' pitch plan for the frontal core
    (0 to 10 degrees) and the oblique module's pitch -5/+5 readings, measured
    on SPIGA's landmarks rather than the mesh.

    Measured against the level head, mm per degree of pitch: zy.r -0.04, +0.19
    and +0.08 at -5, +5 and +10, zy.l -0.50, -0.54 and -0.24, cr.r +0.13,
    +0.16 and +0.05, cr.l -0.16, -0.41 and -0.14 (the left side moves more:
    SPIGA's left contour sits inside this head's silhouette, a bias the
    guards' allowance covers). So in the image the constructed zygion moves
    under 0.6 mm per degree of pitch and the reference points under 0.45,
    against the definitions' "about 1 mm per degree along the arch" for the
    skin point itself; the guards allow 1.0 and 0.6. No clip at any of those
    pitches: the contour tops (0 and 32) sit 2 px below the pupil line on the
    level head and move down as the chin comes up, to 21, 38 and 44 to 46 px
    at +5, +10 and +25, while the 0.3 IPD level sits 47 px below it and rises
    to 44 as the pupils foreshorten (155 to 145 px apart), so at +25 the level
    passes above point 32 by 3 px, CLIP_TOP on the left, and meets point 0
    within rounding on the right. CLIP_BOTTOM (the level below 4/28, 100 px down on
    the level head) is produced by no render: only the stick face reaches it.
    SPIGA's pitch reads chin up positive, 2.57, 5.39, 7.28, 8.62 and 14.52 at
    -5, 0, +5, +10 and +25: monotonic, offset by 5.4 on this head and
    compressed to about 0.4 of the pitch, so the true 25 degrees reads 14.5,
    inside PITCH_GROSS_RAW (20), and cheek.pitch.ok is 1 on every render here
    (the gross gate would take a chin-up of about 40 degrees on this head; the
    mock in test_measures_cheek exercises it). The landmark yaw holds within
    FRONTAL_YAW_GATE at every pitch of the level head (1.19, 2.28, 2.03 and
    0.21 against 2.26 level) and classify_quarter stays frontal; on the turned
    head the nose-offset cue reads chin up as more turn, 45.02, 47.79 and
    49.55 at -5, +5 and +10 (about 0.45 degrees per degree of pitch, the tip
    dropping toward the far contour), where the model's calibrated yaw reads
    44.06, 44.03 and 43.00, so the two agree within YAW_AGREE_DEG at -5 and +5
    and disagree by 6.5 at +10, which is what would fail apex.ok there; the
    far side is the same at every pitch.
    """

    LEVEL = ("yaw0_pitch-5", "yaw0_pitch5", "yaw0_pitch10")
    TURNED = ("yaw45_pitch-5", "yaw45_pitch5", "yaw45_pitch10")

    @staticmethod
    def pitch_of(name):
        return float(name.split("_pitch")[1])

    def test_frontal_constructions_under_pitch(self):
        mm = float(renders()["mm_per_px"])
        level = cheek.frontal_constructions(raw_landmarks("yaw0"))
        for name in self.LEVEL:
            with self.subTest(name):
                fc = cheek.frontal_constructions(raw_landmarks(name))
                self.assertEqual(fc.zygion_r_clip, cheek.CLIP_NONE)
                self.assertEqual(fc.zygion_l_clip, cheek.CLIP_NONE)
                pitch = self.pitch_of(name)
                for attr, limit in (
                    ("zygion_r_px", 1.0),
                    ("zygion_l_px", 1.0),
                    ("cheek_ref_r_px", 0.6),
                    ("cheek_ref_l_px", 0.6),
                ):
                    per_degree = (getattr(fc, attr) - getattr(level, attr)) * mm
                    self.assertLess(abs(per_degree / pitch), limit, attr)

    def test_gross_pitch_clips_the_level_at_the_top(self):
        pts = np.asarray(raw_landmarks("yaw0_pitch25"), dtype=np.float64)
        fc = cheek.frontal_constructions(pts)
        self.assertEqual(fc.zygion_l_clip, cheek.CLIP_TOP)
        np.testing.assert_allclose(fc.zygion_l, pts[32])
        self.assertEqual(fc.zygion_r_clip, cheek.CLIP_NONE)
        # The level passed above the left contour's top by about 3 px, and the
        # right contour's top sits on it within a pixel.
        self.assertGreater(fc.frame.level(pts[32]) - fc.zygion_level, 1.0)
        self.assertLess(abs(fc.frame.level(pts[0]) - fc.zygion_level), 1.0)

    def test_model_pitch_is_monotonic_and_compressed(self):
        names = ("yaw0_pitch-5", "yaw0", "yaw0_pitch5", "yaw0_pitch10", "yaw0_pitch25")
        raw = [float(entry(name)["raw"]["pitch"]) for name in names]
        self.assertTrue(all(b > a for a, b in zip(raw, raw[1:])), raw)
        self.assertAlmostEqual(raw[1], 5.39, delta=0.05)
        slope = (raw[-1] - raw[0]) / 30.0
        self.assertGreater(slope, 0.25)
        self.assertLess(slope, 0.5)
        for value in raw:
            self.assertLessEqual(abs(value), cheek.PITCH_GROSS_RAW)

    def test_level_head_stays_frontal_under_pitch(self):
        for name in self.LEVEL + ("yaw0_pitch25",):
            with self.subTest(name):
                pts = raw_landmarks(name)
                frame = cheek.pupil_frame(pts)
                self.assertLess(
                    abs(cheek.landmark_yaw(pts, frame)), cheek.FRONTAL_YAW_GATE
                )
                self.assertLess(
                    abs(cheek.calibrate_yaw(raw_yaw(name))), cheek.FRONTAL_YAW_GATE
                )
                self.assertEqual(
                    cheek.classify_quarter(pts, raw_yaw(name)), (False, False)
                )

    def test_turned_head_stays_quarter_under_pitch(self):
        yaws, models = [], []
        for name in self.TURNED:
            with self.subTest(name):
                pts = raw_landmarks(name)
                frame = cheek.pupil_frame(pts)
                self.assertEqual(
                    cheek.classify_quarter(pts, raw_yaw(name)), (True, True)
                )
                self.assertTrue(cheek.far_left_of(pts, raw_yaw(name)))
                yaws.append(cheek.landmark_yaw(pts, frame))
                models.append(cheek.calibrate_yaw(raw_yaw(name)))
                self.assertAlmostEqual(models[-1], 45.0, delta=2.5)
        self.assertAlmostEqual(yaws[0], 45.02, delta=0.05)
        self.assertAlmostEqual(yaws[1], 47.79, delta=0.05)
        self.assertAlmostEqual(yaws[2], 49.55, delta=0.05)
        for yaw in yaws[:2]:
            self.assertLessEqual(abs(yaw - cheek.OBLIQUE_YAW), cheek.OBLIQUE_YAW_GATE)
        for yaw, model in zip(yaws[:2], models[:2]):
            self.assertLessEqual(abs(yaw - model), cheek.YAW_AGREE_DEG)
        self.assertGreater(abs(yaws[2] - models[2]), cheek.YAW_AGREE_DEG)


class TestClassifyQuarter(unittest.TestCase):
    """
    classify_quarter() on the fixture's raw landmarks and raw yaw, before any
    crop. Measured landmark yaws: img4 -1.0, img1 11.2, the sketch -31.9,
    yaw30 30.0, yaw35 35.003 (admitted: the window's lower edge is inclusive,
    and its margin is 0.129 with a model yaw of 35.0), yaw45 45.0, yaw48 48.0,
    yaw50 50.0 (margin 0.000), yaw-45 -45.1, yaw-50 -49.5, yaw55 55.0 with
    margin -0.064 (the margin rule, since 55 is the window's inclusive edge),
    yaw60 60.0, yaw65 58.0, img2 73.8 and img3 -76.1 with margins -0.15 and
    -0.13.
    """

    def classify(self, name, previous_quarter=False):
        return cheek.classify_quarter(
            raw_landmarks(name), raw_yaw(name), previous_quarter=previous_quarter
        )

    def test_frontal_faces(self):
        for name in (
            "img4",
            "img1",
            "quarter_sketch",
            "quarter_sketch_forced",
            "yaw0",
            "yaw5",
            "yaw10",
            "yaw15",
            "yaw20",
            "yaw25",
            "yaw30",
            "yaw-15",
            "yaw-30",
            "yaw0_pitch-5",
            "yaw0_pitch5",
            "yaw0_pitch10",
            "yaw0_pitch25",
        ):
            self.assertEqual(self.classify(name), (False, False), name)

    def test_quarter_faces(self):
        for name in ("yaw35", "yaw40", "yaw42", "yaw45", "yaw48", "yaw50"):
            self.assertEqual(self.classify(name), (True, True), name)
        for name in ("yaw-45", "yaw-50"):
            self.assertEqual(self.classify(name), (True, False), name)
        for name in (
            "yaw45_hollow1.5",
            "yaw45_hollow3",
            "yaw45_hollow4.5",
            "yaw50_hollow3",
            "yaw45_bump3",
            "yaw45_pitch-5",
            "yaw45_pitch5",
            "yaw45_pitch10",
        ):
            self.assertEqual(self.classify(name), (True, True), name)

    def test_not_quarter(self):
        for name in ("yaw55", "yaw60", "yaw65", "img2", "img3"):
            self.assertEqual(self.classify(name), (False, False), name)
        # The window alone would admit yaw55 (landmark yaw 55.0): its nose is
        # past the far contour, which is profile logic.
        pts = raw_landmarks("yaw55")
        frame = cheek.pupil_frame(pts)
        self.assertAlmostEqual(cheek.landmark_yaw(pts, frame), 55.0, delta=0.05)
        self.assertLess(cheek.nose_margin(pts, frame), cheek.QUARTER_MARGIN_MIN)

    @staticmethod
    def raw_for(calibrated):
        """The raw model yaw that calibrate_yaw() reads as this many degrees."""
        low, high = 0.0, 80.0
        for _ in range(80):
            middle = (low + high) / 2
            if cheek.calibrate_yaw(middle) < calibrated:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    def test_model_reading_past_the_window(self):
        """
        A profile whose landmark yaw collapsed into the window: the model still
        reads the turn, and past QUARTER_MAX_MODEL_YAW the face isn't a quarter
        face, except within the video hold's few degrees.
        """
        pts = raw_landmarks("yaw45")
        limit = cheek.QUARTER_MAX_MODEL_YAW
        inside, outside = self.raw_for(limit - 0.1), self.raw_for(limit + 0.1)
        self.assertEqual(cheek.classify_quarter(pts, inside), (True, True))
        self.assertEqual(cheek.classify_quarter(pts, outside), (False, False))
        held = dict(previous_quarter=True)
        self.assertEqual(cheek.classify_quarter(pts, outside, **held), (True, True))
        past_hold = self.raw_for(limit + cheek.QUARTER_YAW_HOLD + 0.1)
        self.assertEqual(cheek.classify_quarter(pts, past_hold, **held), (False, False))

    def test_the_margin_rule_can_be_waived(self):
        """
        yaw55's nose is past the far contour, so the margin rule alone declines
        it; without the rule (a face the profile test has declined) it is a
        quarter face. The rest of the rule still applies.
        """
        pts, raw = raw_landmarks("yaw55"), raw_yaw("yaw55")
        self.assertLessEqual(abs(cheek.calibrate_yaw(raw)), cheek.QUARTER_MAX_MODEL_YAW)
        self.assertEqual(cheek.classify_quarter(pts, raw), (False, False))
        waived = dict(require_margin=False)
        self.assertEqual(cheek.classify_quarter(pts, raw, **waived), (True, True))
        for name in ("img4", "yaw30", "yaw60"):
            self.assertEqual(
                cheek.classify_quarter(raw_landmarks(name), raw_yaw(name), **waived),
                (False, False),
                name,
            )

    def test_forged_model_yaw(self):
        """A frontal face with a model yaw of 28 is still frontal."""
        self.assertEqual(
            cheek.classify_quarter(raw_landmarks("img4"), 28.0), (False, False)
        )
        self.assertEqual(
            cheek.classify_quarter(raw_landmarks("yaw10"), 28.28), (False, False)
        )

    def test_displaced_nose_without_the_model(self):
        """A quarter face's landmarks with a frontal model yaw: not a quarter."""
        pts = raw_landmarks("yaw45")
        self.assertEqual(cheek.classify_quarter(pts, 5.0), (False, False))
        self.assertEqual(cheek.classify_quarter(pts, 0.0), (False, False))
        self.assertEqual(cheek.classify_quarter(pts, math.nan), (False, False))
        # The model's calibrated minimum, 20 degrees (raw 10.41), with the sign.
        self.assertEqual(cheek.classify_quarter(pts, 10.41), (True, True))
        self.assertEqual(cheek.classify_quarter(pts, 10.0), (False, False))

    def test_cues_disagreeing_in_sign(self):
        self.assertEqual(
            cheek.classify_quarter(raw_landmarks("yaw45"), -raw_yaw("yaw45")),
            (False, False),
        )
        self.assertEqual(
            cheek.classify_quarter(raw_landmarks("yaw-45"), -raw_yaw("yaw-45")),
            (False, False),
        )

    def test_no_frame(self):
        self.assertEqual(cheek.classify_quarter([], 28.28), (False, False))
        self.assertEqual(
            cheek.classify_quarter(raw_landmarks("yaw45")[:97], 28.28), (False, False)
        )
        pts = np.array(raw_landmarks("yaw45"), dtype=np.float64)
        pts[97] = pts[96]
        self.assertEqual(cheek.classify_quarter(pts, 28.28), (False, False))

    def test_hold_keeps_one_view(self):
        """
        Landmark yaws 34, 36, 34, 36 (the yaw35 render's nose slid along the
        pupil axis) with previous_quarter carried: the view enters at 36 and the
        hold keeps it at 34; started inside the view, every frame stays in it.
        """
        base = raw_landmarks("yaw35")
        model = raw_yaw("yaw35")
        for target in (34.0, 36.0, 31.0, 33.0):
            pts = with_landmark_yaw(base, target)
            self.assertAlmostEqual(
                cheek.landmark_yaw(pts, cheek.pupil_frame(pts)), target, places=6
            )
        previous = False
        seen = []
        for target in (34.0, 36.0, 34.0, 36.0):
            previous, _ = cheek.classify_quarter(
                with_landmark_yaw(base, target), model, previous_quarter=previous
            )
            seen.append(previous)
        self.assertEqual(seen, [False, True, True, True])
        previous = True
        seen = []
        for target in (34.0, 36.0, 34.0, 36.0):
            previous, _ = cheek.classify_quarter(
                with_landmark_yaw(base, target), model, previous_quarter=previous
            )
            seen.append(previous)
        self.assertEqual(seen, [True, True, True, True])

    def test_hold_leaves_below_the_widened_edge(self):
        base = raw_landmarks("yaw35")
        model = raw_yaw("yaw35")
        at_36 = with_landmark_yaw(base, 36.0)
        at_33 = with_landmark_yaw(base, 33.0)
        at_31 = with_landmark_yaw(base, 31.0)
        previous, _ = cheek.classify_quarter(at_36, model)
        self.assertTrue(previous)
        self.assertEqual(
            cheek.classify_quarter(at_31, model, previous_quarter=True), (False, False)
        )
        self.assertEqual(
            cheek.classify_quarter(at_33, model, previous_quarter=True), (True, True)
        )
        self.assertEqual(cheek.classify_quarter(at_33, model), (False, False))
        # The hold never relaxes the margin rule or the model's minimum.
        self.assertEqual(
            cheek.classify_quarter(at_36, 5.0, previous_quarter=True), (False, False)
        )
        self.assertEqual(
            cheek.classify_quarter(
                raw_landmarks("yaw55"), raw_yaw("yaw55"), previous_quarter=True
            ),
            (False, False),
        )

    def test_hold_at_the_top_edge(self):
        """
        56 degrees stays a quarter face only while held, 59 never. A tip slid
        that far on a real contour (yaw50's) lands past the contour top (margin
        -0.07), which the margin rule rejects whatever the hold, so the edge is
        exercised on the stick face with its contour widened by half.
        """
        base = stick_face()
        base[:33, 0] = 510.0 + (base[:33, 0] - 510.0) * 1.5
        model = raw_yaw("yaw55")  # 37.79, calibrated 55: past the minimum
        at_56 = with_landmark_yaw(base, 56.0)
        at_59 = with_landmark_yaw(base, 59.0)
        for pts in (at_56, at_59):
            frame = cheek.pupil_frame(pts)
            self.assertGreater(cheek.nose_margin(pts, frame), 0.05)
        self.assertEqual(cheek.classify_quarter(at_56, model), (False, False))
        self.assertEqual(
            cheek.classify_quarter(at_56, model, previous_quarter=True), (True, True)
        )
        self.assertEqual(
            cheek.classify_quarter(at_59, model, previous_quarter=True), (False, False)
        )
        self.assertEqual(
            cheek.classify_quarter(with_landmark_yaw(base, 54.0), model), (True, True)
        )


class TestLocateAll(unittest.TestCase):
    """locate_all() on the three poses, with observed_points() stubbed."""

    FRONTAL_ROWS = (0, 1, 2, 3, 4, 5)
    OBLIQUE_ROWS = (6, 7, 8, 9)

    def test_frontal_rows(self):
        pts = entry("img4")["pipeline"]["landmarks"]
        points = cheek.locate_all(pts, Pose.FRONTAL, False, None, None, None)
        self.assertEqual(points.shape, (10, 2))
        self.assertEqual(points.dtype, np.float64)
        fc = cheek.frontal_constructions(pts)
        np.testing.assert_allclose(points[cheek.CHEEK_LM_ZYGION_R], fc.zygion_r)
        np.testing.assert_allclose(points[cheek.CHEEK_LM_ZYGION_L], fc.zygion_l)
        np.testing.assert_allclose(points[cheek.CHEEK_LM_CHEEK_REF_R], fc.cheek_ref_r)
        np.testing.assert_allclose(points[cheek.CHEEK_LM_CHEEK_REF_L], fc.cheek_ref_l)
        np.testing.assert_allclose(
            points[cheek.CHEEK_LM_SUBMALAR_REF_R], fc.submalar_ref_r
        )
        np.testing.assert_allclose(
            points[cheek.CHEEK_LM_SUBMALAR_REF_L], fc.submalar_ref_l
        )
        self.assertTrue(np.isnan(points[list(self.OBLIQUE_ROWS)]).all())
        self.assertEqual(len(cheek.CHEEK_LANDMARK_NAMES), 10)

    def test_lateral_is_all_nan(self):
        pts = entry("img4")["pipeline"]["landmarks"]
        self.assertTrue(
            np.isnan(cheek.locate_all(pts, Pose.LATERAL, False, None, None, None)).all()
        )
        self.assertTrue(
            np.isnan(cheek.locate_all(pts, Pose.DETECT, False, None, None, None)).all()
        )

    def test_degenerate_landmarks_are_all_nan(self):
        pts = np.array(entry("img4")["pipeline"]["landmarks"], dtype=np.float64)
        for pose in (Pose.FRONTAL, Pose.QUARTER):
            self.assertTrue(
                np.isnan(
                    cheek.locate_all(pts[:97], pose, False, None, None, None)
                ).all()
            )
            self.assertTrue(
                np.isnan(cheek.locate_all([], pose, False, None, None, None)).all()
            )
            coincident = pts.copy()
            coincident[97] = coincident[96]
            self.assertTrue(
                np.isnan(
                    cheek.locate_all(coincident, pose, True, None, None, None)
                ).all()
            )

    def _stub(self, depth=0.01):
        return SimpleNamespace(
            apex=np.array([101.0, 202.0]),
            submalar=np.array([303.0, 404.0]),
            submalar_depth=depth,
        )

    def test_quarter_rows_follow_the_far_side(self):
        pts = raw_landmarks("yaw45")
        xs = np.arange(5, dtype=np.float32)
        ys = np.arange(5, dtype=np.int32)
        src = np.zeros(5, dtype=np.uint8)
        for far_left, apex_row, sub_row in ((True, 7, 9), (False, 6, 8)):
            with mock.patch.object(
                cheek_outline, "observed_points", return_value=self._stub()
            ) as observed:
                points = cheek.locate_all(pts, Pose.QUARTER, far_left, xs, ys, src)
            observed.assert_called_once()
            args = observed.call_args[0]
            np.testing.assert_allclose(args[0], np.asarray(pts, dtype=float))
            self.assertIs(args[1], xs)
            self.assertIs(args[2], ys)
            self.assertIs(args[3], src)
            self.assertEqual(args[4], far_left)
            np.testing.assert_allclose(points[apex_row], (101.0, 202.0))
            np.testing.assert_allclose(points[sub_row], (303.0, 404.0))
            other = [r for r in range(10) if r not in (apex_row, sub_row)]
            self.assertTrue(np.isnan(points[other]).all())

    def test_quarter_submalar_nan_when_its_depth_is(self):
        pts = raw_landmarks("yaw45")
        with mock.patch.object(
            cheek_outline, "observed_points", return_value=self._stub(math.nan)
        ):
            points = cheek.locate_all(pts, Pose.QUARTER, True, None, None, None)
        np.testing.assert_allclose(points[cheek.CHEEK_LM_APEX_L], (101.0, 202.0))
        self.assertTrue(np.isnan(points[cheek.CHEEK_LM_SUBMALAR_L]).all())

    def test_quarter_without_observed_points(self):
        pts = raw_landmarks("yaw45")
        with mock.patch.object(cheek_outline, "observed_points", return_value=None):
            points = cheek.locate_all(pts, Pose.QUARTER, True, None, None, None)
        self.assertTrue(np.isnan(points).all())
        with self.assertLogs("dynaface.cheek", level="WARNING") as logs:
            with mock.patch.object(
                cheek_outline, "observed_points", side_effect=ValueError("mid-edit")
            ):
                points = cheek.locate_all(pts, Pose.QUARTER, True, None, None, None)
        self.assertTrue(np.isnan(points).all())
        # The swallowed exception is logged with its text, so a real bug in the
        # oblique search shows in a video import's log.
        self.assertEqual(len(logs.output), 1)
        self.assertIn("ValueError: mid-edit", logs.output[0])

    def test_quarter_ragged_outline_means_the_fallback(self):
        pts = raw_landmarks("yaw45")
        xs = np.arange(5, dtype=np.float32)
        ys = np.arange(4, dtype=np.int32)
        src = np.zeros(5, dtype=np.uint8)
        with mock.patch.object(
            cheek_outline, "observed_points", return_value=self._stub()
        ) as observed:
            cheek.locate_all(pts, Pose.QUARTER, True, xs, ys, src)
        self.assertEqual(observed.call_args[0][1:4], (None, None, None))
        with mock.patch.object(
            cheek_outline, "observed_points", return_value=self._stub()
        ) as observed:
            cheek.locate_all(pts, Pose.QUARTER, True, xs, None, src)
        self.assertEqual(observed.call_args[0][1:4], (None, None, None))

    def test_landmark_array(self):
        """The one rule the array is filled by: rows 0-5 from the constructions,
        the far side's rows 6-9 from the observed points, NaN elsewhere."""
        pts = entry("img4")["pipeline"]["landmarks"]
        fc = cheek.frontal_constructions(pts)
        frontal = cheek.locate_all(pts, Pose.FRONTAL, False, None, None, None)
        both = cheek.landmark_array(fc, self._stub(), True)
        self.assertEqual(both.shape, (10, 2))
        self.assertEqual(both.dtype, np.float64)
        rows = list(self.FRONTAL_ROWS)
        np.testing.assert_array_equal(both[rows], frontal[rows])
        self.assertFalse(np.isnan(both[rows]).any())
        np.testing.assert_allclose(both[cheek.CHEEK_LM_APEX_L], (101.0, 202.0))
        np.testing.assert_allclose(both[cheek.CHEEK_LM_SUBMALAR_L], (303.0, 404.0))
        self.assertTrue(np.isnan(both[[6, 8]]).all())
        np.testing.assert_array_equal(cheek.landmark_array(fc, None, True), frontal)
        # The right side's rows, and no submalar row when its depth is NaN.
        right = cheek.landmark_array(None, self._stub(math.nan), False)
        np.testing.assert_allclose(right[cheek.CHEEK_LM_APEX_R], (101.0, 202.0))
        self.assertTrue(np.isnan(right[[r for r in range(10) if r != 6]]).all())
        # Nothing located: all NaN, and a fresh array every time.
        none = cheek.landmark_array(None, None, False)
        self.assertTrue(np.isnan(none).all())
        self.assertIsNot(none, cheek.landmark_array(None, None, False))

    def test_observe_far_cheek_needs_a_pupil_frame(self):
        """No pupil frame, no observed points, and observed_points() isn't even
        asked: the rule locate_all(), AnalyzeFace and the measure share."""
        pts = np.array(raw_landmarks("yaw45"), dtype=np.float64)
        with mock.patch.object(
            cheek_outline, "observed_points", return_value=self._stub()
        ) as observed:
            self.assertIsNotNone(cheek.observe_far_cheek(pts, None, None, None, True))
        observed.assert_called_once()
        coincident = pts.copy()
        coincident[97] = coincident[96]
        broken = pts.copy()
        broken[cheek.LM_CHIN, 1] = math.nan
        for landmarks in (coincident, broken, pts[:97], []):
            with mock.patch.object(
                cheek_outline, "observed_points", return_value=self._stub()
            ) as observed:
                self.assertIsNone(
                    cheek.observe_far_cheek(landmarks, None, None, None, True)
                )
            observed.assert_not_called()
        # On the real module, locate_all() is landmark_array() of its result.
        ob = cheek.observe_far_cheek(pts, None, None, None, True)
        np.testing.assert_array_equal(
            cheek.locate_all(pts, Pose.QUARTER, True, None, None, None),
            cheek.landmark_array(None, ob, True),
        )

    def test_quarter_path_runs_on_the_real_module(self):
        """Whatever cheek_outline returns today, the quarter path never raises."""
        pts = raw_landmarks("yaw45")
        points = cheek.locate_all(pts, Pose.QUARTER, True, None, None, None)
        self.assertEqual(points.shape, (10, 2))
        self.assertTrue(np.isnan(points[list(self.FRONTAL_ROWS)]).all())
        self.assertTrue(np.isnan(points[[6, 8]]).all())
        # A NaN chin or an inf canthus: no observed points, and no exception to
        # swallow (nothing is logged).
        for index, value in ((cheek.LM_CHIN, math.nan), (cheek.LM_CANTHUS_L, math.inf)):
            broken = np.array(pts, dtype=np.float64)
            broken[index, 1] = value
            with self.assertNoLogs("dynaface.cheek", level="WARNING"):
                points = cheek.locate_all(broken, Pose.QUARTER, True, None, None, None)
            self.assertTrue(np.isnan(points).all())


if __name__ == "__main__":
    unittest.main()
