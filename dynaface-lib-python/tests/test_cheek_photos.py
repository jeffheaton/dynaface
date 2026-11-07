"""
The cheek landmarks through the real pipeline, with the models: the test photos,
the app's three-quarter sketch and the synthetic-head renders, loaded with
config.AUTO_QUARTER on (set in setUpClass and restored in tearDownClass; a forced
pose is only ever passed to one load_image() call, so none persists).

What the pipeline gave on CPU when these numbers were measured, kept here so a
change shows against them (the asserted tolerances are wider). A quarter face
is cropped at the band-fit scale: the landmarks' band plus its pads fills the
1024 px crop whatever the frame, so H (cheek.scale) is 600 to 700 px on every
quarter image here, and the pixel-sized rules read alike from any frame.

- img4 (frontal): zy.w 130.59 mm (R 65.53 / L 65.06), cr.r.d and cr.l.d 28.835
  mm, the zygion level 18.90 mm below the pupil line, no clip, cheek.yaw -0.89
  against the model's 0.00 (raw 1.08, inside the level head's reading of 1.5),
  zy.ratio.ok 1.
- img1 (frontal, cheek.yaw 10.84): zy.ok 0 with zy.w 151.36 mm.
- The sketch (tests_data/pose-quarter-512.png): frontal under detection
  (cheek.yaw -32.0); forced quarter, far_left False, scaled 3.8 times (H 604
  px), where the background model separates the hair beside the far cheek from
  the cheek: 660 of 664 outline rows from the mask and 4 from the contour, none
  from the intensity edge, the mask rows a median 3.1 px outside the landmark
  contour (the hair's edge lies 15 px and more out, past the walk's reach on
  all but a few rows), cheek.mask 1.0, a local-maximum apex at h 0.748 (dev
  0.056) with
  apex.ok 0 on the yaw gate. (At the frame-filling scale of 2.0 the same
  model kept the hair as foreground, and 317 of 351 rows came from the
  intensity edge, none from the mask.)
- yaw0 and yaw10 (frontal, zy.ok 1 and 0): each zygion sits inside the render's
  silhouette at its row, on the subject's right by 2.80 and 1.48 mm, on the left
  by 8.36 and 4.61 mm: SPIGA's left contour reads about 19 raw px (7.6 mm)
  inside this synthetic head's silhouette (the 1.08 side ratio of the
  detector's known contour bias), so the "within 4 mm" bound holds on the right
  side only; the left offsets are asserted as measured. The landmark yaw reads yaw0
  as 2.10 degrees (SPIGA's nose tip sits 0.02 eye-corner spans off this head's
  center: the cue's bias at zero, which the mirrored turns don't share), so
  zy.ratio.ok is 0 on the level head while zy.ok is 1, as the caveat in
  docs/measures.md states.
- yaw45 and yaw-45 (quarter, far_left True and False, H 665 and 696 px):
  cheek.yaw 44.88 and -45.05 (0.17 degrees apart), cheek.yaw.model 45.00 and
  -44.77 (0.23 apart; the raw yaws 28.28 and -25.63 read less
  cheek.YAW_RAW_OFFSET), every band row from the mask, each stored x equal to
  the alpha-32 crossing of the raw alpha recomputed in the test (0.00003 px
  apart at most, float32 rounding) and outside the render's true silhouette by
  a median 0.32 px (min -1.58, max 2.16) and 2.27 px (min 0.79, max 4.03); the
  apex x 0.81 px inside and 1.47 px outside the true silhouette at its row;
  apex.l.h 0.7935 and apex.r.h 0.7931 (0.0004 apart), both the prior; submalar
  depth -0.0000 H (-0.02 px) and -0.0004 H (-0.31 px), present 0.
- yaw45_hair (quarter, far_left True, H 667 px): yaw45 with hair painted beside
  the far cheek (scripts/make_cheek_hair_render.py), which the background model
  keeps as foreground, where it drops the sketch's own hair, so no band row
  comes from the mask: 365 of the 399 band rows from the intensity edge, a
  median 1.5 px inside the landmark contour (the plain render's mask rows sit
  4.5 px outside it), the 34 at the band's top, above the edge source's
  EDGE_MAX_H, from the contour, and 247 mask rows below the stomion level, past
  the hair; cheek.mask 0, the prior apex on an edge row at h 0.7927 with
  apex.ok 0, no depth (NaN), cheek.yaw 45.26 against the model's 43.62.
- yaw45_hollow3 (quarter): submalar depth 0.0043 H (2.85 px) at h 0.491,
  present 1, 0.0043 H above the plain render's, the apex the prior at h
  0.7935; the stored x outside the true silhouette by a median 0.28 px (max
  2.06). The render's own submalar oracle (cheek_renders.json: 2.21 raw px
  deep at h 0.557, under the same chord rule on the true silhouette) reads
  deeper and higher than the pipeline's point (2.85 crop px, 1.22 raw px at
  the crop's 2.34 scale, at h 0.491); test_cheek_outline's TestRenders asserts
  the oracle within 30% on the raw frame, and this file deliberately asserts
  only presence and the shipped threshold on the pipeline, because the
  threshold and the chord rule were checked on photos (408 Face Research Lab
  London photos, both turns at rest and smiling: SUBMALAR_MIN_DEPTH is 2.7 to
  3.6 times the no-hollow floor there by statistic, and stays 0.003; see its
  comment in cheek_outline.py) and the pipeline reads a pressed hollow at 0.6
  to 0.8 of its silhouette depth.
- yaw45_hollow1.5 (quarter): the 1.5 mm hollow moves the render's own
  silhouette by half a raw pixel (its oracle reads 0.45 px, 0.18 mm, at h
  0.56) and the pipeline reads no depth at all, -0.0002 H (-0.11 px) at h
  0.386 against the plain render's -0.0000, present 0: the false negative the
  depth rule's own bound predicts (0.0012 to 0.0027 H there) and
  docs/measures.md's caveat states.
- yaw45_hollow4.5 (quarter): submalar depth 0.0088 H (5.80 px) at h 0.504,
  present 1, about twice the 3 mm hollow's reading, where its oracle reads
  3.51 raw px (1.41 mm) at h 0.55, 1.6 times the 3 mm hollow's oracle.
- yaw50 (quarter): cheek.yaw 50.06 (the model's 50.00), 65 nose rows,
  cheek.nose 1, cheek.nose.gap 0.0022, cheek.mask 0.84, apex.ok 0 with the prior
  apex at h 0.793; the stored mask rows outside the true silhouette by a median
  1.31 px (max 3.04).
"""

import math
import os
import sys
import unittest
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import (  # noqa: E402
    cheek,
    cheek_outline,
    config,
    facial,
    lateral,
    measures,
    models,
    util,
)
from dynaface.const import Pose  # noqa: E402
from dynaface.image import load_image  # noqa: E402
from dynaface.measures_cheek import ITEMS, AnalyzeCheekMeasures  # noqa: E402

# The cheek outputs analyze() returns, by view: the Cheek Measures items'
# outputs with the coordinates the Landmarks measure's Cheek Landmarks item
# reports for them (the frontal six's, then the far cheek's apex and submalar
# point), and the head turn's, numeric in either view.
_OUTPUTS_OF = {name: outputs for name, _, outputs in ITEMS}
_COORDS = tuple(cheek.CHEEK_LANDMARK_OUTPUTS)
FRONTAL_OUTPUTS = (
    _OUTPUTS_OF["Zygion width"] + _OUTPUTS_OF["Reference points"] + _COORDS[:12]
)
OBLIQUE_OUTPUTS = (
    _OUTPUTS_OF["Far cheek outline"]
    + _OUTPUTS_OF["Malar apex"]
    + _OUTPUTS_OF["Submalar point"]
    + _COORDS[12:]
)
SHARED_OUTPUTS = _OUTPUTS_OF["Head turn"]

TESTS_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests_data"
)
PATHS = {
    "img4": "img4-1024-frontal.jpg",
    "img1": "img1-512.jpg",
    "img2": "img2-1024-right-lateral.jpg",
    "img3": "img3-1024-left-lateral.jpg",
    "sketch": "pose-quarter-512.png",
    "yaw0": os.path.join("cheek", "yaw0.png"),
    "yaw10": os.path.join("cheek", "yaw10.png"),
    "yaw45": os.path.join("cheek", "yaw45.png"),
    "yaw-45": os.path.join("cheek", "yaw-45.png"),
    "yaw50": os.path.join("cheek", "yaw50.png"),
    "yaw45_hollow1.5": os.path.join("cheek", "yaw45_hollow1.5.png"),
    "yaw45_hollow3": os.path.join("cheek", "yaw45_hollow3.png"),
    "yaw45_hollow4.5": os.path.join("cheek", "yaw45_hollow4.5.png"),
    "yaw45_hair": os.path.join("cheek", "yaw45_hair.png"),
}
RENDERS = (
    "yaw0",
    "yaw10",
    "yaw45",
    "yaw-45",
    "yaw50",
    "yaw45_hollow1.5",
    "yaw45_hollow3",
    "yaw45_hollow4.5",
    "yaw45_hair",
)
QUARTER_RENDERS = (
    "yaw45",
    "yaw-45",
    "yaw50",
    "yaw45_hollow1.5",
    "yaw45_hollow3",
    "yaw45_hollow4.5",
    "yaw45_hair",
)
# The renders' background color (tests_data/cheek/cheek_renders.json) and how
# far a resampled pixel may stray from it, per channel, and still be background.
BACKGROUND = np.array([238, 240, 243], dtype=np.int32)
BACKGROUND_TOL = 12
FRONTAL_VIEW = {"pose": "frontal", "flipped": False, "lateral": False}
# The measured distance of each frontal render's zygion inside the silhouette,
# mm at the face's pix2mm (see the module docstring).
ZYGION_INSIDE_MM = {
    ("yaw0", "r"): 2.80,
    ("yaw0", "l"): 8.36,
    ("yaw10", "r"): 1.48,
    ("yaw10", "l"): 4.61,
}
# The library's bound on the zygion's distance from the silhouette, which the
# synthetic head meets on the subject's right side only.
ZYGION_SILHOUETTE_MM = 4.0
# How far the sketch's mask rows may sit from the landmark contour, as a median
# over the band, px: measured 3.1 (the skin's edge; the hair's edge beside it
# lies 15 px and more out, past the walk's reach on all but a few rows).
SKETCH_SEED_DEPARTURE_PX = 6.0
# How far the hair render's intensity-edge rows may sit from the landmark
# contour, as a median over the band, px: measured 2.1 (the skin's edge, which
# the painted hair meets a pixel outside the contour).
HAIR_EDGE_DEPARTURE_PX = 4.0
# How far a stored mask row's x may sit from the alpha-32 crossing recomputed
# on the same alpha, px: measured 0.00003 at most on every render (the float32
# the outline is stored in), so a sub-pixel bound that still admits another
# rounding of the interpolation, where a tracer that stopped interpolating
# would be a whole pixel off on every row.
ALPHA_CROSSING_PX = 0.25


def _tolerance(expected: float) -> float:
    # test_facial's tolerance: 30% relative, with a 1.0 absolute floor, sized
    # against the swing between OpenCV versions and execution providers.
    return max(abs(expected) * 0.30, 1.0)


def _same(a: Any, b: Any) -> bool:
    """Equal values, where NaN equals NaN."""
    return bool(a == b) or (a != a and b != b)


def _foreground(image: np.ndarray) -> np.ndarray:
    """Every pixel of a cropped render that is not the background color."""
    return np.any(np.abs(image.astype(np.int32) - BACKGROUND) > BACKGROUND_TOL, axis=2)


def _silhouette(foreground_row: np.ndarray, x: float, out: int) -> float:
    """
    The render's true silhouette in one row, as the tracer would walk to it from
    x: outward while the pixels are foreground, inward while they are not, to
    the boundary between the last foreground pixel and the background.
    """
    width = len(foreground_row)
    col = min(max(int(round(x)), 0), width - 1)
    if foreground_row[col]:
        while 0 <= col + out < width and foreground_row[col + out]:
            col += out
    else:
        while 0 <= col - out < width and not foreground_row[col]:
            col -= out
        if not foreground_row[col]:
            return math.nan
    return col + 0.5 * out


def _alpha_crossing(alpha_row: np.ndarray, x: float, out: int) -> float:
    """
    Where the raw alpha crosses MASK_ALPHA nearest x along the walk, interpolated
    between the two pixels that bracket it, as cheek_outline's tracer does.
    """
    width = len(alpha_row)
    col = min(max(int(round(x)), 0), width - 1)
    alpha = alpha_row.astype(np.float64)
    if alpha[col] > cheek_outline.MASK_ALPHA:
        while 0 <= col + out < width and alpha[col + out] > cheek_outline.MASK_ALPHA:
            col += out
    else:
        while 0 <= col - out < width and alpha[col] <= cheek_outline.MASK_ALPHA:
            col -= out
    fg, bg = col, col + out
    if not (0 <= fg < width and 0 <= bg < width):
        return math.nan
    if not alpha[fg] > cheek_outline.MASK_ALPHA >= alpha[bg]:
        return math.nan
    t = (alpha[fg] - cheek_outline.MASK_ALPHA) / (alpha[fg] - alpha[bg])
    return fg + t * (bg - fg)


class TestCheekPhotos(unittest.TestCase):
    """The cheek landmarks on real detections (see the module docstring)."""

    _auto_quarter: bool
    # Each image loaded once, with a copy of its cropped image before anything
    # is drawn on it, keyed by name and the forced pose.
    _faces: Dict[Tuple[str, Pose], Tuple[facial.AnalyzeFace, np.ndarray]]

    @classmethod
    def setUpClass(cls) -> None:
        models.init_models(models.download_models(), models.detect_device())
        cls._auto_quarter = config.AUTO_QUARTER
        config.AUTO_QUARTER = True
        cls._faces = {}

    @classmethod
    def tearDownClass(cls) -> None:
        config.AUTO_QUARTER = cls._auto_quarter
        cls._faces = {}

    @classmethod
    def load(
        cls, name: str, force_pose: Pose = Pose.DETECT
    ) -> Tuple[facial.AnalyzeFace, np.ndarray]:
        """The face loaded from the named image with every measure, and a clean
        copy of its cropped image."""
        key = (name, force_pose)
        if key not in cls._faces:
            face = facial.AnalyzeFace(measures=measures.all_measures())
            image = load_image(os.path.join(TESTS_DATA, PATHS[name]))
            face.load_image(image, crop=True, force_pose=force_pose)
            cls._faces[key] = (face, face.original_img.copy())
        return cls._faces[key]

    @classmethod
    def face(cls, name: str, force_pose: Pose = Pose.DETECT) -> facial.AnalyzeFace:
        return cls.load(name, force_pose)[0]

    def results(self, face: facial.AnalyzeFace) -> Dict[str, Any]:
        results = face.analyze(render=False)
        self.assertIsNotNone(results)
        assert results is not None
        return results

    def assert_frontal_face(
        self, face: facial.AnalyzeFace, results: Dict[str, Any]
    ) -> None:
        """A frontal face: its view, its cheek landmarks and the measure's
        frontal core filled, the oblique module NaN."""
        self.assertEqual(face.pose, Pose.FRONTAL)
        self.assertFalse(face.quarter or face.lateral or face.flipped)
        self.assertFalse(face.far_left)
        self.assertIsNone(face.cheek_outline_x)
        self.assertEqual(face.dump_state()[6], FRONTAL_VIEW)
        for name in FRONTAL_OUTPUTS + SHARED_OUTPUTS:
            self.assertTrue(math.isfinite(results[name]), name)
        for name in OBLIQUE_OUTPUTS:
            self.assertTrue(math.isnan(results[name]), name)
        fc = face.frontal_cheek()
        self.assertIsNotNone(fc)
        assert fc is not None
        self.assertIsNone(face.oblique_cheek())
        # The coordinates (the Landmarks measure's Cheek Landmarks item) are the
        # constructions', and so are the cheek landmarks' rows 0 to 5.
        for key, point in (
            ("zy.r", fc.zygion_r),
            ("zy.l", fc.zygion_l),
            ("cr.r", fc.cheek_ref_r),
            ("cr.l", fc.cheek_ref_l),
            ("sr.r", fc.submalar_ref_r),
            ("sr.l", fc.submalar_ref_l),
        ):
            self.assertEqual((results[f"{key}.x"], results[f"{key}.y"]), tuple(point))
        np.testing.assert_array_equal(
            face.cheek_landmarks,
            cheek.landmark_array(fc, None, False),
        )
        self.assertTrue(np.isnan(face.cheek_landmarks[6:]).all())
        # The scale contract: the measure's millimeters use the pupil scale of
        # the landmarks being measured.
        self.assertAlmostEqual(face.pix2mm, results["px2mm"], places=9)

    def assert_quarter_face(
        self, face: facial.AnalyzeFace, results: Dict[str, Any], far_left: bool
    ) -> Tuple[cheek_outline.ObliqueCheek, str]:
        """A quarter face: its view, its outline, the measure's oblique module
        filled for the far side only and the frontal core NaN; the observed
        points and the far side's letter."""
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertTrue(face.quarter)
        self.assertFalse(face.lateral or face.flipped)
        self.assertEqual(face.far_left, far_left)
        self.assertIsNotNone(face.cheek_outline_x)
        assert face.cheek_outline_x is not None
        assert face.cheek_outline_y is not None
        assert face.cheek_outline_src is not None
        self.assertEqual(face.cheek_outline_x.dtype, np.float32)
        self.assertEqual(face.cheek_outline_y.dtype, np.int32)
        self.assertEqual(face.cheek_outline_src.dtype, np.uint8)
        self.assertEqual(
            len(face.cheek_outline_x),
            len(face.cheek_outline_y),
        )
        self.assertEqual(len(face.cheek_outline_x), len(face.cheek_outline_src))
        self.assertGreater(len(face.cheek_outline_x), 0)
        view = face.dump_state()[6]
        self.assertEqual(view["pose"], "quarter")
        self.assertIs(view["far_left"], far_left)
        self.assertEqual(results["cheek.far"], float(far_left))
        for name in FRONTAL_OUTPUTS:
            self.assertTrue(math.isnan(results[name]), name)
        for name in SHARED_OUTPUTS + ("cheek.mask", "cheek.nose", "cheek.scale"):
            self.assertTrue(math.isfinite(results[name]), name)
        # The pupil scale, foreshortened, is kept; Position reports no
        # millimeter per pixel on a quarter face, and the pixel pd stays.
        self.assertTrue(math.isnan(results["px2mm"]))
        self.assertTrue(math.isfinite(results["pd"]))
        self.assertEqual(results["pd"], face.pupillary_distance)
        self.assertTrue(math.isfinite(results["tilt"]))
        ob = face.oblique_cheek()
        self.assertIsNotNone(ob)
        assert ob is not None
        self.assertIsNone(face.frontal_cheek())
        s = "l" if far_left else "r"
        other = "r" if far_left else "l"
        for name in ("x", "y", "h", "dev", "prior", "near_nose"):
            self.assertTrue(math.isnan(results[f"apex.{other}.{name}"]), name)
            self.assertTrue(math.isfinite(results[f"apex.{s}.{name}"]), name)
        for name in ("x", "y", "h", "depth", "depth.px", "present"):
            self.assertTrue(math.isnan(results[f"sub.{other}.{name}"]), name)
        self.assertEqual(
            (results[f"apex.{s}.x"], results[f"apex.{s}.y"]), tuple(ob.apex)
        )
        apex_row = cheek.CHEEK_LM_APEX_L if far_left else cheek.CHEEK_LM_APEX_R
        np.testing.assert_array_equal(face.cheek_landmarks[apex_row], ob.apex)
        self.assertTrue(np.isnan(face.cheek_landmarks[:6]).all())
        # The apex lies on the stored outline: the outline's x at its row.
        rows = np.flatnonzero(face.cheek_outline_y == int(round(float(ob.apex[1]))))
        self.assertEqual(len(rows), 1)
        self.assertLessEqual(
            abs(float(face.cheek_outline_x[rows[0]]) - float(ob.apex[0])), 1.0
        )
        self.assertGreater(results["cheek.scale"], 0.0)
        return ob, s

    def band_offsets(
        self, name: str, ob: cheek_outline.ObliqueCheek, far_left: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each band row (canthus level to stomion level) that came from the
        mask, the stored x minus the alpha-32 crossing of the raw alpha from
        lateral.foreground_mask() recomputed on the same cropped image, and the
        stored x minus the render's true silhouette from the background color
        (outward positive).
        """
        face, clean = self.load(name)
        self.assertTrue(np.array_equal(clean, face.original_img))
        pil = util.cv2_to_pil(cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
        _, alpha = lateral.foreground_mask(pil)
        foreground = _foreground(clean)
        out = 1 if far_left else -1
        from_alpha, from_true = [], []
        for (x, y), src in zip(ob.outline, ob.outline_src):
            if src != cheek_outline.OUTLINE_SRC_MASK:
                continue
            row = int(round(float(y)))
            from_alpha.append(float(x) - _alpha_crossing(alpha[row], float(x), out))
            from_true.append(
                (float(x) - _silhouette(foreground[row], float(x), out)) * out
            )
        return np.array(from_alpha), np.array(from_true)

    def test_img4_frontal_constructions(self):
        face = self.face("img4")
        results = self.results(face)
        self.assert_frontal_face(face, results)
        fc = face.frontal_cheek()
        assert fc is not None
        self.assertEqual(results["zy.ok"], 1.0)
        self.assertEqual((results["zy.r.ok"], results["zy.l.ok"]), (1.0, 1.0))
        self.assertEqual((results["zy.r.clip"], results["zy.l.clip"]), (0.0, 0.0))
        self.assertEqual((results["cr.ok"], results["sr.ok"]), (1.0, 1.0))
        self.assertEqual(results["zy.ratio.ok"], 1.0)
        self.assertEqual(results["cheek.pitch.ok"], 1.0)
        self.assertLessEqual(abs(results["cheek.yaw"]), cheek.ZYGION_RATIO_YAW_GATE)
        # The frontal reference readings, measured on img4.
        expected = {
            "zy.w": 130.6,
            "zy.r": 65.5,
            "zy.l": 65.1,
            "cr.r.d": 28.8,
            "cr.l.d": 28.8,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(
                results[key], value, delta=_tolerance(value), msg=key
            )
        level_mm = fc.zygion_level * face.pix2mm
        self.assertAlmostEqual(level_mm, 18.9, delta=_tolerance(18.9))
        self.assertAlmostEqual(
            fc.zygion_level, cheek.ZYGION_LEVEL_IPD * fc.frame.ipd, places=6
        )
        self.assertAlmostEqual(
            results["zy.ratio"], util.symmetry_ratio(results["zy.r"], results["zy.l"])
        )
        self.assertEqual(results["zy.w"], results["zy.r"] + results["zy.l"])

    def test_img1_turned_a_little_is_flagged(self):
        face = self.face("img1")
        results = self.results(face)
        self.assert_frontal_face(face, results)
        self.assertGreater(abs(results["cheek.yaw"]), cheek.FRONTAL_YAW_GATE)
        self.assertEqual(results["zy.ok"], 0.0)
        self.assertEqual((results["zy.r.ok"], results["zy.l.ok"]), (0.0, 0.0))
        self.assertEqual(results["zy.ratio.ok"], 0.0)
        self.assertEqual((results["cr.ok"], results["sr.ok"]), (0.0, 0.0))
        # Flagged, not blanked: the numbers are still reported.
        self.assertEqual((results["zy.r.clip"], results["zy.l.clip"]), (0.0, 0.0))
        self.assertGreater(results["zy.w"], 0.0)
        self.assertGreater(results["cr.r.d"], 0.0)
        self.assertGreater(results["cr.l.d"], 0.0)

    def test_profiles_have_no_cheek_outputs(self):
        for name, flipped in (("img2", True), ("img3", False)):
            with self.subTest(name):
                face = self.face(name)
                self.assertEqual(face.pose, Pose.LATERAL)
                self.assertTrue(face.lateral)
                self.assertEqual(face.flipped, flipped)
                self.assertFalse(face.quarter)
                self.assertIsNone(face.cheek_outline_x)
                self.assertTrue(np.isnan(face.cheek_landmarks).all())
                self.assertIsNone(face.frontal_cheek())
                self.assertIsNone(face.oblique_cheek())
                measure = AnalyzeCheekMeasures()
                self.assertEqual(measure.calc(face, render=False), {})
                self.assertEqual(measure.calc(face, render=True), {})
                results = self.results(face)
                # No Cheek Measures output; the Landmarks measure (which suits
                # no profile, but analyze() runs every enabled measure) reports
                # the cheek landmarks' coordinates, all NaN.
                for name in AnalyzeCheekMeasures().outputs():
                    self.assertNotIn(name, results)
                for name in cheek.CHEEK_LANDMARK_OUTPUTS:
                    self.assertTrue(math.isnan(results[name]), name)
                self.assertIn("nfa", results)

    def test_sketch_is_frontal_under_detection(self):
        # The app's sketch turns about 30 degrees: below the quarter window.
        face = self.face("sketch")
        results = self.results(face)
        self.assert_frontal_face(face, results)
        self.assertGreater(abs(results["cheek.yaw"]), cheek.FRONTAL_YAW_GATE)
        self.assertLess(abs(results["cheek.yaw"]), cheek.QUARTER_YAW_RANGE[0])
        self.assertEqual(results["zy.ok"], 0.0)

    def test_sketch_forced_quarter_traces_the_far_cheek(self):
        # The app's sketch, turned about 30 degrees, forced to the quarter path.
        # Hair hangs beside the far cheek; at the band-fit scale (3.8 times the
        # 512 px sketch) the background model separates it from the cheek, so
        # nearly every band row comes from the mask, close to the landmark
        # contour rather than at the hair's edge, and a local-maximum apex is
        # found; the turn is outside the yaw gate, so nothing is validated.
        # (The contour and intensity-edge fallbacks, for hair the mask doesn't
        # separate, are exercised on synthetic masks in test_cheek_outline and
        # on the mocked mask in test_pose_quarter.)
        face = self.face("sketch", Pose.QUARTER)
        results = self.results(face)
        ob, s = self.assert_quarter_face(face, results, far_left=False)
        self.assertEqual(s, "r")
        assert face.cheek_outline_src is not None
        assert face.cheek_outline_x is not None
        src = face.cheek_outline_src
        mask_rows = src == cheek_outline.OUTLINE_SRC_MASK
        self.assertGreaterEqual(float(mask_rows.mean()), 0.95)
        self.assertFalse(np.any(src == cheek_outline.OUTLINE_SRC_NOSE))
        seed, _, _ = cheek_outline.contour_outline(face.landmarks, False)
        departure = (face.cheek_outline_x[mask_rows] - seed[mask_rows]) * -1.0
        self.assertLessEqual(
            float(np.median(np.abs(departure))), SKETCH_SEED_DEPARTURE_PX
        )
        self.assertGreaterEqual(results["cheek.mask"], 0.95)
        self.assertEqual(results["cheek.mask"], ob.mask_fraction)
        self.assertEqual(results["cheek.nose"], 0.0)
        self.assertGreater(
            abs(abs(results["cheek.yaw"]) - cheek.OBLIQUE_YAW), cheek.OBLIQUE_YAW_GATE
        )
        self.assertEqual(results["apex.ok"], 0.0)
        self.assertEqual(results["sub.ok"], 0.0)
        self.assertEqual(results["apex.r.prior"], 0.0)
        self.assertGreater(results["apex.r.dev"], cheek_outline.APEX_MIN_DEV)
        self.assertTrue(
            cheek_outline.APEX_H_WINDOW[0]
            <= results["apex.r.h"]
            <= cheek_outline.APEX_H_WINDOW[1]
        )
        self.assertTrue(math.isfinite(results["sub.r.depth"]))
        self.assertIn(results["sub.r.present"], (0.0, 1.0))

    def test_level_renders_are_frontal(self):
        for name, ok in (("yaw0", 1.0), ("yaw10", 0.0)):
            with self.subTest(name):
                face, clean = self.load(name)
                results = self.results(face)
                self.assert_frontal_face(face, results)
                self.assertEqual(results["zy.ok"], ok)
                self.assertEqual((results["zy.r.clip"], results["zy.l.clip"]), (0, 0))
                # The ratio gate needs a landmark yaw within 2 degrees, and the
                # level head reads 2.10 (the cue's bias at zero, which the
                # mirrored turns don't share, so the table's zero wasn't moved
                # to it; the caveat in docs/measures.md), so it fails there as
                # it does on yaw10.
                self.assertEqual(results["zy.ratio.ok"], 0.0)
                if name == "yaw0":
                    self.assertAlmostEqual(results["cheek.yaw"], 2.10, delta=0.1)
                    self.assertGreater(
                        abs(results["cheek.yaw"]), cheek.ZYGION_RATIO_YAW_GATE
                    )
                foreground = _foreground(clean)
                for side, out in (("r", -1), ("l", 1)):
                    x, y = results[f"zy.{side}.x"], results[f"zy.{side}.y"]
                    edge = _silhouette(foreground[int(round(y))], x, out)
                    self.assertTrue(math.isfinite(edge))
                    # Inside the silhouette (the contour never lies past it),
                    # by the measured distance (see the module docstring).
                    inside_mm = (edge - x) * out * face.pix2mm
                    expected = ZYGION_INSIDE_MM[(name, side)]
                    self.assertGreater(inside_mm, 0.0, side)
                    self.assertAlmostEqual(
                        inside_mm, expected, delta=_tolerance(expected), msg=side
                    )
                    if side == "r":
                        self.assertLessEqual(inside_mm, ZYGION_SILHOUETTE_MM)

    def test_45_degree_renders_are_quarter_faces(self):
        plus = self.face("yaw45")
        minus = self.face("yaw-45")
        results_plus = self.results(plus)
        results_minus = self.results(minus)
        ob_plus, _ = self.assert_quarter_face(plus, results_plus, far_left=True)
        ob_minus, _ = self.assert_quarter_face(minus, results_minus, far_left=False)
        for results, s in ((results_plus, "l"), (results_minus, "r")):
            self.assertEqual(results["cheek.nose"], 0.0)
            self.assertGreater(results["cheek.nose.gap"], 0.0)
            self.assertEqual(results["apex.ok"], 1.0)
            self.assertEqual(results["sub.ok"], 1.0)
            self.assertEqual(results[f"apex.{s}.near_nose"], 0.0)
            # The generic mesh has no distinct malar prominence, so the plain 45s
            # take the prior.
            self.assertEqual(results[f"apex.{s}.prior"], 1.0)
            self.assertLessEqual(
                abs(abs(results["cheek.yaw"]) - cheek.OBLIQUE_YAW),
                cheek.OBLIQUE_YAW_GATE,
            )
            self.assertEqual(results[f"sub.{s}.present"], 0.0)
            self.assertLess(results[f"sub.{s}.depth"], cheek_outline.SUBMALAR_MIN_DEPTH)
            self.assertAlmostEqual(
                results[f"sub.{s}.depth.px"],
                results[f"sub.{s}.depth"] * results["cheek.scale"],
                places=6,
            )
            self.assertGreaterEqual(results["cheek.mask"], 0.95)
        # The two turns read alike: the landmark yaw within a degree, the
        # model's within one too (measured 0.23 apart with the offset; 3.13
        # without it), the apex height within 0.006 H. h stands in for the
        # definitions' "mirroring within 1 mm" because a quarter face has no
        # millimeter: at the renders' 0.4 mm/px and the crop's 2.3 to 2.4
        # scale, 1 mm is 0.0083 to 0.0090 H, so 0.006 H is about 0.7 mm, 15
        # times the measured 0.0004 gap.
        self.assertGreater(results_plus["cheek.yaw"], 0.0)
        self.assertLess(results_minus["cheek.yaw"], 0.0)
        self.assertLessEqual(
            abs(abs(results_plus["cheek.yaw"]) - abs(results_minus["cheek.yaw"])), 1.0
        )
        self.assertLessEqual(
            abs(
                abs(results_plus["cheek.yaw.model"])
                - abs(results_minus["cheek.yaw.model"])
            ),
            1.0,
        )
        self.assertLessEqual(
            abs(results_plus["apex.l.h"] - results_minus["apex.r.h"]), 0.006
        )
        # The stored outline is the sub-pixel alpha crossing of the raw alpha
        # (within ALPHA_CROSSING_PX: float32 rounding, where a tracer that
        # stopped interpolating would be a whole pixel off on every row), and
        # lies on the render's own silhouette.
        for name, ob, far_left in (
            ("yaw45", ob_plus, True),
            ("yaw-45", ob_minus, False),
        ):
            with self.subTest(name):
                from_alpha, from_true = self.band_offsets(name, ob, far_left)
                self.assertGreaterEqual(len(from_alpha), 0.95 * len(ob.outline))
                self.assertFalse(np.isnan(from_alpha).any())
                self.assertFalse(np.isnan(from_true).any())
                self.assertLessEqual(
                    float(np.max(np.abs(from_alpha))), ALPHA_CROSSING_PX
                )
                self.assertLessEqual(float(np.max(np.abs(from_true))), 5.0)

    def test_hair_beside_the_far_cheek_falls_back_to_the_edge_and_the_contour(self):
        # The hair case through the real pipeline: yaw45 with hair painted
        # beside the far cheek (scripts/make_cheek_hair_render.py), which the
        # background model keeps as foreground at the band-fit crop, where it
        # drops the sketch's own hair. No band row comes from the mask: the
        # intensity edge stands in where the gray image has one, within a few
        # px of the landmark contour, the contour above EDGE_MAX_H at the
        # band's top, and the mask rows lie below the stomion level, past the
        # hair. The stand-in rows count for nothing: cheek.mask is 0, the
        # window gate zeroes apex.ok, the prior apex sits on an edge row at
        # Kaptein's height, and no depth is read below it.
        face = self.face("yaw45_hair")
        results = self.results(face)
        ob, s = self.assert_quarter_face(face, results, far_left=True)
        self.assertEqual(s, "l")
        self.assertAlmostEqual(abs(results["cheek.yaw"]), cheek.OBLIQUE_YAW, delta=2.0)
        assert face.cheek_outline_src is not None
        assert face.cheek_outline_x is not None
        assert face.cheek_outline_y is not None
        src, band_src = face.cheek_outline_src, ob.outline_src
        in_band = np.isin(face.cheek_outline_y, ob.outline[:, 1])
        self.assertFalse(np.any(band_src == cheek_outline.OUTLINE_SRC_MASK))
        self.assertFalse(np.any(band_src == cheek_outline.OUTLINE_SRC_NOSE))
        edge = band_src == cheek_outline.OUTLINE_SRC_EDGE
        contour = band_src == cheek_outline.OUTLINE_SRC_CONTOUR
        self.assertGreaterEqual(float(edge.mean()), 0.75)
        self.assertTrue(contour.any())
        self.assertLess(ob.outline[contour, 1].max(), ob.outline[edge, 1].min())
        seed, _, _ = cheek_outline.contour_outline(face.landmarks, True)
        edge_rows = in_band & (src == cheek_outline.OUTLINE_SRC_EDGE)
        departure = face.cheek_outline_x[edge_rows] - seed[edge_rows]
        self.assertLessEqual(
            float(np.median(np.abs(departure))), HAIR_EDGE_DEPARTURE_PX
        )
        mask = src == cheek_outline.OUTLINE_SRC_MASK
        self.assertTrue(np.any(mask & ~in_band))
        self.assertGreater(float(face.cheek_outline_y[mask].max()), ob.outline[-1, 1])
        self.assertEqual(results["cheek.mask"], 0.0)
        self.assertEqual(results["cheek.nose"], 0.0)
        self.assertEqual(results["apex.ok"], 0.0)
        self.assertEqual(results["apex.l.prior"], 1.0)
        self.assertEqual(results["apex.l.near_nose"], 0.0)
        self.assertAlmostEqual(
            results["apex.l.h"],
            cheek_outline.APEX_H_PRIOR,
            delta=2.0 / results["cheek.scale"],
        )
        apex_src = band_src[np.flatnonzero(ob.outline[:, 1] == ob.apex[1])[0]]
        self.assertEqual(apex_src, cheek_outline.OUTLINE_SRC_EDGE)
        for name in ("x", "y", "h", "depth", "depth.px", "present"):
            self.assertTrue(math.isnan(results[f"sub.l.{name}"]), name)
        self.assertEqual(results["sub.ok"], 0.0)
        self.assertTrue(np.isnan(face.cheek_landmarks[cheek.CHEEK_LM_SUBMALAR_L]).all())

    def test_hollow_render_has_a_submalar_point(self):
        face = self.face("yaw45_hollow3")
        results = self.results(face)
        ob, s = self.assert_quarter_face(face, results, far_left=True)
        self.assertEqual(s, "l")
        self.assertEqual(results["apex.ok"], 1.0)
        self.assertEqual(results["sub.ok"], 1.0)
        self.assertEqual(results["cheek.nose"], 0.0)
        plain = self.results(self.face("yaw45"))
        depth, plain_depth = results["sub.l.depth"], plain["sub.l.depth"]
        self.assertTrue(math.isfinite(depth))
        self.assertGreaterEqual(depth, cheek_outline.SUBMALAR_MIN_DEPTH)
        # The hollow is separated from the plain render by the shipped
        # threshold (measured 0.0043 H above its -0.0000; a bound on the ratio
        # would be vacuous, since the plain depth is not positive).
        self.assertGreaterEqual(depth - plain_depth, cheek_outline.SUBMALAR_MIN_DEPTH)
        self.assertEqual(results["sub.l.present"], 1.0)
        self.assertAlmostEqual(
            results["sub.l.depth.px"], depth * results["cheek.scale"], places=6
        )
        # The submalar point lies on the outline below the apex, above the
        # stomion level, and fills its cheek landmark.
        self.assertGreater(results["sub.l.y"], results["apex.l.y"])
        self.assertLess(results["sub.l.h"], results["apex.l.h"])
        self.assertGreater(results["sub.l.h"], 0.0)
        assert face.cheek_outline_y is not None and face.cheek_outline_x is not None
        rows = np.flatnonzero(face.cheek_outline_y == int(round(results["sub.l.y"])))
        self.assertEqual(len(rows), 1)
        self.assertLessEqual(
            abs(float(face.cheek_outline_x[rows[0]]) - results["sub.l.x"]), 1.0
        )
        np.testing.assert_array_equal(
            face.cheek_landmarks[cheek.CHEEK_LM_SUBMALAR_L], ob.submalar
        )
        from_alpha, from_true = self.band_offsets("yaw45_hollow3", ob, True)
        self.assertGreaterEqual(len(from_alpha), 0.95 * len(ob.outline))
        self.assertLessEqual(float(np.nanmax(np.abs(from_alpha))), ALPHA_CROSSING_PX)
        self.assertLessEqual(float(np.nanmax(np.abs(from_true))), 5.0)

    def test_hollow_depth_series(self):
        """
        The hollows pressed 1.5, 3 and 4.5 mm into the 45 degree render read
        present 0, 1 and 1. The 1.5 mm hollow is the documented false negative
        (docs/measures.md's caveat): it moves the silhouette
        by half a raw pixel and the pipeline reads no depth (-0.0002 H measured,
        the plain render's -0.0000), so no threshold at or above the
        render-derived lower bound would call it. The 4.5 mm hollow reads about
        twice the 3 mm one (0.0088 against 0.0043 H measured; both above the
        threshold).
        """
        names = ("yaw45", "yaw45_hollow1.5", "yaw45_hollow3", "yaw45_hollow4.5")
        results = {name: self.results(self.face(name)) for name in names}
        for name in names:
            with self.subTest(name):
                self.assert_quarter_face(self.face(name), results[name], far_left=True)
                self.assertEqual(results[name]["cheek.nose"], 0.0)
                self.assertEqual(results[name]["apex.ok"], 1.0)
                self.assertEqual(results[name]["sub.ok"], 1.0)
                self.assertTrue(math.isfinite(results[name]["sub.l.depth"]))
        depths = {name: results[name]["sub.l.depth"] for name in names}
        self.assertEqual(results["yaw45_hollow1.5"]["sub.l.present"], 0.0)
        self.assertLess(abs(depths["yaw45_hollow1.5"]), 0.001)
        self.assertLess(abs(depths["yaw45_hollow1.5"] - depths["yaw45"]), 0.001)
        self.assertEqual(results["yaw45_hollow3"]["sub.l.present"], 1.0)
        self.assertEqual(results["yaw45_hollow4.5"]["sub.l.present"], 1.0)
        self.assertGreaterEqual(
            depths["yaw45_hollow3"], cheek_outline.SUBMALAR_MIN_DEPTH
        )
        self.assertGreater(depths["yaw45_hollow4.5"], 1.5 * depths["yaw45_hollow3"])

    def test_50_degree_render_has_the_nose_on_the_outline(self):
        face = self.face("yaw50")
        results = self.results(face)
        ob, s = self.assert_quarter_face(face, results, far_left=True)
        assert face.cheek_outline_src is not None
        self.assertTrue(
            np.any(face.cheek_outline_src == cheek_outline.OUTLINE_SRC_NOSE)
        )
        self.assertEqual(results["cheek.nose"], 1.0)
        self.assertTrue(ob.nose_on)
        self.assertLessEqual(results["cheek.nose.gap"], 0.01)
        self.assertEqual(results["apex.ok"], 0.0)
        self.assertEqual(results["sub.ok"], 0.0)
        self.assertLess(results["apex.l.h"], 0.9)
        self.assertGreater(results["apex.l.h"], 0.0)
        # The turn reads its true 50 degrees, the very edge of the yaw gate
        # (45 +/- 5), so whether that gate passes is a matter of rounding
        # (50.06 measured); the nose is what fails this render, and it would
        # at any turn.
        self.assertAlmostEqual(abs(results["cheek.yaw"]), 50.0, delta=0.5)
        self.assertAlmostEqual(abs(results["cheek.yaw.model"]), 50.0, delta=1.0)
        self.assertLess(results["cheek.mask"], 1.0)
        from_alpha, from_true = self.band_offsets("yaw50", ob, True)
        self.assertGreater(len(from_alpha), 0)
        self.assertLessEqual(float(np.nanmax(np.abs(from_alpha))), ALPHA_CROSSING_PX)
        self.assertLessEqual(float(np.nanmax(np.abs(from_true))), 5.0)

    def test_position_on_a_quarter_face(self):
        for name in QUARTER_RENDERS:
            with self.subTest(name):
                face = self.face(name)
                self.assertTrue(face.quarter)
                results = self.results(face)
                self.assertTrue(math.isnan(results["px2mm"]))
                self.assertTrue(math.isfinite(results["pd"]))
                self.assertGreater(results["pd"], 0.0)
                self.assertEqual(results["pd"], face.pupillary_distance)
                self.assertTrue(math.isfinite(results["tilt"]))
                pd, pix2mm = facial.util_calc_pd(face.get_pupils(), face.pd)
                self.assertEqual((face.pupillary_distance, face.pix2mm), (pd, pix2mm))

    def test_each_render_survives_saving(self):
        for name in RENDERS:
            with self.subTest(name):
                face = self.face(name)
                expected = self.results(face)
                state = face.dump_state()
                restored = facial.AnalyzeFace(measures=measures.all_measures())
                restored.load_state(state)
                self.assertEqual(restored.pose, face.pose)
                self.assertEqual(restored.quarter, face.quarter)
                self.assertEqual(restored.far_left, face.far_left)
                self.assertEqual(restored.landmarks, face.landmarks)
                self.assertEqual(restored.pix2mm, face.pix2mm)
                actual = self.results(restored)
                self.assertEqual(set(actual), set(expected))
                for key, value in expected.items():
                    self.assertTrue(_same(value, actual[key]), f"{key}: {value}")
                np.testing.assert_array_equal(
                    restored.cheek_landmarks, face.cheek_landmarks
                )
                if not face.quarter:
                    self.assertEqual(state[6], FRONTAL_VIEW)
                    self.assertIsNone(restored.cheek_outline_x)
                    continue
                for key in ("cheek_outline_x", "cheek_outline_y", "cheek_outline_src"):
                    original: Optional[np.ndarray] = getattr(face, key)
                    copy: Optional[np.ndarray] = getattr(restored, key)
                    assert original is not None and copy is not None
                    np.testing.assert_array_equal(copy, original, key)
                    self.assertEqual(copy.dtype, original.dtype, key)
                    self.assertFalse(np.shares_memory(copy, original), key)
                    self.assertFalse(np.shares_memory(state[6][key], original), key)

    def test_render_changes_no_value(self):
        for name, force_pose in (
            ("img4", Pose.DETECT),
            ("sketch", Pose.QUARTER),
            ("yaw45", Pose.DETECT),
            ("yaw45_hollow3", Pose.DETECT),
            ("yaw50", Pose.DETECT),
        ):
            with self.subTest(name):
                face = self.face(name, force_pose)
                expected = self.results(face)
                drawn = face.analyze(render=True)
                assert drawn is not None
                self.assertEqual(set(drawn), set(expected))
                for key, value in expected.items():
                    self.assertTrue(_same(value, drawn[key]), key)


if __name__ == "__main__":
    unittest.main()
