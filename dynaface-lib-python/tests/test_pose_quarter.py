"""
The quarter path of AnalyzeFace without the models: a three-quarter face's
detection under config.AUTO_QUARTER, the forced poses, the centered band crop,
the far-cheek trace, the saved view and its restore, a reused face, the pose
label and a smoothed video through VideoAnalyzer.

The synthetic-head renders stand in for photos: their raw SPIGA landmarks and
head poses in tests_data/cheek_landmarks.json are the mocked detections, and
the foreground mask is every pixel differing from the background color, as
tests_data/cheek/cheek_renders.json defines it. _find_landmarks and
dynaface.lateral.foreground_mask are mocked the way test_persist.TestSavedView
mocks the detector and analyze_lateral; load_image() finds the landmarks once
per image (a crop remaps them, it never detects again), so each mocked
detection answers in the frame's own pixels.
"""

import inspect
import json
import math
import os
import sys
import unittest
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import cheek, cheek_outline, config, facial, measures  # noqa: E402
from dynaface.const import DEFAULT_TILT_THRESHOLD, STD_PUPIL_DIST, Pose  # noqa: E402
from dynaface.measures_cheek import AnalyzeCheekMeasures  # noqa: E402
from dynaface.measures_frontal import AnalyzeLandmarks  # noqa: E402
from dynaface.video import FrameSeries, LandmarkSmoother, VideoAnalyzer  # noqa: E402

TESTS_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests_data"
)
LANDMARKS_JSON = os.path.join(TESTS_DATA, "cheek_landmarks.json")
RENDERS_JSON = os.path.join(TESTS_DATA, "cheek", "cheek_renders.json")

STANDARD = 1024
FRONTAL_VIEW = {"pose": "frontal", "flipped": False, "lateral": False}
QUARTER_VIEW_KEYS = {
    "pose",
    "flipped",
    "lateral",
    "far_left",
    "cheek_outline_x",
    "cheek_outline_y",
    "cheek_outline_src",
}
# Each cheek landmark's row that exists in the view: the frontal constructions,
# and the far cheek's observed points of a turn to each side.
FRONTAL_ROWS = (0, 1, 2, 3, 4, 5)
OBLIQUE_ROWS = (6, 7, 8, 9)

Found = Tuple[List[Tuple[int, int]], np.ndarray]


@lru_cache(maxsize=None)
def fixture() -> Dict[str, Any]:
    with open(LANDMARKS_JSON, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data["images"]


@lru_cache(maxsize=None)
def background() -> np.ndarray:
    with open(RENDERS_JSON, encoding="utf-8") as f:
        return np.array(json.load(f)["background"], dtype=np.uint8)


def image_of(name: str) -> np.ndarray:
    """The fixture's image as an RGB array, as load_image() takes it."""
    path = os.path.join(TESTS_DATA, fixture()[name]["file"])
    return np.asarray(Image.open(path).convert("RGB"))


def detection(name: str, dx: int = 0, dy: int = 0) -> Found:
    """What the models found in the image: its raw landmarks, shifted by (dx, dy),
    and its head pose, as _find_landmarks() returns them."""
    entry = fixture()[name]
    landmarks = [(int(x) + dx, int(y) + dy) for x, y in entry["raw"]["landmarks"]]
    raw = entry["raw"]
    pose = np.array([raw["yaw"], raw["pitch"], raw["roll"], 0.0, 0.0, 0.0])
    return landmarks, pose


def wide_frame(
    name: str, width: int = 1600, height: int = 1024
) -> Tuple[np.ndarray, Found]:
    """
    The render in the middle of a wider, background-colored frame, like a video
    frame, with its detection shifted to match. The frame is as tall as the
    standard image; the band crop scales it by the landmarks' band (about 1.8 for
    these renders, whatever the frame) and pads nothing, and the crop's horizontal
    placement is what the landmarks decide.
    """
    image = image_of(name)
    h, w = image.shape[:2]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = background()
    dx, dy = (width - w) // 2, (height - h) // 2
    frame[dy : dy + h, dx : dx + w] = image
    return frame, detection(name, dx, dy)


def wflw_mirror_index() -> List[int]:
    """The landmark each WFLW landmark becomes when the image is mirrored: the
    outline reverses, each side's brow, eye, nose base and lip points swap with the
    other side's, and the midline points stay."""
    index = list(range(98))

    def swap(a: int, b: int) -> None:
        index[a], index[b] = b, a

    for i in range(16):
        swap(i, 32 - i)  # the outline, 0-32, around the chin (16)
    for k in range(5):
        swap(33 + k, 46 - k)  # the brows' tops, 33-37 and 42-46
    for k in range(4):
        swap(38 + k, 50 - k)  # the brows' bottoms, 38-41 and 47-50
    swap(55, 59)  # the base of the nose, 55-59 around 57
    swap(56, 58)
    for k in range(5):
        swap(60 + k, 72 - k)  # the eyes' corners and tops, 60-64 and 68-72
    for k in range(3):
        swap(65 + k, 75 - k)  # the eyes' bottoms, 65-67 and 73-75
    for k in range(3):
        swap(76 + k, 82 - k)  # the outer lips' tops, 76-78 and 80-82
    for k in range(2):
        swap(83 + k, 87 - k)  # the outer lips' bottoms, 83-84 and 86-87
    for k in range(2):
        swap(88 + k, 92 - k)  # the inner lips' tops, 88-89 and 91-92
    swap(93, 95)  # the inner lips' bottoms, around 94
    swap(96, 97)  # the pupils
    return index


MIRROR = wflw_mirror_index()


def mirrored(frame: np.ndarray, found: Found) -> Tuple[np.ndarray, Found]:
    """The frame and its detection mirrored left to right: the photo of the other
    turn, with the landmarks renumbered for the swapped sides and the yaw and roll
    negated."""
    landmarks, pose = found
    width = frame.shape[1]
    flipped: List[Tuple[int, int]] = [(0, 0)] * 98
    for i, (x, y) in enumerate(landmarks):
        flipped[MIRROR[i]] = (width - 1 - x, y)
    pose = np.array(pose, dtype=np.float64)
    pose[0], pose[2] = -pose[0], -pose[2]
    return np.ascontiguousarray(frame[:, ::-1]), (flipped, pose)


def mask_from_background(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """What lateral.foreground_mask() would return for a render: every pixel that
    differs from the background color is foreground, with a full alpha."""
    pixels = np.asarray(image.convert("RGB"))
    foreground = np.any(pixels != background(), axis=2)
    alpha = np.where(foreground, 255, 0).astype(np.uint8)
    return foreground, alpha


def load(
    frame: np.ndarray,
    found: Found,
    force_pose: Pose = Pose.DETECT,
    face: Optional[facial.AnalyzeFace] = None,
    previous_pose: Optional[Pose] = None,
    auto_quarter: bool = True,
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD,
    crop: bool = True,
) -> Tuple[facial.AnalyzeFace, int]:
    """
    Load the frame with the detection in place of the models and the mask from
    the background color in place of the background model, with quarter
    detection switched on (and restored afterward) unless told otherwise; crop
    is load_image()'s, which a quarter face ignores.

    Returns:
        The face, and how many times the far cheek was traced.
    """
    if face is None:
        face = facial.AnalyzeFace(
            measures=measures.all_measures(), tilt_threshold=tilt_threshold
        )
    with mock.patch.object(
        facial.AnalyzeFace, "_find_landmarks", return_value=found
    ), mock.patch(
        "dynaface.lateral.foreground_mask", side_effect=mask_from_background
    ) as traced, mock.patch.object(
        config, "AUTO_QUARTER", auto_quarter
    ):
        face.load_image(
            frame, crop=crop, force_pose=force_pose, previous_pose=previous_pose
        )
    return face, traced.call_count


def restore(
    state: List[Any], face: Optional[facial.AnalyzeFace] = None
) -> facial.AnalyzeFace:
    face = face or facial.AnalyzeFace(measures=measures.all_measures())
    face.load_state(state)
    return face


def same(a: Any, b: Any) -> bool:
    """Equal values, where NaN equals NaN."""
    return bool(a == b) or (a != a and b != b)


def assert_same_results(
    test: unittest.TestCase, expected: Dict[str, Any], actual: Dict[str, Any]
) -> None:
    test.assertEqual(set(expected), set(actual))
    for key, value in expected.items():
        test.assertTrue(same(value, actual[key]), f"{key}: {value} != {actual[key]}")


def assert_rows(
    test: unittest.TestCase, points: np.ndarray, finite: Tuple[int, ...]
) -> None:
    """The cheek landmarks' rows that exist are exactly these, the rest NaN."""
    test.assertEqual(points.shape, (10, 2))
    test.assertEqual(points.dtype, np.float64)
    present = tuple(i for i in range(10) if np.isfinite(points[i]).all())
    absent = tuple(i for i in range(10) if np.isnan(points[i]).all())
    test.assertEqual(present, finite)
    test.assertEqual(len(present) + len(absent), 10)


def located(face: facial.AnalyzeFace) -> np.ndarray:
    """cheek.locate_all() on the face's own landmarks and view, which
    face.cheek_landmarks must equal exactly."""
    return cheek.locate_all(
        face.landmarks,
        face.pose,
        face.far_left,
        face.cheek_outline_x,
        face.cheek_outline_y,
        face.cheek_outline_src,
    )


class TestQuarterDetection(unittest.TestCase):
    """Detection, the forced poses, the crop, the scale and the trace."""

    def test_public_names(self):
        face = facial.AnalyzeFace(measures=[])
        self.assertFalse(face.quarter)
        self.assertFalse(face.far_left)
        self.assertIsNone(face.cheek_outline_x)
        self.assertIsNone(face.cheek_outline_y)
        self.assertIsNone(face.cheek_outline_src)
        assert_rows(self, face.cheek_landmarks, ())
        self.assertIsNone(face.frontal_cheek())
        self.assertIsNone(face.oblique_cheek())
        assert_rows(self, face.find_cheek_landmarks(), ())
        center = inspect.signature(facial.AnalyzeFace.crop_lateral).parameters["center"]
        self.assertIs(center.default, False)
        previous = inspect.signature(facial.AnalyzeFace.load_image).parameters[
            "previous_pose"
        ]
        self.assertIsNone(previous.default)

    def test_45_degree_render_is_a_quarter_face(self):
        frame, found = wide_frame("yaw45")
        face, traced = load(frame, found)
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertTrue(face.quarter)
        # The nose points to the image's right, so the far cheek is the subject's
        # left: the photo was taken from the subject's right.
        self.assertTrue(face.far_left)
        self.assertFalse(face.flipped)
        self.assertFalse(face.lateral)
        self.assertEqual(traced, 1)
        self.assertEqual((face.width, face.height), (STANDARD, STANDARD))

        # The centered band crop: every landmark inside, the chin at least 10%
        # above the bottom, the landmarks' middle (as pixel centers) within two
        # pixels of the crop's middle (measured 1.0: the landmarks are truncated
        # to integers after the scaling by about 1.8), and the band scaled to
        # fill the crop with its pads whatever the frame: here the 800 px
        # render's 1024-tall frame is scaled up, not taken as it is.
        xs = [x for x, _ in face.landmarks]
        ys = [y for _, y in face.landmarks]
        self.assertTrue(0 <= min(xs) and max(xs) < STANDARD)
        self.assertTrue(0 <= min(ys) and max(ys) < STANDARD)
        self.assertLessEqual(face.landmarks[cheek.LM_CHIN][1], 0.9 * STANDARD)
        self.assertLessEqual(abs((min(xs) + max(xs) + 1) / 2 - STANDARD / 2), 2.0)
        pad_top = round(STANDARD * facial.LATERAL_PAD_TOP_RATIO)
        pad_bottom = round(STANDARD * facial.LATERAL_PAD_BOTTOM_RATIO)
        self.assertLessEqual(abs(min(ys) - pad_top), 2)
        self.assertLessEqual(abs(max(ys) - (STANDARD - 1 - pad_bottom)), 2)
        raw_ys = [y for _, y in found[0]]
        self.assertGreater((max(ys) - min(ys)) / (max(raw_ys) - min(raw_ys)), 1.5)

        # The scale: the pupil scale measured on the cropped landmarks, as for a
        # frontal face, so analyze() measures it again the same way.
        pd, pix2mm = facial.util_calc_pd(face.get_pupils(), face.pd)
        self.assertEqual((face.pupillary_distance, face.pix2mm), (pd, pix2mm))
        self.assertIsNone(face.face_rotation)

        # The far cheek's outline: one row from the far outer eye corner's row
        # (or the far contour's top) to the chin row, every x inside the crop,
        # nearly every row from the mask on a plain render.
        rows = face.cheek_outline_y
        top = min(face.landmarks[72][1], face.landmarks[32][1])
        np.testing.assert_array_equal(
            rows, np.arange(top, face.landmarks[cheek.LM_CHIN][1] + 1)
        )
        self.assertEqual(len(face.cheek_outline_x), len(rows))
        self.assertEqual(len(face.cheek_outline_src), len(rows))
        self.assertTrue(np.all(face.cheek_outline_x >= 0))
        self.assertTrue(np.all(face.cheek_outline_x < STANDARD))
        mask_rows = face.cheek_outline_src == cheek_outline.OUTLINE_SRC_MASK
        self.assertGreater(mask_rows.mean(), 0.9)

        # The cheek landmarks: the far cheek's observed points, nothing frontal.
        assert_rows(
            self,
            face.cheek_landmarks,
            (cheek.CHEEK_LM_APEX_L, cheek.CHEEK_LM_SUBMALAR_L),
        )
        np.testing.assert_array_equal(face.cheek_landmarks, located(face))
        observed = face.oblique_cheek()
        self.assertIsNotNone(observed)
        self.assertTrue(observed.far_left)
        self.assertEqual(observed.mask_fraction, 1.0)
        np.testing.assert_array_equal(
            face.cheek_landmarks[cheek.CHEEK_LM_APEX_L], observed.apex
        )
        self.assertIsNone(face.frontal_cheek())

    def test_a_turn_the_profile_test_declines_is_a_quarter_face(self):
        """
        A 35 to 55 degree turn whose nose crossed the far contour goes to the
        profile test first, and when that declines it too, the face is a quarter
        face (its cheek.nose flags it), never a frontal one, whose millimeter
        measures would run on a turned head.
        """
        frame, found = wide_frame("yaw45")
        declined = (False, False)
        with mock.patch.object(cheek, "nose_margin", return_value=-0.1):
            with mock.patch.object(
                facial.AnalyzeFace, "_is_lateral", return_value=declined
            ) as profile_test:
                face, traced = load(frame, found)
        profile_test.assert_called_once()
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertTrue(face.quarter)
        self.assertTrue(face.far_left)
        self.assertEqual(traced, 1)

    def test_mirrored_turn_gets_the_mirrored_crop(self):
        frame, found = wide_frame("yaw45")
        plus, _ = load(frame, found)
        minus, traced = load(*mirrored(frame, found))
        self.assertEqual(minus.pose, Pose.QUARTER)
        self.assertFalse(minus.far_left)
        self.assertFalse(minus.flipped)
        self.assertEqual(traced, 1)
        for i in range(98):
            x, y = plus.landmarks[i]
            x2, y2 = minus.landmarks[MIRROR[i]]
            self.assertLessEqual(abs(x2 - (STANDARD - 1 - x)), 2, i)
            self.assertLessEqual(abs(y2 - y), 2, i)
        # So is the outline: the same rows, the edge mirrored. The band crop
        # scales the frame by the landmarks' band, and the mirrored frame's
        # integer landmarks place its crop a pixel off the exact mirror, so a
        # few rows at the chin, past the measured band, can change their source.
        np.testing.assert_array_equal(minus.cheek_outline_y, plus.cheek_outline_y)
        same_source = minus.cheek_outline_src == plus.cheek_outline_src
        self.assertGreaterEqual(same_source.mean(), 0.98)
        np.testing.assert_array_equal(
            minus.oblique_cheek().outline_src, plus.oblique_cheek().outline_src
        )
        both = (plus.cheek_outline_src == cheek_outline.OUTLINE_SRC_MASK) & (
            minus.cheek_outline_src == cheek_outline.OUTLINE_SRC_MASK
        )
        self.assertTrue(both.any())
        self.assertLessEqual(
            float(
                np.max(
                    np.abs(
                        minus.cheek_outline_x[both]
                        - (STANDARD - 1 - plus.cheek_outline_x[both])
                    )
                )
            ),
            2.0,
        )
        assert_rows(
            self,
            minus.cheek_landmarks,
            (cheek.CHEEK_LM_APEX_R, cheek.CHEEK_LM_SUBMALAR_R),
        )
        # The apex height reads alike: measured 1.2e-4 apart, from the integer
        # landmarks mirroring up to 2 px off, which tilts the levels a hair.
        self.assertAlmostEqual(
            minus.oblique_cheek().apex_h, plus.oblique_cheek().apex_h, delta=5e-4
        )
        self.assertAlmostEqual(
            minus.oblique_cheek().submalar_depth,
            plus.oblique_cheek().submalar_depth,
            delta=1e-4,
        )
        # And the crops themselves are mirror images, to the rounding of the
        # scaled frame's width.
        mirror = np.abs(
            minus.original_img[:, ::-1].astype(int) - plus.original_img.astype(int)
        )
        self.assertLess(float(mirror.mean()), 1.0)

    def test_portrait_frames_crop_alike(self):
        # The band crop scales the landmarks' band to fill the crop whatever
        # the frame, so one face reads alike from a phone frame and a 4K one:
        # two portrait frames of the yaw45 render against the wide frame, the
        # 800 px render in an 800 by 1600 frame, and the render cut to 400 px
        # around the face in a 400 by 1600 frame, whose scaled width (956 px at
        # the band's 2.4 scale) is below the crop, so the crop starts left of
        # the frame and safe_clip pads both sides with FILL_COLOR (31 and 36
        # columns, measured). Measured against the wide frame: the landmarks
        # within 1 px, the pupil distance equal, cheek.scale 662.59 and 662.49
        # against 662.66, apex_h within 6e-5, the same 709 rows, and the mask
        # rows' x within 1 px.
        frame, found = wide_frame("yaw45")
        wide, _ = load(frame, found)
        wide_ob = wide.oblique_cheek()
        image = image_of("yaw45")
        landmarks, pose = detection("yaw45")
        xs = [x for x, _ in landmarks]
        cut = (min(xs) + max(xs)) // 2 - 200
        narrow = np.empty((1600, 400, 3), dtype=np.uint8)
        narrow[:] = background()
        narrow[400:1200, :] = image[:, cut : cut + 400]
        narrow_found = ([(x - cut, y + 400) for x, y in landmarks], pose)
        cases = {
            "800 x 1600": wide_frame("yaw45", width=800, height=1600),
            "400 x 1600": (narrow, narrow_found),
        }
        for label, (portrait, portrait_found) in cases.items():
            with self.subTest(label):
                face, traced = load(portrait, portrait_found)
                self.assertEqual(face.pose, Pose.QUARTER)
                self.assertTrue(face.far_left)
                self.assertEqual(traced, 1)
                self.assertEqual((face.width, face.height), (STANDARD, STANDARD))
                for i in range(98):
                    x, y = face.landmarks[i]
                    x2, y2 = wide.landmarks[i]
                    self.assertLessEqual(abs(x - x2), 2, i)
                    self.assertLessEqual(abs(y - y2), 2, i)
                self.assertLessEqual(
                    abs(face.pupillary_distance - wide.pupillary_distance), 2.0
                )
                ob = face.oblique_cheek()
                self.assertLessEqual(abs(ob.scale - wide_ob.scale), 2.0)
                self.assertAlmostEqual(ob.apex_h, wide_ob.apex_h, delta=1e-3)
                np.testing.assert_array_equal(
                    face.cheek_outline_y, wide.cheek_outline_y
                )
                both = (face.cheek_outline_src == cheek_outline.OUTLINE_SRC_MASK) & (
                    wide.cheek_outline_src == cheek_outline.OUTLINE_SRC_MASK
                )
                self.assertGreater(both.mean(), 0.9)
                self.assertLessEqual(
                    float(
                        np.max(
                            np.abs(
                                face.cheek_outline_x[both] - wide.cheek_outline_x[both]
                            )
                        )
                    ),
                    2.0,
                )
        # The narrow frame's crop is padded at both sides with the fill color,
        # not the frame's background, and the face sits between the pads.
        face, _ = load(narrow, narrow_found)
        fill = np.array(facial.FILL_COLOR, dtype=np.uint8)
        self.assertTrue(np.all(face.original_img[:, :20] == fill))
        self.assertTrue(np.all(face.original_img[:, -20:] == fill))
        self.assertFalse(np.all(face.original_img[:, 100:-100] == fill))

    def test_quarter_faces_are_always_cropped(self):
        # crop=False leaves a frontal face at the frame's own size, and does
        # nothing to a quarter face, which is always cropped with the centered
        # band crop, like a profile: under detection and forced alike.
        frame, found = wide_frame("yaw45")
        cropped, _ = load(frame, found)
        for force_pose in (Pose.DETECT, Pose.QUARTER):
            with self.subTest(force_pose.value):
                face, traced = load(frame, found, force_pose=force_pose, crop=False)
                self.assertEqual(face.pose, Pose.QUARTER)
                self.assertTrue(face.quarter)
                self.assertEqual((face.width, face.height), (STANDARD, STANDARD))
                self.assertEqual(traced, 1)
                self.assertEqual(face.landmarks, cropped.landmarks)
                np.testing.assert_array_equal(
                    face.cheek_outline_x, cropped.cheek_outline_x
                )
                np.testing.assert_array_equal(
                    face.cheek_outline_src, cropped.cheek_outline_src
                )
        frame, found = wide_frame("yaw0")
        face, traced = load(frame, found, crop=False)
        self.assertEqual(face.pose, Pose.FRONTAL)
        self.assertEqual((face.width, face.height), (frame.shape[1], frame.shape[0]))
        self.assertEqual(traced, 0)

    def test_the_other_turns_render(self):
        # The -45 degree render, with its own landmarks (SPIGA reads the two
        # turns a little differently), is the mirrored view.
        face, traced = load(*wide_frame("yaw-45"))
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertTrue(face.quarter)
        self.assertFalse(face.far_left)
        self.assertFalse(face.flipped)
        self.assertEqual(traced, 1)
        xs = [x for x, _ in face.landmarks]
        ys = [y for _, y in face.landmarks]
        self.assertTrue(0 <= min(xs) and max(xs) < STANDARD)
        self.assertTrue(0 <= min(ys) and max(ys) < STANDARD)
        self.assertLessEqual(face.landmarks[cheek.LM_CHIN][1], 0.9 * STANDARD)
        self.assertLessEqual(abs((min(xs) + max(xs) + 1) / 2 - STANDARD / 2), 2.0)
        assert_rows(
            self,
            face.cheek_landmarks,
            (cheek.CHEEK_LM_APEX_R, cheek.CHEEK_LM_SUBMALAR_R),
        )
        np.testing.assert_array_equal(face.cheek_landmarks, located(face))
        self.assertFalse(face.oblique_cheek().far_left)

    def test_forced_quarter_takes_the_quarter_path_on_a_frontal_face(self):
        face, traced = load(image_of("img4"), detection("img4"), Pose.QUARTER)
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertTrue(face.quarter)
        self.assertFalse(face.lateral or face.flipped)
        # The far side comes from the landmarks: img4's nose sits a hair to the
        # subject's right of the canthi, so the far cheek is the right.
        self.assertFalse(face.far_left)
        self.assertEqual(traced, 1)
        self.assertIsNotNone(face.cheek_outline_x)
        self.assertGreater(len(face.cheek_outline_y), 0)
        # A photo's background isn't the renders', so the mask is all foreground
        # and no row comes from it; the contour and the intensity edge stand in.
        self.assertFalse(
            np.any(face.cheek_outline_src == cheek_outline.OUTLINE_SRC_MASK)
        )
        assert_rows(self, face.cheek_landmarks, (cheek.CHEEK_LM_APEX_R,))
        np.testing.assert_array_equal(face.cheek_landmarks, located(face))
        self.assertIsNotNone(face.oblique_cheek())
        self.assertIsNone(face.frontal_cheek())
        pd, pix2mm = facial.util_calc_pd(face.get_pupils(), face.pd)
        self.assertEqual(face.pix2mm, pix2mm)

    def test_forced_frontal_never_takes_the_quarter_path(self):
        face, traced = load(*wide_frame("yaw45"), force_pose=Pose.FRONTAL)
        self.assertEqual(face.pose, Pose.FRONTAL)
        self.assertFalse(face.quarter)
        self.assertFalse(face.far_left)
        self.assertEqual(traced, 0)
        self.assertIsNone(face.cheek_outline_x)
        self.assertIsNone(face.cheek_outline_y)
        self.assertIsNone(face.cheek_outline_src)
        assert_rows(self, face.cheek_landmarks, FRONTAL_ROWS)
        np.testing.assert_array_equal(face.cheek_landmarks, located(face))
        self.assertIsNotNone(face.frontal_cheek())
        self.assertIsNone(face.oblique_cheek())
        self.assertEqual(face.dump_state()[6], FRONTAL_VIEW)

    def test_detection_needs_both_switches(self):
        # Shipped off: a 45 degree turn is a frontal face, as before.
        self.assertFalse(config.AUTO_QUARTER)
        self.assertTrue(config.AUTO_LATERAL)
        frame, found = wide_frame("yaw45")
        face, traced = load(frame, found, auto_quarter=False)
        self.assertEqual((face.pose, face.quarter, traced), (Pose.FRONTAL, False, 0))
        self.assertEqual(face.dump_state()[6], FRONTAL_VIEW)
        assert_rows(self, face.cheek_landmarks, FRONTAL_ROWS)
        # AUTO_LATERAL off turns all detection off, quarter detection included.
        with mock.patch.object(config, "AUTO_LATERAL", False):
            face, traced = load(frame, found, auto_quarter=True)
        self.assertEqual((face.pose, face.quarter, traced), (Pose.FRONTAL, False, 0))
        # Both on: detected.
        face, traced = load(frame, found, auto_quarter=True)
        self.assertEqual((face.pose, face.quarter, traced), (Pose.QUARTER, True, 1))
        # A level render is frontal either way.
        face, traced = load(*wide_frame("yaw0"), auto_quarter=True)
        self.assertEqual((face.pose, face.quarter, traced), (Pose.FRONTAL, False, 0))

    def test_previous_pose_reaches_the_classification(self):
        frame, found = wide_frame("yaw45")
        for previous, held in (
            (None, False),
            (Pose.FRONTAL, False),
            (Pose.QUARTER, True),
        ):
            with self.subTest(previous=previous), mock.patch.object(
                cheek, "classify_quarter", wraps=cheek.classify_quarter
            ) as classify:
                face, _ = load(frame, found, previous_pose=previous)
            classify.assert_called_once()
            self.assertEqual(classify.call_args.kwargs["previous_quarter"], held)
            self.assertEqual(classify.call_args.args[0], found[0])  # raw landmarks
            self.assertEqual(classify.call_args.args[1], found[1][0])  # raw yaw
            self.assertTrue(face.quarter)

    def test_trace_runs_once_per_quarter_load_and_never_otherwise(self):
        frame, found = wide_frame("yaw45")
        face, traced = load(frame, found)
        self.assertEqual(traced, 1)
        self.assertEqual(load(frame, found, Pose.QUARTER)[1], 1)
        self.assertEqual(load(frame, found, Pose.FRONTAL)[1], 0)
        self.assertEqual(load(*wide_frame("yaw0"))[1], 0)

        # A profile traces its sagittal outline, not a cheek.
        chart = np.full((1024, 600, 3), 255, dtype=np.uint8)
        lateral = (
            chart,
            np.array([[300, 200], [290, 300], [200, 450]]),
            np.arange(1024, dtype=np.int32),
            np.arange(1024, dtype=np.int32),
        )
        with mock.patch("dynaface.lateral.analyze_lateral", return_value=lateral):
            face, traced = load(frame, found, Pose.LATERAL)
        self.assertTrue(face.lateral)
        self.assertFalse(face.quarter)
        self.assertEqual(traced, 0)
        self.assertIsNone(face.cheek_outline_x)
        assert_rows(self, face.cheek_landmarks, ())
        self.assertIsNone(face.frontal_cheek())
        self.assertIsNone(face.oblique_cheek())

        # No face: nothing to crop or trace, whatever the pose.
        no_face: Found = ([], np.array([0.0, 0.0, 0.0]))
        face, traced = load(frame, no_face, Pose.QUARTER)
        self.assertTrue(face.is_no_face())
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertFalse(face.quarter)
        self.assertEqual(traced, 0)
        self.assertIsNone(face.cheek_outline_x)
        assert_rows(self, face.cheek_landmarks, ())
        self.assertIsNone(face.analyze(render=False))

    def test_cheek_landmarks_are_filled_from_one_result(self):
        """find_cheek_landmarks() observes the far cheek once (constructs the
        frontal points once) and fills the array from that result by
        cheek.landmark_array(), the rule cheek.locate_all() fills by, without
        calling it; the array equals locate_all's."""
        frame, found = wide_frame("yaw45")
        with mock.patch.object(
            cheek_outline, "observed_points", wraps=cheek_outline.observed_points
        ) as observed, mock.patch.object(
            cheek, "locate_all", wraps=cheek.locate_all
        ) as locate:
            face, _ = load(frame, found)
        observed.assert_called_once()
        locate.assert_not_called()
        self.assertIsNotNone(face.oblique_cheek())
        np.testing.assert_array_equal(face.cheek_landmarks, located(face))
        np.testing.assert_array_equal(
            face.cheek_landmarks,
            cheek.landmark_array(None, face.oblique_cheek(), face.far_left),
        )
        # analyze() locates again: one more observation, the same array.
        before = face.cheek_landmarks.copy()
        with mock.patch.object(
            cheek_outline, "observed_points", wraps=cheek_outline.observed_points
        ) as observed:
            face.analyze(render=False)
        observed.assert_called_once()
        np.testing.assert_array_equal(face.cheek_landmarks, before)
        # A frontal face: the constructions once, and no observation at all.
        with mock.patch.object(
            cheek, "frontal_constructions", wraps=cheek.frontal_constructions
        ) as constructed, mock.patch.object(
            cheek_outline, "observed_points"
        ) as observed:
            face, _ = load(*wide_frame("yaw0"))
        constructed.assert_called_once()
        observed.assert_not_called()
        self.assertIsNotNone(face.frontal_cheek())
        np.testing.assert_array_equal(face.cheek_landmarks, located(face))
        np.testing.assert_array_equal(
            face.cheek_landmarks,
            cheek.landmark_array(face.frontal_cheek(), None, False),
        )

    def test_coincident_pupils_on_a_quarter_face(self):
        """No pupil frame, no cheek landmark, in the quarter view too: a quarter
        state whose pupils coincide restores with no observed points, all NaN,
        equal to locate_all's, and the oblique outputs NaN (the Cheek Measures', and
        the Landmarks measure's cheek coordinates)."""
        face, _ = load(*wide_frame("yaw45"))
        state = face.dump_state()
        landmarks = list(state[2])
        landmarks[97] = landmarks[96]
        state[2] = landmarks
        restored = restore(
            state,
            facial.AnalyzeFace(measures=[AnalyzeLandmarks(), AnalyzeCheekMeasures()]),
        )
        self.assertTrue(restored.quarter)
        self.assertTrue(restored.far_left)
        self.assertIsNotNone(restored.cheek_outline_x)
        self.assertIsNone(restored.oblique_cheek())
        self.assertTrue(np.isnan(restored.cheek_landmarks).all())
        np.testing.assert_array_equal(restored.cheek_landmarks, located(restored))
        results = restored.analyze(render=False)
        self.assertEqual(results["cheek.far"], 1.0)
        for name in ("apex.l.x", "apex.l.h", "sub.l.x", "cheek.scale", "cheek.yaw"):
            self.assertTrue(math.isnan(results[name]), name)
        self.assertFalse(math.isnan(results["cheek.yaw.model"]))


class TestQuarterState(unittest.TestCase):
    """The saved view of a quarter face and its restore, and a reused face."""

    def test_view_holds_far_left_and_three_numeric_arrays_only(self):
        face, _ = load(*wide_frame("yaw45"))
        view = face.dump_state()[6]
        self.assertEqual(set(view), QUARTER_VIEW_KEYS)
        self.assertEqual(view["pose"], "quarter")
        self.assertIs(view["flipped"], False)
        self.assertIs(view["lateral"], False)
        self.assertIs(view["far_left"], True)
        for key in ("cheek_outline_x", "cheek_outline_y", "cheek_outline_src"):
            self.assertIsInstance(view[key], np.ndarray, key)
            self.assertFalse(view[key].dtype.hasobject, key)
        self.assertEqual(view["cheek_outline_x"].dtype, np.float32)
        self.assertEqual(view["cheek_outline_y"].dtype, np.int32)
        self.assertEqual(view["cheek_outline_src"].dtype, np.uint8)
        for key, value in view.items():
            if not isinstance(value, np.ndarray):
                self.assertIn(type(value), (str, bool), key)

    def test_round_trip(self):
        face, _ = load(*wide_frame("yaw45"))
        expected = face.analyze(render=False)
        state = face.dump_state()
        restored = restore(state)
        self.assertEqual(restored.pose, Pose.QUARTER)
        self.assertTrue(restored.quarter)
        self.assertTrue(restored.far_left)
        self.assertFalse(restored.flipped or restored.lateral)
        self.assertEqual(restored.landmarks, face.landmarks)
        self.assertEqual(restored.pix2mm, face.pix2mm)
        self.assertEqual(restored.pd, face.pd)
        np.testing.assert_array_equal(restored.cheek_outline_x, face.cheek_outline_x)
        np.testing.assert_array_equal(restored.cheek_outline_y, face.cheek_outline_y)
        np.testing.assert_array_equal(
            restored.cheek_outline_src, face.cheek_outline_src
        )
        # Restored without running the models: the cheek landmarks are located
        # again from the restored landmarks and outline, and equal the original's.
        np.testing.assert_array_equal(restored.cheek_landmarks, face.cheek_landmarks)
        np.testing.assert_array_equal(restored.cheek_landmarks, located(restored))
        self.assertIsNotNone(restored.oblique_cheek())
        assert_same_results(self, expected, restored.analyze(render=False))

        # The arrays are copies at both ends: the state's and the restored face's
        # are the original's values, not its memory.
        restored.cheek_outline_x[0] += 1
        restored.cheek_outline_y[0] += 1
        restored.cheek_outline_src[0] = 7
        np.testing.assert_array_equal(state[6]["cheek_outline_x"], face.cheek_outline_x)
        np.testing.assert_array_equal(state[6]["cheek_outline_y"], face.cheek_outline_y)
        np.testing.assert_array_equal(
            state[6]["cheek_outline_src"], face.cheek_outline_src
        )
        state[6]["cheek_outline_x"][1] += 1
        self.assertNotEqual(state[6]["cheek_outline_x"][1], face.cheek_outline_x[1])

    def test_quarter_state_saved_before_the_outline_was_kept(self):
        # A development build's forced Pose.QUARTER took the frontal crop and
        # saved a view with the pose alone (pose "quarter", no far_left, no
        # cheek_outline_* keys): no far side, no outline. It restores as a
        # quarter face whose far side comes from its landmarks and whose outline
        # is the contour alone (the fallback), never an error.
        face, _ = load(*wide_frame("yaw45"), force_pose=Pose.FRONTAL)
        state = face.dump_state()
        state[6] = {"pose": "quarter", "flipped": False, "lateral": False}
        restored = restore(state)
        self.assertEqual(restored.pose, Pose.QUARTER)
        self.assertTrue(restored.quarter)
        self.assertTrue(restored.far_left)  # from the landmarks: the nose points right
        self.assertIsNone(restored.cheek_outline_x)
        self.assertIsNone(restored.cheek_outline_y)
        self.assertIsNone(restored.cheek_outline_src)
        observed = restored.oblique_cheek()
        self.assertIsNotNone(observed)
        self.assertEqual(observed.mask_fraction, 0.0)
        self.assertTrue(np.isfinite(observed.apex).all())
        self.assertTrue(math.isnan(observed.submalar_depth))
        assert_rows(self, restored.cheek_landmarks, (cheek.CHEEK_LM_APEX_L,))
        np.testing.assert_array_equal(restored.cheek_landmarks, located(restored))
        # Its pupil pair restores the pupil scale, the quarter rule anyway.
        pd, pix2mm = facial.util_calc_pd(restored.get_pupils(), restored.pd)
        self.assertEqual(restored.pix2mm, pix2mm)
        results = restored.analyze(render=False)
        self.assertEqual(results["cheek.mask"], 0.0)
        self.assertEqual(results["apex.ok"], 0.0)
        self.assertTrue(math.isnan(results["sub.l.depth"]))
        # Saved again, the view says which cheek but holds no outline.
        view = restored.dump_state()[6]
        self.assertEqual(
            view,
            {"pose": "quarter", "flipped": False, "lateral": False, "far_left": True},
        )
        again = restore(restored.dump_state())
        self.assertTrue(again.quarter and again.far_left)
        self.assertIsNone(again.cheek_outline_x)
        np.testing.assert_array_equal(again.cheek_landmarks, restored.cheek_landmarks)

    def test_no_face_forced_quarter_state_restores_as_saved(self):
        """An image with no face, forced to the quarter pose: the pose is
        quarter and quarter is False (as lateral is for a forced profile without
        a face), its view says so, and the state restores just so."""
        frame, found = wide_frame("yaw45")
        none: Found = ([], np.array([0.0, 0.0, 0.0]))
        face, traced = load(frame, none, Pose.QUARTER)
        self.assertTrue(face.is_no_face())
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertFalse(face.quarter)
        self.assertFalse(face.far_left)
        self.assertEqual(traced, 0)
        self.assertIsNone(face.cheek_outline_x)
        self.assertTrue(np.isnan(face.cheek_landmarks).all())
        self.assertIsNone(face.oblique_cheek())
        self.assertIsNone(face.analyze(render=False))
        state = face.dump_state()
        self.assertEqual(
            state[6], {"pose": "quarter", "flipped": False, "lateral": False}
        )
        restored = restore(state)
        self.assertEqual(restored.landmarks, [])
        self.assertEqual(restored.pose, Pose.QUARTER)
        self.assertFalse(restored.quarter)
        self.assertFalse(restored.far_left)
        self.assertIsNone(restored.cheek_outline_x)
        self.assertIsNone(restored.cheek_outline_y)
        self.assertIsNone(restored.cheek_outline_src)
        self.assertTrue(np.isnan(restored.cheek_landmarks).all())
        self.assertIsNone(restored.oblique_cheek())
        self.assertEqual(restored.dump_state()[6], state[6])
        # Restored onto a face holding a quarter face, it leaves no quarter data.
        holder, _ = load(frame, found)
        self.assertTrue(holder.quarter)
        restore(state, holder)
        self.assertFalse(holder.quarter)
        self.assertFalse(holder.far_left)
        self.assertIsNone(holder.cheek_outline_x)
        self.assertIsNone(holder.oblique_cheek())

    def test_old_state_and_frontal_view_restore_no_outline(self):
        quarter, _ = load(*wide_frame("yaw45"))
        # A state saved by Dynaface 2.0.4 and earlier ends with the face rotation
        # and holds no view: on a fresh face, nothing says quarter.
        fresh = restore(quarter.dump_state()[:6])
        self.assertEqual(fresh.pose, Pose.FRONTAL)
        self.assertFalse(fresh.quarter)
        self.assertFalse(fresh.far_left)
        self.assertIsNone(fresh.cheek_outline_x)
        assert_rows(self, fresh.cheek_landmarks, FRONTAL_ROWS)
        self.assertIsNone(fresh.oblique_cheek())

        # A frontal view restored onto a face that holds a quarter face leaves no
        # quarter data behind.
        frontal, _ = load(*wide_frame("yaw0"))
        holder = restore(frontal.dump_state(), quarter)
        self.assertIs(holder, quarter)
        self.assertEqual(holder.pose, Pose.FRONTAL)
        self.assertFalse(holder.quarter)
        self.assertFalse(holder.far_left)
        self.assertIsNone(holder.cheek_outline_x)
        self.assertIsNone(holder.cheek_outline_y)
        self.assertIsNone(holder.cheek_outline_src)
        assert_rows(self, holder.cheek_landmarks, FRONTAL_ROWS)
        np.testing.assert_array_equal(holder.cheek_landmarks, frontal.cheek_landmarks)
        self.assertIsNotNone(holder.frontal_cheek())
        self.assertIsNone(holder.oblique_cheek())
        self.assertEqual(holder.dump_state()[6], FRONTAL_VIEW)

    def test_a_reused_face_saves_each_images_own_view(self):
        # One face loads every frame of a video. A frontal frame after a quarter
        # frame saves no cheek keys and keeps no outline.
        face, _ = load(*wide_frame("yaw45"))
        self.assertTrue(face.quarter)
        load(*wide_frame("yaw0"), face=face)
        self.assertEqual(face.pose, Pose.FRONTAL)
        self.assertFalse(face.quarter)
        self.assertFalse(face.far_left)
        self.assertIsNone(face.cheek_outline_x)
        self.assertIsNone(face.cheek_outline_y)
        self.assertIsNone(face.cheek_outline_src)
        self.assertEqual(face.dump_state()[6], FRONTAL_VIEW)
        assert_rows(self, face.cheek_landmarks, FRONTAL_ROWS)
        self.assertIsNone(face.oblique_cheek())

        # A quarter frame after a straightened frontal frame saves no rotation:
        # only the frontal crop straightens, and it isn't this frame's.
        frame, (landmarks, pose) = wide_frame("yaw0")
        tilted = list(landmarks)
        x, y = tilted[97]
        tilted[97] = (x, y + 40)  # the left pupil 40 px lower: about 20 degrees
        face, _ = load(frame, (tilted, pose), Pose.FRONTAL, tilt_threshold=2)
        self.assertIsNotNone(face.face_rotation)
        self.assertNotEqual(face.face_rotation, 0)
        self.assertIsNotNone(face.dump_state()[5])
        load(*wide_frame("yaw45"), face=face)
        self.assertTrue(face.quarter)
        self.assertIsNone(face.face_rotation)
        self.assertIsNone(face.dump_state()[5])
        self.assertEqual(face.orig_pupils, ((0, 0), (0, 0)))

    def test_a_profile_after_a_tilted_frontal_frame_has_no_rotation(self):
        # Only the frontal crop straightens: a profile loaded next on the same
        # face has no rotation, and saves None, as a fresh face's profile does.
        frame, (landmarks, pose) = wide_frame("yaw0")
        tilted = list(landmarks)
        x, y = tilted[97]
        tilted[97] = (x, y + 40)
        face, _ = load(frame, (tilted, pose), Pose.FRONTAL, tilt_threshold=2)
        self.assertIsNotNone(face.face_rotation)
        self.assertNotEqual(face.face_rotation, 0)
        no_chart = (None, np.zeros((6, 2), dtype=int), None, None)
        fresh = facial.AnalyzeFace(measures=measures.all_measures())
        for profile in (face, fresh):
            with mock.patch.object(
                facial.AnalyzeFace, "_find_landmarks", return_value=(landmarks, pose)
            ), mock.patch("dynaface.lateral.analyze_lateral", return_value=no_chart):
                profile.load_image(frame, crop=True, force_pose=Pose.LATERAL)
            self.assertTrue(profile.lateral)
            self.assertEqual(profile.pose, Pose.LATERAL)
            self.assertFalse(profile.quarter)
            self.assertIsNone(profile.face_rotation)
            self.assertIsNone(profile.dump_state()[5])
            self.assertEqual(profile.orig_pupils, ((0, 0), (0, 0)))
            self.assertEqual(profile.pix2mm, 0.24)

    def test_a_frame_without_a_face_keeps_no_scale(self):
        # The last frame's pupillary distance and scale don't carry over to a
        # frame without a face: it holds a fresh face's values, and saves them.
        face, _ = load(*wide_frame("yaw45"))
        self.assertGreater(face.pupillary_distance, 0.0)
        self.assertNotEqual(face.pix2mm, 1.0)
        nothing = ([], np.array([0.0, 0.0, 0.0]))
        face, traced = load(wide_frame("yaw0")[0], nothing, face=face)
        self.assertTrue(face.is_no_face())
        self.assertEqual(traced, 0)
        fresh = facial.AnalyzeFace()
        self.assertEqual((fresh.pupillary_distance, fresh.pix2mm), (0.0, 1.0))
        self.assertEqual((face.pupillary_distance, face.pix2mm), (0.0, 1.0))
        self.assertEqual(face.dump_state()[3:5], [0.0, 1.0])
        self.assertEqual(face.pose, Pose.FRONTAL)
        self.assertFalse(face.quarter or face.far_left or face.lateral)
        self.assertIsNone(face.cheek_outline_x)
        self.assertTrue(np.isnan(face.cheek_landmarks).all())
        self.assertEqual(face.dump_state()[6], FRONTAL_VIEW)

    def test_a_bad_pupillary_distance_leaves_the_face_untouched(self):
        # The setting is checked before anything else, so a mistaken one fails
        # at once and the face still holds the previous image, its landmarks
        # and its view: a quarter face stays one, outline and all.
        face, _ = load(*wide_frame("yaw45"))
        before = face.dump_state()
        frame, found = wide_frame("yaw0")
        with mock.patch.object(facial.AnalyzeFace, "pd", 300.0):
            with self.assertRaises(ValueError):
                load(frame, found, face=face)
        self.assertEqual(face.pd, STD_PUPIL_DIST)
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertTrue(face.quarter)
        self.assertTrue(face.far_left)
        after = face.dump_state()
        np.testing.assert_array_equal(after[0], before[0])
        self.assertEqual(after[1:6], before[1:6])
        self.assertEqual(set(after[6]), QUARTER_VIEW_KEYS)
        for key in ("cheek_outline_x", "cheek_outline_y", "cheek_outline_src"):
            np.testing.assert_array_equal(after[6][key], before[6][key], key)
        self.assertIsNotNone(face.oblique_cheek())
        assert_rows(
            self,
            face.cheek_landmarks,
            (cheek.CHEEK_LM_APEX_L, cheek.CHEEK_LM_SUBMALAR_L),
        )

    def test_an_old_state_restored_onto_a_quarter_face_keeps_no_quarter_view(self):
        # A state saved by Dynaface 2.0.4 or earlier (six entries, no view)
        # restored onto a face that last loaded a quarter image: none of those
        # states was a quarter face, and the outline belongs to that image, so
        # the face measures the restored landmarks as a fresh face would.
        frontal, _ = load(*wide_frame("yaw0"))
        state = frontal.dump_state()[:6]
        fresh = restore(state)
        expected = fresh.analyze(render=True)
        holder, _ = load(*wide_frame("yaw45"))
        self.assertTrue(holder.quarter)
        restore(state, holder)
        self.assertEqual(holder.pose, Pose.FRONTAL)
        self.assertFalse(holder.quarter)
        self.assertFalse(holder.far_left)
        self.assertIsNone(holder.cheek_outline_x)
        self.assertIsNone(holder.cheek_outline_y)
        self.assertIsNone(holder.cheek_outline_src)
        self.assertIsNone(holder.oblique_cheek())
        self.assertIsNotNone(holder.frontal_cheek())
        assert_rows(self, holder.cheek_landmarks, FRONTAL_ROWS)
        np.testing.assert_array_equal(holder.cheek_landmarks, fresh.cheek_landmarks)
        assert_same_results(self, expected, holder.analyze(render=True))
        np.testing.assert_array_equal(holder.render_img, fresh.render_img)
        self.assertEqual(holder.dump_state()[6], FRONTAL_VIEW)

    def test_draw_static_names_the_cheek(self):
        for name, label in (
            ("yaw45", "Quarter (left cheek)"),
            ("yaw-45", "Quarter (right cheek)"),
        ):
            with self.subTest(name):
                face, _ = load(*wide_frame(name))
                with mock.patch.object(face, "write_text") as write:
                    face.draw_static()
                write.assert_called_once()
                self.assertEqual(write.call_args.args[1], label)
        face, _ = load(*wide_frame("yaw0"))
        with mock.patch.object(face, "write_text") as write:
            face.draw_static()
        self.assertEqual(write.call_args.args[1], "Frontal")


class TestQuarterVideo(unittest.TestCase):
    """A video that turns: frontal, quarter, quarter, frontal, smoothed over two
    frames, imported by VideoAnalyzer and measured again from its states."""

    def test_smoothed_video_over_both_views(self):
        names = ("yaw0", "yaw45", "yaw45_hollow3", "yaw0")
        frames = [wide_frame(name) for name in names]
        alone = [load(frame, found)[0] for frame, found in frames]

        previous: List[Optional[Pose]] = []
        original = facial.AnalyzeFace.load_image

        def spy(
            face: facial.AnalyzeFace, image: np.ndarray, *args: Any, **kwargs: Any
        ) -> bool:
            previous.append(kwargs.get("previous_pose"))
            return original(face, image, *args, **kwargs)

        analyzer = VideoAnalyzer(smoothing=2)
        with mock.patch.object(
            facial.AnalyzeFace, "_find_landmarks", side_effect=[f for _, f in frames]
        ), mock.patch(
            "dynaface.lateral.foreground_mask", side_effect=mask_from_background
        ) as traced, mock.patch.object(
            config, "AUTO_QUARTER", True
        ), mock.patch.object(
            facial.AnalyzeFace, "load_image", spy
        ):
            for frame, _ in frames:
                self.assertTrue(analyzer.add(frame))

        # Each frame is loaded with the previous kept frame's pose, and each
        # quarter frame traces its outline once.
        self.assertEqual(previous, [None, Pose.FRONTAL, Pose.QUARTER, Pose.QUARTER])
        self.assertEqual(traced.call_count, 2)
        states = analyzer.states
        self.assertEqual(analyzer.frames, [0, 1, 2, 3])
        self.assertEqual(
            [state[6]["pose"] for state in states],
            ["frontal", "quarter", "quarter", "frontal"],
        )

        # The smoothing starts afresh at both view changes: frames 1 and 3 keep
        # their own landmarks, and only frame 2 averages with the frame before.
        self.assertEqual(states[0][2], alone[0].landmarks)
        self.assertEqual(states[1][2], alone[1].landmarks)
        self.assertEqual(states[3][2], alone[3].landmarks)
        smoother = LandmarkSmoother(2)
        smoother.add(alone[1].landmarks)
        averaged = smoother.add(alone[2].landmarks)
        self.assertEqual(states[2][2], averaged)
        self.assertNotEqual(averaged, alone[2].landmarks)
        # The saved scale is the averaged landmarks' own pupil scale.
        pd, pix2mm = facial.util_calc_pd(facial.util_get_pupils(averaged))
        self.assertEqual(states[2][3:5], [pd, pix2mm])

        # Each quarter frame keeps an outline traced from its own mask on the
        # landmarks its state saves: frame 1's on its own detection, since the
        # smoothing started afresh there, and frame 2's traced again on the
        # averaged landmarks (retrace_far_cheek(), no second model pass), so
        # its rows are the averaged band's, not its detection's; the frontal
        # frames keep none.
        for index in (1, 2):
            view = states[index][6]
            self.assertEqual(set(view), QUARTER_VIEW_KEYS)
            self.assertIs(view["far_left"], True)
        outline_keys = ("cheek_outline_x", "cheek_outline_y", "cheek_outline_src")
        for key in outline_keys:
            np.testing.assert_array_equal(states[1][6][key], getattr(alone[1], key))
        foreground, alpha = mask_from_background(Image.fromarray(alone[2].original_img))
        retraced = cheek_outline.trace_far_cheek(
            foreground,
            alpha,
            averaged,
            True,
            gray=cheek_outline.gray_image(alone[2].original_img),
        )
        for key, expected in zip(outline_keys, retraced):
            np.testing.assert_array_equal(states[2][6][key], expected, key)
        self.assertEqual(
            (states[2][6]["cheek_outline_y"][0], states[2][6]["cheek_outline_y"][-1]),
            (min(averaged[72][1], averaged[32][1]), averaged[16][1]),
        )
        self.assertFalse(
            np.array_equal(states[2][6]["cheek_outline_y"], alone[2].cheek_outline_y)
        )
        for index in (0, 3):
            self.assertEqual(states[index][6], FRONTAL_VIEW)

        # Measured again from the states, each view's outputs are floats and the
        # other view's are NaN.
        series = FrameSeries.from_states(states, [AnalyzeCheekMeasures()])
        self.assertEqual(series.frames, [0, 1, 2, 3])
        width, apex_h = series.values["zy.w"], series.values["apex.l.h"]
        for row in range(4):
            self.assertIsInstance(width[row], float)
            self.assertIsInstance(apex_h[row], float)
        for row in (0, 3):
            self.assertTrue(math.isfinite(width[row]), row)
            self.assertTrue(math.isnan(apex_h[row]), row)
        for row in (1, 2):
            self.assertTrue(math.isnan(width[row]), row)
            self.assertTrue(math.isfinite(apex_h[row]), row)
        self.assertTrue(all(math.isnan(v) for v in series.values["apex.r.h"]))

        # The second quarter frame's apex comes from the averaged landmarks on
        # that frame's own outline.
        view = states[2][6]
        expected = cheek_outline.observed_points(
            states[2][2],
            view["cheek_outline_x"],
            view["cheek_outline_y"],
            view["cheek_outline_src"],
            True,
        )
        self.assertIsNotNone(expected)
        self.assertEqual(apex_h[2], expected.apex_h)
        restored = restore(states[2])
        self.assertEqual(restored.landmarks, averaged)
        self.assertEqual(restored.oblique_cheek().apex_h, expected.apex_h)
        np.testing.assert_array_equal(
            restored.cheek_landmarks[cheek.CHEEK_LM_APEX_L], expected.apex
        )

    def test_far_side_flip_resets_the_smoothing(self):
        """
        A quarter video whose far cheek flips (a cut between the two mirrored
        turns): the two band crops are mirrored, so the smoothing starts
        afresh at the flip, as it does when a profile's facing flips, and each
        turn's rows fill its own side's outputs.
        """
        frames = [wide_frame("yaw45"), wide_frame("yaw45_hollow3")]
        frames += [mirrored(*frame) for frame in frames]
        alone = [load(frame, found)[0] for frame, found in frames]
        self.assertEqual([face.far_left for face in alone], [True, True, False, False])

        analyzer = VideoAnalyzer(smoothing=2)
        with mock.patch.object(
            facial.AnalyzeFace, "_find_landmarks", side_effect=[f for _, f in frames]
        ), mock.patch(
            "dynaface.lateral.foreground_mask", side_effect=mask_from_background
        ) as traced, mock.patch.object(
            config, "AUTO_QUARTER", True
        ):
            for frame, _ in frames:
                self.assertTrue(analyzer.add(frame))
        self.assertEqual(traced.call_count, 4)
        states = analyzer.states
        self.assertEqual([state[6]["pose"] for state in states], ["quarter"] * 4)
        self.assertEqual(
            [state[6]["far_left"] for state in states], [True, True, False, False]
        )
        # Frames 0 and 2 start an average; 1 and 3 average with the frame before.
        smoother = LandmarkSmoother(2)
        self.assertEqual(states[0][2], alone[0].landmarks)
        smoother.add(alone[0].landmarks)
        self.assertEqual(states[1][2], smoother.add(alone[1].landmarks))
        self.assertEqual(states[2][2], alone[2].landmarks)
        smoother.reset()
        smoother.add(alone[2].landmarks)
        self.assertEqual(states[3][2], smoother.add(alone[3].landmarks))
        self.assertNotEqual(states[3][2], alone[3].landmarks)
        # Flip-free frames keep their own outline; the averaged ones are traced
        # on the averaged landmarks.
        for index in (0, 2):
            np.testing.assert_array_equal(
                states[index][6]["cheek_outline_x"], alone[index].cheek_outline_x
            )
        for index in (1, 3):
            expected = cheek_outline.trace_far_cheek(
                *mask_from_background(Image.fromarray(alone[index].original_img)),
                states[index][2],
                alone[index].far_left,
                gray=cheek_outline.gray_image(alone[index].original_img),
            )
            np.testing.assert_array_equal(
                states[index][6]["cheek_outline_x"], expected[0]
            )
            np.testing.assert_array_equal(
                states[index][6]["cheek_outline_y"], expected[1]
            )
        series = FrameSeries.from_states(states, [AnalyzeCheekMeasures()])
        left, right = series.values["apex.l.h"], series.values["apex.r.h"]
        for row in (0, 1):
            self.assertTrue(math.isfinite(left[row]) and math.isnan(right[row]), row)
        for row in (2, 3):
            self.assertTrue(math.isnan(left[row]) and math.isfinite(right[row]), row)
        for row in range(4):
            self.assertAlmostEqual(
                abs(series.values["cheek.yaw"][row]), 45.0, delta=1.0
            )


class TestRetrace(unittest.TestCase):
    """retrace_far_cheek(): the outline traced again from the kept mask."""

    def test_retrace_follows_the_landmarks_without_a_model_pass(self):
        face, _ = load(*wide_frame("yaw45"))
        self.assertTrue(face.quarter)
        assert face.cheek_outline_x is not None and face.cheek_outline_y is not None
        before_x, before_y = face.cheek_outline_x.copy(), face.cheek_outline_y.copy()
        before_src = face.cheek_outline_src.copy()
        face.landmarks = [(x, y + 3) for x, y in face.landmarks]
        with mock.patch(
            "dynaface.lateral.foreground_mask", side_effect=mask_from_background
        ) as traced:
            self.assertTrue(face.retrace_far_cheek())
        self.assertEqual(traced.call_count, 0)
        # The band follows the landmarks by the 3 rows.
        np.testing.assert_array_equal(face.cheek_outline_y, before_y + 3)
        # The mask is the frame's: a row both traces read from the mask has
        # the same edge.
        common = {int(y): i for i, y in enumerate(before_y)}
        for i, y in enumerate(face.cheek_outline_y):
            j = common.get(int(y))
            if j is None:
                continue
            if (
                face.cheek_outline_src[i] == cheek_outline.OUTLINE_SRC_MASK
                and before_src[j] == cheek_outline.OUTLINE_SRC_MASK
            ):
                self.assertEqual(float(face.cheek_outline_x[i]), float(before_x[j]))
        # The cheek landmarks follow once located again, as analyze() does.
        face.find_cheek_landmarks()
        np.testing.assert_array_equal(face.cheek_landmarks, located(face))
        self.assertTrue(np.isfinite(face.cheek_landmarks[cheek.CHEEK_LM_APEX_L]).all())
        # A state carries the outline, never the mask: a restored face keeps
        # the outline it was given, and says so.
        restored = restore(face.dump_state())
        saved = restored.cheek_outline_y.copy()
        restored.landmarks = [(x, y + 5) for x, y in restored.landmarks]
        self.assertFalse(restored.retrace_far_cheek())
        np.testing.assert_array_equal(restored.cheek_outline_y, saved)
        # A frontal face has nothing to trace, and a reused face drops the mask
        # with the rest of the quarter view.
        frontal, _ = load(*wide_frame("yaw0"), face=face)
        self.assertFalse(frontal.quarter)
        self.assertFalse(frontal.retrace_far_cheek())
        self.assertIsNone(frontal.cheek_outline_x)


if __name__ == "__main__":
    unittest.main()
