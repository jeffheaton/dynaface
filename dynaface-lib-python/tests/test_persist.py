import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

from dynaface import facial, measures, models
from dynaface.const import Pose
from dynaface.image import load_image
from dynaface.measures_cheek import AnalyzeCheekMeasures
from dynaface.measures_lateral import AnalyzeLateral

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


LATERAL_OUTPUTS = ("nfa", "nla", "at", "nt", "at_nt")
# The types a saved view may hold: bool, str, None and arrays of numbers, which
# the desktop app's .dyfc files store as JSON and .npy files.
PLAIN_TYPES = (str, bool, type(None))
FRONTAL_VIEW = {"pose": "frontal", "flipped": False, "lateral": False}


def _same(a, b):
    """Equal values, where NaN equals NaN."""
    return a == b or (a != a and b != b)


def _tolerance(expected: float) -> float:
    # 30% relative, with a 1.0 absolute floor for small/near-zero baselines
    # (e.g. fai, dental_ratio). Sized against the actual swing observed when
    # opencv-python moved 4.13.0.92 -> 5.0.0.93 (worst case ~30% on fai),
    # not an arbitrary guess -- see dynaface_onnx.py's warpAffine/resize
    # cross-version sensitivity.
    return max(abs(expected) * 0.30, 1.0)


class TestFaceSave(unittest.TestCase):

    def test_persist_frontal(self):
        img = load_image("./tests_data/img1-512.jpg")

        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Analyze face
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(img, crop=True)
        face.analyze()
        data = face.dump_state()

        face2 = facial.AnalyzeFace(measures=measures.all_measures())
        face2.load_state(data)
        stats = face2.analyze()

        # Expected values (rounded to 2 decimals)
        # Calibrated with the first-principles resamplers (dynaface.resample,
        # replacing cv2.resize/warpAffine) under the CPU execution provider; other
        # providers/platforms stay within _tolerance(). See dynaface.resample.
        expected_values = {
            "fai": 0.95,
            "oce.l": 36.83,
            "oce.r": 46.48,
            "brow.d": 1.91,
            "dental_area": 577.59,
            "dental_left": 303.51,
            "dental_right": 274.08,
            "dental_ratio": 0.9,
            "dental_diff": 29.43,
            "eye.left": 121.64,
            "eye.right": 121.69,
            "eye.diff": 0.05,
            "eye.ratio": 1.0,
            "id": 33.73,
            "ml": 74.15,
            "oe": 88.46,
            "tilt": 1.16,
            "px2mm": 0.32,
            "pd": 198.04,
        }

        # Check expected values (rounded)

        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            self.assertAlmostEqual(
                actual,
                expected,
                delta=_tolerance(expected),
                msg=f"{key}: expected {expected}, got {actual}",
            )

    def test_persist_left_lateral(self):
        img = load_image("./tests_data/img3-1024-left-lateral.jpg")

        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Analyze face
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(img, crop=True)
        face.analyze()

        original = face.analyze()
        data = face.dump_state()

        face2 = facial.AnalyzeFace(measures=measures.all_measures())
        face2.load_state(data)
        stats = face2.analyze()

        # The profile is restored as a profile, so AnalyzeLateral measures it as
        # it measured the original. Up to Dynaface 2.0.4 it was restored as a
        # frontal face, and AnalyzeLateral returned nothing.
        self.assertTrue(face2.lateral)
        self.assertEqual(face2.pose, Pose.LATERAL)
        for key in LATERAL_OUTPUTS:
            self.assertIn(key, stats)
            self.assertTrue(_same(stats[key], original[key]), key)

        # The frontal measures still run on a profile, as on the original, since
        # every measure given to AnalyzeFace runs.
        # Expected values (rounded to 2 decimals)
        # Calibrated with the first-principles resamplers (dynaface.resample,
        # replacing cv2.resize/warpAffine) under the CPU execution provider; other
        # providers/platforms stay within _tolerance(). See dynaface.resample.
        expected_values = {
            "fai": 2.19,
            "oce.l": 22.68,
            "oce.r": 16.65,
            "brow.d": 8.64,
            "dental_area": 15.98,
            "dental_left": 2.21,
            "dental_right": 13.77,
            "dental_ratio": 0.16,
            "dental_diff": 11.56,
            "eye.left": 73.87,
            "eye.right": 0.17,
            "eye.diff": 73.7,
            "eye.ratio": 0.0,
            "id": 15.0,
            "tilt": 0.78,
        }

        # Check expected values (rounded)
        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            self.assertAlmostEqual(
                actual,
                expected,
                delta=_tolerance(expected),
                msg=f"{key}: expected {expected}, got {actual}",
            )


class TestSavedHeadPose(unittest.TestCase):
    """A restored state reports the head pose its face was analyzed with."""

    # yaw, pitch and roll in degrees, then the translation, as the landmark model gives
    POSE = np.array([12.5, -4.25, 3.0, 10.0, 20.0, 30.0])

    def load(self):
        """A gray image loaded with made-up landmarks and pose instead of the models."""
        face = facial.AnalyzeFace(measures=[measures.AnalyzePose()])
        img = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        landmarks = [(512, 512)] * 96 + [(400, 500), (600, 500)]
        with mock.patch.object(
            facial.AnalyzeFace, "_find_landmarks", return_value=(landmarks, self.POSE)
        ):
            face.load_image(img, crop=True, force_pose=Pose.FRONTAL)
        return face

    def restore(self, state):
        face = facial.AnalyzeFace(measures=[measures.AnalyzePose()])
        face.load_state(state)
        return face

    def test_pose_survives_saving(self):
        face = self.load()
        expected = face.analyze()
        restored = self.restore(face.dump_state())
        self.assertEqual(restored.analyze(), expected)
        self.assertEqual(restored.headpose, list(self.POSE))

    def test_state_saved_before_the_pose_was_kept(self):
        # Dynaface 2.0.4 and earlier saved [0, 0, 0] in its place, and such a state
        # restores a level head, as it always did.
        state = self.load().dump_state()
        state[1] = [0, 0, 0]
        restored = self.restore(state)
        self.assertEqual((restored.yaw, restored.pitch, restored.roll), (0.0, 0.0, 0.0))

    def test_flipped_profile(self):
        # A profile facing right is flipped and found again, but it reports, and
        # saves, the pose found in the photo as loaded.
        models.init_models(models.download_models(), models.detect_device())
        img = load_image("./tests_data/img2-1024-right-lateral.jpg")
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(img, crop=True)
        self.assertTrue(face.flipped)
        # The photo's own pose, facing right; the mirror image's yaw is below 0. The
        # .NET port's AFlippedProfileReportsThePhotosOwnPose checks the same.
        self.assertGreater(face.yaw, 0)

        restored = facial.AnalyzeFace(measures=measures.all_measures())
        restored.load_state(face.dump_state())
        for name in ("yaw", "pitch", "roll"):
            self.assertAlmostEqual(
                getattr(restored, name), getattr(face, name), msg=name
            )
        self.assertTrue(restored.flipped)


class TestSavedView(unittest.TestCase):
    """
    A restored state keeps its view: the pose, whether a profile was flipped, and a
    profile's lateral landmarks and sagittal outline.
    """

    # A profile facing right (yaw > 0), so it is flipped to face left.
    POSE = np.array([60.0, -4.0, 2.0, 0.0, 0.0, 0.0])
    LATERAL_LANDMARKS = np.array(
        [[300, 200], [290, 300], [200, 450], [260, 520], [280, 640], [270, 760]]
    )
    SAGITTAL_Y = np.arange(1024, dtype=np.int32)
    SAGITTAL_X = (300 + SAGITTAL_Y % 50).astype(np.int32)

    def load(self, pose, face=None, lateral=None):
        """A face loaded with made-up landmarks and lateral data instead of the models.

        lateral is what analyze_lateral() returns: by default a chart, the lateral
        landmarks and the sagittal outline."""
        face = face or facial.AnalyzeFace(
            measures=[AnalyzeLateral(), measures.AnalyzePose()]
        )
        img = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        landmarks = [(400 + i, 300 + 4 * i) for i in range(96)]
        landmarks += [(450, 400), (550, 400)]  # the pupils
        if lateral is None:
            chart = np.full((1024, 600, 3), 255, dtype=np.uint8)
            chart[:, 100:500] = 0
            lateral = (chart, self.LATERAL_LANDMARKS, self.SAGITTAL_X, self.SAGITTAL_Y)
        with mock.patch.object(
            facial.AnalyzeFace, "_find_landmarks", return_value=(landmarks, self.POSE)
        ), mock.patch("dynaface.lateral.analyze_lateral", return_value=lateral):
            face.load_image(img, crop=True, force_pose=pose)
        return face

    def load_quarter(self):
        """A three-quarter face loaded without the models: the 45 degree render
        with the raw landmarks and head pose its fixture recorded in place of the
        detector, and the mask from the render's background color in place of
        the background model, as test_pose_quarter mocks them."""
        with open("./tests_data/cheek_landmarks.json", encoding="utf-8") as f:
            entry = json.load(f)["images"]["yaw45"]
        with open("./tests_data/cheek/cheek_renders.json", encoding="utf-8") as f:
            background = np.array(json.load(f)["background"], dtype=np.uint8)
        raw = entry["raw"]
        landmarks = [(int(x), int(y)) for x, y in raw["landmarks"]]
        pose = np.array([raw["yaw"], raw["pitch"], raw["roll"], 0.0, 0.0, 0.0])

        def foreground_mask(image):
            pixels = np.asarray(image.convert("RGB"))
            foreground = np.any(pixels != background, axis=2)
            return foreground, np.where(foreground, 255, 0).astype(np.uint8)

        face = facial.AnalyzeFace(
            measures=[AnalyzeCheekMeasures(), measures.AnalyzePose()]
        )
        with mock.patch.object(
            facial.AnalyzeFace, "_find_landmarks", return_value=(landmarks, pose)
        ), mock.patch("dynaface.lateral.foreground_mask", side_effect=foreground_mask):
            face.load_image(
                load_image(os.path.join("./tests_data", entry["file"])),
                crop=True,
                force_pose=Pose.QUARTER,
            )
        return face

    def restore(self, state, face=None):
        face = face or facial.AnalyzeFace(
            measures=[AnalyzeLateral(), measures.AnalyzePose()]
        )
        face.load_state(state)
        return face

    def test_profile_survives_saving(self):
        face = self.load(Pose.LATERAL)
        self.assertTrue(face.lateral and face.flipped)
        expected = face.analyze(render=False)
        self.assertTrue(set(LATERAL_OUTPUTS) <= set(expected))

        restored = self.restore(face.dump_state())
        self.assertTrue(restored.lateral)
        self.assertTrue(restored.flipped)
        self.assertEqual(restored.pose, Pose.LATERAL)
        np.testing.assert_array_equal(
            restored.lateral_landmarks, self.LATERAL_LANDMARKS
        )
        np.testing.assert_array_equal(restored.sagittal_x, self.SAGITTAL_X)
        np.testing.assert_array_equal(restored.sagittal_y, self.SAGITTAL_Y)
        self.assertEqual(restored.pix2mm, face.pix2mm)
        actual = restored.analyze(render=False)
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            self.assertTrue(_same(actual[key], value), key)

    def test_restoring_does_not_share_the_arrays(self):
        face = self.load(Pose.LATERAL)
        state = face.dump_state()
        restored = self.restore(state)
        restored.lateral_landmarks[0, 0] += 1
        restored.sagittal_x[0] += 1
        np.testing.assert_array_equal(
            state[6]["lateral_landmarks"], face.lateral_landmarks
        )
        np.testing.assert_array_equal(state[6]["sagittal_x"], face.sagittal_x)

    def test_frontal_view(self):
        state = self.load(Pose.FRONTAL).dump_state()
        self.assertEqual(
            state[6], {"pose": "frontal", "flipped": False, "lateral": False}
        )
        # Restored onto a face that holds a profile, it leaves no lateral data.
        face = self.restore(state, self.load(Pose.LATERAL))
        self.assertFalse(face.lateral or face.flipped)
        self.assertEqual(face.pose, Pose.FRONTAL)
        self.assertEqual(face.lateral_landmarks.shape, (0, 2))
        self.assertIsNone(face.sagittal_x)
        self.assertNotIn("nfa", face.analyze(render=False))

    def test_a_reused_face_saves_each_images_own_view(self):
        # One face loads every frame of a video. A frontal frame after a flipped
        # profile saves no flip and no lateral data.
        face = self.load(Pose.LATERAL)
        self.assertTrue(face.flipped)
        self.load(Pose.FRONTAL, face)
        self.assertEqual(
            face.dump_state()[6],
            {"pose": "frontal", "flipped": False, "lateral": False},
        )
        self.assertEqual(face.lateral_landmarks.shape, (0, 2))
        self.assertIsNone(face.sagittal_x)

    def test_view_needs_no_pickle(self):
        # Plain values and arrays of numbers, which the desktop app's .dyfc files
        # store as JSON and .npy files: bool, str, None (a profile whose outline
        # couldn't be traced) and numeric arrays, in every view.
        no_outline = (None, self.LATERAL_LANDMARKS, None, None)
        views = {
            "profile": self.load(Pose.LATERAL).dump_state()[6],
            "profile without an outline": self.load(
                Pose.LATERAL, lateral=no_outline
            ).dump_state()[6],
            "quarter": self.load_quarter().dump_state()[6],
            "frontal": self.load(Pose.FRONTAL).dump_state()[6],
        }
        for name, view in views.items():
            for key, value in view.items():
                with self.subTest(view=name, key=key):
                    if isinstance(value, np.ndarray):
                        self.assertFalse(value.dtype.hasobject)
                        self.assertIn(value.dtype.kind, "iuf")
                    else:
                        self.assertIn(type(value), PLAIN_TYPES)
        # Each view holds what it says: the profile without an outline keeps
        # None where its sagittal outline would be, and the quarter face its far
        # side and three outline arrays, each a numeric array.
        self.assertIsNone(views["profile without an outline"]["sagittal_x"])
        self.assertIsNone(views["profile without an outline"]["sagittal_y"])
        self.assertIsInstance(views["profile"]["sagittal_x"], np.ndarray)
        quarter = views["quarter"]
        self.assertEqual(quarter["pose"], "quarter")
        self.assertIs(quarter["far_left"], True)
        self.assertEqual(
            set(quarter),
            {
                "pose",
                "flipped",
                "lateral",
                "far_left",
                "cheek_outline_x",
                "cheek_outline_y",
                "cheek_outline_src",
            },
        )
        for key in ("cheek_outline_x", "cheek_outline_y", "cheek_outline_src"):
            self.assertIsInstance(quarter[key], np.ndarray, key)
            self.assertGreater(len(quarter[key]), 0, key)
        self.assertEqual(views["frontal"], FRONTAL_VIEW)

    def test_state_saved_before_the_view_was_kept(self):
        # States saved by Dynaface 2.0.4 and earlier end with the face rotation. They
        # leave the face's view as it is: the desktop app sets a profile's lateral
        # data from its own document before restoring its frames.
        state = self.load(Pose.LATERAL).dump_state()[:6]
        face = facial.AnalyzeFace(measures=[AnalyzeLateral()])
        face.lateral = True
        face.lateral_landmarks = self.LATERAL_LANDMARKS
        self.restore(state, face)
        self.assertTrue(face.lateral)
        self.assertIn("nfa", face.analyze(render=False))

        self.assertFalse(self.restore(state).lateral)
