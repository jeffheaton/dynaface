import os
import sys
import unittest
from unittest import mock

import numpy as np

from dynaface import facial, measures, models
from dynaface.const import Pose
from dynaface.image import load_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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

        data = face.dump_state()

        face2 = facial.AnalyzeFace(measures=measures.all_measures())
        face2.load_state(data)
        stats = face2.analyze()

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

        restored = facial.AnalyzeFace(measures=measures.all_measures())
        restored.load_state(face.dump_state())
        for name in ("yaw", "pitch", "roll"):
            self.assertAlmostEqual(
                getattr(restored, name), getattr(face, name), msg=name
            )
