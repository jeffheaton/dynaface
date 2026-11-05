"""
Every millimeter value depends on pix2mm matching the landmarks it's applied to,
and that pairing has broken more than once: the scale was measured before the crop
(from July 2024 until October 2026), an app kept an uncropped run's scale for
landmarks it cropped later, and saved states kept a scale from the old rule. These
measure one made-up face every way a caller can reach its landmarks and require the
same millimeters each time. The .NET port's ScaleConsistencyTests checks the same.
"""

import math
import unittest
from unittest import mock

import numpy as np

from dynaface import facial, measures, models
from dynaface.const import Pose
from dynaface.image import load_image

WIDTH, HEIGHT = 1280, 720
HEADPOSE = np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
RIGHT_PUPIL, LEFT_PUPIL = (530, 320), (760, 330)

# dump_state() keeps self.headpose, which load_image() never fills in, so a restored
# face reports a level head. That's a separate gap from the scale, so the restored
# comparisons leave these out.
POSE_OUTPUTS = ("yaw", "pitch", "roll")


def made_up_landmarks():
    """98 landmarks scattered around pupils 230 pixels apart, in a video-sized frame."""
    landmarks = [
        (RIGHT_PUPIL[0] - 80 + (i * 37) % 380, RIGHT_PUPIL[1] - 120 + (i * 53) % 400)
        for i in range(98)
    ]
    landmarks[96], landmarks[97] = RIGHT_PUPIL, LEFT_PUPIL
    return landmarks


def load(crop):
    """A gray frame loaded with the made-up landmarks in place of the models."""
    face = facial.AnalyzeFace(measures=measures.all_measures())
    frame = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    found = (made_up_landmarks(), HEADPOSE)
    with mock.patch.object(facial.AnalyzeFace, "_find_landmarks", return_value=found):
        face.load_image(frame, crop=crop, force_pose=Pose.FRONTAL)
    return face


def restore(state):
    face = facial.AnalyzeFace(measures=measures.all_measures())
    face.load_state(state)
    return face


def _is_nan(value):
    try:
        return math.isnan(value)
    except TypeError:
        return False


def assert_same_outputs(test, expected, actual, ignore=()):
    test.assertEqual(sorted(expected), sorted(actual))
    for key, value in expected.items():
        if key in ignore:
            continue
        if _is_nan(value):
            test.assertTrue(_is_nan(actual[key]), f"{key}: expected NaN")
        else:
            test.assertAlmostEqual(
                value, actual[key], delta=1e-6 * max(1.0, abs(value)), msg=key
            )


class TestScale(unittest.TestCase):
    """The scale on made-up landmarks, without the models."""

    def test_same_face_same_millimeters(self):
        face = load(crop=True)
        expected = face.analyze()

        # Loaded without the crop, then cropped.
        later = load(crop=False)
        later.crop_stylegan()
        assert_same_outputs(self, expected, later.analyze())

        # Saved, then restored.
        restored = restore(face.dump_state())
        assert_same_outputs(self, expected, restored.analyze(), ignore=POSE_OUTPUTS)

    def test_state_saved_with_the_old_rule_is_corrected(self):
        # Before October 2026 the scale was measured on the uncropped pupils, and
        # that's the pair a saved state holds. Restoring measures it again.
        face = load(crop=True)
        expected = face.analyze()
        uncropped_pd, uncropped_pix2mm = facial.util_calc_pd((LEFT_PUPIL, RIGHT_PUPIL))
        self.assertGreater(abs(uncropped_pix2mm - face.pix2mm), 0.01)

        state = face.dump_state()
        state[3], state[4] = uncropped_pd, uncropped_pix2mm
        assert_same_outputs(
            self, expected, restore(state).analyze(), ignore=POSE_OUTPUTS
        )

    def test_each_face_keeps_its_pupillary_distance(self):
        face = load(crop=True)
        expected = face.analyze()
        state = face.dump_state()
        default_pd = facial.AnalyzeFace.pd
        try:
            facial.AnalyzeFace.pd = 58
            # A face loaded before the change, and one restored after it, keep the
            # pupillary distance they were analyzed with.
            assert_same_outputs(self, expected, face.analyze())
            restored = restore(state)
            self.assertAlmostEqual(restored.pd, default_pd)
            assert_same_outputs(self, expected, restored.analyze(), ignore=POSE_OUTPUTS)
            # A face loaded now uses the new one.
            self.assertAlmostEqual(
                load(crop=True).analyze()["ml"], expected["ml"] * 58 / default_pd
            )
        finally:
            facial.AnalyzeFace.pd = default_pd

    def test_profile_keeps_its_fixed_scale(self):
        # A profile's saved pair holds no pupillary distance (its pupils give none),
        # so its fixed scale comes back as it was.
        state = load(crop=True).dump_state()
        state[3], state[4] = 0.0, 0.24
        restored = restore(state)
        restored.analyze()
        self.assertEqual(restored.pix2mm, 0.24)


class TestScaleOnAPhoto(unittest.TestCase):
    """The first check on a photo. img1-512.jpg is where the scale measured before
    the crop was furthest off, about 2.3 times too large."""

    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())

    def test_same_photo_same_millimeters(self):
        img = load_image("./tests_data/img1-512.jpg")
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(img, crop=True)
        expected = face.analyze()

        later = facial.AnalyzeFace(measures=measures.all_measures())
        later.load_image(img, crop=False)
        later.crop_stylegan()
        assert_same_outputs(self, expected, later.analyze())

        restored = restore(face.dump_state())
        assert_same_outputs(self, expected, restored.analyze(), ignore=POSE_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
