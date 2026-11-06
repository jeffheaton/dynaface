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


def assert_same_outputs(test, expected, actual):
    test.assertEqual(sorted(expected), sorted(actual))
    for key, value in expected.items():
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
        assert_same_outputs(self, expected, restore(face.dump_state()).analyze())

    def test_state_saved_with_the_old_rule_is_corrected(self):
        # Before October 2026 the scale was measured on the uncropped pupils, and
        # that's the pair a saved state holds. Restoring measures it again.
        face = load(crop=True)
        expected = face.analyze()
        uncropped_pd, uncropped_pix2mm = facial.util_calc_pd((LEFT_PUPIL, RIGHT_PUPIL))
        self.assertGreater(abs(uncropped_pix2mm - face.pix2mm), 0.01)

        state = face.dump_state()
        state[3], state[4] = uncropped_pd, uncropped_pix2mm
        assert_same_outputs(self, expected, restore(state).analyze())

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
            assert_same_outputs(self, expected, restored.analyze())
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


class TestPupilDistanceRange(unittest.TestCase):
    """
    A pupillary distance setting outside 40 to 85 mm is a mistake, such as
    centimeters for millimeters, so loading an image refuses it. A saved state keeps
    the distance it was analyzed with. The .NET port's PupilDistanceRangeTests
    checks the same.
    """

    def test_check(self):
        for pd in (40, 63, 85, 58.5, "63"):
            self.assertEqual(facial.check_pupil_distance(pd), float(pd))
        for pd in (39.9, 85.1, 6.3, 630, 0, -63, float("nan"), float("inf"), None, "x"):
            with self.subTest(pd=pd), self.assertRaises(ValueError):
                facial.check_pupil_distance(pd)

    def test_loading_refuses_a_setting_outside(self):
        default_pd = facial.AnalyzeFace.pd
        try:
            facial.AnalyzeFace.pd = 6.3  # centimeters
            face = facial.AnalyzeFace(measures=[])
            with mock.patch.object(facial.AnalyzeFace, "_find_landmarks") as found:
                with self.assertRaisesRegex(ValueError, "40 to 85 mm"):
                    face.load_image(np.full((100, 100, 3), 128, dtype=np.uint8))
            found.assert_not_called()  # before the models run
        finally:
            facial.AnalyzeFace.pd = default_pd

    def test_state_with_a_distance_outside_restores_it(self):
        # A state saved before 2.1 could hold any distance. It restores with that
        # distance, as it always did, and its scale and outputs agree.
        state = load(crop=True).dump_state()
        state[4] = 30.0 / state[3]  # analyzed as if the pupils were 30 mm apart
        face = restore(state)
        self.assertEqual(face.pd, 30.0)
        self.assertAlmostEqual(face.pix2mm, state[4])
        results = face.analyze()
        self.assertAlmostEqual(results["px2mm"], face.pix2mm, places=4)

        # Measuring it again with another distance still works.
        face.pd = 63
        again = face.analyze()
        self.assertAlmostEqual(again["ml"] / results["ml"], 63 / 30, places=4)

    def test_restored_distance_is_exact(self):
        # The product of the saved pair has float error; the face gets 63 back.
        state = load(crop=True).dump_state()
        self.assertEqual(restore(state).pd, 63.0)


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

        assert_same_outputs(self, expected, restore(face.dump_state()).analyze())


if __name__ == "__main__":
    unittest.main()
