"""Tests for AnalyzeFace's options and guards: forcing a pose, straightening a
tilted face, the turned-head correction, saved states and the drawing helpers."""

import os
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

from dynaface import config, facial, measures, models
from dynaface.const import Pose
from dynaface.image import load_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FRONTAL = "./tests_data/img4-1024-frontal.jpg"
PROFILE = "./tests_data/img2-1024-right-lateral.jpg"


def changed(before, after):
    """The (row, column) of every pixel that differs."""
    return np.argwhere((before != after).any(axis=2))


class TestOptionsOnPhotos(unittest.TestCase):
    """Options that need the models, on the sample photos."""

    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())

    def test_straightening_a_tilted_face(self):
        image = load_image(FRONTAL)
        h, w = image.shape[:2]
        turn = cv2.getRotationMatrix2D((w / 2, h / 2), -8, 1.0)  # 8° clockwise
        tilted = cv2.warpAffine(image, turn, (w, h), borderValue=(255, 255, 255))

        def tilt(threshold):
            face = facial.AnalyzeFace(
                [measures.AnalyzePosition()], tilt_threshold=threshold
            )
            face.load_image(tilted, crop=True)
            return face.face_rotation, face.analyze()["tilt"]

        rotation, kept = tilt(-1)  # the default: never straighten
        self.assertIsNone(rotation)
        self.assertAlmostEqual(kept, 8, delta=1.5)
        rotation, left = tilt(2)  # straighten anything tilted more than 2°
        self.assertIsNotNone(rotation)
        self.assertLess(abs(left), 1.5)

    def test_forcing_a_profile(self):
        face = facial.AnalyzeFace(measures=[])
        face.load_image(load_image(FRONTAL), crop=True, force_pose=Pose.LATERAL)
        self.assertTrue(face.lateral)
        self.assertEqual(face.pose, Pose.LATERAL)

    def test_forced_profile_without_a_head_pose(self):
        # With no yaw from the landmark model, a forced profile falls back on the
        # nose's position to decide which way the face looks.
        face = facial.AnalyzeFace(measures=[])
        find = face._find_landmarks

        def no_yaw(img):
            landmarks, pose = find(img)
            pose = pose.copy()
            pose[0] = np.nan
            return landmarks, pose

        face._find_landmarks = no_yaw
        face.load_image(load_image(PROFILE), crop=True, force_pose=Pose.LATERAL)
        self.assertTrue(face.lateral)

    def test_profile_detection_off(self):
        with mock.patch.object(config, "AUTO_LATERAL", False):
            face = facial.AnalyzeFace(measures=[])
            face.load_image(load_image(PROFILE), crop=True)
        self.assertFalse(face.lateral)

    def test_no_face(self):
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(np.full((300, 300, 3), 255, dtype=np.uint8), crop=True)
        self.assertTrue(face.is_no_face())
        self.assertIsNone(face.analyze())
        before = face.render_img.copy()
        face.draw_landmarks()
        face._overlay_lateral_analysis(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(len(changed(before, face.render_img)), 0)
        self.assertEqual(face._is_lateral(), (False, False))
        self.assertEqual(face._force_lateral(), (True, False))

    def test_models_must_be_loaded(self):
        with mock.patch.object(models, "_device", "?"):  # init_models() never ran
            with self.assertRaises(ValueError):
                facial.AnalyzeFace(measures=[]).load_image(load_image(FRONTAL))


def made_up_face(right_pupil=(400, 500), left_pupil=(600, 500), size=1024):
    """A gray image with made-up landmarks, all in the middle except the pupils."""
    face = facial.AnalyzeFace(measures=[])
    face.init_image(np.full((size, size, 3), 128, dtype=np.uint8))
    face.landmarks = [(size // 2, size // 2)] * 96 + [right_pupil, left_pupil]
    face.calc_pd()
    return face


class TestMadeUpFaces(unittest.TestCase):
    """Options and guards on made-up landmarks, without the models."""

    def test_pupils_in_one_place(self):
        for pupil in ((500, 500), (0, 0)):
            face = made_up_face(right_pupil=pupil, left_pupil=pupil)
            with self.assertRaises(ValueError):
                face.crop_stylegan()

    def test_turned_head_correction(self):
        # The crop puts the pupils 260 pixels apart, but when the head is turned
        # more than 5 degrees, it first corrects their distance, making the face
        # smaller in the crop.
        def pupil_distance_in_crop(yaw):
            face = made_up_face()  # pupils 200 pixels apart
            face.yaw = yaw
            face.crop_stylegan()
            return facial.util_calc_pd(facial.util_get_pupils(face.landmarks))[0]

        self.assertAlmostEqual(pupil_distance_in_crop(0), 260, delta=2)
        self.assertLess(pupil_distance_in_crop(10), 240)

    def test_saved_state_restores_the_image(self):
        face = made_up_face()
        face.render_img[:] = 7  # drawing that the state shouldn't carry
        state = face.dump_state()[:5]  # saved before states held a face rotation
        fresh = facial.AnalyzeFace(measures=[])
        fresh.load_state(state)
        self.assertEqual(fresh.face_rotation, 0)
        np.testing.assert_array_equal(fresh.render_img, face.original_img)
        self.assertEqual((fresh.width, fresh.height), (1024, 1024))

    def test_drawing_some_landmarks(self):
        face = made_up_face()
        face.landmarks = [(100 + 8 * i, 300) for i in range(98)]
        before = face.render_img.copy()
        face.draw_landmarks(numbers=True, only=[64, 68])  # at x = 612 and 644
        columns = changed(before, face.render_img)[:, 1]
        self.assertGreater(len(columns), 0)
        self.assertTrue(((columns > 600) & (columns < 680)).all())

    def test_drawing_the_eye_centers(self):
        face = made_up_face()
        face.left_eye, face.right_eye = (300, 700), (700, 700)
        before = face.render_img.copy()
        face.draw_landmarks()
        drawn = {tuple(p) for p in changed(before, face.render_img)}
        self.assertIn((700, 300), drawn)  # (row, column)
        self.assertIn((700, 700), drawn)

    def test_measure_label_above(self):
        face = made_up_face()
        before = face.render_img.copy()
        distance = face.measure((100, 400), (300, 400), dir="a")
        self.assertAlmostEqual(distance, 200 * face.pix2mm)
        rows = changed(before, face.render_img)[:, 0]
        self.assertLess(rows.min(), 390)  # the label is above the line

    def test_quarter_pose_label(self):
        face = made_up_face()
        face.pose = Pose.QUARTER
        before = face.render_img.copy()
        face.draw_static()
        self.assertGreater(len(changed(before, face.render_img)), 0)

    def test_disabled_measures_are_skipped(self):
        fai = measures.AnalyzeFAI()
        fai.enabled = False
        face = made_up_face()
        face.measures = [fai, measures.AnalyzePosition()]
        results = face.analyze()
        self.assertNotIn("fai", results)
        self.assertIn("tilt", results)

    def test_chart_overlay(self):
        face = made_up_face()
        before = face.render_img.copy()
        face._overlay_lateral_analysis(None)
        np.testing.assert_array_equal(face.render_img, before)
        # A chart wider than the image, once scaled to its height, is cut to fit.
        face._overlay_lateral_analysis(np.full((512, 1200, 3), 7, dtype=np.uint8))
        self.assertTrue((face.render_img == 7).all())

    def test_profile_crop_edges(self):
        def crop(height, width, landmarks):
            face = facial.AnalyzeFace(measures=[])
            face.init_image(np.full((height, width, 3), 128, dtype=np.uint8))
            face.landmarks = landmarks
            face.crop_lateral()
            self.assertEqual(face.render_img.shape, (1024, 1024, 3))
            return face.landmarks

        crop(600, 400, [])  # no landmarks: the image is centered
        # Landmarks spanning a whole square image: it's shrunk to leave room for
        # the padding, and padded out to the full size.
        for x, y in crop(1000, 1000, [(300, 0), (200, 500), (400, 999)]):
            self.assertTrue(0 <= y < 1024)
        # Landmarks at the very bottom of a tall image: the crop can't go lower,
        # but keeps them in view.
        for x, y in crop(3000, 1500, [(700, 2990), (750, 2999)]):
            self.assertTrue(0 <= y < 1024)
