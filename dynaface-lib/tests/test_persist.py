import sys
import unittest
import os

from dynaface.image import load_image
from dynaface import facial, measures, models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
        expected_values = {
            "fai": 1.1,
            "oce.l": 82.35,
            "oce.r": 107.47,
            "brow.d": 8.69,
            "dental_area": 3167.49,
            "dental_left": 1501.03,
            "dental_right": 1666.47,
            "dental_ratio": 0.9,
            "dental_diff": 165.44,
            "eye.left": 567.37,
            "eye.right": 588.35,
            "eye.diff": 20.98,
            "eye.ratio": 0.96,
            "id": 78.22,
            "ml": 162.29,
            "oe": 201.34,
            "tilt": 0.0,
            "px2mm": 0.32,
            "pd": 197.0,
        }

        # Check expected values (rounded)

        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            self.assertAlmostEqual(
                actual,
                expected,
                places=2,
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
        expected_values = {
            "fai": 0.19,
            "oce.l": 22.97,
            "oce.r": 15.51,
            "brow.d": 7.68,
            "dental_area": 49.91,
            "dental_left": 45.30,
            "dental_right": 4.61,
            "dental_ratio": 0.10,
            "dental_diff": 40.69,
            "eye.left": 78.65,
            "eye.right": 3.20,
            "eye.diff": 75.46,
            "eye.ratio": 0.04,
            "id": 14.05,
            "tilt": -0.86,
        }

        # Check expected values (rounded)
        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            self.assertAlmostEqual(
                actual,
                expected,
                places=2,
                msg=f"{key}: expected {expected}, got {actual}",
            )
