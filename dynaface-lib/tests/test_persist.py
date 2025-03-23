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
            "fai": 2.01,
            "oce.l": 108.97,
            "oce.r": 140.95,
            "brow.d": 10.86,
            "dental_area": 5549.47,
            "dental_left": 2635.77,
            "dental_right": 2913.69,
            "dental_ratio": 0.90,
            "dental_diff": 277.92,
            "eye.left": 966.69,
            "eye.right": 980.06,
            "eye.diff": 13.37,
            "eye.ratio": 0.99,
            "id": 103.57,
            "ml": 214.44,
            "nw": 116.59,
            "oe": 266.52,
            "tilt": 0.0,
            "px2mm": 0.24,
            "pd": 260.0,
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
            "ml": 20.66,
            "nw": 7.03,
            "oe": 30.05,
            # "nn": 63.64,
            # "nm": 46.78,
            # "np": 56.73,
            "tilt": -0.86,
            # "px2mm": 0.24,
            # "pd": 260.0,
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
