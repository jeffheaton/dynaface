import sys
import unittest
import os

from dynaface.image import load_image
from dynaface import facial, measures, models, lateral

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestFaceAnalysis(unittest.TestCase):

    def test_frontal(self):
        img = load_image("./tests_data/img1-512.jpg")

        lateral.DEBUG = True
        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Analyze face
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(img, crop=True)
        stats = face.analyze()
        face.draw_static()
        face.draw_landmarks()

        # test all items
        items = face.get_all_items()
        assert "fai" in items
        assert isinstance(items, list), "Expected a list"

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
            # print(f'"{key}": {actual},')
            self.assertAlmostEqual(
                actual,
                expected,
                places=3,
                msg=f"{key}: expected {expected}, got {actual}",
            )
        lateral.DEBUG = False

    def test_right_lateral(self):
        img = load_image("./tests_data/img2-1024-right-lateral.jpg")

        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Analyze face
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(img, crop=True)
        stats = face.analyze()
        face.draw_static()
        face.draw_landmarks()

        # Expected values (rounded to 2 decimals)
        expected_values = {
            "fai": 4.18,
            "oce.l": 22.86,
            "oce.r": 16.24,
            "brow.d": 8.16,
            "dental_area": 63.85,
            "dental_left": 63.56,
            "dental_right": 0.29,
            "dental_ratio": 0.00,
            "dental_diff": 63.27,
            "eye.left": 67.45,
            "eye.right": 0.06,
            "eye.diff": 67.39,
            "eye.ratio": 0.00,
            "id": 12.30,
            "ml": 21.95,
            "oe": 28.47,
            "tilt": -7.13,
            "px2mm": 0.24,
            "pd": 64.5,
        }

        # Check expected values (rounded)
        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            # print(f'"{key}": {actual},')
            self.assertAlmostEqual(
                actual,
                expected,
                places=2,
                msg=f"{key}: expected {expected}, got {actual}",
            )

    def test_left_lateral(self):
        img = load_image("./tests_data/img3-1024-left-lateral.jpg")

        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Analyze face
        face = facial.AnalyzeFace()
        face.load_image(img, crop=True)
        face.draw_static()
        face.draw_landmarks()
        stats = face.analyze()

        # Expected values (rounded to 2 decimals)
        expected_values = {
            "fai": 0.19,
            "oce.l": 22.97,
            "oce.r": 15.51,
            "brow.d": 7.68,
            "dental_area": 49.91,
            "dental_left": 45.3,
            "dental_right": 4.61,
            "dental_ratio": 0.1,
            "dental_diff": 40.69,
            "eye.left": 78.65,
            "eye.right": 3.2,
            "eye.diff": 75.46,
            "eye.ratio": 0.04,
            "id": 14.05,
            "ml": 20.66,
            "oe": 30.05,
            "tilt": -0.86,
            "px2mm": 0.24,
            "pd": 67.01,
        }

        # Check expected values (rounded)
        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            # print(f'"{key}": {actual},')
            self.assertAlmostEqual(
                actual,
                expected,
                places=2,
                msg=f"{key}: expected {expected}, got {actual}",
            )

    def test_load_image_local(self):
        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Load image
        face = facial.load_face_image("./tests_data/img1-512.jpg")
        assert face.width == 1024
        assert face.height == 1024
        assert face.render_img is not None
        assert face.render_img.shape == (1024, 1024, 3)

    def test_load_image_url(self):
        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Load image
        face = facial.load_face_image(
            "https://www.heatonresearch.com/images/jeff/about-jeff-heaton-2020.jpg"
        )
        assert face.width == 1024
        assert face.height == 1024
        assert face.render_img is not None
        assert face.render_img.shape == (1024, 1024, 3)

    def test_fail_init_models(self):
        models.unload_models()
        with self.assertRaises(ValueError) as context:
            _ = facial.load_face_image("./tests_data/img1-512.jpg")

        self.assertIn("not initialized", str(context.exception).lower())

    def test_face_rotation(self):
        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Load image
        face = facial.load_face_image("./tests_data/img1-512.jpg")
        assert face.calculate_face_rotation() == 0.0
