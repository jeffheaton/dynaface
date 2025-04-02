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
            "nw": 8.57,
            "oe": 28.47,
            "nn": 56.50,
            "nm": 42.66,
            "np": 51.32,
            "tilt": -7.13,
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
            "nn": 63.64,
            "nm": 46.78,
            "np": 56.73,
            "tilt": -0.86,
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
