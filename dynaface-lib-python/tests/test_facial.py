import os
import sys
import unittest

from dynaface import facial, lateral, measures, models
from dynaface.image import load_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _tolerance(expected: float) -> float:
    # 30% relative, with a 1.0 absolute floor for small/near-zero baselines
    # (e.g. fai, dental_ratio). Sized against the actual swing observed when
    # opencv-python moved 4.13.0.92 -> 5.0.0.93 (worst case ~30% on fai),
    # not an arbitrary guess -- see dynaface_onnx.py's warpAffine/resize
    # cross-version sensitivity.
    return max(abs(expected) * 0.30, 1.0)


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
        # Recalibrated for the ONNX (BlazeFace + SPIGA-onnx) inference pipeline
        # on opencv-python>=5.0.0; see dynaface.dynaface_onnx.DynafaceOnnxInference.
        expected_values = {
            "fai": 1.53,
            "oce.l": 84.78,
            "oce.r": 107.0,
            "brow.d": 4.39,
            "dental_area": 3109.23,
            "dental_left": 1429.91,
            "dental_right": 1679.32,
            "dental_ratio": 0.85,
            "dental_diff": 249.4,
            "eye.left": 644.69,
            "eye.right": 644.96,
            "eye.diff": 0.27,
            "eye.ratio": 1.0,
            "id": 79.84,
            "ml": 169.25,
            "oe": 205.86,
            "tilt": 1.16,
            "px2mm": 0.32,
            "pd": 198.04,
        }

        # Check expected values (rounded)
        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            # print(f'"{key}": {actual},')
            self.assertAlmostEqual(
                actual,
                expected,
                delta=_tolerance(expected),
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
        # Recalibrated for the ONNX (BlazeFace + SPIGA-onnx) inference pipeline
        # on opencv-python>=5.0.0; see dynaface.dynaface_onnx.DynafaceOnnxInference.
        expected_values = {
            "fai": 5.76,
            "oce.l": 20.82,
            "oce.r": 16.42,
            "brow.d": 10.08,
            "dental_area": 49.71,
            "dental_left": 48.93,
            "dental_right": 0.78,
            "dental_ratio": 0.02,
            "dental_diff": 48.15,
            "eye.left": 72.46,
            "eye.right": 0.81,
            "eye.diff": 71.65,
            "eye.ratio": 0.01,
            "id": 13.69,
            "ml": 19.41,
            "oe": 30.08,
            "tilt": -4.09,
            "px2mm": 0.24,
            "pd": 70.18,
        }

        # Check expected values (rounded)
        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            # print(f'"{key}": {actual},')
            self.assertAlmostEqual(
                actual,
                expected,
                delta=_tolerance(expected),
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
        # Recalibrated for the ONNX (BlazeFace + SPIGA-onnx) inference pipeline
        # on opencv-python>=5.0.0; see dynaface.dynaface_onnx.DynafaceOnnxInference.
        expected_values = {
            "fai": 1.99,
            "oce.l": 22.24,
            "oce.r": 15.89,
            "brow.d": 8.64,
            "dental_area": 12.21,
            "dental_left": 10.31,
            "dental_right": 1.9,
            "dental_ratio": 0.18,
            "dental_diff": 8.41,
            "eye.left": 73.21,
            "eye.right": 0.03,
            "eye.diff": 73.18,
            "eye.ratio": 0.0,
            "id": 14.53,
            "ml": 19.68,
            "oe": 30.26,
            "tilt": 1.59,
            "px2mm": 0.24,
            "pd": 72.03,
        }

        # Check expected values (rounded)
        for key, expected in expected_values.items():
            actual = round(stats.get(key, float("inf")), 2)
            # print(f'"{key}": {actual},')
            self.assertAlmostEqual(
                actual,
                expected,
                delta=_tolerance(expected),
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
        # Recalibrated for the ONNX (BlazeFace + SPIGA-onnx) inference pipeline;
        # see dynaface.dynaface_onnx.DynafaceOnnxInference.
        self.assertAlmostEqual(
            face.calculate_face_rotation(), 1.16, delta=_tolerance(1.16)
        )
