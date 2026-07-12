import os
import sys
import unittest

from dynaface import facial, measures, models
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
            "fai": 2.18,
            "oce.l": 84.78,
            "oce.r": 107.0,
            "brow.d": 4.39,
            "dental_area": 3073.29,
            "dental_left": 1447.08,
            "dental_right": 1626.22,
            "dental_ratio": 0.89,
            "dental_diff": 179.14,
            "eye.left": 644.69,
            "eye.right": 644.96,
            "eye.diff": 0.27,
            "eye.ratio": 1.0,
            "id": 77.64,
            "ml": 170.72,
            "oe": 203.66,
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
            "dental_area": 16.39,
            "dental_left": 13.8,
            "dental_right": 2.59,
            "dental_ratio": 0.19,
            "dental_diff": 11.2,
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
