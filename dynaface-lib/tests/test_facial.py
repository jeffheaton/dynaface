import sys
import unittest
import os

from facial_analysis.image import load_image
from facial_analysis import facial, measures, models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestFaceAnalysis(unittest.TestCase):

    def test_complete(self):
        img = load_image("./tests_data/img1-512.jpg")

        # Initialize models
        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        # Analyze face
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(img, crop=True)
        stats = face.analyze()

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


if __name__ == "__main__":
    unittest.main()
