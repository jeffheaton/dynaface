import unittest
import numpy as np
from facial_analysis.image import load_image
from facial_analysis.facial import safe_clip


class TestImage(unittest.TestCase):
    def test_load_image(self):
        img = load_image("./tests_data/img1-512.jpg")
        assert img.shape[0] == 512
        assert img.shape[1] == 512

    def test_crop_image(self):
        test_image = np.zeros((1000, 1000, 3), dtype=np.uint8)  # Dummy image
        clipped_image, x_offset, y_offset = safe_clip(
            test_image, -100, -100, 1024, 1024, (255, 0, 0)
        )

        assert clipped_image.shape[0] == 1024
        assert clipped_image.shape[1] == 1024
        assert clipped_image.shape[2] == 1024
        assert x_offset == 100
        assert y_offset == 100


if __name__ == "__main__":
    unittest.main()
