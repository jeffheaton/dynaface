import sys
import unittest
import os

import numpy as np
from dynaface.util import safe_clip
from dynaface.image import load_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
        assert clipped_image.shape[2] == 3
        assert x_offset == 100
        assert y_offset == 100
