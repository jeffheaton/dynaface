import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# facial must be imported first to break the circular import chain (see
# test_measures_landmarks.py).
from dynaface import facial  # noqa: F401 (side-effect import)
from dynaface.measures_frontal import AnalyzeBrows

PIX2MM = 0.25


class MockFace:
    """Minimal face stub for unit-testing AnalyzeBrows without loaded models."""

    def __init__(self, points):
        self.landmarks = [(i * 2, i * 2 + 1) for i in range(98)]
        for i, p in points.items():
            self.landmarks[i] = p
        self.pix2mm = PIX2MM
        self.width = 1024

    def calc_text_size(self, txt):
        return ((len(txt) * 10, 20), 5)


def rotated_face(deg, raise_left=0.0):
    """
    Symmetric face (pupils at the StyleGAN crop targets, brows 80px above them)
    rotated by deg about the crop center. raise_left raises the image-right brow
    (44) in the face's own frame before rotating.
    """
    pts = {
        96: (380.0, 480.0),
        97: (640.0, 480.0),
        35: (380.0, 400.0),
        36: (410.0, 398.0),
        44: (640.0, 400.0 - raise_left),
    }
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {
        i: (512 + (x - 512) * c - (y - 512) * s, 512 + (x - 512) * s + (y - 512) * c)
        for i, (x, y) in pts.items()
    }


def sloped_face(left_brow_y):
    """
    Integer landmarks on a tilt of slope 13/260 = 0.05 (~2.86 degrees), so the int
    truncation in util.line_to_edge doesn't change the result. Mirrored by
    dynaface-lib-dotnet's MeasureBrowsTests.
    """
    return {
        96: (380, 480),
        97: (640, 493),
        35: (380, 400),
        36: (410, 398),
        44: (640, left_brow_y),
    }


class TestAnalyzeBrows(unittest.TestCase):
    def calc(self, points):
        return AnalyzeBrows().calc(MockFace(points), render=False)["brow.d"]

    def test_tilted_symmetric_face_has_no_brow_difference(self):
        for deg in (2.0, 3.5, 5.0, -4.0, 0.0):
            with self.subTest(deg=deg):
                self.assertAlmostEqual(self.calc(rotated_face(deg)), 0.0, places=3)

    def test_tilted_symmetric_face_integer_landmarks(self):
        # 44 sits on the tilted line through 35: 400 + 0.05 * 260 = 413.
        self.assertAlmostEqual(self.calc(sloped_face(413)), 0.0, places=3)

    def test_tilted_asymmetric_face(self):
        # 44 is 12px above the tilted line through 35: 12 * 0.25 = 3.0 mm.
        self.assertAlmostEqual(self.calc(sloped_face(401)), 3.0, places=3)

    def test_level_face_reports_brow_difference(self):
        self.assertAlmostEqual(
            self.calc(rotated_face(0.0, raise_left=12.0)), 3.0, places=3
        )


if __name__ == "__main__":
    unittest.main()
