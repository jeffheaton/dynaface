import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# facial must be imported first to break the circular import chain (see
# test_measures_landmarks.py).
from dynaface import facial  # noqa: E402
from dynaface.measures_frontal import AnalyzeNoseFrontal  # noqa: E402

PIX2MM = 0.25

# Lower nose outline (WFLW 55-59): alar edges 100 px apart, mid-nostril points 50 px.
# Mirrored by dynaface-lib-dotnet's MeasureNoseFrontalTests.
NOSE = {55: (460, 700), 56: (485, 706), 57: (510, 710), 58: (535, 706), 59: (560, 700)}


def rotate(points, deg, center=(512.0, 512.0)):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    cx, cy = center
    return {
        i: (cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c)
        for i, (x, y) in points.items()
    }


def nose_widths(points, render):
    face = facial.AnalyzeFace(measures=[AnalyzeNoseFrontal()])
    face.init_image(np.zeros((1024, 1024, 3), dtype=np.uint8))
    landmarks = [(510.0, 600.0)] * 98
    for i, p in points.items():
        landmarks[i] = p
    face.landmarks = landmarks
    face.pix2mm = PIX2MM
    return face.analyze() if render else AnalyzeNoseFrontal().calc(face, render=False)


class TestAnalyzeNoseFrontal(unittest.TestCase):
    def test_alar_and_nostril_widths(self):
        for render in (False, True):
            with self.subTest(render=render):
                r = nose_widths(NOSE, render)
                self.assertEqual(set(r), {"nose.alar", "nose.nostril"})
                self.assertAlmostEqual(r["nose.alar"], 100 * PIX2MM)
                self.assertAlmostEqual(r["nose.nostril"], 50 * PIX2MM)

    def test_head_roll_does_not_change_widths(self):
        for deg in (5.0, -8.0):
            with self.subTest(deg=deg):
                r = nose_widths(rotate(NOSE, deg), render=False)
                self.assertAlmostEqual(r["nose.alar"], 100 * PIX2MM, places=6)
                self.assertAlmostEqual(r["nose.nostril"], 50 * PIX2MM, places=6)

    def test_old_columns_are_retired(self):
        names = {item.name for item in AnalyzeNoseFrontal().items}
        self.assertEqual(names, {"nose.alar", "nose.nostril"})
        for old in ("nostril", "nose.tip", "dorsal.base", "dorsal.bridge"):
            self.assertNotIn(old, names)


if __name__ == "__main__":
    unittest.main()
