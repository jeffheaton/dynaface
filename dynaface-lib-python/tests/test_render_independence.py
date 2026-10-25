"""
Every measure must return the same values whether or not it draws: rendering only
shows results, it must not decide whether they are computed. (Tilt used to be 0
unless its label was drawn, and skin tone was only computed while drawing.)
Mirrored by dynaface-lib-dotnet's RenderIndependenceTests.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import facial, measures, models  # noqa: E402
from dynaface.image import load_image  # noqa: E402


def same(a, b):
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    return a == b


class TestRenderIndependence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())

    def check(self, path):
        face = facial.AnalyzeFace(measures=measures.all_measures())
        face.load_image(load_image(path), crop=True)
        off = {}
        for m in face.measures:
            if m.enabled:
                off.update(m.calc(face, render=False))
        on = face.analyze()
        self.assertEqual(set(off), set(on))
        for key in on:
            with self.subTest(image=os.path.basename(path), key=key):
                self.assertTrue(same(off[key], on[key]), f"{off[key]} != {on[key]}")
        return on

    def test_frontal(self):
        on = self.check("./tests_data/img1-512.jpg")
        self.assertNotEqual(on["tilt"], 0.0)
        self.assertIsNotNone(on["hue"])

    def test_lateral(self):
        self.check("./tests_data/img2-1024-right-lateral.jpg")


if __name__ == "__main__":
    unittest.main()
