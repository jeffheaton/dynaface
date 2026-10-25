"""
Every registered measure returns exactly its declared outputs (its items that aren't
display_only), whether or not those items are enabled and whether or not it renders.
Display-only items (drawing switches such as tip_proj) are never returned or listed by
get_all_items(). See the measures_base module docstring for the contract. Mirrored by
dynaface-lib-dotnet's DeclaredOutputsTests.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import facial, measures, models  # noqa: E402
from dynaface.image import load_image  # noqa: E402
from dynaface.measures_lateral import AnalyzeLateral  # noqa: E402


class TestDeclaredOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())
        cls.faces = {}
        for view, path in (
            ("frontal", "./tests_data/img1-512.jpg"),
            ("lateral", "./tests_data/img2-1024-right-lateral.jpg"),
        ):
            face = facial.AnalyzeFace(measures=measures.all_measures())
            face.load_image(load_image(path), crop=True)
            face.analyze()  # sets up text placement for rendering
            cls.faces[view] = face

    def test_every_measure_returns_exactly_its_declared_outputs(self):
        for measure in measures.all_measures():
            declared = {i.name for i in measure.items if not i.display_only}
            for view, face in self.faces.items():
                if not (
                    measure.is_lateral if view == "lateral" else measure.is_frontal
                ):
                    continue
                for item in measure.items:  # disabled items are still returned
                    item.enabled = False
                for render in (False, True):
                    with self.subTest(
                        measure=measure.abbrev(), view=view, render=render
                    ):
                        self.assertEqual(
                            set(measure.calc(face, render=render)), declared
                        )

    def test_display_only_items_are_not_outputs(self):
        measure = AnalyzeLateral()
        tip = [i for i in measure.items if i.name == "tip_proj"]
        self.assertTrue(tip and tip[0].display_only)
        r = measure.calc(self.faces["lateral"], render=False)
        self.assertNotIn("tip_proj", r)
        self.assertIn("at_nt", r)

        face = self.faces["lateral"]
        self.assertNotIn("tip_proj", face.get_all_items())
        self.assertIn("at_nt", face.get_all_items())


if __name__ == "__main__":
    unittest.main()
