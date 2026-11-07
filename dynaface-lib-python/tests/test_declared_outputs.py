"""
Every registered measure returns exactly its declared outputs (measure.outputs(): its
items' output names, a group's outputs listed in its place, display_only items none),
whether or not those items are enabled and whether or not it renders, on each view it
suits: a frontal photo, a profile and a three-quarter (quarter) face. Display-only
items (drawing switches such as tip_proj) are never returned or listed by
get_all_items(). See the measures_base module docstring for the contract. Mirrored by
dynaface-lib-dotnet's DeclaredOutputsTests.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import facial, measures, models  # noqa: E402
from dynaface.const import Pose  # noqa: E402
from dynaface.image import load_image  # noqa: E402
from dynaface.measures_lateral import AnalyzeLateral  # noqa: E402

# Which measures each view picks: the flag that MeasureBase.suits() reads for it.
SUITS = {
    "frontal": lambda measure: measure.is_frontal,
    "lateral": lambda measure: measure.is_lateral,
    "quarter": lambda measure: measure.is_quarter,
}


class TestDeclaredOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())
        cls.faces = {}
        for view, path, force_pose in (
            ("frontal", "./tests_data/img1-512.jpg", Pose.DETECT),
            ("lateral", "./tests_data/img2-1024-right-lateral.jpg", Pose.DETECT),
            # The app's three-quarter sketch, forced: the quarter path with the
            # far cheek's outline traced. At the band-fit crop the background
            # model drops the hair beside the far cheek, so nearly every row
            # comes from the mask (the hair case is tests/test_cheek_photos.py's
            # yaw45_hair render).
            ("quarter", "./tests_data/pose-quarter-512.png", Pose.QUARTER),
        ):
            face = facial.AnalyzeFace(measures=measures.all_measures())
            face.load_image(load_image(path), crop=True, force_pose=force_pose)
            face.analyze()  # sets up text placement for rendering
            cls.faces[view] = face

    def test_the_views(self):
        self.assertEqual(set(self.faces), set(SUITS))
        self.assertFalse(self.faces["frontal"].lateral or self.faces["frontal"].quarter)
        self.assertTrue(self.faces["lateral"].lateral)
        self.assertTrue(self.faces["quarter"].quarter)
        self.assertEqual(self.faces["quarter"].pose, Pose.QUARTER)
        self.assertFalse(self.faces["quarter"].lateral)
        # The quarter view's measures: the ones that need no millimeter.
        self.assertEqual(
            [m.abbrev() for m in measures.all_measures() if SUITS["quarter"](m)],
            ["Position", "Pose", "Landmarks", "Cheek Measures"],
        )

    def test_every_measure_returns_exactly_its_declared_outputs(self):
        for measure in measures.all_measures():
            declared = set(measure.outputs())
            for view, face in self.faces.items():
                if not SUITS[view](measure):
                    continue
                for item in measure.items:  # disabled items are still returned
                    item.enabled = False
                for render in (False, True):
                    with self.subTest(
                        measure=measure.abbrev(), view=view, render=render
                    ):
                        result = measure.calc(face, render=render)
                        self.assertEqual(set(result), declared)
                        # In declared order, a group's outputs together.
                        self.assertEqual(list(result), measure.outputs())

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
