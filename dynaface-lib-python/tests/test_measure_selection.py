"""
MeasureSelection keeps a user's choice of measures and items when the face switches
between frontal and lateral, which update_for_type() can't, and saves and loads it by
name. Mirrored by dynaface-lib-dotnet's MeasureSelectionTests.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import facial, models  # noqa: E402
from dynaface.measures import (  # noqa: E402
    MeasureSelection,
    all_measures,
    get_all_items,
)
from dynaface.measures_frontal import AnalyzeEyeArea  # noqa: E402
from dynaface.measures_lateral import AnalyzeLateral  # noqa: E402

# The outputs of the measures that ticked() leaves on, for each view.
FRONTAL = ["fai", "eye.left", "eye.right", "eye.ratio", "pitch", "roll", "yaw"]
LATERAL = ["nfa", "at", "nt", "at_nt", "pitch", "roll", "yaw"]
ALL_EYES = ["eye.left", "eye.right", "eye.diff", "eye.ratio"]


def find(measures, kind):
    return next(measure for measure in measures if isinstance(measure, kind))


def ticked(measures):
    """
    The measures as a user might tick them: FAI, the eye area without eye.diff, the
    head pose, and the lateral measures without nla.
    """
    keep = {"FAI", "Eye Area", "Pose", "Lateral Measures"}
    for measure in measures:
        measure.set_enabled(measure.abbrev() in keep)
    find(measures, AnalyzeEyeArea).set_item_enabled("eye.diff", False)
    find(measures, AnalyzeLateral).set_item_enabled("nla", False)
    return measures


class TestMeasureSelection(unittest.TestCase):
    def test_switching_views_keeps_the_choices(self):
        measures = ticked(all_measures())
        selection = MeasureSelection.of(measures)
        for lateral, expected in ((False, FRONTAL), (True, LATERAL), (False, FRONTAL)):
            selection.apply(measures, lateral=lateral)
            self.assertEqual(get_all_items(measures), expected)

    def test_update_for_type_loses_them(self):
        """What MeasureSelection is for: update_for_type() turns eye.diff back on."""
        measures = ticked(all_measures())
        for lateral in (True, False):
            for measure in measures:
                measure.update_for_type(lateral)
        self.assertIn("eye.diff", get_all_items(measures))

    def test_a_measure_that_does_not_suit_the_face_is_off(self):
        measures = ticked(all_measures())
        MeasureSelection.of(measures).apply(measures, lateral=True)
        eyes = find(measures, AnalyzeEyeArea)
        self.assertFalse(eyes.enabled)
        self.assertFalse(any(item.enabled for item in eyes.items))

    def test_unmentioned_measures_are_off_and_unmentioned_items_on(self):
        measures = all_measures()
        MeasureSelection().apply(measures, lateral=False)
        self.assertEqual(get_all_items(measures), [])

        selection = MeasureSelection()
        selection.choose(find(measures, AnalyzeEyeArea))
        selection.apply(measures, lateral=False)
        self.assertEqual(get_all_items(measures), ALL_EYES)

    def test_choose_and_unchoose(self):
        eyes = AnalyzeEyeArea()
        selection = MeasureSelection()
        self.assertFalse(selection.is_chosen(eyes))
        self.assertTrue(selection.is_item_chosen(eyes, "eye.diff"))

        selection.choose(eyes)
        selection.choose_item(eyes, "eye.diff", False)
        self.assertTrue(selection.is_chosen(eyes))
        self.assertFalse(selection.is_item_chosen(eyes, "eye.diff"))

        selection.choose(eyes, False)
        self.assertFalse(selection.is_chosen(eyes))
        self.assertFalse(selection.is_item_chosen(eyes, "eye.diff"))

    def test_a_switched_off_measure_shows_all_its_items_when_chosen_again(self):
        measures = all_measures()
        eyes = find(measures, AnalyzeEyeArea)
        eyes.set_enabled(False)  # which switches off every item too
        selection = MeasureSelection.of(measures)
        selection.choose(eyes)
        selection.apply(measures, lateral=False)
        self.assertEqual([item.name for item in eyes.items if item.enabled], ALL_EYES)

    def test_saved_by_name(self):
        saved = MeasureSelection.of(ticked(all_measures())).to_dict()
        self.assertEqual(
            saved["Eye Area"],
            {
                "chosen": True,
                "items": {
                    "eye.left": True,
                    "eye.right": True,
                    "eye.diff": False,
                    "eye.ratio": True,
                },
            },
        )
        self.assertEqual(saved["Brow"], {"chosen": False, "items": {}})
        self.assertEqual(list(saved)[:2], ["FAI", "Oral CE"])

    def test_save_and_load(self):
        selection = MeasureSelection.of(ticked(all_measures()))
        loaded = MeasureSelection.from_dict(json.loads(json.dumps(selection.to_dict())))
        self.assertEqual(loaded.to_dict(), selection.to_dict())

        measures = all_measures()  # every measure and item on, as made
        for lateral, expected in ((False, FRONTAL), (True, LATERAL)):
            loaded.apply(measures, lateral=lateral)
            self.assertEqual(get_all_items(measures), expected)

    def test_items_mentioned_without_their_measure(self):
        eyes = AnalyzeEyeArea()
        selection = MeasureSelection()
        selection.choose_item(eyes, "eye.diff", False)
        self.assertEqual(
            selection.to_dict(),
            {"Eye Area": {"chosen": False, "items": {"eye.diff": False}}},
        )
        self.assertFalse(selection.is_chosen(eyes))


class TestMeasureSelectionAnalysis(unittest.TestCase):
    """apply() decides which measures analyze() runs."""

    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())

    def test_one_selection_for_frontal_photos_and_profiles(self):
        chosen = ticked(all_measures())
        selection = MeasureSelection.of(chosen)
        # Each measure that runs returns all its outputs, items on or off.
        frontal_run = {*FRONTAL, "eye.diff"}
        lateral_run = {*LATERAL, "nla"}
        for path, lateral, shown, run in (
            ("./tests_data/img1-512.jpg", False, FRONTAL, frontal_run),
            ("./tests_data/img2-1024-right-lateral.jpg", True, LATERAL, lateral_run),
            ("./tests_data/img1-512.jpg", False, FRONTAL, frontal_run),
        ):
            with self.subTest(path=path):
                face = facial.load_face_image(path, measures=chosen)
                self.assertEqual(face.lateral, lateral)
                selection.apply(face.measures, lateral=face.lateral)
                results = face.analyze(render=False)
                self.assertEqual(set(results), run)
                self.assertEqual(face.get_all_items(), shown)


if __name__ == "__main__":
    unittest.main()
