"""
MeasureSelection keeps a user's choice of measures and items when the face switches
between frontal, quarter (three-quarter) and lateral, which update_for_type() can't,
and saves and loads it by name, a group item by its label, and keeps an item off on
a view it doesn't suit. Mirrored by dynaface-lib-dotnet's
MeasureSelectionTests.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import facial, models  # noqa: E402
from dynaface.const import Pose  # noqa: E402
from dynaface.image import load_image  # noqa: E402
from dynaface.measures import (  # noqa: E402
    MeasureSelection,
    all_measures,
    get_all_items,
)
from dynaface.measures_cheek import ITEMS as CHEEK_ITEMS  # noqa: E402
from dynaface.measures_cheek import AnalyzeCheekMeasures  # noqa: E402
from dynaface.measures_frontal import (  # noqa: E402
    AnalyzeEyeArea,
    AnalyzeLandmarks,
    AnalyzePose,
    AnalyzePosition,
)
from dynaface.measures_lateral import AnalyzeLateral  # noqa: E402

# The outputs of the measures that ticked() leaves on, for each view.
FRONTAL = ["fai", "eye.left", "eye.right", "eye.ratio", "pitch", "roll", "yaw"]
LATERAL = ["nfa", "at", "nt", "at_nt", "pitch", "roll", "yaw"]
# On a quarter face, only the pose suits it among the ticked measures.
QUARTER = ["pitch", "roll", "yaw"]
ALL_EYES = ["eye.left", "eye.right", "eye.diff", "eye.ratio"]
# The measures a quarter face can use: no millimeter among their outputs.
QUARTER_MEASURES = ["Position", "Pose", "Landmarks", "Cheek Measures"]
# The Cheek Measures items' outputs, by item.
CHEEK = {name: list(item_outputs) for name, _, item_outputs in CHEEK_ITEMS}


def outputs(measure):
    """The names of a measure's outputs, a group's listed in its place."""
    return measure.outputs()


def cheek_outputs(*names):
    """The outputs of these Cheek Measures items, in the measure's order."""
    return [o for name, _, item in CHEEK_ITEMS if name in names for o in item]


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


class TestMeasureSelectionQuarter(unittest.TestCase):
    """
    The third view: apply(measures, lateral=False, quarter=True) enables what is
    chosen among the measures that suit a quarter face, the two-argument calls
    keep their meaning, and a selection naming the cheek measures saves and
    loads by name, their items by label.
    """

    def test_quarter_view_enables_the_measures_that_suit_it(self):
        measures = all_measures()  # every measure and item on, as made
        MeasureSelection.of(measures).apply(measures, lateral=False, quarter=True)
        self.assertEqual(
            [measure.abbrev() for measure in measures if measure.enabled],
            QUARTER_MEASURES,
        )
        for measure in measures:
            with self.subTest(measure.abbrev()):
                self.assertEqual(measure.suits(False, True), measure.is_quarter)
                self.assertEqual(measure.enabled, measure.is_quarter)
                # Each item on where it suits the view: the measure's, and for
                # the Cheek Measures' view-limited items, their own.
                self.assertEqual(
                    [item.enabled for item in measure.items],
                    [measure.is_quarter and item.is_quarter for item in measure.items],
                )
        expected = []
        for kind in (AnalyzePosition, AnalyzePose, AnalyzeLandmarks):
            expected += outputs(find(measures, kind))
        # The Landmarks measure's Cheek Landmarks group is listed as its outputs.
        self.assertEqual(expected[-20:][:2], ["zy.r.x", "zy.r.y"])
        quarter_items = (
            "Far cheek outline",
            "Malar apex",
            "Submalar point",
            "Head turn",
        )
        expected += cheek_outputs(*quarter_items)
        self.assertEqual(get_all_items(measures), expected)
        # The frontal items are off on a quarter face, so their outputs (and no
        # item's label) are listed.
        cheek = find(measures, AnalyzeCheekMeasures)
        self.assertTrue(cheek.enabled)
        self.assertEqual(len(outputs(cheek)), 41)
        self.assertEqual(
            [item.name for item in cheek.items if item.enabled], list(quarter_items)
        )
        for name in CHEEK["Zygion width"] + [item.name for item in cheek.items]:
            self.assertNotIn(name, get_all_items(measures))

    def test_two_argument_calls_give_the_old_lists(self):
        measures = ticked(all_measures())
        selection = MeasureSelection.of(measures)
        for kwargs, expected in (
            ({"lateral": False}, FRONTAL),
            ({"lateral": True}, LATERAL),
            ({"lateral": False, "quarter": True}, QUARTER),
            ({"lateral": True, "quarter": True}, LATERAL),  # quarter is ignored
            ({"lateral": False}, FRONTAL),  # the choices came back
            ({"lateral": False, "quarter": False}, FRONTAL),
        ):
            with self.subTest(**kwargs):
                selection.apply(measures, **kwargs)
                self.assertEqual(get_all_items(measures), expected)
        # update_for_type() keeps its one-argument meaning too, and with
        # quarter=True overwrites the items for the quarter view.
        for measure in measures:
            measure.update_for_type(False)
            self.assertEqual(
                any(item.enabled for item in measure.items),
                measure.is_frontal and bool(measure.items),
                measure.abbrev(),
            )
            measure.update_for_type(False, True)
            self.assertEqual(
                any(item.enabled for item in measure.items),
                measure.is_quarter and bool(measure.items),
                measure.abbrev(),
            )

    def test_a_selection_naming_the_cheek_measures_round_trips(self):
        cheek = AnalyzeCheekMeasures()
        selection = MeasureSelection()
        selection.choose(AnalyzePose())
        selection.choose(cheek)
        selection.choose_item(cheek, "Malar apex", False)
        saved = json.loads(json.dumps(selection.to_dict()))
        self.assertEqual(
            saved["Cheek Measures"], {"chosen": True, "items": {"Malar apex": False}}
        )
        loaded = MeasureSelection.from_dict(saved)
        self.assertEqual(loaded.to_dict(), selection.to_dict())
        self.assertTrue(loaded.is_chosen(cheek))
        self.assertFalse(loaded.is_item_chosen(cheek, "Malar apex"))
        self.assertTrue(loaded.is_item_chosen(cheek, "Submalar point"))

        measures = all_measures()
        on_quarter = cheek_outputs("Far cheek outline", "Submalar point", "Head turn")
        on_frontal = cheek_outputs("Zygion width", "Reference points", "Head turn")
        # The cheek measures suit a frontal face and a quarter face, and never a
        # profile; each item only the views it suits, and the unchosen apex
        # neither.
        for kwargs, expected in (
            ({"lateral": False, "quarter": True}, QUARTER + on_quarter),
            ({"lateral": False}, QUARTER + on_frontal),
            ({"lateral": True}, QUARTER),
            ({"lateral": False, "quarter": True}, QUARTER + on_quarter),
        ):
            with self.subTest(**kwargs):
                loaded.apply(measures, **kwargs)
                self.assertEqual(get_all_items(measures), expected)
        items = find(measures, AnalyzeCheekMeasures).items
        self.assertEqual(
            [(item.name, item.enabled) for item in items],
            [
                ("Zygion width", False),
                ("Reference points", False),
                ("Far cheek outline", True),
                ("Malar apex", False),
                ("Submalar point", True),
                ("Head turn", True),
            ],
        )
        # The choice of the unsuited items is kept for the next frontal face.
        self.assertTrue(loaded.is_item_chosen(cheek, "Zygion width"))

    def test_a_selection_that_never_mentions_them_leaves_them_unchosen(self):
        # A selection saved before the cheek measures existed: of() mentions
        # every measure, so the entry is dropped to make such a file.
        saved = MeasureSelection.of(ticked(all_measures())).to_dict()
        self.assertEqual(saved.pop("Cheek Measures"), {"chosen": False, "items": {}})
        self.assertNotIn("Cheek Measures", saved)
        loaded = MeasureSelection.from_dict(json.loads(json.dumps(saved)))
        self.assertFalse(loaded.is_chosen(AnalyzeCheekMeasures()))
        measures = all_measures()
        loaded.apply(measures, lateral=False, quarter=True)
        self.assertEqual(get_all_items(measures), QUARTER)
        cheek = find(measures, AnalyzeCheekMeasures)
        self.assertFalse(cheek.enabled)
        self.assertFalse(any(item.enabled for item in cheek.items))


class TestMeasureSelectionAnalysis(unittest.TestCase):
    """apply() decides which measures analyze() runs."""

    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())

    def test_one_selection_for_a_quarter_face(self):
        # The app's three-quarter sketch, forced to the quarter path: the
        # ticked selection plus the cheek measures runs the pose and the cheek
        # measures there, and the same selection runs the frontal measures on
        # the frontal photo afterward.
        chosen = ticked(all_measures())
        cheek = find(chosen, AnalyzeCheekMeasures)
        cheek.set_enabled(True)
        cheek.set_item_enabled("Malar apex", False)
        selection = MeasureSelection.of(chosen)
        all_cheek = outputs(cheek)
        face = facial.AnalyzeFace(measures=chosen)
        face.load_image(
            load_image("./tests_data/pose-quarter-512.png"),
            crop=True,
            force_pose=Pose.QUARTER,
        )
        self.assertTrue(face.quarter)
        self.assertFalse(face.lateral)
        selection.apply(face.measures, lateral=face.lateral, quarter=face.quarter)
        results = face.analyze(render=False)
        self.assertEqual(set(results), {*QUARTER, *all_cheek})
        self.assertEqual(
            face.get_all_items(),
            QUARTER + cheek_outputs("Far cheek outline", "Submalar point", "Head turn"),
        )
        self.assertEqual(results["cheek.far"], 0.0)
        # The unchosen item, the measures that don't suit the view and the
        # cheek measure's frontal items are off; every output is still returned
        # (the frontal ones NaN on a quarter face).
        for name in ("apex.ok", "zy.w", "cr.ok", "fai", "eye.left"):
            self.assertNotIn(name, face.get_all_items())
        self.assertTrue(results["zy.w"] != results["zy.w"])  # NaN on a quarter face
        self.assertTrue(results["apex.r.h"] == results["apex.r.h"])  # not NaN

        face = facial.load_face_image("./tests_data/img1-512.jpg", measures=chosen)
        self.assertFalse(face.quarter or face.lateral)
        selection.apply(face.measures, lateral=face.lateral, quarter=face.quarter)
        results = face.analyze(render=False)
        self.assertEqual(set(results), {*FRONTAL, "eye.diff", *all_cheek})
        self.assertEqual(
            face.get_all_items(),
            FRONTAL + cheek_outputs("Zygion width", "Reference points", "Head turn"),
        )

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
