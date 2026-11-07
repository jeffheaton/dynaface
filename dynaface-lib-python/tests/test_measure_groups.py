"""
Grouped and view-limited measure items (the measures_base module docstring): a
group item is a switch named for what it draws that stands for several outputs,
and an item declared with views suits only those views of its measure. Toy
measures, no models. Mirrored by dynaface-lib-dotnet once it ports the groups.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import video  # noqa: E402
from dynaface.const import Pose  # noqa: E402
from dynaface.measures import MeasureSelection, get_all_items  # noqa: E402
from dynaface.measures_base import (  # noqa: E402
    MeasureBase,
    MeasureItem,
    select_outputs,
)


class Grouped(MeasureBase):
    """A frontal and quarter measure with a plain output, two groups (one for each
    view), a group for both views and a drawing switch."""

    def __init__(self):
        super().__init__()
        self.items = [
            MeasureItem("plain"),
            MeasureItem("Width", outputs=["w.r", "w.l", "w"], views=[Pose.FRONTAL]),
            MeasureItem("Apex", outputs=["a.h", "a.ok"], views=[Pose.QUARTER]),
            MeasureItem(
                "Turn", outputs=["yaw", "pitch"], views=[Pose.FRONTAL, Pose.QUARTER]
            ),
            MeasureItem("switch", display_only=True),
        ]
        self.is_frontal, self.is_quarter = True, True
        self.sync_items()

    def abbrev(self):
        return "Grouped"

    def calc(self, face, render=True):
        data = {name: 1.0 for name in self.outputs()}
        self.drawn = [i.name for i in self.items if render and i.enabled]
        return select_outputs(data, self.items)


class Plain(MeasureBase):
    """A frontal measure with plain items only, as every measure was before
    groups."""

    def __init__(self):
        super().__init__()
        self.items = [MeasureItem("a"), MeasureItem("b"), MeasureItem("t", True, True)]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self):
        return "Plain"


class Unsynced(MeasureBase):
    """A frontal measure that sets its flags but never calls sync_items(), as a
    plugin measure written before item views might."""

    def __init__(self):
        super().__init__()
        self.items = [MeasureItem("a"), MeasureItem("b")]
        self.is_frontal = True

    def abbrev(self):
        return "Unsynced"


GROUPED_OUTPUTS = ["plain", "w.r", "w.l", "w", "a.h", "a.ok", "yaw", "pitch"]


class TestOutputNames(unittest.TestCase):
    def test_kinds_of_item(self):
        self.assertEqual(MeasureItem("fai").output_names(), ["fai"])
        self.assertEqual(MeasureItem("tip", display_only=True).output_names(), [])
        group = MeasureItem("Width", outputs=("w.r", "w.l"))
        self.assertEqual(group.output_names(), ["w.r", "w.l"])
        # A copy: changing it doesn't change the item.
        group.output_names().append("x")
        self.assertEqual(group.output_names(), ["w.r", "w.l"])
        # The group's name is a label, never an output.
        self.assertNotIn("Width", group.output_names())
        self.assertEqual(MeasureItem("Empty", outputs=[]).output_names(), [])

    def test_measure_outputs_in_order(self):
        self.assertEqual(Grouped().outputs(), GROUPED_OUTPUTS)
        self.assertEqual(Plain().outputs(), ["a", "b"])

    def test_select_outputs_expands_groups_enabled_or_not(self):
        measure = Grouped()
        for item in measure.items:
            item.enabled = False
        result = select_outputs({"w.r": 2.0, "plain": 3.0}, measure.items)
        self.assertEqual(list(result), GROUPED_OUTPUTS)
        self.assertEqual(result["w.r"], 2.0)
        self.assertEqual(result["plain"], 3.0)
        self.assertIsNone(result["a.h"])  # not computed
        self.assertNotIn("Width", result)
        self.assertNotIn("switch", result)


class TestGetAllItems(unittest.TestCase):
    def test_groups_expand(self):
        measure = Grouped()
        self.assertEqual(get_all_items([measure]), GROUPED_OUTPUTS)

    def test_an_off_group_drops_all_its_outputs_but_calc_returns_them(self):
        measure = Grouped()
        measure.set_item_enabled("Width", False)
        self.assertFalse(measure.is_enabled("Width"))
        self.assertEqual(
            get_all_items([measure]), ["plain", "a.h", "a.ok", "yaw", "pitch"]
        )
        result = measure.calc(None, render=True)
        self.assertEqual(list(result), GROUPED_OUTPUTS)
        self.assertNotIn("Width", measure.drawn)  # an off group isn't drawn

    def test_a_disabled_measure_lists_nothing(self):
        measure = Grouped()
        measure.set_enabled(False)
        self.assertEqual(get_all_items([measure]), [])


class TestViews(unittest.TestCase):
    def flags(self, item):
        return (item.is_frontal, item.is_quarter, item.is_lateral)

    def test_sync_items_narrows_to_the_items_views(self):
        measure = Grouped()
        flags = {item.name: self.flags(item) for item in measure.items}
        self.assertEqual(flags["plain"], (True, True, False))
        self.assertEqual(flags["Width"], (True, False, False))
        self.assertEqual(flags["Apex"], (False, True, False))
        self.assertEqual(flags["Turn"], (True, True, False))
        self.assertEqual(flags["switch"], (True, True, False))
        # Idempotent, and never wider than the measure: a view the measure
        # doesn't suit stays off for an item that names it.
        measure.sync_items()
        self.assertEqual({i.name: self.flags(i) for i in measure.items}, flags)
        measure.is_quarter = False
        measure.sync_items()
        self.assertEqual(self.flags(measure.items[2]), (False, False, False))
        measure.is_quarter = True
        measure.sync_items()
        self.assertEqual({i.name: self.flags(i) for i in measure.items}, flags)

    def test_suits(self):
        width, apex, turn = Grouped().items[1:4]
        self.assertTrue(width.suits(False))
        self.assertFalse(width.suits(False, quarter=True))
        self.assertFalse(width.suits(True))
        self.assertFalse(width.suits(True, quarter=True))  # lateral first
        self.assertFalse(apex.suits(False))
        self.assertTrue(apex.suits(False, quarter=True))
        self.assertTrue(turn.suits(False) and turn.suits(False, quarter=True))

    def test_update_for_type(self):
        measure = Grouped()
        measure.update_for_type(lateral=False, quarter=True)
        enabled = {i.name: i.enabled for i in measure.items}
        self.assertEqual(
            enabled,
            {"plain": True, "Width": False, "Apex": True, "Turn": True, "switch": True},
        )
        self.assertEqual(
            get_all_items([measure]), ["plain", "a.h", "a.ok", "yaw", "pitch"]
        )
        measure.update_for_type(lateral=False)
        self.assertFalse(measure.is_enabled("Apex"))
        self.assertTrue(measure.is_enabled("Width"))
        measure.update_for_type(lateral=True)
        self.assertFalse(any(i.enabled for i in measure.items))

    def test_selection_apply(self):
        measure = Grouped()
        selection = MeasureSelection()
        selection.choose(measure)
        selection.choose_item(measure, "Turn", False)
        selection.apply([measure], lateral=False, quarter=True)
        self.assertTrue(measure.enabled)
        self.assertEqual(
            {i.name: i.enabled for i in measure.items},
            {
                "plain": True,
                "Width": False,  # chosen, but not on a quarter face
                "Apex": True,
                "Turn": False,  # unchosen
                "switch": True,
            },
        )
        selection.apply([measure], lateral=False)
        self.assertTrue(measure.is_enabled("Width"))
        self.assertFalse(measure.is_enabled("Apex"))
        self.assertFalse(measure.is_enabled("Turn"))
        # The choices stay as they were, keyed by the group's label.
        self.assertTrue(selection.is_item_chosen(measure, "Width"))
        self.assertFalse(selection.is_item_chosen(measure, "Turn"))
        self.assertEqual(
            selection.to_dict()["Grouped"], {"chosen": True, "items": {"Turn": False}}
        )

    def test_frame_series_nans_an_unsuited_items_outputs(self):
        class Face:
            lateral, quarter = False, True

            def __init__(self, measures):
                self.measures = measures

        measure = Grouped()
        results = measure.calc(None, render=False)
        in_view = video._results_in_view(Face([measure]), results)
        for name in ("w.r", "w.l", "w"):
            self.assertTrue(math.isnan(in_view[name]), name)
        for name in ("plain", "a.h", "a.ok", "yaw", "pitch"):
            self.assertEqual(in_view[name], 1.0, name)
        Face.quarter = False
        in_view = video._results_in_view(Face([measure]), results)
        self.assertTrue(math.isnan(in_view["a.h"]))
        self.assertEqual(in_view["w"], 1.0)


class TestWithoutGroups(unittest.TestCase):
    """A measure of plain items behaves exactly as before."""

    def test_unchanged(self):
        measure = Plain()
        for item in measure.items:
            self.assertIsNone(item.outputs)
            self.assertIsNone(item.views)
            self.assertEqual(self.flags(item), (True, False, False))
        self.assertEqual(get_all_items([measure]), ["a", "b"])
        self.assertEqual(
            select_outputs({"a": 1, "b": 2, "t": 3}, measure.items), {"a": 1, "b": 2}
        )
        measure.update_for_type(lateral=False, quarter=True)
        self.assertEqual([i.enabled for i in measure.items], [False] * 3)
        measure.update_for_type(lateral=False)
        self.assertEqual([i.enabled for i in measure.items], [True] * 3)
        selection = MeasureSelection()
        selection.choose(measure)
        selection.choose_item(measure, "b", False)
        selection.apply([measure], lateral=False)
        self.assertEqual([i.enabled for i in measure.items], [True, False, True])
        selection.apply([measure], lateral=False, quarter=True)
        self.assertFalse(measure.enabled)
        self.assertEqual([i.enabled for i in measure.items], [False] * 3)

    @staticmethod
    def flags(item):
        return (item.is_frontal, item.is_quarter, item.is_lateral)


class TestWithoutSyncItems(unittest.TestCase):
    """A measure that never calls sync_items(), or gains an item after it, still
    has its items follow its view: update_for_type(), MeasureSelection.apply()
    and the series' view rule sync the items before reading their flags."""

    class Face:
        lateral, quarter = False, False

        def __init__(self, measures):
            self.measures = measures

    def test_update_for_type(self):
        measure = Unsynced()
        measure.update_for_type(lateral=False)
        self.assertEqual([i.enabled for i in measure.items], [True, True])
        measure.update_for_type(lateral=False, quarter=True)
        self.assertEqual([i.enabled for i in measure.items], [False, False])

    def test_selection_apply(self):
        measure = Unsynced()
        selection = MeasureSelection()
        selection.choose(measure)
        selection.apply([measure], lateral=False)
        self.assertTrue(measure.enabled)
        self.assertEqual([i.enabled for i in measure.items], [True, True])
        self.assertEqual(get_all_items([measure]), ["a", "b"])

    def test_frame_series(self):
        measure = Unsynced()
        in_view = video._results_in_view(self.Face([measure]), {"a": 1.0, "b": 2.0})
        self.assertEqual(in_view, {"a": 1.0, "b": 2.0})

    def test_an_unsuited_measure_nans_every_output(self):
        # A disabled item's outputs too, not only the enabled items'.
        measure = Plain()
        measure.set_item_enabled("b", False)
        face = self.Face([measure])
        face.quarter = True
        in_view = video._results_in_view(face, {"a": 1.0, "b": 2.0})
        self.assertTrue(math.isnan(in_view["a"]))
        self.assertTrue(math.isnan(in_view["b"]))

    def test_an_item_added_after_sync_items(self):
        measure = Plain()
        measure.items.append(MeasureItem("c"))
        measure.update_for_type(lateral=False)
        self.assertEqual([i.name for i in measure.items if i.enabled], list("abtc"))
        in_view = video._results_in_view(self.Face([measure]), {"c": 3.0})
        self.assertEqual(in_view["c"], 3.0)


if __name__ == "__main__":
    unittest.main()
