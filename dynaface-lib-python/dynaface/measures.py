"""The registry of Dynaface's measures."""

import logging
from typing import Any, Dict, Iterable, List

from dynaface.measures_base import MeasureBase
from dynaface.measures_cheek import AnalyzeCheekMeasures
from dynaface.measures_frontal import (
    AnalyzeBrows,
    AnalyzeDentalArea,
    AnalyzeEyeArea,
    AnalyzeFAI,
    AnalyzeIntercanthalDistance,
    AnalyzeLandmarks,
    AnalyzeMouthLength,
    AnalyzeNoseFrontal,
    AnalyzeOralCommissureExcursion,
    AnalyzeOuterEyeCorners,
    AnalyzePose,
    AnalyzePosition,
)
from dynaface.measures_lateral import AnalyzeLateral
from dynaface.measures_skin import AnalyzeSkinTone

logger = logging.getLogger(__name__)


def all_measures() -> List["MeasureBase"]:
    """
    Return a new instance of every Dynaface measure, in the order they run.

    The cheek measures come last, after the landmarks, so the outputs before them
    keep their order (CSV columns included) and their drawing lands over the
    landmark dots.

    Returns:
        List[MeasureBase]: One instance of each measure, all enabled.
    """
    return [
        AnalyzeFAI(),
        AnalyzeOralCommissureExcursion(),
        AnalyzeBrows(),
        AnalyzeDentalArea(),
        AnalyzeEyeArea(),
        AnalyzeIntercanthalDistance(),
        AnalyzeMouthLength(),
        AnalyzeNoseFrontal(),
        AnalyzeOuterEyeCorners(),
        AnalyzeLateral(),
        AnalyzePosition(),
        AnalyzePose(),
        AnalyzeSkinTone(),
        AnalyzeLandmarks(),
        AnalyzeCheekMeasures(),  # last: see the docstring
    ]


def get_all_items(measures: List["MeasureBase"]) -> List[str]:
    """
    List the outputs to show or export from these measures: the output names
    (MeasureItem.output_names()) of the enabled items of the enabled measures, so
    a group's outputs are listed together in its place, and a display-only item
    lists none. AnalyzeFace.get_all_items() gives the same list for a face's own
    measures.

    Args:
        measures (List[MeasureBase]): The measures, such as all_measures().

    Returns:
        List[str]: Output names, such as "fai" and "oce.l".
    """
    return [
        name
        for measure in measures
        if measure.enabled
        for item in measure.items
        if item.enabled
        for name in item.output_names()
    ]


class MeasureSelection:
    """
    Which measures, and which of their items, a user has chosen, kept apart from
    the measures themselves, so that switching between a frontal, a quarter
    (three-quarter) and a lateral face never loses a choice.

    MeasureBase.update_for_type() overwrites every item's enabled flag with whether
    the item suits the face (MeasureItem.suits(): its measure's view, narrowed to
    the item's own views). apply() instead enables what was chosen and suits
    the face, and leaves the choices as they were. Save a selection with to_dict(),
    a JSON-friendly dict keyed by each measure's abbrev() and its items' names, and
    restore it with from_dict(). A measure that a selection doesn't mention isn't
    chosen; an item it doesn't mention is, so an item added to a measure later is
    shown.
    """

    def __init__(self) -> None:
        self._measures: Dict[str, bool] = {}
        self._items: Dict[str, Dict[str, bool]] = {}

    @classmethod
    def of(cls, measures: Iterable[MeasureBase]) -> "MeasureSelection":
        """
        The choices that the measures' enabled flags hold now, such as after a
        user ticks them: each measure's own flag, and its items' flags if it is
        enabled. A measure switched off with set_enabled(False) has every item
        switched off too, which isn't a choice of items, so choosing it again
        later shows all of them.

        Use it on measures whose flags hold the user's choices, not on ones that
        apply() has set for a face, which hold only what suits that face.
        """
        selection = cls()
        for measure in measures:
            selection.choose(measure, measure.enabled)
            if measure.enabled:
                for item in measure.items:
                    selection.choose_item(measure, item.name, item.enabled)
        return selection

    def is_chosen(self, measure: MeasureBase) -> bool:
        """Whether the measure is chosen; a measure never mentioned isn't."""
        return self._measures.get(measure.abbrev(), False)

    def is_item_chosen(self, measure: MeasureBase, item: str) -> bool:
        """Whether one of the measure's items is chosen; an item never mentioned is."""
        return self._items.get(measure.abbrev(), {}).get(item, True)

    def choose(self, measure: MeasureBase, chosen: bool = True) -> None:
        """Choose a measure, or with chosen=False, unchoose it."""
        self._measures[measure.abbrev()] = chosen

    def choose_item(self, measure: MeasureBase, item: str, chosen: bool = True) -> None:
        """Choose one of a measure's items, or with chosen=False, unchoose it."""
        self._items.setdefault(measure.abbrev(), {})[item] = chosen

    def apply(
        self, measures: Iterable[MeasureBase], lateral: bool, quarter: bool = False
    ) -> None:
        """
        Set the measures' and items' enabled flags for a frontal, a quarter
        (three-quarter) or a lateral face: on where chosen and the measure suits
        the face (MeasureBase.suits()), and for an item, where it suits the face
        too (MeasureItem.suits(), which also honors an item's own views), off
        everywhere else. It calls each measure's sync_items() first, so the items'
        flags follow the measure's as they are now. The choices stay as they were,
        so applying another view later brings them back.

        Args:
            measures (Iterable[MeasureBase]): The measures to set, such as a face's.
            lateral (bool): True for a lateral face.
            quarter (bool): True for a quarter face (face.quarter); ignored when
                lateral is True. False, the default, keeps the two-argument calls
                as they were.
        """
        for measure in measures:
            measure.sync_items()
            measure.enabled = measure.suits(lateral, quarter) and self.is_chosen(
                measure
            )
            for item in measure.items:
                item.enabled = (
                    measure.enabled
                    and item.suits(lateral, quarter)
                    and self.is_item_chosen(measure, item.name)
                )

    def to_dict(self) -> Dict[str, Any]:
        """
        The selection as {abbrev: {"chosen": bool, "items": {name: bool}}}, one
        entry for each measure it mentions, ready for json.dump().
        """
        names = list(dict.fromkeys([*self._measures, *self._items]))
        return {
            name: {
                "chosen": self._measures.get(name, False),
                "items": dict(self._items.get(name, {})),
            }
            for name in names
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeasureSelection":
        """A selection saved with to_dict()."""
        selection = cls()
        for name, entry in data.items():
            selection._measures[name] = bool(entry.get("chosen", False))
            selection._items[name] = {
                item: bool(chosen) for item, chosen in entry.get("items", {}).items()
            }
        return selection
