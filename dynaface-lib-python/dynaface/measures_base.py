"""
Measurement plugin base classes.

Contract for measures (mirrored by dynaface-lib-dotnet's FaceMeasureBase):

- calc() always returns every declared output, whether or not its item is enabled
  and whether or not render is True. A value that can't be determined is NaN, or
  None when the measure couldn't run at all. MeasureBase.outputs() lists them.
- An item's enabled flag gates its drawing and, for callers that honor it, which
  outputs are shown or exported: AnalyzeFace.get_all_items() lists the outputs
  of the enabled items, which is what the desktop app uses for its chart and CSV
  columns.
- An item marked display_only (such as AnalyzeLateral's tip_proj) is a drawing
  switch, not an output: calc() never returns it and get_all_items() never lists it.
- An item declared with outputs is a group: a switch named for what it draws
  (such as the Landmarks measure's "Cheek Landmarks", or the Cheek Measures'
  "Zygion width"), whose name is a label shown in lists and kept by
  MeasureSelection and saved documents, never an output. It stands for its
  outputs, which are returned, listed and exported together
  (MeasureItem.output_names()), and turning it off drops all of them from
  get_all_items() but not from calc().
- A measure declares the views it suits with is_frontal, is_lateral and
  is_quarter (a three-quarter face, Pose.QUARTER, whose far cheek the cheek
  measures read; no millimeter is defined on a turned head, so only the
  Position, Pose, Landmarks and Cheek Measures measures suit it). suits(lateral,
  quarter) reads them with one rule, "lateral, else quarter, else frontal", and
  update_for_type() and dynaface.measures.MeasureSelection.apply() enable a
  measure through it. Each item carries the same three flags, copied by
  sync_items(), which update_for_type(), MeasureSelection.apply() and
  FrameSeries call before reading them, so a measure that sets its flags without
  calling it, or adds an item afterward, still has its items follow its view; an
  item declared with views (some of Pose.FRONTAL, Pose.QUARTER
  and Pose.LATERAL) suits only those views of its measure, so sync_items() keeps
  a flag only where the view is among them, and MeasureItem.suits() reads the
  item's flags with the same rule. update_for_type() and MeasureSelection.apply()
  then leave such an item off on a view it doesn't suit, and
  dynaface.video.FrameSeries reports its outputs as NaN there.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dynaface.const import Pose

logger = logging.getLogger(__name__)


def select_outputs(data: Dict[str, Any], items: List["MeasureItem"]) -> Dict[str, Any]:
    """
    Return a measure's outputs: every output name of every declared item
    (MeasureItem.output_names(): a group's outputs, each listed, and nothing for
    a display-only item), whether or not the item is enabled, with None for any
    the measure didn't compute.

    Args:
        data (Dict[str, Any]): Values computed by the measure.
        items (List[MeasureItem]): The measure's declared items.

    Returns:
        Dict[str, Any]: One entry per declared output, in the items' order.
    """
    return {name: data.get(name) for item in items for name in item.output_names()}


# Earlier name, kept for callers outside this package. It never filtered by the
# enabled flag; see the module docstring for the contract.
filter_measurements = select_outputs


class MeasureItem:
    """
    One of a measure's items: an output, a group of outputs or a drawing switch,
    with its enabled state and the views it suits.
    """

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        display_only: bool = False,
        outputs: Optional[Sequence[str]] = None,
        views: Optional[Sequence[Pose]] = None,
    ) -> None:
        """
        Initialize a MeasureItem.

        Args:
            name (str): The output's name, or for a group or a drawing switch, its
                label, shown in lists and kept by MeasureSelection.
            enabled (bool): Whether the item is drawn (and, for callers that honor
                it, shown or exported).
            display_only (bool): True for a drawing switch that has no value of its
                own; it is never returned or listed as an output.
            outputs (Optional[Sequence[str]]): For a group, the outputs it stands
                for, in order; its name is then a label, never an output. None for
                an item that is its own output.
            views (Optional[Sequence[Pose]]): The views of its measure the item
                suits, some of Pose.FRONTAL, Pose.QUARTER and Pose.LATERAL; None
                for every view its measure suits.
        """
        self.name: str = name
        self.enabled: bool = enabled
        self.display_only: bool = display_only
        self.outputs: Optional[Tuple[str, ...]] = (
            None if outputs is None else tuple(outputs)
        )
        self.views: Optional[Tuple[Pose, ...]] = None if views is None else tuple(views)
        self.is_lateral: bool = False
        self.is_frontal: bool = False
        self.is_quarter: bool = False

    def output_names(self) -> List[str]:
        """
        The outputs the item stands for.

        Returns:
            List[str]: Nothing for a display-only item, a group's outputs in
                order, otherwise the item's own name.
        """
        if self.display_only:
            return []
        if self.outputs is not None:
            return list(self.outputs)
        return [self.name]

    def suits(self, lateral: bool, quarter: bool = False) -> bool:
        """
        Whether the item suits the view, by MeasureBase.suits()' rule (lateral,
        else quarter, else frontal) on the item's own flags, which
        MeasureBase.sync_items() sets (update_for_type(), MeasureSelection.apply()
        and FrameSeries call it before asking).

        Args:
            lateral (bool): True for a lateral face.
            quarter (bool): True for a quarter (three-quarter) face; ignored when
                lateral is True.

        Returns:
            bool: is_lateral for a lateral face, is_quarter for a quarter face,
                is_frontal otherwise.
        """
        if lateral:
            return self.is_lateral
        return self.is_quarter if quarter else self.is_frontal

    def __str__(self) -> str:
        """
        Return a string representation of the MeasureItem.

        Returns:
            str: String representation.
        """
        return f"(name={self.name},enabled={self.enabled})"


class MeasureBase:
    """
    Base class for measurement analysis objects.
    """

    # The class-level default, for a subclass that doesn't call the base
    # initializer: a measure never suits a quarter face unless it says so.
    is_quarter: bool = False

    def __init__(self) -> None:
        """
        Initialize the MeasureBase with default settings.
        """
        self.enabled: bool = True
        self.items: List[MeasureItem] = []
        self.is_lateral: bool = False
        self.is_frontal: bool = False
        self.is_quarter: bool = False

    def abbrev(self) -> str:
        """
        Return the measure's short name, as shown in lists and saved selections.

        Each Dynaface measure has its own; the default is the class name.

        Returns:
            str: The short name, such as "FAI".
        """
        return type(self).__name__

    def outputs(self) -> List[str]:
        """
        Every output the measure declares, in order: its items' output_names(),
        a group's outputs listed in its place. calc() returns all of them.

        Returns:
            List[str]: The output names, such as "oce.l" and "oce.r".
        """
        return [name for item in self.items for name in item.output_names()]

    def suits(self, lateral: bool, quarter: bool = False) -> bool:
        """
        Whether the measure suits the view: lateral, else quarter, else frontal.

        Args:
            lateral (bool): True for a lateral face.
            quarter (bool): True for a quarter (three-quarter) face; ignored when
                lateral is True.

        Returns:
            bool: is_lateral for a lateral face, is_quarter for a quarter face,
                is_frontal otherwise.
        """
        if lateral:
            return self.is_lateral
        return self.is_quarter if quarter else self.is_frontal

    def update_for_type(self, lateral: bool, quarter: bool = False) -> None:
        """
        Update the enabled state of each measurement item based on view type.

        This overwrites every item's enabled flag with MeasureItem.suits(lateral,
        quarter): the measure's view, narrowed to the item's own views when it
        declares any (it calls sync_items() first, so the items' flags follow the
        measure's as they are now), so a user's choice of items is lost. To keep
        the choices, use dynaface.measures.MeasureSelection.apply().

        Args:
            lateral (bool): True if the face is lateral.
            quarter (bool): True if the face is a quarter face; False, the
                default, keeps the two-view calls as they were.
        """
        self.sync_items()
        for item in self.items:
            item.enabled = item.suits(lateral, quarter)

    def set_item_enabled(self, name: str, enabled: bool) -> None:
        """
        Set the enabled state for a specific measurement item by name.

        Args:
            name (str): The name of the measurement item.
            enabled (bool): True to enable, False to disable.
        """
        for item in self.items:
            if item.name == name:
                item.enabled = enabled

    def is_enabled(self, name: str) -> bool:
        """
        Check if a specific measurement item is enabled.

        Args:
            name (str): The name of the measurement item.

        Returns:
            bool: True if the item is enabled, False otherwise.
        """
        for item in self.items:
            if item.name == name:
                return item.enabled
        return True

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the MeasureBase derived class and its items.

        Args:
            enabled (bool): True to enable, False to disable.
        """
        self.enabled = enabled
        for item in self.items:
            item.enabled = enabled

    def sync_items(self) -> None:
        """
        Synchronize each measurement item with the base class view settings: each
        of the item's flags is the measure's, and, for an item declared with
        views, only where that view is among them. Idempotent.
        """
        for item in self.items:
            views = item.views
            item.is_lateral = self.is_lateral and (
                views is None or Pose.LATERAL in views
            )
            item.is_frontal = self.is_frontal and (
                views is None or Pose.FRONTAL in views
            )
            item.is_quarter = self.is_quarter and (
                views is None or Pose.QUARTER in views
            )
