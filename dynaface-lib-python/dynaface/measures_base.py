"""
Measurement plugin base classes.

Contract for measures (mirrored by dynaface-lib-dotnet's FaceMeasureBase):

- calc() always returns every declared output, whether or not its item is enabled
  and whether or not render is True. A value that can't be determined is NaN, or
  None when the measure couldn't run at all.
- An item's enabled flag only controls drawing. Choosing which outputs to show or
  export is the caller's job: AnalyzeFace.get_all_items() lists the enabled
  outputs, which is what the desktop app uses for its chart and CSV columns.
- An item marked display_only (such as AnalyzeLateral's tip_proj) is a drawing
  switch, not an output: calc() never returns it and get_all_items() never lists it.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def select_outputs(data: Dict[str, Any], items: List["MeasureItem"]) -> Dict[str, Any]:
    """
    Return a measure's outputs: every declared item that isn't display-only, whether
    or not it is enabled, with None for any the measure didn't compute.

    Args:
        data (Dict[str, Any]): Values computed by the measure.
        items (List[MeasureItem]): The measure's declared items.

    Returns:
        Dict[str, Any]: One entry per non-display-only item.
    """
    return {item.name: data.get(item.name) for item in items if not item.display_only}


# Earlier name, kept for callers outside this package. It never filtered by the
# enabled flag; see the module docstring for the contract.
filter_measurements = select_outputs


class MeasureItem:
    """
    Represents an individual measurement item with a name and enabled state.
    """

    def __init__(
        self, name: str, enabled: bool = True, display_only: bool = False
    ) -> None:
        """
        Initialize a MeasureItem.

        Args:
            name (str): The name of the measurement.
            enabled (bool): Whether the item is drawn (and, for callers that honor
                it, shown or exported).
            display_only (bool): True for a drawing switch that has no value of its
                own; it is never returned or listed as an output.
        """
        self.name: str = name
        self.enabled: bool = enabled
        self.display_only: bool = display_only
        self.is_lateral: bool = False
        self.is_frontal: bool = False

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

    def __init__(self) -> None:
        """
        Initialize the MeasureBase with default settings.
        """
        self.enabled: bool = True
        self.items: List[MeasureItem] = []
        self.is_lateral: bool = False
        self.is_frontal: bool = False

    def update_for_type(self, lateral: bool) -> None:
        """
        Update the enabled state of each measurement item based on view type.

        Args:
            lateral (bool): True if the measurement is lateral, False if frontal.
        """
        for item in self.items:
            item.enabled = self.is_lateral if lateral else self.is_frontal

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
        Synchronize each measurement item with the base class view settings.
        """
        for item in self.items:
            item.is_lateral = self.is_lateral
            item.is_frontal = self.is_frontal
