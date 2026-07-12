import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def filter_measurements(
    data: Dict[str, Any], items: List["MeasureItem"]
) -> Dict[str, Any]:
    """
    Filter measurements from a data dictionary based on the provided measurement items.

    Args:
        data (Dict[str, Any]): Dictionary containing measurement values.
        items (List[MeasureItem]): List of measurement items to filter.

    Returns:
        Dict[str, Any]: A dictionary with keys from measurement item names and their corresponding values from data.
    """
    return {item.name: data.get(item.name, None) for item in items}


class MeasureItem:
    """
    Represents an individual measurement item with a name and enabled state.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        """
        Initialize a MeasureItem.

        Args:
            name (str): The name of the measurement.
            enabled (bool): Whether the measurement is enabled.
        """
        self.name: str = name
        self.enabled: bool = enabled
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
