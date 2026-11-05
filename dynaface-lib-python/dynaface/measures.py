"""The registry of Dynaface's measures."""

import logging
import math
from typing import Any, List

import numpy as np
from dynaface.measures_skin import AnalyzeSkinTone

from dynaface.measures_base import MeasureBase
from dynaface.measures_frontal import (
    AnalyzeFAI,
    AnalyzeOralCommissureExcursion,
    AnalyzeBrows,
    AnalyzeDentalArea,
    AnalyzeEyeArea,
    AnalyzeIntercanthalDistance,
    AnalyzeMouthLength,
    AnalyzeNoseFrontal,
    AnalyzeOuterEyeCorners,
    AnalyzePosition,
    AnalyzePose,
    AnalyzeLandmarks,
)
from dynaface.measures_lateral import AnalyzeLateral

logger = logging.getLogger(__name__)


def all_measures() -> List["MeasureBase"]:
    """
    Return a new instance of every Dynaface measure, in the order they run.

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
    ]


def get_all_items(measures: List["MeasureBase"]) -> List[str]:
    """
    List the outputs to show or export from these measures: the enabled items of
    the enabled measures, without display-only items. AnalyzeFace.get_all_items()
    gives the same list for a face's own measures.

    Args:
        measures (List[MeasureBase]): The measures, such as all_measures().

    Returns:
        List[str]: Output names, such as "fai" and "oce.l".
    """
    return [
        item.name
        for measure in measures
        if measure.enabled
        for item in measure.items
        if item.enabled and not item.display_only
    ]
