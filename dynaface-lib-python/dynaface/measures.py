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
