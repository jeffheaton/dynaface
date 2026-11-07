"""
Cheek landmarks: the frame, the frontal constructions, the yaw calibration and
the quarter-face classification behind AnalyzeFace.cheek_landmarks, the
Landmarks measure's "Cheek Landmarks" item (CHEEK_LANDMARK_OUTPUTS) and
measures_cheek.AnalyzeCheekMeasures.

On a frontal face, the zygion, the cheek reference point and the submalar
reference point of each side are constructed from the 98 WFLW landmarks in a
frame on the pupil axis (Frame), so head roll never enters a number. On a
three-quarter (quarter) face, the far cheek's malar apex and submalar point are
observed on its outline by dynaface.cheek_outline (observe_far_cheek());
locate() is the one rule that turns a face's landmarks and view into the two
results, landmark_array() fills the (10, 2) cheek_landmarks array from them,
NaN where a point doesn't exist in the view, and locate_all() does both from
a Pose. calibrate_yaw(), nose_offset(), landmark_yaw() and nose_margin() are
the pose cues, and classify_quarter() is the rule AnalyzeFace.load_image()
uses to call a face a quarter face (off unless config.AUTO_QUARTER).

This module imports nothing beyond numpy, math, logging, dataclasses, typing,
dynaface.const and dynaface.cheek_outline (the leaf module, which never imports
this one: observed_points(), the WFLW indices, the Landmarks alias and
as_landmarks() come from it): no scipy, no cv2, and none of facial, models or
lateral, so it is testable without the models and ports to one C# file. Every
number here is a pixel of the image the landmarks came from, a degree, or a
ratio; millimeters are applied by the measure (* face.pix2mm), never here.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dynaface import cheek_outline
from dynaface.cheek_outline import Landmarks, as_landmarks
from dynaface.const import LM_LEFT_PUPIL, LM_RIGHT_PUPIL, Pose

logger = logging.getLogger(__name__)

#: The ten cheek landmarks, in the order of AnalyzeFace.cheek_landmarks' rows.
#: "Right" and "left" are the subject's, like oce.r. Rows 0 to 5 exist on a
#: frontal face, rows 6 to 9 on a quarter face, for its far cheek only; only the
#: observed oblique point is called an apex.
CHEEK_LANDMARK_NAMES = [
    "Zygion (right)",  # 0, frontal
    "Zygion (left)",  # 1, frontal
    "Cheek reference point (right)",  # 2, frontal
    "Cheek reference point (left)",  # 3, frontal
    "Submalar reference point (right)",  # 4, frontal
    "Submalar reference point (left)",  # 5, frontal
    "Malar apex (right)",  # 6, quarter, far cheek
    "Malar apex (left)",  # 7, quarter, far cheek
    "Submalar point (right)",  # 8, quarter, far cheek
    "Submalar point (left)",  # 9, quarter, far cheek
]
#: Each cheek landmark's code, in CHEEK_LANDMARK_NAMES order: the prefix of its
#: outputs (apex.l.x is "apex.l"'s x), and the label drawn beside it
#: (AnalyzeFace.draw_cheek_landmarks()).
CHEEK_LANDMARK_CODES = [
    "zy.r",
    "zy.l",
    "cr.r",
    "cr.l",
    "sr.r",
    "sr.l",
    "apex.r",
    "apex.l",
    "sub.r",
    "sub.l",
]
#: The cheek landmarks' coordinates as outputs, px of the standard image: each
#: code's x then y, in CHEEK_LANDMARK_NAMES order (zy.r.x, zy.r.y, zy.l.x, ...
#: sub.l.y), NaN where the landmark doesn't exist in the view. The Landmarks
#: measure's "Cheek Landmarks" item stands for them (AnalyzeLandmarks).
CHEEK_LANDMARK_OUTPUTS = [
    f"{code}.{axis}" for code in CHEEK_LANDMARK_CODES for axis in ("x", "y")
]
CHEEK_LM_ZYGION_R, CHEEK_LM_ZYGION_L = 0, 1
CHEEK_LM_CHEEK_REF_R, CHEEK_LM_CHEEK_REF_L = 2, 3
CHEEK_LM_SUBMALAR_REF_R, CHEEK_LM_SUBMALAR_REF_L = 4, 5
CHEEK_LM_APEX_R, CHEEK_LM_APEX_L = 6, 7
CHEEK_LM_SUBMALAR_R, CHEEK_LM_SUBMALAR_L = 8, 9

# The zygion level below the interpupillary line, in IPD units (an orbitale
# proxy).
ZYGION_LEVEL_IPD = 0.3
# Wilkinson: the cheek reference point is this far down the canthus vertical to
# the contour.
CHEEK_REF_FRACTION = 1.0 / 3.0
# The submalar reference point is this far from the cheek reference point down to
# the stomion level.
SUBMALAR_REF_FRACTION = 0.5
# The polylines the zygion level crosses.
RIGHT_CONTOUR_TOP = (0, 1, 2, 3, 4)
LEFT_CONTOUR_TOP = (32, 31, 30, 29, 28)
# The half contours the canthus vertical crosses, the WFLW landmarks the
# constructions and the pose cues read and the landmark count: defined once in
# cheek_outline (the leaf module, whose tracer seeds on the same half contours)
# and re-exported here.
NUM_LANDMARKS = cheek_outline.NUM_LANDMARKS
RIGHT_CONTOUR = cheek_outline.RIGHT_FAR_CONTOUR
LEFT_CONTOUR = cheek_outline.LEFT_FAR_CONTOUR
LM_CANTHUS_R, LM_CANTHUS_L = cheek_outline.LM_CANTHUS_R, cheek_outline.LM_CANTHUS_L
LM_ALA_R, LM_ALA_L = cheek_outline.LM_ALA_R, cheek_outline.LM_ALA_L
LM_NOSE_TIP, LM_CHIN = cheek_outline.LM_NOSE_TIP, cheek_outline.LM_CHIN
LM_STOMION = cheek_outline.LM_STOMION
# The contour's top points: the nose margin is read from the nearer of them.
LM_CONTOUR_TOP_R, LM_CONTOUR_TOP_L = 0, 32
# zy.*.clip: the level crossed the contour; passed above point 0/32; passed below
# point 4/28.
CLIP_NONE, CLIP_TOP, CLIP_BOTTOM = 0, 1, 2
FRONTAL_YAW_GATE = 5.0  # |cheek.yaw| for zy.r.ok, zy.l.ok, cr.ok, sr.ok
ZYGION_RATIO_YAW_GATE = 2.0  # for zy.ratio.ok (the ratio moves ~3% per degree)
# cheek.yaw and cheek.yaw.model must agree within this for any yaw gate.
YAW_AGREE_DEG = 5.0
# |raw SPIGA pitch| above this fails cheek.pitch.ok, and nothing else: pitch is
# uncalibrated, so no other flag folds it in.
PITCH_GROSS_RAW = 20.0
# apex.ok needs |cheek.yaw| within OBLIQUE_YAW +/- OBLIQUE_YAW_GATE.
OBLIQUE_YAW, OBLIQUE_YAW_GATE = 45.0, 5.0
# |cheek.yaw| (the landmark yaw) that load_image() calls a quarter face.
QUARTER_YAW_RANGE = (35.0, 55.0)
# A face that was a quarter face on the previous frame stays one this far past
# both edges of the range.
QUARTER_YAW_HOLD = 3.0
# The model's calibrated |yaw| must reach this, with the landmark yaw's sign.
QUARTER_MIN_MODEL_YAW = 20.0
# ... and stay at or below this (widened by QUARTER_YAW_HOLD with the hold): on
# a profile the landmark yaw collapses into the quarter window while the model
# still reads the turn. Over 816 Face Research Lab London photos (detection on)
# no three-quarter photo read above 51.8, and 8 of the 14 profiles the window
# took read 57 to 83 (the other 6 read 29 to 55 by both cues).
QUARTER_MAX_MODEL_YAW = 55.0
# A nose past the far contour by more than this (nose_margin): profile logic
# instead.
QUARTER_MARGIN_MIN = -0.03
# (raw SPIGA yaw, true yaw) on the synthetic renders: the positive sweep's
# raw.yaw in tests_data/cheek_landmarks.json (scripts/make_cheek_fixtures.py),
# the level head's reading first, where SPIGA reads about 0.63 of the turn and
# saturates past 60 degrees. calibrate_yaw() reads it less YAW_RAW_OFFSET.
YAW_CALIBRATION = (
    (1.5, 0.0),
    (3.04, 5.0),
    (5.87, 10.0),
    (7.82, 15.0),
    (10.41, 20.0),
    (12.89, 25.0),
    (15.57, 30.0),
    (19.59, 35.0),
    (23.73, 40.0),
    (25.77, 42.0),
    (28.28, 45.0),
    (30.54, 48.0),
    (32.63, 50.0),
    (37.79, 55.0),
    (43.04, 60.0),
    (44.31, 65.0),
)
# Degrees of true turn per reported degree past the table.
YAW_SLOPE_BEYOND = 1.0 / 0.62
# The constant SPIGA's yaw reads over the truth on every render, degrees: the
# level head reads 1.50, and the half-sums of the mirrored pairs' raw yaws
# (+7.82 / -5.10 at 15 degrees, +15.57 / -14.24 at 30, +28.28 / -25.63 at 45,
# +32.63 / -30.01 at 50) read 1.36, 0.67, 1.33 and 1.31, their mean; the nose
# offset reads the same pairs within 0.02 spans of each other. So the model
# over-reads a turn toward the subject's left and under-reads the mirrored one
# by this much, and calibrate_yaw() subtracts it before the table, whose raw
# column was measured with it on the positive sweep: that sweep calibrates as
# it did, and the mirrored turns read 0.7, -1.4, 0.2 and 0.2 degrees from true
# instead of 6.4, 2.5, 3.1 and 2.7 low (the negative sweep's raw yaws, in the
# fixture, had no say before). A reading of exactly this is a level head.
YAW_RAW_OFFSET = 1.23
# (|nose offset| in canthus spans, true |yaw|) on the same renders: the sweep's
# roll-corrected cues.nose_offset (nose_offset()), the landmark yaw
# (landmark_yaw()). The table stops at 60 degrees: the cue falls back at 65
# (0.735 after 0.770), and np.interp needs it increasing. Its zero is the plain
# (0, 0), unlike the yaw table's measured first row: the level head reads
# 0.0207 spans (2.26 degrees through the table, the landmark yaw's bias at
# zero, which docs/measures.md states and which fails the 2 degree ratio gate
# on that head), but the mirrored pairs' half-sums (0.0044 spans at 15 degrees,
# -0.0062 at 30, -0.0007 at 45, 0.0036 at 50) show no constant offset in the
# cue, so a zero fitted to that one reading would read every offset within
# 0.0207 spans as a level head and widen the ratio gate to about 3 degrees
# while the turns read as they do now. On 204 mirrored pairs of three-quarter
# photos (102 subjects, two expressions, turns not measured) the landmark
# yaw's half-sum is +0.75 and +0.91 degrees (se 0.25, 0.28) and the model's
# +0.05 and +0.46 (se 0.20, 0.21): the two cues differ by about 0.7 degrees
# per side in the level head's direction (0.006 to 0.016 spans at the table's
# 45 degree slopes), too small for any gate, and whether it is a cue constant
# or the subjects, and which cue carries it, needs measured turns.
NOSE_OFFSET_CALIBRATION = (
    (0.0, 0.0),
    (0.0458, 5.0),
    (0.0908, 10.0),
    (0.122, 15.0),
    (0.167, 20.0),
    (0.2069, 25.0),
    (0.2573, 30.0),
    (0.3289, 35.0),
    (0.3861, 40.0),
    (0.4181, 42.0),
    (0.4449, 45.0),
    (0.514, 48.0),
    (0.5431, 50.0),
    (0.6842, 55.0),
    (0.7696, 60.0),
)
# Degrees per canthus span past the table (its last segment reads 58.5).
NOSE_OFFSET_SLOPE_BEYOND = 57.0

# The tables as np.interp's xp and fp (both strictly increasing); the yaw
# table's cue column is the raw reading less the offset.
_YAW_CAL_REPORTED = np.array(
    [reported - YAW_RAW_OFFSET for reported, _ in YAW_CALIBRATION]
)
_YAW_CAL_TRUE = np.array([true for _, true in YAW_CALIBRATION])
_NOSE_CAL_OFFSET = np.array([offset for offset, _ in NOSE_OFFSET_CALIBRATION])
_NOSE_CAL_TRUE = np.array([true for _, true in NOSE_OFFSET_CALIBRATION])


@dataclass(frozen=True)
class Frame:
    """
    A face frame on the pupil axis, so head roll never enters a number.

    Attributes:
        origin: The pupil midpoint, (x, y) px.
        across: Unit vector from the right pupil (96) toward the left (97), the
            subject's left.
        down: Its normal toward the chin, (-across.y, across.x): image-down for a
            level head.
        ipd: The pupil distance, px.
    """

    origin: NDArray[np.float64]
    across: NDArray[np.float64]
    down: NDArray[np.float64]
    ipd: float

    def level(self, p: ArrayLike) -> float:
        """How far below the interpupillary line the point is, px."""
        return float((np.asarray(p, dtype=float) - self.origin) @ self.down)

    def offset(self, p: ArrayLike) -> float:
        """How far toward the subject's left of the midline the point is, px."""
        return float((np.asarray(p, dtype=float) - self.origin) @ self.across)

    def point(self, offset: float, level: float) -> NDArray[np.float64]:
        """The point at this offset and level, (x, y) px."""
        return self.origin + self.across * offset + self.down * level


def pupil_frame(landmarks: Landmarks) -> Optional[Frame]:
    """
    The frame of the 98 landmarks, or None when the pupils coincide.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.

    Returns:
        Optional[Frame]: The frame on the pupil axis (landmarks 96 and 97), or None
            when the pupils coincide, there are fewer than 98 landmarks or any
            landmark isn't finite.
    """
    pts = as_landmarks(landmarks)
    if pts is None:
        return None
    right = pts[LM_RIGHT_PUPIL]
    left = pts[LM_LEFT_PUPIL]
    axis = left - right
    ipd = float(np.hypot(axis[0], axis[1]))
    if not np.isfinite(ipd) or ipd <= 0.0:
        return None
    across = axis / ipd
    down = np.array([-across[1], across[0]], dtype=np.float64)
    return Frame(origin=(right + left) / 2.0, across=across, down=down, ipd=ipd)


def cross_polyline(
    points: ArrayLike, values: ArrayLike, target: float
) -> Tuple[NDArray[np.float64], int]:
    """
    Where a polyline reaches values == target: the first segment whose ends
    bracket it, interpolated; status 0. When none does, the end nearer the target,
    clipped: status -1 (the first point) or +1 (the last).

    Args:
        points (ArrayLike): The polyline's vertices, (N, 2), in order.
        values (ArrayLike): A value at each vertex, (N,), such as its level in a
            Frame.
        target (float): The value to reach.

    Returns:
        Tuple[NDArray[np.float64], int]: The point, (x, y), and the status: 0
            crossed, -1 clipped to the first vertex, +1 clipped to the last.
    """
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float)
    for k in range(len(pts) - 1):
        a, b = vals[k], vals[k + 1]
        if (a - target) * (b - target) <= 0 and a != b:
            t = (target - a) / (b - a)
            return pts[k] + t * (pts[k + 1] - pts[k]), 0
    nearer_first = abs(vals[0] - target) <= abs(vals[-1] - target)
    return (pts[0], -1) if nearer_first else (pts[-1], 1)


def contour_below(
    points: ArrayLike, offsets: ArrayLike, levels: ArrayLike, offset: float
) -> NDArray[np.float64]:
    """
    Where the vertical (pupil-axis normal) at this offset crosses the contour
    polyline: the crossing with the greatest level (the mandibular border, not a
    crossing near the contour's top on a turned or hair-occluded face),
    interpolated. The vertex nearest in offset when nothing crosses. The earlier
    prototype (the review folder's synthetic_head/photo_prototype/
    overlay_proposed.py, not in the repository) took the nearest contour vertex
    instead of the crossing: identical on img4's right side, 0.34 mm apart on its
    left (depth 28.59 against 28.83 mm), so the doc figure is regenerated from
    the measure, not copied.

    Args:
        points (ArrayLike): The half contour's vertices, (N, 2) px, in order
            (RIGHT_CONTOUR or LEFT_CONTOUR).
        offsets (ArrayLike): Each vertex's Frame.offset(), (N,).
        levels (ArrayLike): Each vertex's Frame.level(), (N,).
        offset (float): The vertical's offset, such as the lateral canthus's.

    Returns:
        NDArray[np.float64]: The crossing, (x, y) px.
    """
    pts = np.asarray(points, dtype=float)
    offs = np.asarray(offsets, dtype=float)
    lvls = np.asarray(levels, dtype=float)
    best: Optional[NDArray[np.float64]] = None
    best_level = -math.inf
    for k in range(len(pts) - 1):
        a, b = offs[k], offs[k + 1]
        if (a - offset) * (b - offset) <= 0 and a != b:
            t = (offset - a) / (b - a)
            level = lvls[k] + t * (lvls[k + 1] - lvls[k])
            if level > best_level:
                best_level = level
                best = pts[k] + t * (pts[k + 1] - pts[k])
    if best is None:
        best = np.array(pts[int(np.argmin(np.abs(offs - offset)))], dtype=float)
    return best


def zygion(
    landmarks: Landmarks,
    frame: Frame,
    left: bool,
    level_ipd: float = ZYGION_LEVEL_IPD,
) -> Tuple[NDArray[np.float64], int]:
    """
    The constructed zygion of one side: the level level_ipd * ipd below the
    interpupillary line, crossed with the contour's top polyline (0-4, or 32-28),
    interpolated by cross_polyline(). The level is a convention (an orbitale
    proxy): the lid and canthus levels meet the contour at its very end on the
    library's own test photo, so the point would collapse to 0/32, the noisiest
    contour point.

    The clip code says what happened when the level missed the polyline:
    CLIP_NONE when it crossed, CLIP_TOP when it passed above point 0/32 (the
    definitions' and the prototype's clip: the point is 0/32) or CLIP_BOTTOM when
    it passed below point 4/28 (the point is 4/28; unseen on 64 faces, but a 16
    degree pitch moved the terminus-lid gap by 13 mm, so it gets its own code
    rather than sharing the top clip's flag).

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.
        frame (Frame): Their pupil-axis frame.
        left (bool): The subject's left side (contour 32-28), else the right.
        level_ipd (float): The level below the interpupillary line, in IPD units.

    Returns:
        Tuple[NDArray[np.float64], int]: The zygion, (x, y) px, and its clip code.
    """
    pts = np.asarray(landmarks, dtype=float)
    idx = LEFT_CONTOUR_TOP if left else RIGHT_CONTOUR_TOP
    top = np.array([pts[i] for i in idx], dtype=float)
    levels = np.array([frame.level(p) for p in top])
    point, status = cross_polyline(top, levels, level_ipd * frame.ipd)
    return point, {0: CLIP_NONE, -1: CLIP_TOP, 1: CLIP_BOTTOM}[status]


@dataclass
class FrontalCheek:
    """
    The six frontal constructions and the lines they are drawn with, in pixels of
    the standard image. Millimeters are applied by the measure (* face.pix2mm),
    never here.

    Attributes:
        frame: The pupil-axis frame they were built in.
        zygion_level: The zygion level, px below the interpupillary line (for the
            level line).
        zygion_r, zygion_l: Each zygion, (x, y) px.
        zygion_r_clip, zygion_l_clip: CLIP_NONE, CLIP_TOP or CLIP_BOTTOM.
        zygion_r_px, zygion_l_px: Each zygion's distance from the midline, px
            (both positive).
        canthus_r, canthus_l, foot_r, foot_l: The ends of the canthus verticals:
            each lateral canthus (60 / 72) and its foot on the contour.
        cheek_ref_r, cheek_ref_l: The cheek reference points, (x, y) px.
        cheek_ref_r_px, cheek_ref_l_px: Their depth below the canthus along the
            vertical, px.
        submalar_ref_r, submalar_ref_l: The submalar reference points, (x, y) px.
        stomion_level: The stomion level, px below the interpupillary line.
    """

    frame: Frame
    zygion_level: float
    zygion_r: NDArray[np.float64]
    zygion_l: NDArray[np.float64]
    zygion_r_clip: int
    zygion_l_clip: int
    zygion_r_px: float
    zygion_l_px: float
    canthus_r: NDArray[np.float64]
    canthus_l: NDArray[np.float64]
    foot_r: NDArray[np.float64]
    foot_l: NDArray[np.float64]
    cheek_ref_r: NDArray[np.float64]
    cheek_ref_l: NDArray[np.float64]
    cheek_ref_r_px: float
    cheek_ref_l_px: float
    submalar_ref_r: NDArray[np.float64]
    submalar_ref_l: NDArray[np.float64]
    stomion_level: float


@dataclass(frozen=True)
class _CanthusVertical:
    """
    One side's canthus vertical and the two reference points on it, px.

    Attributes:
        canthus: The lateral canthus, the vertical's top.
        foot: Its foot on the contour, exactly on the vertical.
        cheek_ref: The cheek reference point, CHEEK_REF_FRACTION of the way down.
        depth: The cheek reference point's depth below the canthus.
        submalar_ref: The submalar reference point, SUBMALAR_REF_FRACTION of the
            way from the cheek reference point down to the stomion level.
    """

    canthus: NDArray[np.float64]
    foot: NDArray[np.float64]
    cheek_ref: NDArray[np.float64]
    depth: float
    submalar_ref: NDArray[np.float64]


def _canthus_vertical(
    pts: NDArray[np.float64],
    frame: Frame,
    canthus: int,
    contour: Tuple[int, ...],
    stomion_level: float,
) -> _CanthusVertical:
    """
    Drop one side's vertical from its lateral canthus to the contour and place
    the two reference points on it.

    Args:
        pts (NDArray[np.float64]): The 98 WFLW landmarks, (98, 2) px.
        frame (Frame): Their pupil-axis frame.
        canthus (int): The side's lateral canthus, LM_CANTHUS_R or LM_CANTHUS_L.
        contour (Tuple[int, ...]): Its half contour, RIGHT_CONTOUR or
            LEFT_CONTOUR.
        stomion_level (float): The stomion level, px below the interpupillary
            line.

    Returns:
        _CanthusVertical: The vertical's ends and the two points.
    """
    ex = np.array(pts[canthus], dtype=np.float64)
    cont = pts[list(contour)]
    rel = cont - frame.origin
    ex_offset = frame.offset(ex)
    crossing = contour_below(cont, rel @ frame.across, rel @ frame.down, ex_offset)
    # The crossing's level at the canthus's own offset: exactly on the vertical,
    # which the interpolation's rounding alone would not guarantee.
    foot = frame.point(ex_offset, frame.level(crossing))
    cheek_ref = ex + (foot - ex) * CHEEK_REF_FRACTION
    depth = frame.level(cheek_ref) - frame.level(ex)
    submalar_ref = cheek_ref + frame.down * (
        (stomion_level - frame.level(cheek_ref)) * SUBMALAR_REF_FRACTION
    )
    return _CanthusVertical(ex, foot, cheek_ref, depth, submalar_ref)


def frontal_constructions(landmarks: Landmarks) -> Optional[FrontalCheek]:
    """
    All six frontal points, or None when the pupils coincide.

    Per side, in the pupil-axis frame: the zygion is zygion()'s crossing of the
    ZYGION_LEVEL_IPD level with the contour top, and its midline distance is its
    offset, made positive on both sides (the right side's offsets are negative).
    The canthus vertical runs from the lateral canthus (60 / 72) down the frame's
    normal to its foot on the half contour, contour_below()'s lowest crossing,
    taken at the canthus's own offset so the foot lies exactly on the vertical;
    the cheek reference point is CHEEK_REF_FRACTION of the way down that
    vertical (Wilkinson's third, with no lateral offset, so it is never called an
    apex), and its depth is the level difference along the vertical. The stomion
    level is the mean level of the inner-lip midpoints (90 and 94: they coincide
    at rest and separate in a smile, and unilateral commissure droop disturbs
    them less than the commissures), and the submalar reference point lies
    SUBMALAR_REF_FRACTION of the way from the cheek reference point down to it,
    on the same vertical. Millimeters are applied by the measure (* face.pix2mm),
    never here.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.

    Returns:
        Optional[FrontalCheek]: The constructions, or None when the pupils coincide,
            there are fewer than 98 landmarks or any landmark isn't finite (no
            frame, no point: a NaN chin gives None, not NaN points).
    """
    pts = as_landmarks(landmarks)
    if pts is None:
        return None
    frame = pupil_frame(pts)
    if frame is None:
        return None
    zygion_r, clip_r = zygion(pts, frame, left=False)
    zygion_l, clip_l = zygion(pts, frame, left=True)
    stomion_level = float(
        np.mean([frame.level(pts[i]) for i in LM_STOMION], dtype=np.float64)
    )
    right = _canthus_vertical(pts, frame, LM_CANTHUS_R, RIGHT_CONTOUR, stomion_level)
    left = _canthus_vertical(pts, frame, LM_CANTHUS_L, LEFT_CONTOUR, stomion_level)
    return FrontalCheek(
        frame=frame,
        zygion_level=ZYGION_LEVEL_IPD * frame.ipd,
        zygion_r=zygion_r,
        zygion_l=zygion_l,
        zygion_r_clip=clip_r,
        zygion_l_clip=clip_l,
        zygion_r_px=-frame.offset(zygion_r),
        zygion_l_px=frame.offset(zygion_l),
        canthus_r=right.canthus,
        canthus_l=left.canthus,
        foot_r=right.foot,
        foot_l=left.foot,
        cheek_ref_r=right.cheek_ref,
        cheek_ref_l=left.cheek_ref,
        cheek_ref_r_px=right.depth,
        cheek_ref_l_px=left.depth,
        submalar_ref_r=right.submalar_ref,
        submalar_ref_l=left.submalar_ref,
        stomion_level=stomion_level,
    )


def _through_table(
    magnitude: float,
    xp: NDArray[np.float64],
    fp: NDArray[np.float64],
    slope_beyond: float,
) -> float:
    """
    A cue's magnitude through a calibration table: np.interp inside it, linear
    at slope_beyond past its last row; NaN stays NaN.

    Args:
        magnitude (float): The cue's magnitude, >= 0.
        xp (NDArray[np.float64]): The table's cue column, strictly increasing.
        fp (NDArray[np.float64]): The table's true-yaw column.
        slope_beyond (float): Degrees per unit of the cue past the table.

    Returns:
        float: The true yaw's magnitude, degrees.
    """
    if math.isnan(magnitude):
        return math.nan
    if magnitude > xp[-1]:
        return float(fp[-1] + (magnitude - xp[-1]) * slope_beyond)
    return float(np.interp(magnitude, xp, fp))


def calibrate_yaw(reported: float) -> float:
    """
    SPIGA's yaw as a true turn: the reading less YAW_RAW_OFFSET (the constant
    the model reads over the truth, so a reading of the offset is a level head
    and one below it a turn toward the subject's right), then np.interp on its
    magnitude through YAW_CALIBRATION, linear at YAW_SLOPE_BEYOND past the
    table's last point, with that difference's sign; NaN stays NaN. Odd about
    the offset, not about zero: 28.28 reads 45.0 and -25.63 reads -44.8, the
    two 45 degree renders.

    Args:
        reported (float): The model's yaw, degrees (AnalyzeFace.yaw).

    Returns:
        float: The calibrated yaw, degrees.
    """
    shifted = float(reported) - YAW_RAW_OFFSET
    if math.isnan(shifted):
        return math.nan
    true = _through_table(
        abs(shifted), _YAW_CAL_REPORTED, _YAW_CAL_TRUE, YAW_SLOPE_BEYOND
    )
    # A level head is 0.0, never -0.0.
    return math.copysign(true, shifted) if true != 0.0 else 0.0


def nose_offset(landmarks: Landmarks, frame: Frame) -> float:
    """
    The nose tip's offset from the midpoint of the lateral canthi, along the pupil
    axis, in canthus-to-canthus spans: positive toward the subject's left (image
    right). Roll-corrected; 0.26 at a 30 degree turn, 0.44 at 45, within 0.02 of
    the mirrored turn (0.4449 and -0.4463 on the two 45 degree renders), so it is
    the symmetric cue (SPIGA reads +28.3 and -25.6 on them).

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.
        frame (Frame): Their pupil-axis frame.

    Returns:
        float: The offset, in canthus spans; NaN with fewer than 98 landmarks, a
            landmark that isn't finite, or when the canthi don't span the axis.
    """
    pts = as_landmarks(landmarks)
    if pts is None:
        return math.nan
    canthus_r = frame.offset(pts[LM_CANTHUS_R])
    canthus_l = frame.offset(pts[LM_CANTHUS_L])
    span = canthus_l - canthus_r
    if not span > 0.0:
        return math.nan
    return (frame.offset(pts[LM_NOSE_TIP]) - (canthus_r + canthus_l) / 2.0) / span


def landmark_yaw(landmarks: Landmarks, frame: Frame) -> float:
    """
    The turn measured from the landmarks, as the definitions ask: nose_offset
    through NOSE_OFFSET_CALIBRATION (np.interp on its magnitude,
    NOSE_OFFSET_SLOPE_BEYOND past the table's last point, 60 degrees), signed
    like SPIGA's yaw. The mirrored 45 degree renders read 0.4449 and -0.4463 (45.0
    and 45.1 degrees) against the model's 28.28 and -25.63 (45.0 and 44.8 through
    calibrate_yaw), and a visit's two obliques must match within about 3 degrees,
    so this is cheek.yaw and the yaw every gate uses; calibrate_yaw(face.yaw) is
    cheek.yaw.model, the independent check (the gates need the two within
    YAW_AGREE_DEG). The eye-width ratio the definitions also name (0.983 at 0;
    0.571 and 0.593 at +/-45: less symmetric on the renders, and tied at 0 and 5
    degrees) is measured by the fixture script and asserted monotonic (non-strict),
    a cross-check only.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.
        frame (Frame): Their pupil-axis frame.

    Returns:
        float: The yaw, degrees, positive when the nose points to the image right;
            NaN when nose_offset() is.
    """
    offset = nose_offset(landmarks, frame)
    if math.isnan(offset):
        return math.nan
    true = _through_table(
        abs(offset), _NOSE_CAL_OFFSET, _NOSE_CAL_TRUE, NOSE_OFFSET_SLOPE_BEYOND
    )
    return math.copysign(true, offset)


def nose_margin(landmarks: Landmarks, frame: Frame) -> float:
    """
    The nose tip's margin from the nearer contour top (0 or 32), along the pupil
    axis, as a fraction of the contour width: on the renders 0.47 face-on, 0.05
    at 45 degrees, 0 at 50 where the tip meets the far silhouette, negative past
    it (-0.06 at 55).

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.
        frame (Frame): Their pupil-axis frame.

    Returns:
        float: The margin, as a fraction of the contour width; NaN with fewer
            than 98 landmarks, a landmark that isn't finite, or when the contour
            tops don't span the axis.
    """
    pts = as_landmarks(landmarks)
    if pts is None:
        return math.nan
    top_r = frame.offset(pts[LM_CONTOUR_TOP_R])
    top_l = frame.offset(pts[LM_CONTOUR_TOP_L])
    width = top_l - top_r
    if not width > 0.0:
        return math.nan
    tip = frame.offset(pts[LM_NOSE_TIP])
    return min(top_l - tip, tip - top_r) / width


def far_left_of(landmarks: Landmarks, yaw_reported: float = math.nan) -> bool:
    """
    True when the nose points to the image's right (nose_offset > 0): the camera
    sees the subject's right side, so the far cheek is the left. Falls back to the
    yaw's sign without a frame (coincident pupils, fewer than 98 landmarks or a
    landmark that isn't finite) or when the canthi don't span the pupil axis
    (nose_offset() is NaN); a right-cheek apex needs the photo taken from the
    subject's left.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.
        yaw_reported (float): The model's yaw, degrees, for the fallback; NaN (the
            default) falls back to False.

    Returns:
        bool: Whether the far (measured) cheek is the subject's left.
    """
    frame = pupil_frame(landmarks)
    if frame is not None:
        offset = nose_offset(landmarks, frame)
        if not math.isnan(offset):
            return offset > 0.0
    return float(yaw_reported) > 0.0


def classify_quarter(
    landmarks: Landmarks,
    yaw_reported: float,
    previous_quarter: bool = False,
    require_margin: bool = True,
) -> Tuple[bool, bool]:
    """
    (is_quarter, far_left) from the raw landmarks and head pose, before any crop
    or flip. Quarter when |landmark_yaw| is within QUARTER_YAW_RANGE, widened by
    QUARTER_YAW_HOLD at both edges when previous_quarter (hysteresis: a video
    whose yaw hovers at 35 degrees enters at 35 and leaves below 32, instead of
    alternating the pupil-anchored and the band crop frame by frame, each flip
    resetting the smoother and jumping every pixel output between two frames),
    the calibrated model yaw has the same sign, reaches QUARTER_MIN_MODEL_YAW (a
    frontal face with a displaced nose is not a quarter face) and stays at or
    below QUARTER_MAX_MODEL_YAW, widened by the same hold (a profile whose
    landmark yaw collapsed into the window is not one either), and nose_margin
    >= QUARTER_MARGIN_MIN (a nose past the far contour is left to _is_lateral(),
    unchanged by the hold) unless require_margin is False. A NaN yaw fails every
    comparison.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks of the uncropped image.
        yaw_reported (float): The model's yaw, degrees (AnalyzeFace.yaw).
        previous_quarter (bool): Whether the previous kept frame of a video was a
            quarter face (VideoAnalyzer passes its pose as load_image's
            previous_pose).
        require_margin (bool): Whether the nose margin rule applies. False is
            for a face the profile test has already declined, which
            load_image() takes as a quarter face when the turn and the model
            say so, rather than as a frontal face.

    Returns:
        Tuple[bool, bool]: Whether the face is a quarter face, and whether its far
            cheek is the subject's left (False unless a quarter face).
    """
    frame = pupil_frame(landmarks)
    if frame is None:
        return False, False
    yaw = landmark_yaw(landmarks, frame)
    model_yaw = calibrate_yaw(yaw_reported)
    low, high = QUARTER_YAW_RANGE
    model_high = QUARTER_MAX_MODEL_YAW
    if previous_quarter:
        low -= QUARTER_YAW_HOLD
        high += QUARTER_YAW_HOLD
        model_high += QUARTER_YAW_HOLD
    in_window = low <= abs(yaw) <= high
    model_agrees = QUARTER_MIN_MODEL_YAW <= abs(model_yaw) <= model_high and (
        model_yaw > 0.0
    ) == (yaw > 0.0)
    margin_ok = (
        not require_margin or nose_margin(landmarks, frame) >= QUARTER_MARGIN_MIN
    )
    if not (in_window and model_agrees and margin_ok):
        return False, False
    return True, far_left_of(landmarks)


def observe_far_cheek(
    landmarks: Landmarks,
    outline_x: Optional[NDArray[np.float32]],
    outline_y: Optional[NDArray[np.int32]],
    outline_src: Optional[NDArray[np.uint8]],
    far_left: bool,
) -> Optional[cheek_outline.ObliqueCheek]:
    """
    The far cheek's observed points as the cheek landmarks take them: None
    without a pupil frame (fewer than 98 landmarks, a landmark that isn't
    finite, or coincident pupils: the rule the frontal constructions follow, so
    a face without a frame has no cheek landmark in either view), else
    cheek_outline.observed_points(). The outline goes in whole or not at all (a
    partial or ragged one means the contour fallback), and a search that fails
    is a face without observed points, never an exception (this runs on every
    kept video frame, and a measure that reads the points gets NaN rather than
    an aborted import). The exception is logged at warning level with its text,
    so a real bug in the oblique search shows in the import's log instead of
    vanishing into NaN. locate() observes through it, for every caller.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px of the standard
            image.
        outline_x (Optional[NDArray[np.float32]]): The far cheek's outline
            (AnalyzeFace.cheek_outline_x), or None for the contour fallback.
        outline_y (Optional[NDArray[np.int32]]): Its rows (cheek_outline_y).
        outline_src (Optional[NDArray[np.uint8]]): Each row's source
            (cheek_outline_src).
        far_left (bool): Whether the far cheek is the subject's left.

    Returns:
        Optional[cheek_outline.ObliqueCheek]: The observed points, or None.
    """
    pts = as_landmarks(landmarks)
    if pts is None or pupil_frame(pts) is None:
        return None
    if outline_x is None or outline_y is None or outline_src is None:
        outline_x = outline_y = outline_src = None
    elif not len(outline_x) == len(outline_y) == len(outline_src):
        outline_x = outline_y = outline_src = None
    try:
        return cheek_outline.observed_points(
            pts, outline_x, outline_y, outline_src, far_left
        )
    except Exception as exc:
        logger.warning(
            "The far-cheek search failed, so the face has no observed cheek "
            "points: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None


def landmark_array(
    frontal: Optional[FrontalCheek],
    observed: Optional[cheek_outline.ObliqueCheek],
    far_left: bool,
) -> NDArray[np.float64]:
    """
    The (10, 2) cheek_landmarks array from what was located: rows 0-5 from the
    frontal constructions when there are any, the far side's rows 6-9 from the
    observed points when there are any (the submalar point only when its depth
    isn't NaN), NaN elsewhere. The one rule locate_all() and
    AnalyzeFace.find_cheek_landmarks() fill the array by, from locate()'s two
    results.

    Args:
        frontal (Optional[FrontalCheek]): The frontal constructions, or None.
        observed (Optional[cheek_outline.ObliqueCheek]): The far cheek's observed
            points, or None.
        far_left (bool): Whether the far cheek is the subject's left
            (AnalyzeFace.far_left): the side whose rows the observed points fill.

    Returns:
        NDArray[np.float64]: The ten points in CHEEK_LANDMARK_NAMES order, (x, y)
            px, NaN where absent.
    """
    points = np.full((len(CHEEK_LANDMARK_NAMES), 2), math.nan, dtype=np.float64)
    if frontal is not None:
        points[CHEEK_LM_ZYGION_R] = frontal.zygion_r
        points[CHEEK_LM_ZYGION_L] = frontal.zygion_l
        points[CHEEK_LM_CHEEK_REF_R] = frontal.cheek_ref_r
        points[CHEEK_LM_CHEEK_REF_L] = frontal.cheek_ref_l
        points[CHEEK_LM_SUBMALAR_REF_R] = frontal.submalar_ref_r
        points[CHEEK_LM_SUBMALAR_REF_L] = frontal.submalar_ref_l
    if observed is not None:
        points[CHEEK_LM_APEX_L if far_left else CHEEK_LM_APEX_R] = observed.apex
        if not math.isnan(observed.submalar_depth):
            points[CHEEK_LM_SUBMALAR_L if far_left else CHEEK_LM_SUBMALAR_R] = (
                observed.submalar
            )
    return points


def locate(
    landmarks: Landmarks,
    quarter: bool,
    lateral: bool,
    far_left: bool,
    outline_x: Optional[NDArray[np.float32]],
    outline_y: Optional[NDArray[np.int32]],
    outline_src: Optional[NDArray[np.uint8]],
) -> Tuple[Optional[FrontalCheek], Optional[cheek_outline.ObliqueCheek]]:
    """
    What a face's landmarks and view give the cheek landmarks: the far cheek's
    observed points (observe_far_cheek()) on a quarter face, the frontal
    constructions (frontal_constructions()) on any other face that isn't a
    profile, nothing on a profile; the other result is always None, and so is
    either with fewer than 98 landmarks, a landmark that isn't finite or
    coincident pupils. Never raises. The one rule behind locate_all(),
    AnalyzeFace.find_cheek_landmarks() and the fallback of the cheek measures
    and the Landmarks measure's "Cheek Landmarks" item for a face without the
    frontal_cheek() and oblique_cheek() accessors or cheek_landmarks, so they
    can't drift apart.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px of the standard
            image.
        quarter (bool): Whether the face is a quarter face (AnalyzeFace.quarter).
        lateral (bool): Whether it is a profile (AnalyzeFace.lateral); read only
            when quarter is False.
        far_left (bool): On a quarter face, whether the far cheek is the subject's
            left (AnalyzeFace.far_left).
        outline_x (Optional[NDArray[np.float32]]): The far cheek's outline
            (AnalyzeFace.cheek_outline_x), or None for the contour fallback.
        outline_y (Optional[NDArray[np.int32]]): Its rows (cheek_outline_y).
        outline_src (Optional[NDArray[np.uint8]]): Each row's source
            (cheek_outline_src).

    Returns:
        Tuple[Optional[FrontalCheek], Optional[cheek_outline.ObliqueCheek]]: The
            frontal constructions and the far cheek's observed points, each None
            when the view doesn't carry it or the landmarks give none.
    """
    if quarter:
        observed = observe_far_cheek(
            landmarks, outline_x, outline_y, outline_src, far_left
        )
        return None, observed
    if lateral:
        return None, None
    return frontal_constructions(landmarks), None


def locate_all(
    landmarks: Landmarks,
    pose: Pose,
    far_left: bool,
    outline_x: Optional[NDArray[np.float32]],
    outline_y: Optional[NDArray[np.int32]],
    outline_src: Optional[NDArray[np.uint8]],
) -> NDArray[np.float64]:
    """
    The (10, 2) cheek_landmarks array: landmark_array() of locate()'s two
    results, with a Pose for the view (QUARTER is a quarter face, FRONTAL a
    frontal one, and LATERAL or an undecided DETECT gives all NaN, as does a
    quarter or frontal face with fewer than 98 landmarks, a landmark that isn't
    finite or coincident pupils). Never raises. AnalyzeFace.find_cheek_landmarks()
    keeps the same two results (frontal_cheek() and oblique_cheek()) and fills
    its array by the same landmark_array(), so the two agree exactly.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px of the standard
            image.
        pose (Pose): The face's pose: FRONTAL, QUARTER or LATERAL.
        far_left (bool): On a quarter face, whether the far cheek is the subject's
            left (AnalyzeFace.far_left).
        outline_x (Optional[NDArray[np.float32]]): The far cheek's outline
            (AnalyzeFace.cheek_outline_x), or None for the contour fallback.
        outline_y (Optional[NDArray[np.int32]]): Its rows (cheek_outline_y).
        outline_src (Optional[NDArray[np.uint8]]): Each row's source
            (cheek_outline_src).

    Returns:
        NDArray[np.float64]: The ten points in CHEEK_LANDMARK_NAMES order, (x, y)
            px, NaN where absent.
    """
    frontal, observed = locate(
        landmarks,
        pose == Pose.QUARTER,
        pose != Pose.FRONTAL,
        far_left,
        outline_x,
        outline_y,
        outline_src,
    )
    return landmark_array(frontal, observed, far_left)
