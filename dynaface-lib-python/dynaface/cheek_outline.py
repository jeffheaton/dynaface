"""
The far-cheek outline of a three-quarter (quarter) face and the observed cheek
points on it: the malar apex and the submalar point.

trace_far_cheek() follows the background model's silhouette row by row from the
far canthus to the chin, seeded and bounded by the far WFLW contour, keeping
the contour where the mask has no edge (or, given the gray image, the strongest
intensity edge beside the seed) and marking the mask rows the nose disturbs:
those in the nose band that depart outward from the contour by NOSE_STEP more
than the band's reference, the median departure of the mask rows between the
stomion and canthus levels (the model's edge and the contour each sit a pixel
or two off the silhouette along the whole outline, so the departure itself is
not the nose); contour_outline() is the same band from the contour alone, the
fallback. When the cheek landmarks are located
(AnalyzeFace.find_cheek_landmarks(), through cheek.locate()), oblique_frame(),
level_through() and height_ratio() draw the canthus, stomion and chin levels
from the vanishing point of the canthi and alar-base lines and read Kaptein's
height ratio h along the roll-corrected vertical, and observed_points() finds
the apex (the outline's local prominence maximum nearest h = APEX_H_PRIOR
within APEX_H_WINDOW, else the prior, on a mask row whenever the window has
one) and the submalar point (the deepest point below it against the
apex-to-stomion chord) with every flag the measure reports, reading neither
extremum within the smoothing filter's reach of a row the mask didn't give.
gray_image() is the gray conversion the second edge source reads.

This module imports nothing beyond numpy, math, dataclasses and typing: no
scipy, no cv2, and nothing from dynaface (the foreground mask comes in as an
array, from lateral.foreground_mask()), so it is testable without the models
and ports to one C# file. It is the leaf of the cheek modules: the WFLW indices
the cheek landmarks read, the Landmarks alias and as_landmarks() are defined
here once, and dynaface.cheek imports them along with observed_points(); this
module never imports dynaface.cheek.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: The 98 WFLW landmarks as (x, y) pairs: AnalyzeFace.landmarks (a list of integer
#: tuples) or an (N, 2) array.
Landmarks = Union[Sequence[Tuple[float, float]], NDArray[Any]]

#: How many WFLW landmarks a face has; fewer give no frame and no point.
NUM_LANDMARKS = 98

# The WFLW landmarks the cheek landmarks read (dynaface.cheek re-exports them):
# the chin, the lateral canthi, the alar bases, the nose tip, the subnasale and
# the inner-lip midpoints whose mean is the stomion.
LM_CHIN = 16
LM_CANTHUS_R, LM_CANTHUS_L = 60, 72
LM_ALA_R, LM_ALA_L = 55, 59
LM_NOSE_TIP, LM_SUBNASALE = 54, 57
LM_STOMION = (90, 94)
# The far half contour in polyline order, from the contour's top point to the
# chin: the seed polyline when the far cheek is the right, and the left (the
# half contours the frontal constructions cross too, as cheek.RIGHT_CONTOUR and
# cheek.LEFT_CONTOUR).
RIGHT_FAR_CONTOUR = tuple(range(0, 17))
LEFT_FAR_CONTOUR = tuple(range(32, 15, -1))
# Keyed by far_left: the far lateral canthus.
FAR_CANTHUS = {False: LM_CANTHUS_R, True: LM_CANTHUS_L}

APEX_H_PRIOR = 0.793  # Kaptein 2015: chin-to-apex over chin-to-canthus
APEX_H_WINDOW = (0.71, 0.83)  # Kaptein's observed range: the search window
APEX_MIN_DEV = 0.01  # of H: a bulge smaller than this is not an observed apex
# The definitions' flag, unconditional: apex.*.near_nose within this h of the tip.
APEX_NEAR_NOSE_H = 0.02
# The tip's and the subnasale's levels bound the nose rows (OUTLINE_SRC_NOSE);
# see trace_far_cheek().
NOSE_ROWS = (LM_NOSE_TIP, LM_SUBNASALE)
# Of the tracer's span, the chin-to-far-canthus distance along the roll-corrected
# vertical read between those two landmarks (trace_far_cheek()): not cheek.scale,
# the H every reported height and depth is a fraction of, which height_ratio()
# reads on the far canthus's vertical through the levels and which differs from
# the span by up to 5% on the fixtures. A mask row in the nose band whose
# outward departure from the seed exceeds the band's reference by this is the
# nose. The reference is the median departure of the OUTLINE_SRC_MASK rows
# between the stomion level and the canthus level (h in [sto, 1.0]) outside the
# band, 0.0 when there are none: on
# the renders the model's edge sits 1 to 3 px outside the true silhouette and
# SPIGA's far contour 1 to 2 px inside it, near-uniformly along the outline, so
# the departure alone straddles 0.01 H at 45 degrees where the nose is clear of
# the silhouette, while the nose's own jump (2 px at 50 degrees, 4 px at 52, on
# a 360 px H) stands out against the reference. The chin and jaw rows, below the
# stomion level, depart 5 to 11 px on the renders and 12 to 19 px on photos for
# reasons unrelated to the cheek, so they are not in the reference. Contour and
# intensity-edge rows are never nose rows (_nose_rows()).
NOSE_STEP = 0.01
# Of the tracer's span, like NOSE_STEP: how far the per-row walk may go from the
# seed (the walk only).
OUTLINE_TOL = 0.05
OUTLINE_MIN_MASK = 0.8  # fraction of mask rows a window needs for apex.ok / sub.ok
# Box filter on the deviation profiles before extrema; no extremum is read
# within its halo of a row the mask didn't give (see _HALO_ROWS).
SMOOTH_ROWS = 9
# The rows on each side of a row the mask didn't give (the contour, a nose row
# or the intensity edge) where no local maximum or deepest point is read: the
# box filter's halo (SMOOTH_ROWS // 2 rows whose window reaches such a row) and
# the one row beyond it, where the three-point test reads the halo's first drop
# as a maximum. The contour sits a pixel or two inside the mask's edge, so a
# run of such rows is a step in the outline whose smoothed shoulders would
# otherwise read as an apex or a hollow.
_HALO_ROWS = SMOOTH_ROWS // 2 + 1
# Of H, positive inward. Set on the synthetic renders (no plain face reached
# it; through the band-fit crop the tracer reads a 3 mm hollow at 0.0043 at 45
# degrees and a 4.5 mm hollow at 0.0088, 0.6 to 0.8 of the renders' own
# silhouette depth, and a 1.5 mm hollow as nothing) and checked on 408 Face
# Research Lab London photos, both turns of 102 people at rest and smiling
# (scripts/calibrate_cheek_submalar.py): on faces without a hollow the depth's
# spread over a 3% resize or a 4 px shift is 0.0008 at the 90th percentile
# (0.0015 over all faces, 0.0033 at worst, when the apex changes between the
# prior and an observed bulge), a convex cheek reads no lower than -0.0011,
# and the smiling faces, the nearest thing to a no-hollow population, read
# above it one time in twenty. So this is 2.7 to 3.6 times the floor by
# statistic (three times the typical spread, not the worst case), about 2 px
# in the crop; the 3 mm hollow clears it by 0.0013. Rerun the script after any
# change to SMOOTH_ROWS, the chord rule or the apex rule.
SUBMALAR_MIN_DEPTH = 0.003
# |sin| of the canthi-alar angle below which the solve is singular: a numerical
# guard only.
VANISH_MIN_SIN = 1e-6
# lateral._process_image's foreground threshold (lateral.MASK_ALPHA, the same
# number); the edge is interpolated at this alpha. The closed mask's own edge pair
# rarely brackets it (lateral's even 10 x 10 closing kernel shifts the mask by one
# pixel in x and y, so its edge sits a pixel outside the alpha's crossing on the
# image's right and a pixel inside it on the left), so the raw alpha is searched
# within ALPHA_SEARCH px of the mask's edge for the pair that does
# (_refined_crossing()).
MASK_ALPHA = 32
# Px on either side of the closed mask's edge pixel the pair that brackets
# MASK_ALPHA is looked for, nearest first; when none does, the pixel boundary
# stands in.
ALPHA_SEARCH = 2
# The second edge source: where the mask walk finds no edge (hair beside the
# cheek is foreground too), the strongest horizontal intensity edge in the same
# window stands in when the raw 3 x 3 Sobel response of the binomially blurred
# 8-bit gray image (gray_image()) reaches this (the sum of the Sobel weights'
# magnitudes is 8, so a step of s gray levels reads 2.5 s after the blur). On
# 23 studio photos none of 9,853 mask rows read below it (15 below 60), and a
# floor of 30 would add 9 rows on three photos whose mask had dropped the cheek
# and none on hair.
EDGE_GRAD_MIN = 40.0
# The edge source is off on rows whose plain row ratio (chin row to canthus row)
# is above this: the eye corner's own edge sits beside the canthus row (on
# studio photos the strongest gradient above it wanders to the walk's limit on
# lashes, the eye corner and the brow). Below it the row is OUTLINE_SRC_EDGE,
# above it OUTLINE_SRC_CONTOUR, and both count for nothing, so the cutoff
# changes only the drawn x.
EDGE_MAX_H = 0.95
# The gray image's weights: ITU-R BT.601 luma in OpenCV's fixed point, R, G and
# B over 2 ** _GRAY_SHIFT, rounded (gray_image()).
_GRAY_WEIGHTS = (4899, 9617, 1868)
_GRAY_SHIFT = 14
OUTLINE_SRC_CONTOUR, OUTLINE_SRC_MASK, OUTLINE_SRC_NOSE = 0, 1, 2
# The intensity edge stood in where the mask had none: drawn and flagged, but
# never a mask row for the gates (cheek.mask, apex.ok, the depth) and never a
# row an extremum is read on: on studio photos it sits a median 2 px inside
# the mask's edge on bare skin (40% of rows within 2 px, 87% within 3) and,
# where hair crosses the cheek, 5 to 22 px off on the hair's outer edge,
# reading hollows of 0.006 to 0.047 H that the mask does not.
OUTLINE_SRC_EDGE = 3
# The separable 5-tap binomial blur and the 3 x 3 Sobel's row weights.
_BINOMIAL_5 = (1.0, 4.0, 6.0, 4.0, 1.0)
_SOBEL_ROWS = (1.0, 2.0, 1.0)

_Outline = Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.uint8]]
_Level = Tuple[NDArray[np.float64], NDArray[np.float64]]


def as_landmarks(landmarks: Landmarks) -> Optional[NDArray[np.float64]]:
    """
    The landmarks as an (N, 2) float64 array, N >= NUM_LANDMARKS, or None when
    there are fewer, they aren't (x, y) pairs at all (a ragged list included) or
    any coordinate isn't finite: the one check every cheek construction and the
    oblique search start with, so a NaN chin or an inf canthus gives no point
    rather than raising in the int(round()) of the band rows or running through
    every frontal construction, warning as it goes, behind a good frame.

    Args:
        landmarks (Landmarks): The WFLW landmarks, (x, y) px.

    Returns:
        Optional[NDArray[np.float64]]: The (N, 2) array, or None.
    """
    try:
        pts = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if pts.ndim != 2 or pts.shape[0] < NUM_LANDMARKS or pts.shape[1] != 2:
        return None
    if not np.isfinite(pts).all():
        return None
    return pts


def _unit(v: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
    """The unit vector along v, or None when v is zero or not finite."""
    n = float(np.hypot(v[0], v[1]))
    if n == 0.0 or not math.isfinite(n):
        return None
    return np.asarray(v / n, dtype=np.float64)


def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """The z component of a x b."""
    return float(a[0] * b[1] - a[1] * b[0])


def _empty_outline() -> _Outline:
    """No outline: three empty arrays of the outline's dtypes."""
    return (
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.uint8),
    )


def _halo(rows: NDArray[np.bool_], width: int = _HALO_ROWS) -> NDArray[np.bool_]:
    """
    The flagged rows widened by width rows on each side: where the box filter's
    window reaches a flagged row, and the row beyond.

    Args:
        rows (NDArray[np.bool_]): The flagged rows, (N,).
        width (int): How many rows to add on each side.

    Returns:
        NDArray[np.bool_]: The widened flags, (N,).
    """
    out = np.zeros(len(rows), dtype=bool)
    for i in np.flatnonzero(rows):
        out[max(int(i) - width, 0) : int(i) + width + 1] = True
    return out


def gray_image(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    The 8-bit gray image the second edge source reads (trace_far_cheek()'s
    gray): ITU-R BT.601 luma with OpenCV's fixed-point weights, (R * 4899 +
    G * 9617 + B * 1868 + 8192) >> 14. The library defines it itself so both
    ports trace the same edge rows from one rule: OpenCV's own conversion is
    within one gray level of it, but its rounding differs between its versions
    and code paths, and a shallow edge's row can move with one level.

    Args:
        rgb (NDArray[np.uint8]): The RGB image, (height, width, 3).

    Returns:
        NDArray[np.uint8]: The gray image, (height, width).
    """
    channels = np.asarray(rgb, dtype=np.int32)
    luma = (
        channels[..., 0] * _GRAY_WEIGHTS[0]
        + channels[..., 1] * _GRAY_WEIGHTS[1]
        + channels[..., 2] * _GRAY_WEIGHTS[2]
        + (1 << (_GRAY_SHIFT - 1))
    ) >> _GRAY_SHIFT
    return np.asarray(luma, dtype=np.uint8)


def _smooth(values: NDArray[np.float64], n: int = SMOOTH_ROWS) -> NDArray[np.float64]:
    """
    A box filter of n rows on an edge-padded array (np.convolve), the same length
    as the input.

    Args:
        values (NDArray[np.float64]): The profile, (N,).
        n (int): The box width in rows.

    Returns:
        NDArray[np.float64]: The smoothed profile, (N,).
    """
    v = np.asarray(values, dtype=np.float64)
    if n <= 1 or len(v) == 0:
        return v.copy()
    left = n // 2
    padded = np.pad(v, (left, n - 1 - left), mode="edge")
    return np.asarray(
        np.convolve(padded, np.full(n, 1.0 / n), mode="valid"), dtype=np.float64
    )


def contour_seed(
    cont: ArrayLike, rows: ArrayLike, far_left: Optional[bool] = None
) -> NDArray[np.float64]:
    """
    The far half contour's x in each row, built segment by segment in polyline
    order (the cross_polyline rule the frontal core uses): a row crossed by several
    segments takes the outward-most crossing; rows above the first point
    extrapolate the first segment. No sort and no np.interp: integer landmarks
    give duplicate rows (img4's right contour 15 and 16 share y = 938, img1's left
    17 and 16 y = 884) and the chin reverses (the app's sketch: y 964, 982, 991,
    997, 991), where np.argsort's unstable order and np.interp's tie rule ([4.9,
    5.0, 5.1] on xp [0, 5, 5, 10] gives 98, 200, 202) are implementation-defined
    and a 101 px jump on the sketch, with no C# transliteration. contour_outline()
    uses this same routine, so the fallback and the seed are one rule and the
    persisted arrays match across ports.

    A flat segment (two points on one row) offers both its ends to that row; the
    first and last segments that are not flat extrapolate the rows above the
    first point and below the last.

    Args:
        cont (ArrayLike): The far half contour, (17, 2) px, in polyline order
            (RIGHT_FAR_CONTOUR or LEFT_FAR_CONTOUR).
        rows (ArrayLike): The rows to seed, (N,), top to bottom.
        far_left (Optional[bool]): Whether the far cheek is the subject's left,
            so outward is +x; None reads it from the polyline (its first point
            lies lateral of its last, the chin).

    Returns:
        NDArray[np.float64]: The seed x of each row, (N,).
    """
    pts = np.asarray(cont, dtype=np.float64)
    r = np.asarray(rows, dtype=np.float64)
    xs = np.full(len(r), math.nan, dtype=np.float64)
    if len(r) == 0 or len(pts) == 0:
        return xs
    if len(pts) == 1:
        xs[:] = pts[0, 0]
        return xs
    if far_left is None:
        out = 1.0 if pts[0, 0] >= pts[-1, 0] else -1.0
    else:
        out = 1.0 if far_left else -1.0
    first: Optional[int] = None
    last: Optional[int] = None
    for k in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        if y0 == y1:
            # A flat segment: the row takes the outward-most of its two ends.
            hit = r == y0
            x_flat = x0 if (x0 - x1) * out >= 0 else x1
            cur = xs[hit]
            take = np.isnan(cur) | ((x_flat - cur) * out > 0)
            cur[take] = x_flat
            xs[hit] = cur
            continue
        if first is None:
            first = k
        last = k
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        hit = (r >= lo) & (r <= hi)
        if not hit.any():
            continue
        x = x0 + (r[hit] - y0) * (x1 - x0) / (y1 - y0)
        cur = xs[hit]
        take = np.isnan(cur) | ((x - cur) * out > 0)
        cur[take] = x[take]
        xs[hit] = cur
    missing = np.isnan(xs)
    if not missing.any():
        return xs
    if first is None or last is None:
        # Every point on one row: that row's outward-most x everywhere.
        xs[missing] = pts[np.argmax((pts[:, 0] - pts[0, 0]) * out), 0]
        return xs
    above = missing & (r < pts[0, 1])
    for sel, k in ((above, first), (missing & ~above, last)):
        if sel.any():
            (x0, y0), (x1, y1) = pts[k], pts[k + 1]
            xs[sel] = x0 + (r[sel] - y0) * (x1 - x0) / (y1 - y0)
    return xs


def _band_rows(
    landmarks: NDArray[np.float64], far_left: bool
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    """
    The rows of the far-cheek band, from the far canthus row or the far contour's
    top row (whichever is higher) to the chin row, and the contour seed of each.

    Args:
        landmarks (NDArray[np.float64]): The 98 landmarks, (98, 2) px.
        far_left (bool): Whether the far cheek is the subject's left.

    Returns:
        Tuple[NDArray[np.int32], NDArray[np.float64]]: The rows, (N,), top to
            bottom, and their seed x, (N,).
    """
    idx = LEFT_FAR_CONTOUR if far_left else RIGHT_FAR_CONTOUR
    cont = landmarks[list(idx)]
    canthus_row = int(round(float(landmarks[FAR_CANTHUS[far_left]][1])))
    top = min(canthus_row, int(round(float(cont[0, 1]))))
    chin_row = int(round(float(landmarks[LM_CHIN][1])))
    rows = np.arange(top, chin_row + 1, dtype=np.int32)
    return rows, contour_seed(cont, rows, far_left=far_left)


def _alpha_crossing(alpha_row: NDArray[np.uint8], x_fg: int, x_bg: int) -> float:
    """
    Where the alpha crosses MASK_ALPHA between a foreground pixel and the adjacent
    background pixel, interpolated between their centers; the pixel boundary when
    their alphas don't bracket it (a pixel the closing filled in).

    Args:
        alpha_row (NDArray[np.uint8]): The alpha of the row, (width,).
        x_fg (int): The foreground pixel's column.
        x_bg (int): The background pixel's column, beside it.

    Returns:
        float: The sub-pixel x of the edge.
    """
    a_fg, a_bg = float(alpha_row[x_fg]), float(alpha_row[x_bg])
    if a_fg > MASK_ALPHA >= a_bg:
        t = (a_fg - MASK_ALPHA) / (a_fg - a_bg)
    else:
        t = 0.5
    return x_fg + t * (x_bg - x_fg)


def _refined_crossing(alpha_row: NDArray[np.uint8], x_fg: int, out: int) -> float:
    """
    The sub-pixel edge of a row whose closed mask ends at the foreground pixel
    x_fg, the background beginning at x_fg + out: where the raw alpha crosses
    MASK_ALPHA nearest that edge. The mask's own pair (x_fg, x_fg + out) is tried
    first, then the pairs one and two pixels (ALPHA_SEARCH) inward and outward of
    it, nearest first and inward before outward, and the first pair that brackets
    MASK_ALPHA (foreground side above it, background side at or below it) gives
    the crossing interpolated between its pixel centers (_alpha_crossing). When
    none brackets it (a pixel the closing filled in, or a binary alpha offset by
    more than ALPHA_SEARCH), the mask's pixel boundary x_fg + out / 2.

    Args:
        alpha_row (NDArray[np.uint8]): The alpha of the row, (width,).
        x_fg (int): The closed mask's last foreground pixel on the walk.
        out (int): +1 when outward is +x, else -1: the background pixel is
            x_fg + out.

    Returns:
        float: The sub-pixel x of the edge.
    """
    width = len(alpha_row)
    for distance in range(ALPHA_SEARCH + 1):
        for k in (0,) if distance == 0 else (-distance, distance):
            p = x_fg + k * out
            q = p + out
            if min(p, q) < 0 or max(p, q) >= width:
                continue
            if float(alpha_row[p]) > MASK_ALPHA >= float(alpha_row[q]):
                return _alpha_crossing(alpha_row, p, q)
    return x_fg + 0.5 * out


def _blur5(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """The separable 5-tap binomial blur ([1, 4, 6, 4, 1] / 16), edge-padded."""
    h, w = a.shape
    p = np.pad(a, 2, mode="edge")
    across = np.zeros((h + 4, w), dtype=np.float64)
    for j, k in enumerate(_BINOMIAL_5):
        across += k * p[:, j : j + w]
    down = np.zeros((h, w), dtype=np.float64)
    for j, k in enumerate(_BINOMIAL_5):
        down += k * across[j : j + h, :]
    return down / 256.0


def _sobel_x(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """The 3 x 3 horizontal Sobel response ([-1, 0, 1] across, [1, 2, 1] down)."""
    h, w = a.shape
    p = np.pad(a, 1, mode="edge")
    d = p[:, 2 : w + 2] - p[:, 0:w]
    g = np.zeros((h, w), dtype=np.float64)
    for j, k in enumerate(_SOBEL_ROWS):
        g += k * d[j : j + h, :]
    return g


def _edge_rows(
    gray: NDArray[np.uint8],
    rows: NDArray[np.int32],
    seed: NDArray[np.float64],
    xs: NDArray[np.float64],
    src: NDArray[np.uint8],
    failed: List[int],
    limit: int,
    h_plain: NDArray[np.float64],
) -> None:
    """
    The second edge source: on each failed row whose plain row ratio is at most
    EDGE_MAX_H, the strongest horizontal gradient of the blurred gray image within
    limit px of the seed, when it reaches EDGE_GRAD_MIN, refined to sub-pixel x by
    the parabola through its two neighbors, as OUTLINE_SRC_EDGE.

    Args:
        gray (NDArray[np.uint8]): The gray image, (height, width).
        rows (NDArray[np.int32]): The band rows.
        seed (NDArray[np.float64]): Each row's seed x.
        xs (NDArray[np.float64]): The outline x, updated in place.
        src (NDArray[np.uint8]): The row sources, updated in place.
        failed (List[int]): The indices of the rows the mask walk failed on.
        limit (int): How far from the seed an edge may lie, px.
        h_plain (NDArray[np.float64]): Each row's plain row ratio.
    """
    height, width = gray.shape[:2]
    use = [
        i
        for i in failed
        if h_plain[i] <= EDGE_MAX_H and 0 <= rows[i] < height and math.isfinite(seed[i])
    ]
    if not use or limit <= 0:
        return
    seeds = seed[use]
    r_lo = max(int(rows[use[0]]) - 3, 0)
    r_hi = min(int(rows[use[-1]]) + 3, height - 1)
    c_lo = max(int(math.floor(seeds.min())) - limit - 3, 0)
    c_hi = min(int(math.ceil(seeds.max())) + limit + 3, width - 1)
    if r_hi < r_lo or c_hi < c_lo:
        return
    band = np.asarray(gray[r_lo : r_hi + 1, c_lo : c_hi + 1], dtype=np.float64)
    grad = np.abs(_sobel_x(_blur5(band)))
    for i in use:
        x0 = int(round(seed[i]))
        lo = max(x0 - limit, c_lo)
        hi = min(x0 + limit, c_hi)
        if hi < lo:
            continue
        g = grad[int(rows[i]) - r_lo, lo - c_lo : hi - c_lo + 1]
        j = int(np.argmax(g))
        if g[j] < EDGE_GRAD_MIN:
            continue
        offset = 0.0
        if 0 < j < len(g) - 1:
            denom = g[j - 1] - 2.0 * g[j] + g[j + 1]
            if denom < 0.0:
                offset = min(max(0.5 * (g[j - 1] - g[j + 1]) / denom, -0.5), 0.5)
        xs[i] = lo + j + offset
        src[i] = OUTLINE_SRC_EDGE


def trace_far_cheek(
    foreground: NDArray[np.bool_],
    alpha: NDArray[np.uint8],
    landmarks: Landmarks,
    far_left: bool,
    tol: float = OUTLINE_TOL,
    gray: Optional[NDArray[np.uint8]] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.uint8]]:
    """
    The far cheek's silhouette row by row, from the far canthus row or the far
    contour's top row (landmark 0, or 32), whichever is higher, to the chin row
    (16), seeded and bounded by that half contour.

    Key lines:

    ```text
    idx = LEFT_FAR_CONTOUR (32..16) if far_left else RIGHT_FAR_CONTOUR (0..16)
    cont = L[idx]; top = min(row of L[FAR_CANTHUS[far_left]], row of cont[0])
    rows = arange(top, chin row + 1); seed = contour_seed(cont, rows)
    span = (L[16] - L[FAR_CANTHUS[far_left]]) @ oblique_frame(L).down
        # the tracer's own scale, read between the two landmarks: not
        # cheek.scale (H), which height_ratio() reads through the levels
    limit = int(tol * span); out = +1 if far_left else -1  # outward: away from the face
    for i, y in enumerate(rows):
        x = clip(int(round(seed[i])), 0, width - 1)
        src[i] = OUTLINE_SRC_CONTOUR; xs[i] = seed[i]
        if foreground[y, x]: walk x outward while foreground, up to limit,
                             never past the border
        else:                walk x inward while background, up to limit,
                             never past the border
        if the change was found within limit: xs[i] = the alpha == MASK_ALPHA
             crossing interpolated between the two pixels that bracket it,
             the mask's own edge pair or the nearest pair within ALPHA_SEARCH
             px of it, else the mask's pixel boundary (sub-pixel, float32);
             src[i] = OUTLINE_SRC_MASK
        # a walk that reaches the limit or the border keeps the seed, unless
        # gray gives an intensity edge in the same window (OUTLINE_SRC_EDGE)
    depart = (xs - seed) * out                              # outward, px
    ref = median(depart[MASK rows with h in [sto, 1.0] outside the band])
          or 0.0 when there are none     # the band's reference: the model's
          # edge and the contour each sit a px or two off the silhouette
    nose = MASK rows between the levels of landmarks 54 and 57 where
           depart - ref > NOSE_STEP * span, extended upward from the tip's row
           while that holds       # only the mask rows the nose disturbs
    xs[nose] = seed[nose]; src[nose] = OUTLINE_SRC_NOSE
    return xs, rows, src
    ```

    The levels of 54 and 57 and the stomion's (sto) are read as h along the
    roll-corrected vertical (height_ratio() of each row's seed point), so a
    rolled head keeps the same rows. Rows outside the image keep the seed.

    Args:
        foreground (NDArray[np.bool_]): The closed foreground mask of the standard
            image, (height, width), from lateral.foreground_mask().
        alpha (NDArray[np.uint8]): The raw alpha it was thresholded from, (height,
            width).
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px of the same image.
        far_left (bool): Whether the far cheek is the subject's left.
        tol (float): How far the walk may go from the seed, as a fraction of the
            tracer's span (the chin-to-far-canthus distance along the
            roll-corrected vertical, read between the two landmarks: within 5%
            of cheek.scale on the fixtures, but not the same number).
        gray (Optional[NDArray[np.uint8]]): The gray image, (height, width),
            from gray_image() (the BT.601 luma in OpenCV's fixed point, defined
            here so both ports read the same edge), for the second edge source
            on the rows the mask walk fails on: the strongest horizontal
            gradient within the same window (a 5-tap binomial blur and a 3 x 3
            Sobel, numpy only), accepted at EDGE_GRAD_MIN and marked
            OUTLINE_SRC_EDGE, off on rows whose plain row ratio is above
            EDGE_MAX_H (the eye corner). None keeps the seed on those rows.

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.uint8]]: The
            outline's x in each row (sub-pixel), the rows, and each row's source
            (OUTLINE_SRC_*), in that order (x first, then the rows): the order of
            AnalyzeFace.cheek_outline_x, _y and _src, which contour_outline()
            returns too and observed_points() unpacks.
    """
    pts = as_landmarks(landmarks)
    if pts is None:
        return _empty_outline()
    frame = oblique_frame(pts)
    if frame is None:
        return _empty_outline()
    rows, seed = _band_rows(pts, far_left)
    if len(rows) == 0:
        return _empty_outline()
    height, width = foreground.shape[:2]
    chin, canthus = pts[LM_CHIN], pts[FAR_CANTHUS[far_left]]
    # The tracer's span: the two landmarks' distance along the vertical, which
    # NOSE_STEP and OUTLINE_TOL are fractions of (not cheek.scale).
    span = float((chin - canthus) @ frame.down)
    limit = int(tol * span) if span > 0.0 else 0
    out = 1 if far_left else -1
    xs = np.array(seed, dtype=np.float64)
    src = np.full(len(rows), OUTLINE_SRC_CONTOUR, dtype=np.uint8)
    failed: List[int] = []
    for i, y in enumerate(rows):
        if y < 0 or y >= height or not math.isfinite(seed[i]):
            continue
        row = foreground[y]
        x = min(max(int(round(float(seed[i]))), 0), width - 1)
        inside = bool(row[x])
        step = out if inside else -out
        found: Optional[Tuple[int, int]] = None
        for _ in range(limit):
            nx = x + step
            if nx < 0 or nx >= width:
                break  # the border: keep the seed
            if bool(row[nx]) != inside:
                found = (x, nx) if inside else (nx, x)
                break
            x = nx
        if found is None:
            failed.append(i)
            continue
        xs[i] = _refined_crossing(alpha[y], found[0], out)
        src[i] = OUTLINE_SRC_MASK
    if gray is not None and failed:
        # The edge source's height is read on image rows, from its own span: the
        # tracer's span above stays what NOSE_STEP is a fraction of below.
        chin_row, canthus_row = float(chin[1]), float(canthus[1])
        row_span = chin_row - canthus_row
        if row_span > 0.0:
            h_plain = (chin_row - rows.astype(np.float64)) / row_span
        else:
            h_plain = np.full(len(rows), math.inf, dtype=np.float64)
        _edge_rows(gray, rows, seed, xs, src, failed, limit, h_plain)
    # The nose rows: the mask rows between the tip's and the subnasale's levels
    # where the traced edge departs outward from the seed by NOSE_STEP beyond the
    # band's reference, and upward from the tip's row while it keeps doing so.
    seed_pts = np.stack([seed, rows.astype(np.float64)], axis=1)
    h_rows, _ = _height_ratios(seed_pts, chin, canthus, frame)
    h_tip, _ = height_ratio(pts[NOSE_ROWS[0]], chin, canthus, frame)
    h_sub, _ = height_ratio(pts[NOSE_ROWS[1]], chin, canthus, frame)
    stomion = 0.5 * (pts[LM_STOMION[0]] + pts[LM_STOMION[1]])
    h_sto, _ = height_ratio(stomion, chin, canthus, frame)
    if math.isfinite(h_tip) and math.isfinite(h_sub) and span > 0.0:
        nose = _nose_rows(xs, seed, src, h_rows, out, span, (h_tip, h_sub), h_sto)
        xs[nose] = seed[nose]
        src[nose] = OUTLINE_SRC_NOSE
    return xs.astype(np.float32), rows, src


def _nose_rows(
    xs: NDArray[np.float64],
    seed: NDArray[np.float64],
    src: NDArray[np.uint8],
    h_rows: NDArray[np.float64],
    out: int,
    span: float,
    band_h: Tuple[float, float],
    h_sto: float,
) -> NDArray[np.bool_]:
    """
    The nose rows of a traced outline.

    A row is a nose row when it came from the mask (OUTLINE_SRC_MASK), lies in
    the nose band (h between the tip's and the subnasale's levels, band_h) and
    its outward departure from the seed, (xs - seed) * out, exceeds the reference
    by more than NOSE_STEP * span; the band extends upward from its top row
    while consecutive rows keep departing so. The reference is the median
    departure of the mask rows with h in [h_sto, 1.0] outside the band (the
    cheek between the stomion and canthus levels), 0.0 when there are none, so a
    model edge that sits a near-uniform pixel or two outside the contour along
    the whole outline reads as no nose, and the chin and jaw rows below the
    stomion level, which depart for reasons unrelated to the cheek, have no say.
    Contour and intensity-edge rows are never nose rows, whatever the reference.

    Args:
        xs (NDArray[np.float64]): The outline x of each row, before the nose rows
            are replaced by the seed.
        seed (NDArray[np.float64]): Each row's contour seed x.
        src (NDArray[np.uint8]): Each row's source (OUTLINE_SRC_*).
        h_rows (NDArray[np.float64]): Each row's height ratio at its seed point.
        out (int): +1 when outward is +x (the far cheek the subject's left),
            else -1.
        span (float): The tracer's span, px: the chin-to-far-canthus distance
            along the roll-corrected vertical, read between the two landmarks
            (trace_far_cheek()), not cheek.scale.
        band_h (Tuple[float, float]): The h of the nose tip's and the subnasale's
            levels, in either order.
        h_sto (float): The h of the stomion level.

    Returns:
        NDArray[np.bool_]: Which rows are nose rows, (N,).
    """
    lo_h, hi_h = min(band_h), max(band_h)
    is_mask = src == OUTLINE_SRC_MASK
    with np.errstate(invalid="ignore"):
        outward = (xs - seed) * out
        in_band = (h_rows >= lo_h) & (h_rows <= hi_h)
        cheek = is_mask & ~in_band & (h_rows >= h_sto) & (h_rows <= 1.0)
        reference = float(np.median(outward[cheek])) if cheek.any() else 0.0
        depart = is_mask & (outward - reference > NOSE_STEP * span)
    nose = np.asarray(depart & in_band, dtype=np.bool_)
    band_idx = np.flatnonzero(in_band)
    if len(band_idx) > 0 and depart[band_idx[0]]:
        i = int(band_idx[0]) - 1
        while i >= 0 and depart[i]:
            nose[i] = True
            i -= 1
    return nose


def contour_outline(
    landmarks: Landmarks, far_left: bool
) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.uint8]]:
    """
    The fallback: contour_seed() over the same rows as trace_far_cheek(), all
    OUTLINE_SRC_CONTOUR.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.
        far_left (bool): Whether the far cheek is the subject's left.

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.uint8]]: The
            outline's x in each row, the rows, and each row's source, as
            trace_far_cheek() returns them.
    """
    pts = as_landmarks(landmarks)
    if pts is None:
        return _empty_outline()
    rows, seed = _band_rows(pts, far_left)
    return (
        seed.astype(np.float32),
        rows,
        np.full(len(rows), OUTLINE_SRC_CONTOUR, dtype=np.uint8),
    )


@dataclass(frozen=True)
class ObliqueFrame:
    """
    The level frame of a quarter face: where its horizontals converge.

    Attributes:
        down: Unit normal of the canthi line (60 -> 72), toward the chin: the
            roll-corrected vertical.
        vanishing: Where the canthi line and the alar-base line (55 -> 59) meet,
            (x, y) px, or None when the solve is singular (VANISH_MIN_SIN).
        mean_dir: The mean direction of the two lines, used only when the solve is
            singular.
    """

    down: NDArray[np.float64]
    vanishing: Optional[NDArray[np.float64]]
    mean_dir: NDArray[np.float64]


def oblique_frame(landmarks: Landmarks) -> Optional[ObliqueFrame]:
    """
    The level frame of the 98 landmarks: down from the canthi line, the vanishing
    point of the canthi and alar-base lines (None when |sin| of their angle is
    below VANISH_MIN_SIN) and their mean direction.

    Args:
        landmarks (Landmarks): The 98 WFLW landmarks, (x, y) px.

    Returns:
        Optional[ObliqueFrame]: The frame, or None when the canthi coincide or
            there are fewer than 98 landmarks.
    """
    pts = as_landmarks(landmarks)
    if pts is None:
        return None
    canthus_r, canthus_l = pts[LM_CANTHUS_R], pts[LM_CANTHUS_L]
    across = _unit(canthus_l - canthus_r)
    if across is None:
        return None
    down = np.array([-across[1], across[0]], dtype=np.float64)
    if float((pts[LM_CHIN] - 0.5 * (canthus_r + canthus_l)) @ down) < 0.0:
        down = -down
    alar = _unit(pts[LM_ALA_L] - pts[LM_ALA_R])
    vanishing: Optional[NDArray[np.float64]] = None
    mean_dir = across
    if alar is not None:
        if float(across @ alar) < 0.0:
            alar = -alar
        sin = _cross(across, alar)
        if abs(sin) >= VANISH_MIN_SIN:
            # canthus_r + s * across = ala_r + t * alar: cross both sides with alar.
            s = _cross(pts[LM_ALA_R] - canthus_r, alar) / sin
            vanishing = np.asarray(canthus_r + s * across, dtype=np.float64)
        mean = _unit(across + alar)
        if mean is not None:
            mean_dir = mean
    return ObliqueFrame(down=down, vanishing=vanishing, mean_dir=mean_dir)


def level_through(
    frame: ObliqueFrame, anchor: ArrayLike
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    The level line through anchor: from the vanishing point through it, else along
    mean_dir.

    Args:
        frame (ObliqueFrame): The level frame.
        anchor (ArrayLike): A point on the level, (x, y) px, such as the chin.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: The line as a point (the
            anchor) and a unit direction.
    """
    point = np.asarray(anchor, dtype=np.float64)
    if frame.vanishing is not None:
        direction = _unit(point - frame.vanishing)
        if direction is not None:
            return point, direction
    return point, np.array(frame.mean_dir, dtype=np.float64)


def _levels_at_verticals(
    level: _Level, points: NDArray[np.float64], frame: ObliqueFrame
) -> NDArray[np.float64]:
    """
    Where the level meets the vertical (direction frame.down) through each point:
    a 2x2 solve per point, vectorized.

    Args:
        level (_Level): A level from level_through().
        points (NDArray[np.float64]): The points, (N, 2) px.
        frame (ObliqueFrame): The level frame.

    Returns:
        NDArray[np.float64]: The meeting points, (N, 2) px; NaN when the level
            runs along the vertical.
    """
    point, direction = level
    down = frame.down
    # point + s * direction = p + t * down: cross both sides with down.
    den = _cross(direction, down)
    if abs(den) < 1e-12:
        return np.full(points.shape, math.nan, dtype=np.float64)
    b = points - point
    s = (b[:, 0] * down[1] - b[:, 1] * down[0]) / den
    return np.asarray(point + s[:, None] * direction, dtype=np.float64)


def level_at_vertical(
    level: Tuple[NDArray[np.float64], NDArray[np.float64]],
    p: ArrayLike,
    frame: ObliqueFrame,
) -> NDArray[np.float64]:
    """
    Where that level meets the vertical (direction frame.down) through p: a 2x2
    solve.

    Args:
        level (Tuple[NDArray[np.float64], NDArray[np.float64]]): A level from
            level_through().
        p (ArrayLike): A point on the vertical, (x, y) px.
        frame (ObliqueFrame): The level frame.

    Returns:
        NDArray[np.float64]: The meeting point, (x, y) px.
    """
    points = np.asarray(p, dtype=np.float64).reshape(1, 2)
    return np.asarray(_levels_at_verticals(level, points, frame)[0], dtype=np.float64)


def _height_ratios(
    points: NDArray[np.float64],
    chin: ArrayLike,
    far_canthus: ArrayLike,
    frame: ObliqueFrame,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    height_ratio() of every point at once.

    Args:
        points (NDArray[np.float64]): The points, (N, 2) px.
        chin (ArrayLike): The chin, landmark 16.
        far_canthus (ArrayLike): The far lateral canthus.
        frame (ObliqueFrame): The level frame.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: h and H of each point,
            (N,) each; NaN where H is zero.
    """
    chin_pts = _levels_at_verticals(level_through(frame, chin), points, frame)
    can_pts = _levels_at_verticals(level_through(frame, far_canthus), points, frame)
    up = -frame.down
    scale = (can_pts - chin_pts) @ up
    rise = (points - chin_pts) @ up
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(scale != 0.0, rise / scale, math.nan)
    return np.asarray(h, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def height_ratio(
    p: ArrayLike, chin: ArrayLike, far_canthus: ArrayLike, frame: ObliqueFrame
) -> Tuple[float, float]:
    """
    Kaptein's h at p and H on p's vertical:

    ```text
    chin_pt = level_at_vertical(level_through(frame, chin), p, frame)
    can_pt = level_at_vertical(level_through(frame, far_canthus), p, frame)
    H = (can_pt - chin_pt) @ -frame.down;  h = (p - chin_pt) @ -frame.down / H
    ```

    Args:
        p (ArrayLike): The point, (x, y) px, such as an outline row.
        chin (ArrayLike): The chin, landmark 16.
        far_canthus (ArrayLike): The far lateral canthus, FAR_CANTHUS[far_left].
        frame (ObliqueFrame): The level frame.

    Returns:
        Tuple[float, float]: h, the chin-to-point over chin-to-canthus distance
            along the vertical at p, and H, that chin-to-canthus distance, px.
    """
    points = np.asarray(p, dtype=np.float64).reshape(1, 2)
    h, scale = _height_ratios(points, chin, far_canthus, frame)
    return float(h[0]), float(scale[0])


@dataclass
class ObliqueCheek:
    """
    The observed points of a quarter face's far cheek, with what the measure
    reports and draws. Heights and depths are fractions of H (scale), points are
    pixels of the standard image.

    Attributes:
        far_left: Whether the far cheek is the subject's left.
        outline, outline_src: The band rows, (N, 2) px from the canthus level to
            the stomion level, and each row's source (OUTLINE_SRC_*), for drawing.
        chord: The outline at the canthus level and at the stomion level.
        levels: "canthus", "stomion" and "chin", each a (point, direction) line
            from level_through(), for drawing.
        apex, apex_h, apex_dev, apex_prior, apex_near_nose: The malar apex, its
            height ratio, its outward deviation from the chord, whether the prior
            stood in for a local maximum, and whether its h is within
            APEX_NEAR_NOSE_H of the nose tip's.
        submalar, submalar_h: The submalar point, the deepest row between the
            apex and the stomion level against the chord joining them, never
            either end (NaN when its depth is), and its height ratio.
        submalar_depth, submalar_depth_px, submalar_present: Its depth inward
            from that chord, positive inward, so a convex cheek reads about
            zero or slightly negative (NaN when too few rows came from the
            mask), the same in px, and 1.0 when the depth reaches
            SUBMALAR_MIN_DEPTH, 0.0 below, NaN when the depth is. With
            submalar_present 0.0 the point is still reported, so a caller can
            see where the hollow was looked for: it is then wherever the
            outline comes nearest the chord, on a convex cheek a row beside
            one of the chord's ends, not a landmark, and the measure doesn't
            draw it.
        nose_on, nose_gap: Whether the nose joins the far outline, and the signed
            gap from the nose tip to the outline at the tip's row, positive inside
            the face.
        mask_fraction, window_mask, below_mask: The fraction of mask rows in the
            band, in the apex window, and from the apex down (OUTLINE_SRC_MASK
            only: an OUTLINE_SRC_EDGE row shapes the outline but counts as none).
        scale: H, the chin-to-far-canthus distance along the far canthus's
            vertical, px.
    """

    far_left: bool
    outline: NDArray[np.float64]
    outline_src: NDArray[np.uint8]
    chord: Tuple[NDArray[np.float64], NDArray[np.float64]]
    levels: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]
    apex: NDArray[np.float64]
    apex_h: float
    apex_dev: float
    apex_prior: bool
    apex_near_nose: bool
    submalar: NDArray[np.float64]
    submalar_h: float
    submalar_depth: float
    submalar_depth_px: float
    submalar_present: float
    nose_on: bool
    nose_gap: float
    mask_fraction: float
    window_mask: float
    below_mask: float
    scale: float


def _outward_normal(
    a: NDArray[np.float64], b: NDArray[np.float64], out: float
) -> Optional[NDArray[np.float64]]:
    """The unit normal of the chord a -> b whose x has the sign of out."""
    normal = _unit(np.array([-(b[1] - a[1]), b[0] - a[0]], dtype=np.float64))
    if normal is None:
        return None
    if normal[0] * out < 0.0 or (normal[0] == 0.0 and normal[1] * out < 0.0):
        normal = -normal
    return normal


def observed_points(
    landmarks: Landmarks,
    outline_x: Optional[NDArray[np.float32]],
    outline_y: Optional[NDArray[np.int32]],
    outline_src: Optional[NDArray[np.uint8]],
    far_left: bool,
) -> Optional[ObliqueCheek]:
    """
    The observed points of the far cheek, or None when no frame, band, scale or
    chord can be formed: fewer than 98 landmarks or one that isn't finite,
    coincident canthi (no frame), an empty outline or one whose three arrays
    differ in length, fewer than 3 band rows between the canthus and stomion
    levels, a non-finite or non-positive H, or chord ends that coincide.

    Key lines:

    ```text
    if outline_x is None:
        outline_x, outline_y, outline_src = contour_outline(L, far_left)
    frame = oblique_frame(L); pts = stack([x, y]); out = +1 if far_left else -1
    h, Hs = height_ratio of every row; sto = h of the stomion level at each
        row's x
    band = (h <= 1.0) & (h >= sto); P, S, Hh = pts[band], src[band], h[band]
    a, b = P[argmin |Hh - 1.0|], P[argmin |Hh - sto|]
        # the chord: the band rows nearest the two levels
    n = unit normal of (b - a) with sign out
    dev = smooth(((P - a) @ n), SMOOTH_ROWS) / Hs[band]   # outward, fraction of H
    h_nose = height_ratio(L[54])[0]
    nose_gap = (outline x - tip x) * out at the tip's level / H
        # positive: tip inside the face
    nose_on = any(S == OUTLINE_SRC_NOSE) or nose_gap <= 0
    unsteady = rows with S != MASK widened by SMOOTH_ROWS // 2 + 1 on each side
        # the box filter's halo across a source step (the contour sits a px
        # or two inside the mask's edge), and the row beyond it where the
        # three-point test sees the halo's first drop as a maximum: no local
        # maximum and no deepest point is read there
    excluded = NOSE rows widened the same way
        # where the silhouette is the nose, not the cheek: the prior never
        # lands there
    window = 0.71 <= Hh <= 0.83
    cand = [i for i in 1..len(P)-2 if dev[i] > dev[i-1] and dev[i] >= dev[i+1]
            and APEX_MIN_DEV <= dev[i] and window[i] and not unsteady[i]]
    i_apex = min(cand, key=|Hh - 0.793|) if cand
             else argmin |Hh - 0.793| over the window's MASK rows not
                  excluded (prior), or over every row not excluded when no
                  row of the window came from the mask (hair over all of it:
                  a contour or edge row stands in, apex.ok is 0 and no depth
                  is read)
    apex_near_nose = abs(Hh[i_apex] - h_nose) <= APEX_NEAR_NOSE_H
        # unconditional: the definitions' 0.02
    window_mask = mean(S[window rows] == MASK); below_mask = mean(S[i_apex:] == MASK)
    # submalar: below the apex to the stomion level, inward deviation from the
    # apex-to-stomion chord, over the rows between its ends (the ends deviate
    # by nothing but the filter's edge padding)
    Q = P[i_apex:]; m = unit normal of (Q[-1] - Q[0]) pointing inward
    inward = smooth(((Q - Q[0]) @ m), SMOOTH_ROWS) / Hs
    inward[unsteady rows] = inward[0] = inward[-1] = -inf
    if below_mask >= OUTLINE_MIN_MASK and S[i_apex] == MASK and len(Q) > 4:
        j = argmax(inward); depth = inward[j]; depth_px = depth * H
        present = depth >= SUBMALAR_MIN_DEPTH; submalar = Q[j]
    else: depth = depth_px = present = NaN; submalar = (NaN, NaN)
    ```

    smooth() is np.convolve with a box of SMOOTH_ROWS on an edge-padded array; the
    three-point local-maximum test replaces scipy's find_peaks. Only
    OUTLINE_SRC_MASK rows count toward the three mask fractions: an
    OUTLINE_SRC_EDGE row gives its x to the geometry and nothing to the gates.
    An observed apex (apex_prior False) or a submalar point therefore always
    sits on a mask row with mask rows throughout its smoothing window, and so
    does the prior whenever the window has a mask row, so its x and the chord
    the submalar point is read against rest on a measured edge (a run of
    contour or intensity-edge rows at Kaptein's height would otherwise carry
    the apex by up to the walk tolerance and tilt that chord into a hollow, or
    hide one). Only when no row of the window came from the mask does the
    prior sit on a contour or edge row, at h = APEX_H_PRIOR, and then no depth
    is read. The submalar point is reported whether or not a hollow is
    present: with submalar_present 0.0 it is wherever the outline comes nearest
    the chord (on a convex cheek a row beside one of the chord's ends), so a
    caller can see where the hollow was looked for; the measure draws only a
    present one.

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
        Optional[ObliqueCheek]: The observed points and their flags, or None.
    """
    pts = as_landmarks(landmarks)
    if pts is None:
        return None
    frame = oblique_frame(pts)
    if frame is None:
        return None
    if outline_x is None or outline_y is None or outline_src is None:
        outline_x, outline_y, outline_src = contour_outline(pts, far_left)
    xs = np.asarray(outline_x, dtype=np.float64)
    ys = np.asarray(outline_y, dtype=np.float64)
    src = np.asarray(outline_src, dtype=np.uint8)
    if len(xs) == 0 or len(xs) != len(ys) or len(xs) != len(src):
        return None
    chin, canthus = pts[LM_CHIN], pts[FAR_CANTHUS[far_left]]
    stomion = 0.5 * (pts[LM_STOMION[0]] + pts[LM_STOMION[1]])
    out = 1.0 if far_left else -1.0
    points = np.stack([xs, ys], axis=1)
    h, scales = _height_ratios(points, chin, canthus, frame)
    levels = {
        "canthus": level_through(frame, canthus),
        "stomion": level_through(frame, stomion),
        "chin": level_through(frame, chin),
    }
    sto_pts = _levels_at_verticals(levels["stomion"], points, frame)
    sto, _ = _height_ratios(sto_pts, chin, canthus, frame)
    _, scale = height_ratio(canthus, chin, canthus, frame)
    with np.errstate(invalid="ignore"):
        band = (h <= 1.0) & (h >= sto) & np.isfinite(h) & np.isfinite(sto)
    if int(band.sum()) < 3 or not math.isfinite(scale) or scale <= 0.0:
        return None
    P, S, Hh, Hs = points[band], src[band], h[band], scales[band]
    a = P[int(np.argmin(np.abs(Hh - 1.0)))]
    b = P[int(np.argmin(np.abs(Hh - sto[band])))]
    normal = _outward_normal(a, b, out)
    if normal is None:
        return None
    dev = _smooth((P - a) @ normal) / Hs
    h_nose, _ = height_ratio(pts[NOSE_ROWS[0]], chin, canthus, frame)
    # The signed tip-to-outline gap at the tip's level, positive with the tip
    # inside the face.
    tip_row = int(np.argmin(np.abs(h - h_nose)))
    nose_gap = float((xs[tip_row] - pts[NOSE_ROWS[0]][0]) * out / scale)
    nose_rows = S == OUTLINE_SRC_NOSE
    nose_on = bool(nose_rows.any()) or nose_gap <= 0.0
    is_mask = S == OUTLINE_SRC_MASK
    # No local maximum or deepest point is read within the box filter's halo
    # of a row the mask didn't give (the contour, a nose row or the intensity
    # edge): the contour sits a pixel or two inside the mask's edge, so a run
    # of such rows is a step whose smoothed shoulders would read as an apex or
    # a hollow. The prior keeps clear of the nose rows, where the silhouette is
    # the nose and not the cheek, and prefers the window's mask rows (below).
    unsteady = _halo(~is_mask)
    excluded = _halo(nose_rows)
    window = (Hh >= APEX_H_WINDOW[0]) & (Hh <= APEX_H_WINDOW[1])
    cand = [
        i
        for i in range(1, len(P) - 1)
        if dev[i] > dev[i - 1]
        and dev[i] >= dev[i + 1]
        and dev[i] >= APEX_MIN_DEV
        and window[i]
        and not unsteady[i]
    ]
    if cand:
        i_apex = min(cand, key=lambda i: abs(float(Hh[i]) - APEX_H_PRIOR))
        apex_prior = False
    else:
        # The prior: of the window's mask rows outside the nose rows' halo, the
        # one nearest Kaptein's height, so the apex's x and the chord the
        # submalar point is read against rest on a measured edge whenever the
        # window has one (a run of contour or intensity-edge rows at that
        # height would otherwise carry the apex by up to the walk tolerance
        # and tilt the chord into a hollow, or hide one). When no row of the
        # window came from the mask (hair over all of it: apex.ok is 0 and no
        # depth is read below), the row nearest that height of any source but
        # a nose row.
        distance = np.abs(Hh - APEX_H_PRIOR)
        preferred = window & is_mask & ~excluded
        if preferred.any():
            distance = np.where(preferred, distance, math.inf)
        elif not excluded.all():
            distance = np.where(excluded, math.inf, distance)
        i_apex = int(np.argmin(distance))
        apex_prior = True
    apex_near_nose = abs(float(Hh[i_apex]) - h_nose) <= APEX_NEAR_NOSE_H
    window_mask = float(is_mask[window].mean()) if window.any() else 0.0
    below_mask = float(is_mask[i_apex:].mean())
    mask_fraction = float(is_mask.mean())
    # The submalar point: the deepest row below the apex against the chord from
    # the apex to the band's last row (the stomion level), between those two
    # rows: the chord's ends deviate by nothing but the filter's edge padding,
    # so without this the point could be the apex itself.
    Q = P[i_apex:]
    submalar = np.full(2, math.nan, dtype=np.float64)
    submalar_h = depth = depth_px = present = math.nan
    inward_normal = _outward_normal(Q[0], Q[-1], -out) if len(Q) > 1 else None
    # Read only below an apex the mask gave: a chord from a row the contour or
    # the intensity edge stood in on (the prior with no mask row in the window)
    # measures that row's offset from the silhouette, not a hollow.
    if (
        below_mask >= OUTLINE_MIN_MASK
        and bool(is_mask[i_apex])
        and len(Q) > 4
        and inward_normal is not None
    ):
        inward = _smooth((Q - Q[0]) @ inward_normal) / Hs[i_apex:]
        inward = np.where(unsteady[i_apex:], -math.inf, inward)
        inward[0] = inward[-1] = -math.inf
        if np.isfinite(inward).any():
            j = int(np.argmax(inward))
            depth = float(inward[j])
            depth_px = depth * scale
            present = 1.0 if depth >= SUBMALAR_MIN_DEPTH else 0.0
            submalar = np.array(Q[j], dtype=np.float64)
            submalar_h = float(Hh[i_apex + j])
    return ObliqueCheek(
        far_left=far_left,
        outline=np.array(P, dtype=np.float64),
        outline_src=np.array(S, dtype=np.uint8),
        chord=(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)),
        levels=levels,
        apex=np.array(P[i_apex], dtype=np.float64),
        apex_h=float(Hh[i_apex]),
        apex_dev=float(dev[i_apex]),
        apex_prior=apex_prior,
        apex_near_nose=apex_near_nose,
        submalar=submalar,
        submalar_h=submalar_h,
        submalar_depth=depth,
        submalar_depth_px=depth_px,
        submalar_present=present,
        nose_on=nose_on,
        nose_gap=nose_gap,
        mask_fraction=mask_fraction,
        window_mask=window_mask,
        below_mask=below_mask,
        scale=float(scale),
    )
