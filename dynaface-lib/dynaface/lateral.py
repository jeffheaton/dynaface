from typing import Any, Iterable, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import logging
from enum import Enum, auto
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from PIL import Image
from rembg import remove  # type: ignore
from scipy.signal import find_peaks, savgol_filter  # <-- Savitzky–Golay

from dynaface import models, util

logger = logging.getLogger(__name__)

LATERAL_LANDMARK_NAMES = [
    "Soft Tissue Glabella",
    "Soft Tissue Nasion",
    "Nasal Tip",
    "Subnasal Point",
    "Mento Labial Point",
    "Soft Tissue Pogonion",
]

# ================= CONSTANTS =================
DEBUG = False
CROP_MARGIN_RATIO: float = 0.05

# 1st Derivative (dx) Controls
DX1_SCALE_FACTOR: float = 15.0
DX1_OFFSET: float = 0.0

# 2nd Derivative (ddx) Controls
DX2_SCALE_FACTOR: float = 15.0
DX2_OFFSET: float = 0.0

X_PAD_RATIO: float = 0.1
Y_PAD_RATIO: float = 0.3

# Landmark constants for lateral landmarks (landmark, x/y)
LATERAL_LM_SOFT_TISSUE_GLABELLA = 0
LATERAL_LM_SOFT_TISSUE_NASION = 1
LATERAL_LM_NASAL_TIP = 2
LATERAL_LM_SUBNASAL_POINT = 3
LATERAL_LM_MENTO_LABIAL_POINT = 4
LATERAL_LM_SOFT_TISSUE_POGONION = 5


def _process_image(
    input_image: Image.Image,
) -> Tuple[Image.Image, NDArray[Any], int, int]:
    """
    Remove background, threshold, morph-close, invert.
    """
    output_image: Image.Image = remove(input_image, session=models.rembg_session)  # type: ignore
    grayscale: Image.Image = output_image.convert("L")

    # Binary
    binary_threshold: int = 32
    binary: Image.Image = grayscale.point(lambda p: 255 if p > binary_threshold else 0)  # type: ignore
    binary_np: NDArray[Any] = np.array(binary)

    # Morph close + invert
    kernel: NDArray[Any] = np.ones((10, 10), np.uint8)
    binary_np = cv2.morphologyEx(binary_np, cv2.MORPH_CLOSE, kernel)
    binary_np = 255 - binary_np

    height, width = binary_np.shape
    return input_image, binary_np, width, height


def _shift_sagittal_profile(sagittal_x: NDArray[Any]) -> tuple[NDArray[Any], float]:
    """
    Shift so lowest x becomes 0.
    """
    min_x = np.min(sagittal_x)
    return sagittal_x - min_x, float(min_x)


def _extract_sagittal_profile(
    binary_np: NDArray[np.uint8],
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    For each row (y), take first black pixel column (x).
    """
    height, _ = binary_np.shape
    sagittal_x: List[int] = []
    sagittal_y: List[int] = []
    for y in range(height):
        row = binary_np[y, :]
        black_pixels = np.where(row == 0)[0]
        if len(black_pixels) > 0:
            sagittal_x.append(int(black_pixels[0]))
            sagittal_y.append(int(y))
    return np.array(sagittal_x, dtype=np.int32), np.array(sagittal_y, dtype=np.int32)


def _compute_derivatives(
    sagittal_x: NDArray[Any],
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Return raw and scaled first/second derivatives.
    """
    dx: NDArray[Any] = np.gradient(sagittal_x.astype(float))
    ddx: NDArray[Any] = np.gradient(dx)
    dx_scaled: NDArray[Any] = dx + DX1_OFFSET + DX1_SCALE_FACTOR * dx
    ddx_scaled: NDArray[Any] = ddx + DX2_OFFSET + DX2_SCALE_FACTOR * ddx
    return dx, ddx, dx_scaled, ddx_scaled


def _plot_sagittal_profile(
    ax: Axes,
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    dx_scaled: NDArray[Any],
    ddx_scaled: NDArray[Any],
) -> None:
    ax.plot(  # type: ignore
        sagittal_x,
        sagittal_y,
        color="black",
        linewidth=2,
        label="Sagittal Profile",
    )


def _calculate_quarter_lines(start_y: int, end_y: int) -> tuple[float, float, float]:
    return (
        start_y + 0.25 * (end_y - start_y),
        start_y + 0.50 * (end_y - start_y),
        start_y + 0.75 * (end_y - start_y),
    )


def _plot_quarter_lines(ax: Axes, sagittal_y: NDArray[Any]) -> None:
    start_y, end_y = sagittal_y[0], sagittal_y[-1]
    q1, q2, q3 = _calculate_quarter_lines(start_y, end_y)
    for q, label in zip((q1, q2, q3), ("25% Line", "50% Line", "75% Line")):
        ax.axhline(q, color="green", linestyle="--", linewidth=1, label=label)  # type: ignore


def _find_local_max_min(sagittal_x: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Local extrema of x(y).
    """
    max_indices, _ = find_peaks(sagittal_x)  # maxima
    min_indices, _ = find_peaks(-sagittal_x)  # minima
    return max_indices, min_indices


def _plot_sagittal_minmax(
    ax: Axes,
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[np.int64],
    min_indices: NDArray[np.int64],
) -> None:
    ax.scatter(  # type: ignore
        sagittal_x[max_indices],
        sagittal_y[max_indices],
        color="green",
        s=80,
        label="Local Maxima",
        zorder=3,
    )
    ax.scatter(  # type: ignore
        sagittal_x[min_indices],
        sagittal_y[min_indices],
        color="red",
        s=80,
        label="Local Minima",
        zorder=3,
    )
    for i, idx in enumerate(max_indices):
        ax.annotate(  # type: ignore
            f"max-{i}",
            (float(sagittal_x[idx]), float(sagittal_y[idx])),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
            va="center",
            color="green",
        )
    for i, idx in enumerate(min_indices):
        ax.annotate(  # type: ignore
            f"min-{i}",
            (float(sagittal_x[idx]), float(sagittal_y[idx])),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
            va="center",
            color="red",
        )


def _exclude_near(
    indices: NDArray[np.int64], banned: Iterable[int], radius: int = 5
) -> NDArray[np.int64]:
    if len(indices) == 0:
        return indices
    banned = list(banned) if banned else []
    if not banned:
        return indices
    banned_arr = np.array(banned, dtype=int)
    keep = np.ones(len(indices), dtype=bool)
    for b in banned_arr:
        keep &= np.abs(indices - b) > int(radius)
    return indices[keep]


def _nms_keep_best(
    idxs: NDArray[np.int64], scores: NDArray[np.floating], radius: int = 8
) -> NDArray[np.int64]:
    """Non-max suppression on 1D indices using scores; keep best in ±radius.

    Ensures idxs and scores are aligned in length before proceeding.
    """
    if idxs.size == 0:
        return idxs

    n = min(int(len(idxs)), int(len(scores)))
    idxs = idxs[:n]
    scores = scores[:n]

    if n == 1:
        return idxs

    order = np.argsort(scores)[::-1]  # high → low
    taken = np.zeros(n, dtype=bool)
    kept_vals: List[int] = []

    for o in order:
        if taken[o]:
            continue
        i = int(idxs[o])
        kept_vals.append(i)
        taken |= np.abs(idxs - i) <= int(radius)

    return np.array(sorted(kept_vals), dtype=np.int64)


def _ensure_odd(k: int) -> int:
    k = max(3, int(k))
    return k if k % 2 == 1 else (k + 1)


def _turning_angle(dx: NDArray[np.floating]) -> NDArray[np.floating]:
    """θ = arctan(dx/dy); with dy=1 per row → θ = arctan(dx)."""
    return np.arctan(dx)


def _angle_change(theta: NDArray[np.floating], halfwin: int) -> NDArray[np.floating]:
    """|θ[i+halfwin] − θ[i−halfwin]| with safe edges (0 outside)."""
    n = len(theta)
    out = np.zeros(n, dtype=float)
    if halfwin <= 0 or n < 2 * halfwin + 1:
        return out
    a = theta[: -2 * halfwin]
    b = theta[2 * halfwin :]
    diff = np.abs(b - a)
    out[halfwin:-halfwin] = diff
    return out


def _find_monotonic_corners(
    sagittal_x: NDArray[Any],
    *,
    scales: List[int] = [9, 13, 17],  # Savitzky–Golay windows (odd)
    polyorder: int = 2,
    dx_tol: float = 0.03,  # ignore tiny slopes
    min_run: int = 8,  # demand stable sign neighborhood
    distance_px: int = 28,  # spacing between corners
    angle_percentile: float = 92.0,  # adaptive θ-change percentile
    angle_min_deg: float = 14.0,  # absolute floor for θ change
    kappa_percentile: float = 90.0,  # κ backup percentile
    mix_weight_angle: float = 0.7,  # score fusion: angle vs κ
    exclude_extrema: Iterable[int] = (),
) -> NDArray[np.int64]:
    """
    Multi-scale corner finder:
      1) For each scale, smooth x(y) with Savitzky–Golay, get dx, θ, and |Δθ|.
      2) Keep indices where slope sign is consistent (monotonic) in ±min_run//2.
      3) Threshold by max(angle_percentile, angle_min_deg), peak-pick with distance.
      4) Also get curvature κ peaks at each scale (backup).
      5) Union candidates across scales, exclude near extrema, then NMS using fused score.

    Returns:
        int indices (y-aligned) of corners.
    """
    x = sagittal_x.astype(float)
    n = len(x)
    if n < 7:
        return np.array([], dtype=np.int64)

    all_idx: List[int] = []
    all_scores: List[float] = []

    half_neigh = max(1, int(min_run // 2))

    for w in scales:
        w = _ensure_odd(w)
        if w >= n:
            continue

        # Smooth & derivatives (Savitzky–Golay)
        xs = savgol_filter(
            x, window_length=w, polyorder=polyorder, deriv=0, mode="interp"
        )
        dx = savgol_filter(
            x, window_length=w, polyorder=polyorder, deriv=1, mode="interp"
        )
        ddx = savgol_filter(
            x, window_length=w, polyorder=polyorder, deriv=2, mode="interp"
        )

        # Curvature κ for backup score
        kappa = np.abs(ddx) / np.power(1.0 + dx * dx, 1.5)

        # Turning angle and its change
        theta = _turning_angle(dx)
        halfwin = w // 2
        dtheta = _angle_change(theta, halfwin)

        # Build monotonic mask: strong slope + stable sign neighborhood
        sign_dx = np.sign(dx)
        strong = np.abs(dx) >= dx_tol
        same = np.ones_like(sign_dx, dtype=bool)
        for off in range(1, half_neigh + 1):
            same &= sign_dx == np.roll(sign_dx, off)
            same &= sign_dx == np.roll(sign_dx, -off)
        mono = strong & same

        # Threshold: adaptive percentile among monotonic region, with absolute floor
        mono_vals = dtheta[mono]
        if mono_vals.size == 0:
            continue
        th_angle = float(np.percentile(mono_vals, angle_percentile))
        th_angle = max(th_angle, np.deg2rad(angle_min_deg))

        # Peak pick on dθ (monotone zones only)
        dtheta_peaks = dtheta.copy()
        dtheta_peaks[~mono] = 0.0
        peaks_a, props_a = find_peaks(
            dtheta_peaks,
            height=th_angle,
            distance=int(distance_px),
        )

        # κ peaks as backup (restrict to mono, and above percentile)
        mono_kappa = kappa[mono]
        th_kappa = (
            float(np.percentile(mono_kappa, kappa_percentile))
            if mono_kappa.size
            else 0.0
        )
        kappa_peaks = kappa.copy()
        kappa_peaks[~mono] = 0.0
        peaks_k, props_k = find_peaks(
            kappa_peaks,
            height=th_kappa if th_kappa > 0 else None,
            distance=int(distance_px),
        )

        # Merge sets for this scale
        idxs = np.unique(np.concatenate([peaks_a, peaks_k])).astype(int)
        if idxs.size == 0:
            continue

        # Scores: fuse normalized dθ and κ
        h_a = props_a.get("peak_heights", np.array([]))
        h_k = props_k.get("peak_heights", np.array([]))

        score_a = {int(i): float(h) for i, h in zip(peaks_a, h_a)}
        score_k = {int(i): float(h) for i, h in zip(peaks_k, h_k)}

        def _norm(v: float, vmax: float) -> float:
            return 0.0 if vmax <= 0 else (v / vmax)

        max_a = float(h_a.max()) if h_a.size else 0.0
        max_kv = float(h_k.max()) if h_k.size else 0.0

        for i in idxs:
            sa = _norm(score_a.get(int(i), 0.0), max_a)
            sk = _norm(score_k.get(int(i), 0.0), max_kv)
            s = mix_weight_angle * sa + (1.0 - mix_weight_angle) * sk
            all_idx.append(int(i))
            all_scores.append(float(s))

    if not all_idx:
        return np.array([], dtype=np.int64)

    # Arrays (keep a copy of original indices for masking later)
    orig_idx_arr = np.array(all_idx, dtype=np.int64)
    all_idx_arr = orig_idx_arr.copy()
    all_scores_arr = np.array(all_scores, dtype=float)

    # Exclude near extrema and align scores with surviving indices
    all_idx_arr = _exclude_near(all_idx_arr, exclude_extrema, radius=6)
    if all_idx_arr.size == 0:
        return np.array([], dtype=np.int64)

    mask = np.isin(orig_idx_arr, all_idx_arr)
    all_scores_arr = all_scores_arr[mask]

    # Final NMS across scales
    keep = _nms_keep_best(all_idx_arr, all_scores_arr, radius=max(8, distance_px // 2))
    return keep


def _plot_monotonic_corners(
    ax: Axes,
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    corner_idxs: NDArray[np.int64],
) -> None:
    if len(corner_idxs) == 0:
        return
    ax.scatter(  # type: ignore
        sagittal_x[corner_idxs],
        sagittal_y[corner_idxs],
        s=90,
        zorder=4,
        color="purple",
        label="Monotonic Corner",
        marker="D",
    )
    for i, idx in enumerate(corner_idxs):
        ax.annotate(  # type: ignore
            f"corner-{i}",
            (float(sagittal_x[idx]), float(sagittal_y[idx])),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
            va="center",
            color="purple",
        )


def _plot_lateral_landmarks(ax: Axes, landmarks: NDArray[Any], shift_x: int) -> None:
    """
    Plot the 6 lateral landmarks on the sagittal profile, shifted left by shift_x.
    """
    for i, name in enumerate(LATERAL_LANDMARK_NAMES):
        x, y = landmarks[i]
        if x != -1 and y != -1:
            x -= shift_x
            ax.scatter(x, y, color="green", s=80, zorder=3)  # type: ignore
            ax.annotate(  # type: ignore
                name,
                (x, y),
                textcoords="offset points",
                xytext=(10, 0),
                ha="left",
                color="black",
                fontsize=14,
                fontweight="bold",
            )


def analyze_lateral(
    input_image: Image.Image,
    landmarks: List[Tuple[int, int]],
) -> Tuple[NDArray[Any], NDArray[Any], Any, NDArray[Any]]:
    """
    Render sagittal plot image and return it with landmarks and arrays.
    """
    _, binary_np, _, _ = _process_image(input_image)

    sagittal_x, sagittal_y = _extract_sagittal_profile(binary_np)
    sagittal_x, shift_x = _shift_sagittal_profile(sagittal_x)

    dx, ddx, dx_scaled, ddx_scaled = _compute_derivatives(sagittal_x)

    fig, ax2 = plt.subplots(figsize=(6, 10))  # type: ignore
    _plot_sagittal_profile(ax2, sagittal_x, sagittal_y, dx_scaled, ddx_scaled)

    # Extrema
    max_indices, min_indices = _find_local_max_min(sagittal_x)
    if DEBUG:
        _plot_sagittal_minmax(ax2, sagittal_x, sagittal_y, max_indices, min_indices)

    # Multi-scale “corners” (turning-angle + curvature under monotonic slope)
    extrema_set = set(map(int, np.concatenate([max_indices, min_indices])))
    corner_idxs = _find_monotonic_corners(
        sagittal_x,
        scales=[9, 13, 17],  # try [11, 15, 19, 23] for broader bends
        polyorder=2,
        dx_tol=0.035,  # raise to ignore shallow changes
        min_run=10,  # longer monotonic neighborhood
        distance_px=32,  # spacing between corners
        angle_percentile=93.0,  # stronger angle threshold
        angle_min_deg=16.0,  # absolute angle floor
        kappa_percentile=92.0,  # κ backup gate
        mix_weight_angle=0.75,  # bias toward angle over κ
        exclude_extrema=extrema_set,
    )
    if DEBUG:
        _plot_monotonic_corners(ax2, sagittal_x, sagittal_y, corner_idxs)

    # Compute/plot lateral landmarks
    landmarks_np = _find_lateral_landmarks(
        sagittal_x,
        sagittal_y,
        max_indices,
        min_indices,
        corner_idxs,
        int(shift_x),
        landmarks,
    )
    _plot_lateral_landmarks(ax2, landmarks_np, int(shift_x))
    logging.debug("Lateral Landmarks (x, y):")
    logging.debug(landmarks_np)

    if DEBUG:
        _plot_quarter_lines(ax2, sagittal_y)

    ax2.set_ylim(1024, 0)
    ax2.set_xlim(-25, 512)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.margins(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    legend = ax2.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 1.0))  # type: ignore
    legend.get_frame().set_alpha(0.8)

    return (
        util.convert_matplotlib_to_opencv(ax2),
        landmarks_np,
        sagittal_x + shift_x,
        sagittal_y,
    )


class LateralSearchMode(Enum):
    MAX = auto()  # use max_indices
    MIN = auto()  # use min_indices
    CORNER = auto()  # use corner_idxs (e.g., curvature-based corners)
    NEAREST = auto()  # ignore extrema/corners; nearest sagittal y


def _find_lateral_landmark(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    corner_idxs: NDArray[Any],
    y_coord: float,
    mode: LateralSearchMode = LateralSearchMode.MAX,
    y_forward: Optional[bool] = None,
) -> NDArray[Any]:
    """
    Find a lateral landmark on the sagittal profile according to `mode`.

    Args:
        sagittal_x, sagittal_y: 1D arrays of the sagittal polyline.
        max_indices: indices of local maxima along the sagittal line.
        min_indices: indices of local minima along the sagittal line.
        corner_idxs: indices of detected 'corner' points along the line.
        y_coord: target y to compare against.
        mode: which search to perform (MAX, MIN, CORNER, NEAREST).
        y_forward:
            - True  -> consider only points with y >= y_coord
            - False -> consider only points with y <= y_coord
            - None  -> consider all points

    Returns:
        np.array([x, y]) as float64, or [-1.0, -1.0] if none found.
    """

    def _dir_filter(idxs: NDArray[Any]) -> NDArray[Any]:
        if y_forward is True:
            mask = sagittal_y[idxs] >= y_coord
            return idxs[mask]
        elif y_forward is False:
            mask = sagittal_y[idxs] <= y_coord
            return idxs[mask]
        return idxs

    # Choose candidate indices by mode
    if mode == LateralSearchMode.MAX:
        candidates = _dir_filter(max_indices)
    elif mode == LateralSearchMode.MIN:
        candidates = _dir_filter(min_indices)
    elif mode == LateralSearchMode.CORNER:
        candidates = _dir_filter(corner_idxs)
    elif mode == LateralSearchMode.NEAREST:
        # For NEAREST, we consider *all* points along the sagittal line
        all_idxs = np.arange(len(sagittal_y), dtype=int)
        candidates = _dir_filter(all_idxs)
    else:
        candidates = np.array([], dtype=int)

    if candidates.size == 0:
        return np.array([-1.0, -1.0], dtype=float)

    # Pick the candidate whose y is closest to y_coord
    closest = int(np.argmin(np.abs(sagittal_y[candidates] - y_coord)))
    idx = int(candidates[closest])

    return np.array([float(sagittal_x[idx]), float(sagittal_y[idx])], dtype=float)


def _find_corner_landmark_in_range(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    corner_idxs: NDArray[Any],
    idx_low: Optional[int] = None,
    idx_high: Optional[int] = None,
    *,
    # Optional: derive the sagittal range from a pair of frontal landmarks
    landmarks_frontal: Optional[NDArray[Any]] = None,
    frontal_lo_idx: Optional[int] = None,
    frontal_hi_idx: Optional[int] = None,
    widen_if_same: int = 2,  # widen if both frontal Ys map to same sagittal row
) -> NDArray[Any]:
    """
    Return the first 'event' encountered when scanning indices lo..hi (inclusive):
      - A 'corner' index (member of corner_idxs), OR
      - The index of the maximum X (sagittal_x) in the window,
    whichever occurs first in index order.

    Window [lo, hi] is either provided or derived from frontal landmarks by
    projecting their Y values to nearest sagittal rows (with optional widening).

    Returns:
        np.array([x, y]) as float64, or [-1.0, -1.0] on invalid input.
    """
    n = int(len(sagittal_x))
    if n == 0 or n != int(len(sagittal_y)):
        return np.array([-1.0, -1.0], dtype=float)

    # --- Resolve [lo, hi] range ---
    if idx_low is None or idx_high is None:
        if (
            landmarks_frontal is None
            or frontal_lo_idx is None
            or frontal_hi_idx is None
            or int(max(frontal_lo_idx, frontal_hi_idx))
            >= int(landmarks_frontal.shape[0])
        ):
            return np.array([-1.0, -1.0], dtype=float)

        y_lo = float(landmarks_frontal[int(frontal_lo_idx)][1])
        y_hi = float(landmarks_frontal[int(frontal_hi_idx)][1])

        # Map frontal Ys to nearest sagittal rows
        idx_low = int(np.argmin(np.abs(sagittal_y - y_lo)))
        idx_high = int(np.argmin(np.abs(sagittal_y - y_hi)))

        if idx_low == idx_high and widen_if_same > 0:
            idx_low = max(0, idx_low - widen_if_same)
            idx_high = min(n - 1, idx_high + widen_if_same)

    lo = max(0, min(int(idx_low), n - 1))
    hi = max(0, min(int(idx_high), n - 1))
    if lo > hi:
        lo, hi = hi, lo
    if lo >= n or hi < lo:
        return np.array([-1.0, -1.0], dtype=float)

    # --- Prepare corner set (fast membership) ---
    corner_idxs = np.asarray(corner_idxs, dtype=int)
    if corner_idxs.size == 0:
        # No corners at all → fall back to max in window
        slice_x = np.asarray(sagittal_x[lo : hi + 1], dtype=float)
        # Handle NaNs: treat as -inf so they won't win max
        x_clean = np.where(np.isfinite(slice_x), slice_x, -np.inf)
        if np.all(~np.isfinite(x_clean)):  # all -inf (originally NaN/±inf)
            return np.array([-1.0, -1.0], dtype=float)
        rel = int(np.argmax(x_clean))
        max_idx = lo + rel
        return np.array(
            [float(sagittal_x[max_idx]), float(sagittal_y[max_idx])], dtype=float
        )

    corner_set = set(int(i) for i in corner_idxs if 0 <= int(i) < n)

    # --- Compute the max-X index within [lo, hi] ---
    slice_x = np.asarray(sagittal_x[lo : hi + 1], dtype=float)
    x_clean = np.where(np.isfinite(slice_x), slice_x, -np.inf)
    if np.all(~np.isfinite(x_clean)):
        # If X is unusable in the window, fall back to first corner (if any)
        first_corner_idx = next((i for i in range(lo, hi + 1) if i in corner_set), None)
        if first_corner_idx is None:
            return np.array([-1.0, -1.0], dtype=float)
        chosen = first_corner_idx
        return np.array(
            [float(sagittal_x[chosen]), float(sagittal_y[chosen])], dtype=float
        )

    rel = int(np.argmax(x_clean))
    max_idx = lo + rel

    # --- Find the first corner encountered scanning lo..hi ---
    first_corner_idx = next((i for i in range(lo, hi + 1) if i in corner_set), None)

    # --- Choose whichever comes first in index order ---
    if first_corner_idx is not None:
        chosen = first_corner_idx if first_corner_idx <= max_idx else max_idx
    else:
        chosen = max_idx

    return np.array([float(sagittal_x[chosen]), float(sagittal_y[chosen])], dtype=float)


def _find_lateral_landmark_in_range(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    idx_low: Optional[int] = None,
    idx_high: Optional[int] = None,
    *,
    pick: LateralSearchMode = LateralSearchMode.MAX,  # only MAX or MIN are valid here
    # New optional inputs to derive the sagittal range from frontal landmarks
    landmarks_frontal: Optional[NDArray[Any]] = None,
    frontal_lo_idx: Optional[int] = None,
    frontal_hi_idx: Optional[int] = None,
    widen_if_same: int = 2,  # how much to widen if both map to the same sagittal index
) -> NDArray[Any]:
    """
    Return the point [x, y] on the sagittal polyline whose X is the min or max
    within an inclusive index range. The range can be provided directly (idx_low, idx_high)
    or derived from a pair of frontal landmark indices by projecting their Y values onto
    the sagittal polyline.

    If multiple points share the same extreme (min/max), return the one whose Y
    is closest to the median Y among those candidates.

    Args:
        sagittal_x, sagittal_y: 1D arrays (same length) defining the sagittal profile.
        idx_low, idx_high: inclusive sagittal bounds (order agnostic). If omitted,
            you must provide `landmarks_frontal` plus `frontal_lo_idx` and `frontal_hi_idx`.
        pick: LateralSearchMode.MAX or LateralSearchMode.MIN.
        landmarks_frontal: frontal landmarks array (used only if deriving range).
        frontal_lo_idx, frontal_hi_idx: indices into `landmarks_frontal` whose Y
            coordinates define the range to project onto sagittal.
        widen_if_same: if both frontal Ys map to the same sagittal index, widen by
            this many rows on each side (clamped to array bounds).

    Returns:
        np.array([x, y]) as float64, or [-1.0, -1.0] on invalid input.
    """
    n = int(len(sagittal_x))
    if n == 0 or n != int(len(sagittal_y)):
        return np.array([-1.0, -1.0], dtype=float)

    # If sagittal bounds weren’t provided, derive them from frontal landmark pair
    if idx_low is None or idx_high is None:
        if (
            landmarks_frontal is None
            or frontal_lo_idx is None
            or frontal_hi_idx is None
            or int(max(frontal_lo_idx, frontal_hi_idx)) >= landmarks_frontal.shape[0]
        ):
            return np.array([-1.0, -1.0], dtype=float)

        y_lo = float(landmarks_frontal[int(frontal_lo_idx)][1])
        y_hi = float(landmarks_frontal[int(frontal_hi_idx)][1])

        # Nearest sagittal indices to those Y's
        idx_low = int(np.argmin(np.abs(sagittal_y - y_lo)))
        idx_high = int(np.argmin(np.abs(sagittal_y - y_hi)))

        # Optionally widen if both map to the same row
        if idx_low == idx_high and widen_if_same > 0:
            idx_low = max(0, idx_low - widen_if_same)
            idx_high = min(n - 1, idx_high + widen_if_same)

    # Normalize/clip bounds and ensure low <= high
    lo = max(0, min(int(idx_low), n - 1))
    hi = max(0, min(int(idx_high), n - 1))
    if lo > hi:
        lo, hi = hi, lo

    if hi < lo or lo >= n:
        return np.array([-1.0, -1.0], dtype=float)

    x_slice = np.asarray(sagittal_x[lo : hi + 1], dtype=float)
    y_slice = np.asarray(sagittal_y[lo : hi + 1], dtype=float)

    if np.all(np.isnan(x_slice)):
        return np.array([-1.0, -1.0], dtype=float)

    if pick == LateralSearchMode.MAX:
        extreme_val = np.nanmax(x_slice)
    elif pick == LateralSearchMode.MIN:
        extreme_val = np.nanmin(x_slice)
    else:
        return np.array([-1.0, -1.0], dtype=float)

    # Candidates with the extreme value
    candidates = np.where(x_slice == extreme_val)[0]
    if candidates.size == 0:
        return np.array([-1.0, -1.0], dtype=float)

    # Break ties by picking Y closest to the median Y of the candidates
    if candidates.size > 1:
        candidate_ys = y_slice[candidates]
        median_y = np.median(candidate_ys)
        center_idx = candidates[np.argmin(np.abs(candidate_ys - median_y))]
    else:
        center_idx = candidates[0]

    idx = lo + int(center_idx)
    return np.array([float(sagittal_x[idx]), float(sagittal_y[idx])], dtype=float)


def _find_lateral_landmark_minmax(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    idx_low: Optional[int] = None,
    idx_high: Optional[int] = None,
    *,
    pick: LateralSearchMode = LateralSearchMode.MAX,  # only MAX or MIN are valid here
) -> NDArray[Any]:
    """
    Return [x, y] for the global min/max of sagittal_x within an index range.
    If idx_low/idx_high are None, the full polyline is used.

    Tie-breaks:
      - If multiple indices share the extreme X, choose the one whose Y is
        closest to the median Y among the tied candidates.

    Args:
        sagittal_x, sagittal_y: 1D arrays (same length) of the sagittal profile.
        idx_low, idx_high: inclusive index bounds for the search (order agnostic).
        pick: LateralSearchMode.MAX or LateralSearchMode.MIN.

    Returns:
        np.array([x, y]) as float64, or [-1.0, -1.0] on invalid input.
    """
    n = int(len(sagittal_x))
    if n == 0 or n != int(len(sagittal_y)):
        return np.array([-1.0, -1.0], dtype=float)

    # Default to full range if not provided
    if idx_low is None:
        idx_low = 0
    if idx_high is None:
        idx_high = n - 1

    # Normalize/clip bounds and ensure lo <= hi
    lo = max(0, min(int(idx_low), n - 1))
    hi = max(0, min(int(idx_high), n - 1))
    if lo > hi:
        lo, hi = hi, lo

    x_slice = np.asarray(sagittal_x[lo : hi + 1], dtype=float)
    y_slice = np.asarray(sagittal_y[lo : hi + 1], dtype=float)

    if x_slice.size == 0 or np.all(np.isnan(x_slice)):
        return np.array([-1.0, -1.0], dtype=float)

    if pick == LateralSearchMode.MAX:
        extreme_val = np.nanmax(x_slice)
    elif pick == LateralSearchMode.MIN:
        extreme_val = np.nanmin(x_slice)
    else:
        return np.array([-1.0, -1.0], dtype=float)

    # All indices within the slice that hit the extreme value
    rel_candidates = np.where(x_slice == extreme_val)[0]
    if rel_candidates.size == 0:
        return np.array([-1.0, -1.0], dtype=float)

    # Break ties by Y closest to the median Y of the candidates
    if rel_candidates.size > 1:
        cand_ys = y_slice[rel_candidates]
        median_y = float(np.median(cand_ys))
        rel_idx = int(rel_candidates[np.argmin(np.abs(cand_ys - median_y))])
    else:
        rel_idx = int(rel_candidates[0])

    idx = lo + rel_idx
    return np.array([float(sagittal_x[idx]), float(sagittal_y[idx])], dtype=float)


def _find_lateral_landmarks(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    corner_idxs: NDArray[Any],
    shift_x: int,
    landmarks_frontal: NDArray[Any],
) -> NDArray[Any]:
    """
    Compute lateral landmarks; Subnasal uses corner-or-max (57..79),
    Pogonion uses MIN (14..16), and Mento-labial uses MAX (8..14)
    via find_lateral_landmark_minmax over a sagittal range.

    Output layout (indices):
      0 = Glabella
      1 = Nasion
      2 = Nasal tip
      3 = Subnasal  (57..79, corner-or-max)
      4 = Mento-labial (8..14, MAX X; fallback to CORNER if overlapping Pogonion)
      5 = Pogonion (14..16, MIN X)
    """
    landmarks_frontal = np.array(landmarks_frontal)
    if landmarks_frontal.size == 0:
        return np.full((6, 2), -1, dtype=int)

    n_sag = int(len(sagittal_x))
    landmarks = np.full((6, 2), -1.0, dtype=float)

    # ---- Pogonion via frontal pair (14..16), pick MIN X ----
    if landmarks_frontal.shape[0] > 16:
        pogonion_pt = _find_lateral_landmark_in_range(
            sagittal_x=sagittal_x,
            sagittal_y=sagittal_y,
            pick=LateralSearchMode.MIN,
            landmarks_frontal=landmarks_frontal,
            frontal_lo_idx=14,
            frontal_hi_idx=16,
        )
        landmarks[5] = pogonion_pt

    # ---- Subnasal via frontal pair (57..79), corner-or-max ----
    if landmarks_frontal.shape[0] > 79:
        subnasal_pt = _find_corner_landmark_in_range(
            sagittal_x=sagittal_x,
            sagittal_y=sagittal_y,
            corner_idxs=corner_idxs,
            landmarks_frontal=landmarks_frontal,
            frontal_lo_idx=57,
            frontal_hi_idx=79,
        )
        landmarks[3] = subnasal_pt

    # ---- Mento-labial via frontal pair (8..14), pick MAX X using minmax finder ----
    if landmarks_frontal.shape[0] > 14:
        # Project the two frontal bounds to sagittal indices
        y_upper_bound = float(landmarks_frontal[8][1])  # upper boundary landmark
        y_lower_bound = float(landmarks_frontal[14][1])  # lower boundary landmark

        idx_upper_bound = int(np.argmin(np.abs(sagittal_y - y_upper_bound)))
        idx_lower_bound = int(np.argmin(np.abs(sagittal_y - y_lower_bound)))

        # If both map to the same sagittal row, widen slightly
        if idx_upper_bound == idx_lower_bound:
            widen = 2
            idx_upper_bound = max(0, idx_upper_bound - widen)
            idx_lower_bound = min(n_sag - 1, idx_lower_bound + widen)

        search_lo = min(idx_upper_bound, idx_lower_bound)
        search_hi = max(idx_upper_bound, idx_lower_bound)

        mentolabial_pt = _find_lateral_landmark_minmax(
            sagittal_x=sagittal_x,
            sagittal_y=sagittal_y,
            idx_low=search_lo,
            idx_high=search_hi,
            pick=LateralSearchMode.MAX,
        )
        landmarks[4] = mentolabial_pt

    # ---- Overlap guard: if Mento-labial ~ Pogonion, recompute Mento-labial via CORNER at frontal idx 14 ----
    # Adaptive tolerances based on face height in sagittal space
    if np.all(landmarks[4] >= 0) and np.all(landmarks[5] >= 0):
        ml_x, ml_y = float(landmarks[4][0]), float(landmarks[4][1])
        pog_x, pog_y = float(landmarks[5][0]), float(landmarks[5][1])

        dx = abs(ml_x - pog_x)
        dy = abs(ml_y - pog_y)
        d_euclid = float(np.hypot(dx, dy))

        # Use sagittal_y extent as a scale proxy
        face_height = (
            float(np.max(sagittal_y) - np.min(sagittal_y)) if sagittal_y.size else 0.0
        )

        # Separate axis tolerances + Euclidean backup (with sensible minimums)
        x_tol_px = max(
            3, int(round(0.004 * face_height))
        )  # ~4 px per 1000px face height
        y_tol_px = max(
            5, int(round(0.008 * face_height))
        )  # ~8 px per 1000px face height
        euclid_tol_px = max(
            6, int(round(0.008 * face_height))
        )  # ~8 px per 1000px face height

        cond_axis = (dx <= x_tol_px) and (dy <= y_tol_px)
        cond_euclid = d_euclid <= euclid_tol_px
        overlap = bool(cond_axis or cond_euclid)

        logger.debug(
            "[ML-Pog overlap] ML=(%.1f, %.1f) Pog=(%.1f, %.1f) | dx=%.2f dy=%.2f d=%.2f "
            "| x_tol=%d y_tol=%d euclid_tol=%d | cond_axis=%s cond_euclid=%s | overlap=%s",
            ml_x,
            ml_y,
            pog_x,
            pog_y,
            dx,
            dy,
            d_euclid,
            x_tol_px,
            y_tol_px,
            euclid_tol_px,
            str(cond_axis),
            str(cond_euclid),
            str(overlap),
        )

        if overlap:
            target_y_for_corner = float(landmarks_frontal[14][1])
            logger.debug(
                "[ML-Pog overlap] Recomputing Mento-labial via CORNER at frontal idx 14, target_y=%.1f (y_forward=False)",
                target_y_for_corner,
            )

            ml_corner_pt = _find_lateral_landmark(
                sagittal_x=sagittal_x,
                sagittal_y=sagittal_y,
                max_indices=max_indices,
                min_indices=min_indices,
                corner_idxs=corner_idxs,
                y_coord=target_y_for_corner,
                mode=LateralSearchMode.CORNER,
                y_forward=False,
            )

            if np.all(ml_corner_pt >= 0):
                logger.debug(
                    "[ML-Pog overlap] Moving ML from (%.1f, %.1f) -> (%.1f, %.1f)",
                    ml_x,
                    ml_y,
                    float(ml_corner_pt[0]),
                    float(ml_corner_pt[1]),
                )
                landmarks[4] = ml_corner_pt
            else:
                logger.debug(
                    "[ML-Pog overlap] Corner-based fallback returned invalid point; keeping original ML"
                )

    # ---- Highest frontal landmark (smallest Y): dynamic Glabella anchor ----
    highest_frontal_idx = int(np.argmin(landmarks_frontal[:, 1]))

    # ---- Remaining via existing finder ----
    landmark_mapping = [
        (0, highest_frontal_idx, LateralSearchMode.NEAREST, None),  # Glabella
        (1, 51, LateralSearchMode.NEAREST, False),  # Nasion
        (2, 54, LateralSearchMode.MIN, None),  # Nasal tip
    ]

    for out_idx, frontal_idx, mode, y_forward in landmark_mapping:
        y_target = float(landmarks_frontal[frontal_idx][1])
        pt = _find_lateral_landmark(
            sagittal_x=sagittal_x,
            sagittal_y=sagittal_y,
            max_indices=max_indices,
            min_indices=min_indices,
            corner_idxs=corner_idxs,
            y_coord=y_target,
            mode=mode,
            y_forward=y_forward,
        )
        landmarks[out_idx] = pt

    # Shift X back to full-image coordinates and return ints
    landmarks[:, 0] += float(shift_x)
    return np.array([tuple(map(int, xy)) for xy in landmarks], dtype=int)
