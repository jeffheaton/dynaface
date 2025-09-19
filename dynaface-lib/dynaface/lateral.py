from typing import Any, List, Tuple, Optional, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from PIL import Image
from rembg import remove  # type: ignore
from scipy.signal import find_peaks, savgol_filter  # <-- Savitzky–Golay

from dynaface import models, util
import logging

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
DEBUG = True
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


def process_image(
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


def shift_sagittal_profile(sagittal_x: NDArray[Any]) -> tuple[NDArray[Any], float]:
    """
    Shift so lowest x becomes 0.
    """
    min_x = np.min(sagittal_x)
    return sagittal_x - min_x, float(min_x)


def extract_sagittal_profile(
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


def compute_derivatives(
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


def plot_sagittal_profile(
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


def calculate_quarter_lines(start_y: int, end_y: int) -> tuple[float, float, float]:
    return (
        start_y + 0.25 * (end_y - start_y),
        start_y + 0.50 * (end_y - start_y),
        start_y + 0.75 * (end_y - start_y),
    )


def plot_quarter_lines(ax: Axes, sagittal_y: NDArray[Any]) -> None:
    start_y, end_y = sagittal_y[0], sagittal_y[-1]
    q1, q2, q3 = calculate_quarter_lines(start_y, end_y)
    for q, label in zip((q1, q2, q3), ("25% Line", "50% Line", "75% Line")):
        ax.axhline(q, color="green", linestyle="--", linewidth=1, label=label)  # type: ignore


def find_local_max_min(sagittal_x: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Local extrema of x(y).
    """
    max_indices, _ = find_peaks(sagittal_x)  # maxima
    min_indices, _ = find_peaks(-sagittal_x)  # minima
    return max_indices, min_indices


def plot_sagittal_minmax(
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


# ================= Corner detection (multi-scale, angle + curvature) =================


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


def find_monotonic_corners(
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


def plot_monotonic_corners(
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


def plot_lateral_landmarks(ax: Axes, landmarks: NDArray[Any], shift_x: int) -> None:
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


# ---------------- Main Function ----------------


def save_debug_plot(
    sagittal_x: NDArray[Any], sagittal_y: NDArray[Any], filename: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 10))  # type: ignore
    ax.plot(  # type: ignore
        sagittal_x, sagittal_y, color="black", linewidth=2, label="Sagittal Profile"
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # type: ignore
    plt.close(fig)


def analyze_lateral(
    input_image: Image.Image,
    landmarks: List[Tuple[int, int]],
) -> Tuple[NDArray[Any], NDArray[Any], Any, NDArray[Any]]:
    """
    Render sagittal plot image and return it with landmarks and arrays.
    """
    _, binary_np, _, _ = process_image(input_image)

    sagittal_x, sagittal_y = extract_sagittal_profile(binary_np)
    sagittal_x, shift_x = shift_sagittal_profile(sagittal_x)

    dx, ddx, dx_scaled, ddx_scaled = compute_derivatives(sagittal_x)

    fig, ax2 = plt.subplots(figsize=(6, 10))  # type: ignore
    plot_sagittal_profile(ax2, sagittal_x, sagittal_y, dx_scaled, ddx_scaled)

    # Extrema
    max_indices, min_indices = find_local_max_min(sagittal_x)
    if DEBUG:
        plot_sagittal_minmax(ax2, sagittal_x, sagittal_y, max_indices, min_indices)

    # Multi-scale “corners” (turning-angle + curvature under monotonic slope)
    extrema_set = set(map(int, np.concatenate([max_indices, min_indices])))
    corner_idxs = find_monotonic_corners(
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
        plot_monotonic_corners(ax2, sagittal_x, sagittal_y, corner_idxs)

    # Compute/plot lateral landmarks
    landmarks_np = find_lateral_landmarks(
        sagittal_x, sagittal_y, max_indices, min_indices, int(shift_x), landmarks
    )
    plot_lateral_landmarks(ax2, landmarks_np, int(shift_x))
    logging.debug("Lateral Landmarks (x, y):")
    logging.debug(landmarks_np)

    if DEBUG:
        plot_quarter_lines(ax2, sagittal_y)

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


# ---------------- Utilities for landmark search ----------------


def find_lateral_landmark(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    y_coord: float,
    find_max: bool = True,
    y_forward: Optional[bool] = None,
) -> NDArray[Any]:
    indices = max_indices if find_max else min_indices
    if len(indices) == 0:
        return np.array([-1.0, -1.0])
    if y_forward is True:
        mask = sagittal_y[indices] >= y_coord
        candidates = indices[mask]
    elif y_forward is False:
        mask = sagittal_y[indices] <= y_coord
        candidates = indices[mask]
    else:
        candidates = indices
    if len(candidates) == 0:
        return np.array([-1.0, -1.0])
    diffs = np.abs(sagittal_y[candidates] - y_coord)
    closest_idx = candidates[int(np.argmin(diffs))]
    return np.array([float(sagittal_x[closest_idx]), float(sagittal_y[closest_idx])])


def find_nearest_sagittal_point(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    y_coord: float,
) -> NDArray[Any]:
    if len(sagittal_y) == 0:
        return np.array([-1.0, -1.0])
    closest_idx = np.argmin(np.abs(sagittal_y - y_coord))
    return np.array([sagittal_x[closest_idx], sagittal_y[closest_idx]])


def find_lateral_landmarks(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    shift_x: int,
    landmarks_frontal: NDArray[Any],
) -> NDArray[Any]:
    """
    Compute lateral landmarks; subnasal uses a curvature-based corner.
    """
    landmarks_frontal = np.array(landmarks_frontal)
    if len(landmarks_frontal) == 0:
        return np.full((6, 2), -1.0)

    highest_landmark_idx = int(np.argmin(landmarks_frontal[:, 1]))

    # (out_idx, frontal_idx, find_max, y_forward)
    landmark_mapping = [
        (0, highest_landmark_idx, False, None),  # Glabella (min)
        (1, 51, True, None),  # Nasion (max)
        (2, 54, False, None),  # Nasal tip (min)
        (3, 54, True, None),  # Subnasal placeholder; replaced below
        (4, 16, True, False),  # Mento Labial (max)
        (5, 16, False, False),  # Pogonion (min)
    ]

    landmarks = np.full((6, 2), -1.0)

    for out_idx, lm_index, find_max, y_forward in landmark_mapping:
        y_coord = landmarks_frontal[lm_index][1]
        if find_max is None:
            # Use nearest sagittal point for Subnasal Point
            landmark_point = find_nearest_sagittal_point(
                sagittal_x,
                sagittal_y,
                y_coord=y_coord,
            )
        else:
            landmark_point = find_lateral_landmark(
                sagittal_x,
                sagittal_y,
                max_indices,
                min_indices,
                y_coord=y_coord,
                find_max=find_max,
            )

        landmarks[out_idx] = landmark_point

    # Optionally: recompute mento-labial using Pogonion Y if present
    if landmarks[5, 1] >= 0:
        landmarks[4] = find_lateral_landmark(
            sagittal_x,
            sagittal_y,
            max_indices,
            min_indices,
            y_coord=landmarks[5][1],
            find_max=True,
        )

    landmarks[:, 0] += shift_x
    return np.array([tuple(map(int, point)) for point in landmarks])
