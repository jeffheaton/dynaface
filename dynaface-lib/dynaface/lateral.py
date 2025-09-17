from typing import Any, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from PIL import Image
from rembg import remove  # type: ignore
from scipy.signal import find_peaks  # type: ignore

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
    Process the image by removing the background, converting to grayscale and binary,
    applying morphological closing, and inverting the binary image.
    """
    # Remove background
    output_image: Image.Image = remove(input_image, session=models.rembg_session)  # type: ignore

    # Convert to grayscale
    grayscale: Image.Image = output_image.convert("L")

    # Convert to binary
    binary_threshold: int = 32
    binary: Image.Image = grayscale.point(lambda p: 255 if p > binary_threshold else 0)  # type: ignore
    binary_np: NDArray[Any] = np.array(binary)

    # Apply morphological closing
    kernel: NDArray[Any] = np.ones((10, 10), np.uint8)
    binary_np = cv2.morphologyEx(binary_np, cv2.MORPH_CLOSE, kernel)

    # Invert the binary image
    binary_np = 255 - binary_np

    # Get image dimensions
    height: int
    width: int
    height, width = binary_np.shape

    return input_image, binary_np, width, height


def shift_sagittal_profile(sagittal_x: NDArray[Any]) -> tuple[NDArray[Any], float]:
    """
    Shift the sagittal profile so that the lowest x-coordinate becomes 0.

    Returns:
        tuple[NDArray[Any], float]: A tuple containing:
            - The shifted sagittal profile (NDArray[Any])
            - The minimum x value that was subtracted (float)
    """
    min_x = np.min(sagittal_x)
    return sagittal_x - min_x, min_x


def extract_sagittal_profile(
    binary_np: NDArray[np.uint8],
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Extract the sagittal profile from the binary image. For each row, finds the first black pixel.
    """
    height, _ = binary_np.shape
    sagittal_x: List[int] = []
    sagittal_y: List[int] = []
    for y in range(height):
        row = binary_np[y, :]
        black_pixels = np.where(row == 0)[0]  # Get indices of black pixels
        if len(black_pixels) > 0:
            sagittal_x.append(int(black_pixels[0]))  # Ensure type
            sagittal_y.append(int(y))

    return np.array(sagittal_x, dtype=np.int32), np.array(sagittal_y, dtype=np.int32)


def compute_derivatives(
    sagittal_x: NDArray[Any],
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Compute the first and second derivatives of the sagittal profile and return both the raw and scaled values.
    """
    dx: NDArray[Any] = np.gradient(sagittal_x)
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
    """
    Plot the sagittal profile along with its first and second derivatives on the given axes.
    """
    ax.plot(  # type: ignore
        sagittal_x,
        sagittal_y,
        color="black",
        linewidth=2,
        label="Sagittal Profile",
    )
    # The derivative plots have been commented out in this version.
    # They can be re-enabled if needed.


def calculate_quarter_lines(start_y: int, end_y: int) -> tuple[float, float, float]:
    """
    Calculate the 25%, 50%, and 75% y-coordinates between start_y and end_y.
    """
    return (
        start_y + 0.25 * (end_y - start_y),
        start_y + 0.50 * (end_y - start_y),  # Midpoint
        start_y + 0.75 * (end_y - start_y),
    )


def plot_quarter_lines(ax: Axes, sagittal_y: NDArray[Any]) -> None:
    """
    Plot horizontal lines at 25%, 50%, and 75% of the sagittal profile's vertical span.
    """
    start_y, end_y = sagittal_y[0], sagittal_y[-1]
    q1, q2, q3 = calculate_quarter_lines(start_y, end_y)

    for q, label in zip((q1, q2, q3), ("25% Line", "50% Line", "75% Line")):
        ax.axhline(q, color="green", linestyle="--", linewidth=1, label=label)  # type: ignore


def find_local_max_min(sagittal_x: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Find local maxima and minima on the sagittal profile using peak detection.
    Returns two arrays: indices of maxima and indices of minima.
    """
    max_indices, _ = find_peaks(sagittal_x)  # type: ignore
    min_indices, _ = find_peaks(-sagittal_x)  # type: ignore
    return max_indices, min_indices  # type: ignore


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


def plot_lateral_landmarks(ax: Axes, landmarks: NDArray[Any], shift_x: int) -> None:
    """
    Plot the 6 lateral landmarks on the sagittal profile, shifted to the left by shift_x.
    """
    for i, name in enumerate(LATERAL_LANDMARK_NAMES):
        x, y = landmarks[i]

        # Only plot if a valid point was found.
        if x != -1 and y != -1:
            x -= shift_x  # Shift x-coordinate to the left by shift_x
            ax.scatter(x, y, color="green", s=80, zorder=3)  # type: ignore
            ax.annotate(  # type: ignore
                name,
                (x, y),
                textcoords="offset points",
                xytext=(10, 0),  # Move text 10 points to the right
                ha="left",  # Align text to the left of the point
                color="black",
                fontsize=14,
                fontweight="bold",
            )


# ---------------- Main Function ----------------


def save_debug_plot(
    sagittal_x: NDArray[Any], sagittal_y: NDArray[Any], filename: str
) -> None:
    """
    Helper function to plot and save the sagittal profile for debugging purposes.
    """
    fig, ax = plt.subplots(figsize=(6, 10))  # type: ignore
    ax.plot(  # type: ignore
        sagittal_x, sagittal_y, color="black", linewidth=2, label="Sagittal Profile"
    )
    ax.invert_yaxis()  # Maintain consistency with image coordinates
    ax.set_aspect("equal")
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # type: ignore
    plt.close(fig)


def analyze_lateral(
    input_image: Image.Image,
    landmarks: List[Tuple[int, int]],
) -> Tuple[NDArray[Any], NDArray[Any], Any, NDArray[Any]]:
    """
    Analyze the side profile from a loaded PIL image and return only the far-right plot (sagittal profile).
    The plot will have no axes, margins, or labels, but will still include the legend.
    """
    # Process the image: remove background, threshold, and clean up.
    _, binary_np, _, _ = process_image(input_image)
    # processed_image.save("debug_image1.png")
    # cv2.imwrite("debug_image2.png", binary_np)

    # Extract the sagittal profile.
    sagittal_x, sagittal_y = extract_sagittal_profile(binary_np)
    sagittal_x, shift_x = shift_sagittal_profile(sagittal_x)
    # save_debug_plot(sagittal_x, sagittal_y, "debug_image3.png")

    # Compute derivatives on the sagittal profile.
    _, _, dx_scaled, ddx_scaled = compute_derivatives(sagittal_x)

    # Create the sagittal profile plot.
    fig, ax2 = plt.subplots(figsize=(6, 10))  # type: ignore

    # Plot the sagittal profile.
    plot_sagittal_profile(ax2, sagittal_x, sagittal_y, dx_scaled, ddx_scaled)

    # Find local extrema.
    max_indices, min_indices = find_local_max_min(sagittal_x)
    if DEBUG:
        plot_sagittal_minmax(ax2, sagittal_x, sagittal_y, max_indices, min_indices)

    # Compute and plot lateral landmarks.
    landmarks = find_lateral_landmarks(
        sagittal_x, sagittal_y, max_indices, min_indices, int(shift_x), landmarks
    )
    plot_lateral_landmarks(ax2, landmarks, int(shift_x))
    logging.debug("Lateral Landmarks (x, y):")
    logging.debug(landmarks)

    if DEBUG:
        plot_quarter_lines(ax2, sagittal_y)

    # Finalize the plot appearance.
    ax2.set_ylim(1024, 0)  # Ensures the y-axis is inverted correctly
    ax2.set_xlim(-25, 512)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.margins(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    legend = ax2.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 1.0))  # type: ignore
    legend.get_frame().set_alpha(0.8)

    # Convert the plot to OpenCV format.
    return (
        util.convert_matplotlib_to_opencv(ax2),
        landmarks,
        sagittal_x + shift_x,
        sagittal_y,
    )


from typing import Optional  # make sure this import exists


def find_lateral_landmark(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    y_coord: float,
    find_max: bool = True,
    y_forward: Optional[bool] = None,
) -> NDArray[Any]:
    """
    Finds the lateral landmark nearest to a specified y-coordinate on the sagittal line,
    either as a local maximum or minimum depending on the find_max parameter.

    Direction control (y_forward):
      - None (default): search both directions (nearest by |Δy|, current behavior).
      - True: search only "forward" in y (indices with sagittal_y >= y_coord).
      - False: search only "backward" in y (indices with sagittal_y <= y_coord).

    Note: "Forward" here means increasing y values.

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        max_indices (NDArray[Any]): Indices of local maxima.
        min_indices (NDArray[Any]): Indices of local minima.
        y_coord (float): Target y-coordinate to find nearest landmark.
        find_max (bool): If True, find nearest local maximum; otherwise, nearest local minimum.
        y_forward (Optional[bool]): Directional constraint as described above.

    Returns:
        NDArray[Any]: (x, y) coordinates of the found landmark, or [-1, -1] if none.
    """
    indices = max_indices if find_max else min_indices
    if len(indices) == 0:
        return np.array([-1.0, -1.0])

    # Apply directional filter if requested
    if y_forward is True:
        mask = sagittal_y[indices] >= y_coord
        candidates = indices[mask]
    elif y_forward is False:
        mask = sagittal_y[indices] <= y_coord
        candidates = indices[mask]
    else:
        candidates = indices  # no directional constraint

    if len(candidates) == 0:
        return np.array([-1.0, -1.0])

    # Nearest by vertical distance
    diffs = np.abs(sagittal_y[candidates] - y_coord)
    closest_idx = candidates[int(np.argmin(diffs))]

    return np.array([float(sagittal_x[closest_idx]), float(sagittal_y[closest_idx])])


def find_nearest_sagittal_point(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    y_coord: float,
) -> NDArray[Any]:
    """
    Finds the sagittal landmark nearest to a specified y-coordinate on the sagittal line.

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        y_coord (float): Target y-coordinate to find the nearest landmark.

    Returns:
        NDArray[Any]: (x, y) coordinates of the found landmark.
    """
    if len(sagittal_y) == 0:
        return np.array([-1.0, -1.0])

    closest_idx = np.argmin(np.abs(sagittal_y - y_coord))
    return np.array([sagittal_x[closest_idx], sagittal_y[closest_idx]])


def find_corner_between_landmarks(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    landmarks_frontal: NDArray[Any],
    lm_idx_a: int,
    lm_idx_b: int,
    *,
    smooth_window: int = 7,
    slope_delta_thresh: float = 0.30,
    min_consec: int = 3,
    zero_tol: float = 0.04,
    min_ddx: float = 0.0,
    look_from_baseline: bool = True,
    snap_window: int = 20,  # NEW: how far past max-κ to look for dx zero-crossing
) -> NDArray[Any]:
    """
    Curvature-based 'corner' with a post-step that snaps to the first dx zero-crossing (−→+)
    after the max-curvature index (within snap_window). If no crossing, use max-curvature.
    """
    # ---- guards ----
    if (
        landmarks_frontal is None
        or len(landmarks_frontal) == 0
        or lm_idx_a < 0
        or lm_idx_b < 0
        or lm_idx_a >= len(landmarks_frontal)
        or lm_idx_b >= len(landmarks_frontal)
        or sagittal_x is None
        or sagittal_y is None
        or len(sagittal_x) == 0
        or len(sagittal_y) == 0
        or len(sagittal_x) != len(sagittal_y)
    ):
        return np.array([-1.0, -1.0])

    xa, ya = landmarks_frontal[lm_idx_a]
    xb, yb = landmarks_frontal[lm_idx_b]
    if xa < 0 or ya < 0 or xb < 0 or yb < 0:
        return np.array([-1.0, -1.0])

    # ---- Y slice ----
    y_min, y_max = (ya, yb) if ya <= yb else (yb, ya)
    idxs = np.where((sagittal_y >= y_min) & (sagittal_y <= y_max))[0]
    if idxs.size < 3:
        i_a = int(np.argmin(np.abs(sagittal_y - ya)))
        i_b = int(np.argmin(np.abs(sagittal_y - yb)))
        i0, i1 = (i_a, i_b) if i_a <= i_b else (i_b, i_a)
        if i0 == i1:
            i0 = max(0, i0 - 1)
            i1 = min(len(sagittal_y) - 1, i1 + 1)
        idxs = np.arange(i0, i1 + 1, dtype=int)
        if idxs.size < 3:
            return np.array([-1.0, -1.0])

    seg_x = sagittal_x[idxs].astype(float)

    # ---- smoothing (odd window) ----
    if smooth_window and smooth_window > 1:
        k = int(smooth_window)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        seg_x = np.convolve(
            np.pad(seg_x, (pad, pad), mode="edge"),
            np.ones(k, dtype=float) / k,
            mode="valid",
        )

    # ---- derivatives & curvature ----
    dx = np.gradient(seg_x)
    ddx = np.gradient(dx)
    kappa = np.abs(ddx) / np.power(1.0 + dx * dx, 1.5)

    # ---- gate: wait until slope departs baseline ----
    start_gate = 0
    if look_from_baseline and len(dx) >= 5:
        baseline = float(np.median(dx[:5]))
        delta = np.abs(dx - baseline)
        run = 0
        prev = -9999
        first = None
        for i in np.where(delta >= float(slope_delta_thresh))[0]:
            if first is None or i != prev + 1:
                run, first = 1, i
            else:
                run += 1
            if run >= int(min_consec):
                start_gate = int(first)
                break
            prev = i

    # ---- pick max curvature past the gate ----
    kappa2 = kappa.copy()
    kappa2[:start_gate] = -1.0
    j_star = int(np.argmax(kappa2))
    if kappa2[j_star] <= 0:
        return np.array([-1.0, -1.0])

    # ---- NEW: snap to first zero-crossing of dx (−→+) after j_star ----
    j_end = min(len(dx) - 2, j_star + int(snap_window))
    snapped = None
    for j in range(j_star, j_end + 1):
        if dx[j] <= 0.0 and dx[j + 1] >= 0.0:
            # linear interpolate between j and j+1 to dx=0
            a, b = float(dx[j]), float(dx[j + 1])
            t = 0.0 if (a == b) else (abs(a) / (abs(a) + abs(b)))
            # map back to ORIGINAL (unsmoothed) arrays for coordinates
            g0, g1 = int(idxs[j]), int(idxs[j + 1])
            x0, x1 = float(sagittal_x[g0]), float(sagittal_x[g1])
            y0, y1 = float(sagittal_y[g0]), float(sagittal_y[g1])
            xz = (1.0 - t) * x0 + t * x1
            yz = (1.0 - t) * y0 + t * y1
            snapped = np.array([xz, yz])
            break

    if snapped is not None:
        return snapped

    # fallback: use the max-curvature index
    best_global_idx = int(idxs[j_star])
    return np.array(
        [float(sagittal_x[best_global_idx]), float(sagittal_y[best_global_idx])]
    )


def find_lateral_landmarks(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    shift_x: int,
    landmarks_frontal: NDArray[Any],
    *,
    # subnasal/corner tuning knobs
    corner_smooth_window: int = 7,
    corner_slope_delta_thresh: float = 0.25,
    corner_min_consec: int = 3,
    corner_zero_tol: float = 0.05,
    corner_min_ddx: float = 0.0,
    corner_look_from_baseline: bool = True,
) -> NDArray[Any]:
    """
    Compute lateral landmarks; Subnasal now uses curvature-based 'corner' detection.
    """
    landmarks_frontal = np.array(landmarks_frontal)

    highest_landmark_idx = np.argmin(landmarks_frontal[:, 1])

    # (out_idx, frontal_idx, find_max)
    landmark_mapping = [
        (0, highest_landmark_idx, False, None),  # Glabella (min)
        (1, 51, True, None),  # Nasion (max)
        (2, 54, False, None),  # Nasal tip (min)
        (3, 54, True, None),  # Subnasal (max)
        (4, 16, True, False),  # Mento Labial Point (max)
        (5, 16, False, False),  # Pogonion (min)
    ]

    landmarks = np.full((6, 2), -1.0)

    # extrema-driven
    for out_idx, lm_index, find_max, y_forward in landmark_mapping:
        y_coord = landmarks_frontal[lm_index][1]
        pt = find_lateral_landmark(
            sagittal_x,
            sagittal_y,
            max_indices,
            min_indices,
            y_coord=y_coord,
            find_max=find_max,
            y_forward=y_forward,
        )
        landmarks[out_idx] = pt

    # ---- Subnasal (index 3): corner between 57 and 79 ----
    idx_a, idx_b = 57, 79
    subnasal_pt = find_corner_between_landmarks(
        sagittal_x,
        sagittal_y,
        landmarks_frontal,
        idx_a,
        idx_b,
        smooth_window=corner_smooth_window,
        slope_delta_thresh=corner_slope_delta_thresh,
        min_consec=corner_min_consec,
        zero_tol=corner_zero_tol,
        min_ddx=corner_min_ddx,
        look_from_baseline=corner_look_from_baseline,
    )

    # Fallback: nearest sagittal at Y(57)
    if subnasal_pt[0] < 0 or subnasal_pt[1] < 0:
        if 0 <= idx_a < len(landmarks_frontal) and landmarks_frontal[idx_a][1] >= 0:
            subnasal_pt = find_nearest_sagittal_point(
                sagittal_x=sagittal_x,
                sagittal_y=sagittal_y,
                y_coord=landmarks_frontal[idx_a][1],
            )

    # landmarks[3] = subnasal_pt
    landmarks[4] = find_lateral_landmark(
        sagittal_x,
        sagittal_y,
        max_indices,
        min_indices,
        y_coord=landmarks[5][1],
        find_max=True,
    )

    # shift back and cast
    landmarks[:, 0] += shift_x
    return np.array([tuple(map(int, point)) for point in landmarks])
