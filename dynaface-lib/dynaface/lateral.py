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


def find_nasal_tip(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    min_indices: NDArray[Any],
    q2: float,
    q3: float,
) -> NDArray[Any]:
    """
    Finds the nasal tip as the smallest local minimum between the 50th (Q2) and 75th (Q3) percentile lines.

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        min_indices (NDArray[Any]): Indices of local minima.
        q2 (float): 50th percentile line.
        q3 (float): 75th percentile line.

    Returns:
        NDArray[Any]: (x, y) coordinates of the nasal tip, or [-1, -1] if no valid minimum is found.
    """
    valid_min_indices = min_indices[
        (sagittal_y[min_indices] >= q2) & (sagittal_y[min_indices] <= q3)
    ]

    if len(valid_min_indices) > 0:
        nasal_tip_idx = valid_min_indices[np.argmin(sagittal_y[valid_min_indices])]
        return np.array([sagittal_x[nasal_tip_idx], sagittal_y[nasal_tip_idx]])

    return np.array([-1.0, -1.0])


def find_soft_tissue_pogonion(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    min_indices: NDArray[Any],
    q3: float,
) -> NDArray[Any]:
    """
    Finds the soft tissue pogonion as the last local minimum between the 75th (Q3) and 100th percentile lines.

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        min_indices (NDArray[Any]): Indices of local minima.
        q3 (float): 75th percentile line.

    Returns:
        NDArray[Any]: (x, y) coordinates of the soft tissue pogonion, or [-1, -1] if no valid minimum is found.
    """
    valid_min_indices = min_indices[sagittal_y[min_indices] >= q3]

    if len(valid_min_indices) > 0:
        # Select the last local minimum in the range (closest to the end)
        pogonion_idx = valid_min_indices[-1]
        return np.array([sagittal_x[pogonion_idx], sagittal_y[pogonion_idx]])

    return np.array([-1.0, -1.0])


def find_soft_tissue_glabella(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    min_indices: NDArray[Any],
    q1: float,
    q2: float,
) -> NDArray[Any]:
    """
    Finds the soft tissue glabella as the local minimum closest to the 25% (Q1) line but within the 25%-50% (Q1-Q2) range.

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        min_indices (NDArray[Any]): Indices of local minima.
        q1 (float): 25th percentile line.
        q2 (float): 50th percentile line.

    Returns:
        NDArray[Any]: (x, y) coordinates of the soft tissue glabella, or [-1, -1] if no valid minimum is found.
    """
    valid_min_indices = min_indices[
        (sagittal_y[min_indices] >= q1) & (sagittal_y[min_indices] <= q2)
    ]  # Only consider minima between Q1 and Q2

    if len(valid_min_indices) > 0:
        # Find the local minimum closest to Q1
        glabella_idx = valid_min_indices[
            np.argmin(np.abs(sagittal_y[valid_min_indices] - q1))
        ]
        return np.array([sagittal_x[glabella_idx], sagittal_y[glabella_idx]])

    return np.array([-1.0, -1.0])


def find_soft_tissue_nasion(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    glabella_x: float,
    glabella_y: float,
    q1: float,
    q2: float,
) -> NDArray[Any]:
    """
    Finds the soft tissue nasion as the next local maximum after glabella, moving toward the nasal tip,
    but within the 25%-50% (Q1-Q2) range. Ensures nasion is **above** glabella (`y_nasion > y_glabella`).

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        max_indices (NDArray[Any]): Indices of local maxima.
        glabella_x (float): X-coordinate of the glabella.
        glabella_y (float): Y-coordinate of the glabella.
        q1 (float): 25th percentile line.
        q2 (float): 50th percentile line.

    Returns:
        NDArray[Any]: (x, y) coordinates of the soft tissue nasion, or [-1, -1] if no valid max is found.
    """
    valid_max_indices = max_indices[
        (sagittal_y[max_indices] >= q1) & (sagittal_y[max_indices] <= q2)
    ]

    # Filter for points occurring after glabella and higher than glabella
    valid_max_indices = valid_max_indices[
        (sagittal_x[valid_max_indices] > glabella_x)
        & (sagittal_y[valid_max_indices] > glabella_y)
    ]

    if len(valid_max_indices) > 0:
        nasion_idx = valid_max_indices[0]  # First maximum after Glabella
        return np.array([sagittal_x[nasion_idx], sagittal_y[nasion_idx]])

    return np.array([-1.0, -1.0])


def find_subnasal_point(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    nasal_tip_x: float,
    nasal_tip_y: float,
) -> NDArray[Any]:
    """
    Finds the subnasal point as the next local maximum after the nasal tip and above it.

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        max_indices (NDArray[Any]): Indices of local maxima.
        nasal_tip_x (float): X-coordinate of the nasal tip.
        nasal_tip_y (float): Y-coordinate of the nasal tip.

    Returns:
        NDArray[Any]: (x, y) coordinates of the subnasal point, or [-1, -1] if no valid max is found.
    """
    valid_max_indices = max_indices[
        sagittal_x[max_indices] > nasal_tip_x
    ]  # Must be after Nasal Tip

    # Ensure subnasal point is **above** the nasal tip
    valid_max_indices = valid_max_indices[sagittal_y[valid_max_indices] > nasal_tip_y]

    if len(valid_max_indices) > 0:
        subnasal_idx = valid_max_indices[0]  # First max after Nasal Tip
        return np.array([sagittal_x[subnasal_idx], sagittal_y[subnasal_idx]])

    return np.array([-1.0, -1.0])


def find_mento_labial_point(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    pogonion_y: float,
    q3: float,
) -> NDArray[Any]:
    """
    Finds the mento-labial point as the first local maximum below the soft tissue pogonion,
    within the 75%-100% (Q3 to end) range.

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        max_indices (NDArray[Any]): Indices of local maxima.
        pogonion_y (float): Y-coordinate of the pogonion.
        q3 (float): 75th percentile line.

    Returns:
        NDArray[Any]: (x, y) coordinates of the mento-labial point, or [-1, -1] if no valid max is found.
    """
    # Filter for maxima that are in the Q3 to end range and **below** the Pogonion
    valid_max_indices = max_indices[
        (sagittal_y[max_indices] >= q3) & (sagittal_y[max_indices] < pogonion_y)
    ]

    if len(valid_max_indices) > 0:
        mento_idx = valid_max_indices[0]  # First valid max below Pogonion
        return np.array([sagittal_x[mento_idx], sagittal_y[mento_idx]])

    return np.array([-1.0, -1.0])


def find_lateral_landmarks(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    shift_x: int,
) -> NDArray[Any]:
    """
    Using the local extrema, compute the 6 lateral landmarks.

    Returns:
        NDArray[Any]: A 6x2 array containing the (x, y) coordinates for each landmark:
          - LATERAL_LM_SOFT_TISSUE_GLABELLA (0): Soft Tissue Glabella
          - LATERAL_LM_SOFT_TISSUE_NASION (1): Soft Tissue Nasion
          - LATERAL_LM_NASAL_TIP (2): Nasal Tip
          - LATERAL_LM_SUBNASAL_POINT (3): Subnasal Point
          - LATERAL_LM_MENTO_LABIAL_POINT (4): Mento Labial Point
          - LATERAL_LM_SOFT_TISSUE_POGONION (5): Soft Tissue Pogonion
    """
    if len(max_indices) == 0:
        raise ValueError("No local maxima found.")

    if len(min_indices) == 0:
        raise ValueError("No local minima found.")

    # Compute quartile lines
    start_y, end_y = sagittal_y[0], sagittal_y[-1]
    q1, q2, q3 = calculate_quarter_lines(start_y, end_y)

    # Initialize landmarks array with placeholder values
    landmarks = np.full((6, 2), -1.0)

    # Compute Soft Tissue Glabella
    glabella = find_soft_tissue_glabella(sagittal_x, sagittal_y, min_indices, q1, q2)
    landmarks[LATERAL_LM_SOFT_TISSUE_GLABELLA] = glabella

    # Compute Soft Tissue Nasion
    landmarks[LATERAL_LM_SOFT_TISSUE_NASION] = find_soft_tissue_nasion(
        sagittal_x, sagittal_y, max_indices, glabella[0], glabella[1], q1, q2
    )

    # Compute Nasal Tip
    nasal_tip = find_nasal_tip(sagittal_x, sagittal_y, min_indices, q2, q3)
    landmarks[LATERAL_LM_NASAL_TIP] = nasal_tip

    # Compute Subnasal Point
    landmarks[LATERAL_LM_SUBNASAL_POINT] = find_subnasal_point(
        sagittal_x, sagittal_y, max_indices, nasal_tip[0], nasal_tip[1]
    )

    # Compute Soft Tissue Pogonion
    pogonion = find_soft_tissue_pogonion(sagittal_x, sagittal_y, min_indices, q3)
    landmarks[LATERAL_LM_SOFT_TISSUE_POGONION] = pogonion

    # Compute Mento-Labial Point (below Pogonion)
    landmarks[LATERAL_LM_MENTO_LABIAL_POINT] = find_mento_labial_point(
        sagittal_x, sagittal_y, max_indices, pogonion[1], q3
    )

    # Shift all x-coordinates to the left by shift_x
    landmarks[:, 0] += shift_x

    # return [tuple(map(int, point)) for point in landmarks]
    return np.array([tuple(map(int, point)) for point in landmarks])


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
    landmark_names = [
        "Soft Tissue Glabella",
        "Soft Tissue Nasion",
        "Nasal Tip",
        "Subnasal Point",
        "Mento Labial Point",
        "Soft Tissue Pogonion",
    ]

    for i, name in enumerate(landmark_names):
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
        sagittal_x, sagittal_y, max_indices, min_indices, int(shift_x)
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
