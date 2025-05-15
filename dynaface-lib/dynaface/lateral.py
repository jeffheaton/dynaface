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


def find_lateral_landmark(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    y_coord: float,
    find_max: bool = True,
) -> NDArray[Any]:
    """
    Finds the lateral landmark nearest to a specified y-coordinate on the sagittal line,
    either as a local maximum or minimum depending on the find_max parameter.

    Args:
        sagittal_x (NDArray[Any]): X-coordinates of the sagittal profile.
        sagittal_y (NDArray[Any]): Y-coordinates of the sagittal profile.
        max_indices (NDArray[Any]): Indices of local maxima.
        min_indices (NDArray[Any]): Indices of local minima.
        y_coord (float): Target y-coordinate to find nearest landmark.
        find_max (bool): If True, find nearest local maximum; otherwise, find nearest local minimum.

    Returns:
        NDArray[Any]: (x, y) coordinates of the found landmark, or [-1, -1] if no valid landmark is found.
    """
    indices = max_indices if find_max else min_indices

    if len(indices) == 0:
        return np.array([-1.0, -1.0])

    # Find the index of the landmark nearest to the provided y-coordinate
    closest_idx = indices[np.argmin(np.abs(sagittal_y[indices] - y_coord))]

    return np.array([sagittal_x[closest_idx], sagittal_y[closest_idx]])


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


def find_lateral_landmarks(
    sagittal_x: NDArray[Any],
    sagittal_y: NDArray[Any],
    max_indices: NDArray[Any],
    min_indices: NDArray[Any],
    shift_x: int,
    landmarks_frontal: NDArray[Any],
) -> NDArray[Any]:
    """
    Compute lateral landmarks based on y-coordinates from known frontal landmarks.

    Returns:
        NDArray[Any]: A 6x2 array containing the (x, y) coordinates for each landmark:
          - 0: Soft Tissue Glabella (highest landmark on face, local min)
          - 1: Soft Tissue Nasion (landmark #53, local max)
          - 2: Nasal Tip (landmark #54, local min)
          - 3: Subnasal Point (landmark #57, nearest sagittal point)
          - 4: Mento Labial Point (landmark #85, local max)
          - 5: Soft Tissue Pogonion (landmark #16, local min)
    """

    landmarks_frontal = np.array(landmarks_frontal)

    # Dynamically select highest landmark for Glabella (smallest y-value)
    highest_landmark_idx = np.argmin(landmarks_frontal[:, 1])

    landmark_mapping = [
        (highest_landmark_idx, False),  # Glabella (min) dynamically chosen
        (53, True),  # Nasion (max)
        (54, False),  # Nasal tip (min)
        (58, None),  # Subnasal point (nearest sagittal point, no min/max)
        (85, True),  # Mento Labial (max)
        (16, False),  # Pogonion (min)
    ]

    landmarks = np.full((6, 2), -1.0)

    for i, (lm_index, find_max) in enumerate(landmark_mapping):
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

        landmarks[i] = landmark_point

    # Shift all x-coordinates by shift_x
    landmarks[:, 0] += shift_x

    return np.array([tuple(map(int, point)) for point in landmarks])
