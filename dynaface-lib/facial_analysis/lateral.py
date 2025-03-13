import io
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from facial_analysis import util
from matplotlib.axes import Axes
from PIL import Image
from rembg import remove
from scipy.signal import find_peaks  # <-- New import for peak detection

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


def process_image(input_image: Image.Image) -> Tuple[Image.Image, np.ndarray, int, int]:
    """
    Process the image by removing the background, converting to grayscale and binary,
    applying morphological closing, and inverting the binary image.
    """
    # Remove background
    output_image: Image.Image = remove(input_image)

    # Convert to grayscale
    grayscale: Image.Image = output_image.convert("L")

    # Convert to binary
    binary_threshold: int = 32
    binary: Image.Image = grayscale.point(lambda p: 255 if p > binary_threshold else 0)
    binary_np: np.ndarray = np.array(binary)

    # Apply morphological closing
    kernel: np.ndarray = np.ones((10, 10), np.uint8)
    binary_np = cv2.morphologyEx(binary_np, cv2.MORPH_CLOSE, kernel)

    # Invert the binary image
    binary_np = 255 - binary_np

    # Get image dimensions
    height, width = binary_np.shape

    return input_image, binary_np, width, height


def shift_sagittal_profile(sagittal_x: np.ndarray) -> np.ndarray:
    """
    Shift the sagittal profile so that the lowest x-coordinate becomes 0.
    """
    min_x = np.min(sagittal_x)
    return sagittal_x - min_x, min_x


def extract_sagittal_profile(binary_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the sagittal profile from the binary image. For each row, finds the first black pixel.
    """
    height, width = binary_np.shape
    sagittal_x: list = []
    sagittal_y: list = []
    for y in range(height):
        row = binary_np[y, :]
        black_pixels = np.where(row == 0)[0]  # Get indices of black pixels
        if len(black_pixels) > 0:
            sagittal_x.append(black_pixels[0])  # Take the first black pixel (leftmost)
            sagittal_y.append(y)

    return np.array(sagittal_x), np.array(sagittal_y)


def compute_derivatives(
    sagittal_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the first and second derivatives of the sagittal profile and return both the raw and scaled values.
    """
    dx: np.ndarray = np.gradient(sagittal_x)
    ddx: np.ndarray = np.gradient(dx)
    dx_scaled: np.ndarray = dx + DX1_OFFSET + DX1_SCALE_FACTOR * dx
    ddx_scaled: np.ndarray = ddx + DX2_OFFSET + DX2_SCALE_FACTOR * ddx
    return dx, ddx, dx_scaled, ddx_scaled


def plot_sagittal_profile(
    ax: Axes,
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    dx_scaled: np.ndarray,
    ddx_scaled: np.ndarray,
) -> None:
    """
    Plot the sagittal profile along with its first and second derivatives on the given axes.
    """
    ax.plot(
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


def plot_quarter_lines(ax: Axes, sagittal_y: np.ndarray) -> None:
    """
    Plot horizontal lines at 25%, 50%, and 75% of the sagittal profile's vertical span.
    """
    start_y, end_y = sagittal_y[0], sagittal_y[-1]
    q1, q2, q3 = calculate_quarter_lines(start_y, end_y)

    for q, label in zip((q1, q2, q3), ("25% Line", "50% Line", "75% Line")):
        ax.axhline(q, color="green", linestyle="--", linewidth=1, label=label)


def find_local_max_min(sagittal_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima and minima on the sagittal profile using peak detection.
    Returns two arrays: indices of maxima and indices of minima.
    """
    max_indices, _ = find_peaks(sagittal_x)
    min_indices, _ = find_peaks(-sagittal_x)
    return max_indices, min_indices


def find_nasal_tip(
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    min_indices: np.ndarray,
    q2: float,
    q3: float,
) -> np.ndarray:
    """
    Finds the nasal tip as the smallest local minimum between the 50th (Q2) and 75th (Q3) percentile lines.

    Args:
        sagittal_x (np.ndarray): X-coordinates of the sagittal profile.
        sagittal_y (np.ndarray): Y-coordinates of the sagittal profile.
        min_indices (np.ndarray): Indices of local minima.
        q2 (float): 50th percentile line.
        q3 (float): 75th percentile line.

    Returns:
        np.ndarray: (x, y) coordinates of the nasal tip, or [-1, -1] if no valid minimum is found.
    """
    valid_min_indices = min_indices[
        (sagittal_y[min_indices] >= q2) & (sagittal_y[min_indices] <= q3)
    ]

    if len(valid_min_indices) > 0:
        nasal_tip_idx = valid_min_indices[np.argmin(sagittal_y[valid_min_indices])]
        return np.array([sagittal_x[nasal_tip_idx], sagittal_y[nasal_tip_idx]])

    return np.array([-1.0, -1.0])


def find_soft_tissue_pogonion(
    sagittal_x: np.ndarray, sagittal_y: np.ndarray, min_indices: np.ndarray, q3: float
) -> np.ndarray:
    """
    Finds the soft tissue pogonion as the last local minimum between the 75th (Q3) and 100th percentile lines.

    Args:
        sagittal_x (np.ndarray): X-coordinates of the sagittal profile.
        sagittal_y (np.ndarray): Y-coordinates of the sagittal profile.
        min_indices (np.ndarray): Indices of local minima.
        q3 (float): 75th percentile line.

    Returns:
        np.ndarray: (x, y) coordinates of the soft tissue pogonion, or [-1, -1] if no valid minimum is found.
    """
    valid_min_indices = min_indices[sagittal_y[min_indices] >= q3]

    if len(valid_min_indices) > 0:
        # Select the last local minimum in the range (closest to the end)
        pogonion_idx = valid_min_indices[-1]
        return np.array([sagittal_x[pogonion_idx], sagittal_y[pogonion_idx]])

    return np.array([-1.0, -1.0])


def find_soft_tissue_glabella(
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    min_indices: np.ndarray,
    q1: float,
    q2: float,
) -> np.ndarray:
    """
    Finds the soft tissue glabella as the local minimum closest to the 25% (Q1) line but within the 25%-50% (Q1-Q2) range.

    Args:
        sagittal_x (np.ndarray): X-coordinates of the sagittal profile.
        sagittal_y (np.ndarray): Y-coordinates of the sagittal profile.
        min_indices (np.ndarray): Indices of local minima.
        q1 (float): 25th percentile line.
        q2 (float): 50th percentile line.

    Returns:
        np.ndarray: (x, y) coordinates of the soft tissue glabella, or [-1, -1] if no valid minimum is found.
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
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    max_indices: np.ndarray,
    glabella_x: float,
    glabella_y: float,
    q1: float,
    q2: float,
) -> np.ndarray:
    """
    Finds the soft tissue nasion as the next local maximum after glabella, moving toward the nasal tip,
    but within the 25%-50% (Q1-Q2) range. Ensures nasion is **above** glabella (`y_nasion > y_glabella`).

    Args:
        sagittal_x (np.ndarray): X-coordinates of the sagittal profile.
        sagittal_y (np.ndarray): Y-coordinates of the sagittal profile.
        max_indices (np.ndarray): Indices of local maxima.
        glabella_x (float): X-coordinate of the glabella.
        glabella_y (float): Y-coordinate of the glabella.
        q1 (float): 25th percentile line.
        q2 (float): 50th percentile line.

    Returns:
        np.ndarray: (x, y) coordinates of the soft tissue nasion, or [-1, -1] if no valid max is found.
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
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    max_indices: np.ndarray,
    nasal_tip_x: float,
    nasal_tip_y: float,
) -> np.ndarray:
    """
    Finds the subnasal point as the next local maximum after the nasal tip and above it.

    Args:
        sagittal_x (np.ndarray): X-coordinates of the sagittal profile.
        sagittal_y (np.ndarray): Y-coordinates of the sagittal profile.
        max_indices (np.ndarray): Indices of local maxima.
        nasal_tip_x (float): X-coordinate of the nasal tip.
        nasal_tip_y (float): Y-coordinate of the nasal tip.

    Returns:
        np.ndarray: (x, y) coordinates of the subnasal point, or [-1, -1] if no valid max is found.
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
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    max_indices: np.ndarray,
    pogonion_y: float,
    q3: float,
) -> np.ndarray:
    """
    Finds the mento-labial point as the first local maximum below the soft tissue pogonion,
    within the 75%-100% (Q3 to end) range.

    Args:
        sagittal_x (np.ndarray): X-coordinates of the sagittal profile.
        sagittal_y (np.ndarray): Y-coordinates of the sagittal profile.
        max_indices (np.ndarray): Indices of local maxima.
        pogonion_y (float): Y-coordinate of the pogonion.
        q3 (float): 75th percentile line.

    Returns:
        np.ndarray: (x, y) coordinates of the mento-labial point, or [-1, -1] if no valid max is found.
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
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    max_indices: np.ndarray,
    min_indices: np.ndarray,
    shift_x: int,
) -> np.ndarray:
    """
    Using the local extrema, compute the 6 lateral landmarks.

    Returns:
        np.ndarray: A 2D numpy array of shape (6, 2), where each row contains [x, y] for a landmark:
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

    return [tuple(map(int, point)) for point in landmarks]


def plot_sagittal_minmax(
    ax: Axes,
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    max_indices: np.ndarray,
    min_indices: np.ndarray,
) -> None:
    """
    Plot the local minima and maxima points (with index annotations) on the sagittal profile.
    """
    ax.scatter(
        sagittal_x[max_indices],
        sagittal_y[max_indices],
        color="green",
        s=80,
        label="Local Maxima",
        zorder=3,
    )
    ax.scatter(
        sagittal_x[min_indices],
        sagittal_y[min_indices],
        color="red",
        s=80,
        label="Local Minima",
        zorder=3,
    )

    # Enumerate for positional index labels
    for i, idx in enumerate(max_indices):
        ax.annotate(
            f"max-{i}",
            (sagittal_x[idx], sagittal_y[idx]),
            textcoords="offset points",
            xytext=(10, 0),  # Shift right
            ha="left",
            va="center",
            color="green",
        )

    for i, idx in enumerate(min_indices):
        ax.annotate(
            f"min-{i}",
            (sagittal_x[idx], sagittal_y[idx]),
            textcoords="offset points",
            xytext=(10, 0),  # Shift right
            ha="left",
            va="center",
            color="red",
        )


def plot_lateral_landmarks(ax: Axes, landmarks: np.ndarray, shift_x: int) -> None:
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
            ax.scatter(x, y, color="green", s=80, zorder=3)
            ax.annotate(
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
    sagittal_x: np.ndarray, sagittal_y: np.ndarray, filename: str
) -> None:
    """
    Helper function to plot and save the sagittal profile for debugging purposes.
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.plot(
        sagittal_x, sagittal_y, color="black", linewidth=2, label="Sagittal Profile"
    )
    ax.invert_yaxis()  # Maintain consistency with image coordinates
    ax.set_aspect("equal")
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_lateral(input_image: Image.Image) -> np.ndarray:
    """
    Analyze the side profile from a loaded PIL image and return only the far-right plot (sagittal profile).
    The plot will have no axes, margins, or labels, but will still include the legend.
    """
    # Process the image: remove background, threshold, and clean up.
    processed_image, binary_np, _, _ = process_image(input_image)
    # processed_image.save("debug_image1.png")
    # cv2.imwrite("debug_image2.png", binary_np)

    # Extract the sagittal profile.
    sagittal_x, sagittal_y = extract_sagittal_profile(binary_np)
    sagittal_x, shift_x = shift_sagittal_profile(sagittal_x)
    # save_debug_plot(sagittal_x, sagittal_y, "debug_image3.png")

    # Compute derivatives on the sagittal profile.
    dx, ddx, dx_scaled, ddx_scaled = compute_derivatives(sagittal_x)

    # Create the sagittal profile plot.
    fig, ax2 = plt.subplots(figsize=(6, 10))

    # Plot the sagittal profile.
    plot_sagittal_profile(ax2, sagittal_x, sagittal_y, dx_scaled, ddx_scaled)

    # Find local extrema.
    max_indices, min_indices = find_local_max_min(sagittal_x)
    if DEBUG:
        plot_sagittal_minmax(ax2, sagittal_x, sagittal_y, max_indices, min_indices)

    # Compute and plot lateral landmarks.
    landmarks = find_lateral_landmarks(
        sagittal_x, sagittal_y, max_indices, min_indices, shift_x
    )
    plot_lateral_landmarks(ax2, landmarks, shift_x)
    if DEBUG:
        print("Lateral Landmarks (x, y):")
        print(landmarks)

    if DEBUG:
        plot_quarter_lines(ax2, sagittal_y)

    # Finalize the plot appearance.
    ax2.set_ylim(1024, 0)  # Ensures the y-axis is inverted correctly
    ax2.set_xlim(-25, 512)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.margins(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    legend = ax2.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    legend.get_frame().set_alpha(0.8)

    # Convert the plot to OpenCV format.
    return (
        util.convert_matplotlib_to_opencv(ax2),
        landmarks,
        sagittal_x + shift_x,
        sagittal_y,
    )
