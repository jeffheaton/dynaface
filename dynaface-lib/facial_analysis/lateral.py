import io
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from facial_analysis import util
from matplotlib.axes import Axes
from PIL import Image
from rembg import remove
from scipy.signal import find_peaks  # <-- New import for peak detection

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


def process_image(input_image: Image.Image) -> Tuple[Image.Image, np.ndarray, int, int]:
    """
    Process the image by removing the background, converting to grayscale and binary,
    applying morphological closing, and inverting the binary image.

    Args:
        input_image (Image.Image): The loaded PIL image.

    Returns:
        Tuple[Image.Image, np.ndarray, int, int]: A tuple containing the processed input image,
        the processed binary numpy array, the image width, and the image height.
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

    Args:
        sagittal_x (np.ndarray): Array of x-values from the sagittal profile.

    Returns:
        np.ndarray: Shifted x-values with the lowest x-coordinate set to 0.
    """
    min_x = np.min(sagittal_x)
    return sagittal_x - min_x


def extract_sagittal_profile(binary_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the sagittal profile from the binary image. For each row, finds the first black pixel.

    Args:
        binary_np (np.ndarray): Binary image as a numpy array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two numpy arrays representing the x and y coordinates of the sagittal profile.
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

    Args:
        sagittal_x (np.ndarray): Array of x-values from the sagittal profile.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The first derivative (dx), second derivative (ddx),
        scaled first derivative (dx_scaled), and scaled second derivative (ddx_scaled).
    """
    dx: np.ndarray = np.gradient(sagittal_x)
    ddx: np.ndarray = np.gradient(dx)
    dx_scaled: np.ndarray = dx + DX1_OFFSET + DX1_SCALE_FACTOR * dx
    ddx_scaled: np.ndarray = ddx + DX2_OFFSET + DX2_SCALE_FACTOR * ddx
    return dx, ddx, dx_scaled, ddx_scaled


def plot_sagittal_profile(
    ax: Axes,
    sagittal_x_adj: np.ndarray,
    sagittal_y_adj: np.ndarray,
    dx_scaled: np.ndarray,
    ddx_scaled: np.ndarray,
) -> None:
    """
    Plot the sagittal profile along with its first and second derivatives on the given axes.

    Args:
        ax (Axes): Matplotlib Axes object.
        sagittal_x_adj (np.ndarray): Adjusted x-values of the sagittal profile.
        sagittal_y_adj (np.ndarray): Adjusted y-values of the sagittal profile.
        dx_scaled (np.ndarray): Scaled first derivative.
        ddx_scaled (np.ndarray): Scaled second derivative.
    """
    ax.plot(
        sagittal_x_adj,
        sagittal_y_adj,
        color="black",
        linewidth=2,
        label="Sagittal Profile",
    )


"""
    ax.plot(
        dx_scaled,
        sagittal_y_adj,
        color="red",
        linewidth=1.5,
        linestyle="--",
        label="1st Derivative",
    )
    ax.plot(
        ddx_scaled,
        sagittal_y_adj,
        color="blue",
        linewidth=1.5,
        linestyle="--",
        label="2nd Derivative",
    )
    if DEBUG:
        ax.scatter(
            sagittal_x_adj[0],
            sagittal_y_adj[0],
            color="green",
            s=100,
            label="Start Point",
            zorder=3,
        )
        ax.scatter(
            sagittal_x_adj[-1],
            sagittal_y_adj[-1],
            color="purple",
            s=100,
            label="End Point",
            zorder=3,
        )
"""


def plot_quarter_lines(ax: Axes, sagittal_y_adj: np.ndarray) -> None:
    """
    Plot horizontal lines at 25%, 50%, and 75% of the sagittal profile's vertical span.

    Args:
        ax (Axes): Matplotlib Axes object.
        sagittal_y_adj (np.ndarray): Adjusted y-values of the sagittal profile.
    """
    start_y: int = sagittal_y_adj[0]
    end_y: int = sagittal_y_adj[-1]
    q1: float = start_y + 0.25 * (end_y - start_y)
    q2: float = start_y + 0.50 * (end_y - start_y)  # Midpoint
    q3: float = start_y + 0.75 * (end_y - start_y)
    ax.axhline(q1, color="green", linestyle="--", linewidth=1, label="25% Line")
    ax.axhline(q2, color="green", linestyle="--", linewidth=1, label="50% Line")
    ax.axhline(q3, color="green", linestyle="--", linewidth=1, label="75% Line")


def plot_and_print_local_extrema(
    ax: Axes, sagittal_x_adj: np.ndarray, sagittal_y_adj: np.ndarray
) -> None:
    """
    Identify local extrema (maxima and minima) in the sagittal profile, plot them, annotate the points,
    and print their details sorted by value descending if DEBUG is True.

    Args:
        ax (Axes): Matplotlib Axes object.
        sagittal_x_adj (np.ndarray): Adjusted x-values of the sagittal profile.
        sagittal_y_adj (np.ndarray): Adjusted y-values of the sagittal profile.
    """
    max_indices, _ = find_peaks(sagittal_x_adj)
    min_indices, _ = find_peaks(-sagittal_x_adj)

    if DEBUG:
        ax.scatter(
            sagittal_x_adj[max_indices],
            sagittal_y_adj[max_indices],
            color="magenta",
            s=80,
            label="Local Maxima",
            zorder=3,
        )
        ax.scatter(
            sagittal_x_adj[min_indices],
            sagittal_y_adj[min_indices],
            color="cyan",
            s=80,
            label="Local Minima",
            zorder=3,
        )

    def annotate_point(label, x, y, color):
        ax.scatter(x, y, color=color, s=80, zorder=3)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 7),  # Adjusted to 7 for even closer labels
            ha="center",
            va="bottom",  # Ensures text is right above the point
            color=color,
            fontsize=12,
            fontweight="bold",
        )

    # Identify chin as the first maximum (lowest index)
    if len(max_indices) > 0:
        chin_idx = min(max_indices)
        annotate_point(
            "Soft Tissue Pogonion",
            sagittal_x_adj[chin_idx],
            sagittal_y_adj[chin_idx],
            "blue",
        )

    # Identify tip as the largest maximum (excluding chin)
    max_info = [(idx, sagittal_x_adj[idx]) for idx in max_indices if idx != chin_idx]
    max_info_sorted = sorted(max_info, key=lambda x: x[1], reverse=True)

    if max_info_sorted:
        tip_idx = max_info_sorted[0][0]
        annotate_point(
            "Nasal Tip", sagittal_x_adj[tip_idx], sagittal_y_adj[tip_idx], "red"
        )

    # Identify minima between tip and chin
    min_between = [idx for idx in min_indices if chin_idx < idx < tip_idx]
    if min_between:
        subnasal_idx = min(min_between, key=lambda x: abs(x - tip_idx))
        mento_labial_idx = min(min_between, key=lambda x: abs(x - chin_idx))

        annotate_point(
            "Subnasal Point",
            sagittal_x_adj[subnasal_idx],
            sagittal_y_adj[subnasal_idx],
            "green",
        )
        annotate_point(
            "Mento Labial Point",
            sagittal_x_adj[mento_labial_idx],
            sagittal_y_adj[mento_labial_idx],
            "purple",
        )

    # Identify closest min and max to tip in the direction opposite the chin
    min_opposite = [idx for idx in min_indices if idx > tip_idx]
    max_opposite = [idx for idx in max_indices if idx > tip_idx]

    if min_opposite:
        soft_tissue_nason_idx = min(min_opposite, key=lambda x: abs(x - tip_idx))
        annotate_point(
            "Soft Tissue Nason",
            sagittal_x_adj[soft_tissue_nason_idx],
            sagittal_y_adj[soft_tissue_nason_idx],
            "orange",
        )

    if max_opposite:
        soft_tissue_glabella_idx = min(max_opposite, key=lambda x: abs(x - tip_idx))
        annotate_point(
            "Soft Tissue Glabella",
            sagittal_x_adj[soft_tissue_glabella_idx],
            sagittal_y_adj[soft_tissue_glabella_idx],
            "brown",
        )

    if DEBUG:
        print("Local Maxima (Sorted by Value Descending):")
        for idx, val in sorted(
            [(idx, sagittal_x_adj[idx]) for idx in max_indices],
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"Max: Index {idx}, Value {val}")

        print("Local Minima (Sorted by Value Descending):")
        min_info = [(idx, sagittal_x_adj[idx]) for idx in min_indices]
        min_info_sorted = sorted(min_info, key=lambda x: x[1], reverse=True)
        for idx, val in min_info_sorted:
            print(f"Min: Index {idx}, Value {val}")


# ---------------- Main Function ----------------


def save_debug_plot(
    sagittal_x: np.ndarray, sagittal_y: np.ndarray, filename: str
) -> None:
    """
    Helper function to plot and save the sagittal profile for debugging purposes.

    Args:
        sagittal_x (np.ndarray): Array of x-values from the sagittal profile.
        sagittal_y (np.ndarray): Array of y-values from the sagittal profile.
        filename (str): Filename to save the debug image.
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.plot(
        sagittal_x, sagittal_y, color="black", linewidth=2, label="Sagittal Profile"
    )
    ax.invert_yaxis()  # Maintain consistency with image coordinates
    ax.set_aspect("equal")
    # ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_lateral(input_image: Image.Image) -> np.ndarray:
    """
    Analyze the side profile from a loaded PIL image and return only the far-right plot (sagittal profile).
    The plot will have no axes, margins, or labels, but will still include the legend.

    Args:
        input_image (Image.Image): The loaded PIL image to analyze.

    Returns:
        np.ndarray: OpenCV image of the sagittal profile plot.
    """
    # Process the image: flip, remove background, threshold, and clean up.
    processed_image, binary_np, _, _ = process_image(input_image)
    processed_image.save("debug_image1.png")
    cv2.imwrite("debug_image2.png", binary_np)

    # Extract the sagittal profile.
    sagittal_x, sagittal_y = extract_sagittal_profile(binary_np)
    sagittal_x = shift_sagittal_profile(sagittal_x)
    save_debug_plot(sagittal_x, sagittal_y, "debug_image3.png")

    # Compute derivatives on the original trimmed sagittal_x.
    dx, ddx, dx_scaled, ddx_scaled = compute_derivatives(sagittal_x)

    # Create the sagittal profile plot.
    fig, ax2 = plt.subplots(figsize=(6, 10))

    # Plot the sagittal profile with derivatives and extrema.
    plot_sagittal_profile(ax2, sagittal_x, sagittal_y, dx_scaled, ddx_scaled)
    plot_and_print_local_extrema(ax2, sagittal_x, sagittal_y)
    if DEBUG:
        plot_quarter_lines(ax2, sagittal_y)

    # Finalize the plot appearance.
    ax2.set_ylim(1024, 0)  # Ensures the y-axis is inverted correctly
    ax2.set_xlim(-25, 512)
    ax2.set_aspect("equal")
    ax2.axis("off")

    # Remove any extra margins inside the plot
    ax2.margins(0)

    # **Aggressively remove extra figure space**
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Ensure the legend is visible but does not introduce whitespace.
    legend = ax2.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    legend.get_frame().set_alpha(0.8)

    # Convert the plot to OpenCV format.
    return util.convert_matplotlib_to_opencv(ax2)
