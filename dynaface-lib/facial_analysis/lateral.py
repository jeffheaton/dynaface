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

# ---------------- Helper Functions ----------------


def download_image(url: str) -> Image.Image:
    """
    Download an image from the given URL and return it as a PIL Image.

    Args:
        url (str): URL of the image to download.

    Returns:
        Image.Image: The downloaded image.

    Raises:
        Exception: If the image cannot be downloaded.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception("Failed to download image")


def crop_image(
    image: Union[Image.Image, np.ndarray],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    is_array: bool = False,
) -> Union[Image.Image, np.ndarray]:
    """
    Crop an image or numpy array given the bounding coordinates.

    Args:
        image (Union[Image.Image, np.ndarray]): The image to crop.
        x_min (float): Minimum x-coordinate.
        x_max (float): Maximum x-coordinate.
        y_min (float): Minimum y-coordinate.
        y_max (float): Maximum y-coordinate.
        is_array (bool): Flag indicating if the image is a numpy array. Defaults to False.

    Returns:
        Union[Image.Image, np.ndarray]: The cropped image.
    """
    if is_array:
        return image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
    else:
        return image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))


def find_end_point(
    idx: int, sagittal_x: np.ndarray, sagittal_y: np.ndarray, threshold: float = 0.05
) -> Optional[int]:
    """
    Find the end point index in the sagittal profile where the x-value drops
    below a threshold relative to the starting x value.

    Args:
        idx (int): Starting index in the profile.
        sagittal_x (np.ndarray): Array of x-values from the profile.
        sagittal_y (np.ndarray): Array of y-values from the profile (unused in computation).
        threshold (float): Threshold factor to determine the endpoint. Defaults to 0.05.

    Returns:
        Optional[int]: The index where the condition is met, or None if not found.
    """
    start_x = sagittal_x[idx]
    for i in range(idx - 1, -1, -1):
        if sagittal_x[i] <= start_x * (1 - threshold):
            return i
    return None


def find_max_point(
    sagittal_x: np.ndarray, start_index: int, end_index: int
) -> Optional[int]:
    """
    Find the index of the maximum x-value in the sagittal profile between end_index and start_index.

    Args:
        sagittal_x (np.ndarray): Array of x-values from the profile.
        start_index (int): The starting index.
        end_index (int): The ending index.

    Returns:
        Optional[int]: The index of the maximum value, or None if conditions are not met.
    """
    if end_index is None or start_index is None or (start_index - end_index) <= 1:
        return None
    return max(range(end_index + 1, start_index), key=lambda i: sagittal_x[i])


def find_bounding_points(
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    min_distance: int = 50,
    threshold: float = 0.05,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Find bounding points (start index, end index, and maximum index) for the sagittal profile.

    Args:
        sagittal_x (np.ndarray): Array of x-values from the profile.
        sagittal_y (np.ndarray): Array of y-values from the profile.
        min_distance (int): Minimum distance between start and end points. Defaults to 50.
        threshold (float): Threshold factor to determine the endpoint. Defaults to 0.05.

    Returns:
        Tuple[Optional[int], Optional[int], Optional[int]]: A tuple containing the start index, end index,
        and the maximum index. Returns (None, None, None) if not found.
    """
    idx: int = len(sagittal_x) - 1
    while idx > 0:
        end_index: Optional[int] = find_end_point(
            idx, sagittal_x, sagittal_y, threshold=threshold
        )
        if end_index is not None and (idx - end_index) >= min_distance:
            max_index: Optional[int] = find_max_point(sagittal_x, idx, end_index)
            if max_index is not None and (end_index < max_index < idx):
                return idx, end_index, max_index
        idx -= 1
    return None, None, None


def trim_sagittal_profile(
    sagittal_x: np.ndarray, sagittal_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim the sagittal profile based on bounding points and print the indices.

    Args:
        sagittal_x (np.ndarray): Array of x-values from the profile.
        sagittal_y (np.ndarray): Array of y-values from the profile.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The trimmed sagittal x and y arrays.
    """
    start_idx, end_idx, max_idx = find_bounding_points(sagittal_x, sagittal_y)

    if DEBUG:
        print("Start index:", start_idx)
        print("End index:", end_idx)
        print("Max index:", max_idx)
    if start_idx is not None and end_idx is not None:
        sagittal_x = sagittal_x[end_idx : start_idx + 1]
        sagittal_y = sagittal_y[end_idx : start_idx + 1]
    return sagittal_x, sagittal_y


def process_image(input_image: Image.Image) -> Tuple[Image.Image, np.ndarray, int, int]:
    """
    Process the image by flipping, removing background, converting to grayscale and binary,
    applying morphological closing, and inverting the binary image.

    Note:
        The input image should already be loaded. This function then handles the image processing.

    Args:
        input_image (Image.Image): The loaded PIL image.

    Returns:
        Tuple[Image.Image, np.ndarray, int, int]: A tuple containing the flipped input image,
        the processed binary numpy array, the image width, and the image height.
    """
    # Flip the image as in the original code.
    input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM).transpose(
        Image.FLIP_LEFT_RIGHT
    )
    output_image: Image.Image = remove(input_image)
    grayscale: Image.Image = output_image.convert("L")
    binary_threshold: int = 32
    binary: Image.Image = grayscale.point(lambda p: 255 if p > binary_threshold else 0)
    binary_np: np.ndarray = np.array(binary)
    kernel: np.ndarray = np.ones((10, 10), np.uint8)
    binary_np = cv2.morphologyEx(binary_np, cv2.MORPH_CLOSE, kernel)
    binary_np = 255 - binary_np
    height, width = binary_np.shape
    return input_image, binary_np, width, height


def extract_sagittal_profile(binary_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the sagittal profile from the binary image. For each row, finds the last black pixel.

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
        black_pixels = np.where(row == 0)[0]
        if len(black_pixels) > 0:
            sagittal_x.append(black_pixels[-1])
            sagittal_y.append(y)
    return np.array(sagittal_x), np.array(sagittal_y)


def calculate_crop_boundaries(
    sagittal_x: np.ndarray, sagittal_y: np.ndarray, width: int, height: int
) -> Tuple[int, int, int, int, float, float]:
    """
    Calculate the crop boundaries and adjusted x-axis limits for the cropped image using padding ratios.

    Args:
        sagittal_x (np.ndarray): Array of x-values from the sagittal profile.
        sagittal_y (np.ndarray): Array of y-values from the sagittal profile.
        width (int): Width of the original image.
        height (int): Height of the original image.

    Returns:
        Tuple[int, int, int, int, float, float]: The crop boundaries (x_min_crop, x_max_crop, y_min_crop, y_max_crop)
        and the adjusted x-axis limits (x_min_adj, x_max_adj) for plotting.
    """
    x_min: float = np.min(sagittal_x)
    x_max: float = np.max(sagittal_x)
    y_min: float = np.min(sagittal_y)
    y_max: float = np.max(sagittal_y)
    x_pad: float = X_PAD_RATIO * (x_max - x_min)
    y_pad: float = Y_PAD_RATIO * (y_max - y_min)
    x_min_crop: int = max(0, int(x_min - x_pad))
    x_max_crop: int = min(width, int(x_max + x_pad))
    y_min_crop: int = max(0, int(y_min - y_pad))
    y_max_crop: int = min(height, int(y_max + y_pad))
    # New x-axis limits for the cropped image.
    x_min_adj: float = -X_PAD_RATIO * (x_max_crop - x_min_crop)
    x_max_adj: float = (x_max_crop - x_min_crop) + X_PAD_RATIO * (
        x_max_crop - x_min_crop
    )
    return x_min_crop, x_max_crop, y_min_crop, y_max_crop, x_min_adj, x_max_adj


def crop_and_adjust(
    input_image: Image.Image,
    binary_np: np.ndarray,
    sagittal_x: np.ndarray,
    sagittal_y: np.ndarray,
    x_min_crop: int,
    x_max_crop: int,
    y_min_crop: int,
    y_max_crop: int,
) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop the input image and binary image, and adjust the sagittal coordinates relative to the crop.

    Args:
        input_image (Image.Image): The original PIL image.
        binary_np (np.ndarray): The binary numpy array representation of the image.
        sagittal_x (np.ndarray): Array of x-values from the sagittal profile.
        sagittal_y (np.ndarray): Array of y-values from the sagittal profile.
        x_min_crop (int): Minimum x-coordinate for cropping.
        x_max_crop (int): Maximum x-coordinate for cropping.
        y_min_crop (int): Minimum y-coordinate for cropping.
        y_max_crop (int): Maximum y-coordinate for cropping.

    Returns:
        Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]: The cropped PIL image, cropped binary array,
        and adjusted sagittal x and y arrays.
    """
    input_image_cropped: Image.Image = crop_image(
        input_image, x_min_crop, x_max_crop, y_min_crop, y_max_crop
    )
    binary_np_cropped: np.ndarray = crop_image(
        binary_np, x_min_crop, x_max_crop, y_min_crop, y_max_crop, is_array=True
    )
    sagittal_x_adj: np.ndarray = sagittal_x - x_min_crop
    sagittal_y_adj: np.ndarray = sagittal_y - y_min_crop
    return input_image_cropped, binary_np_cropped, sagittal_x_adj, sagittal_y_adj


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


def plot_original(
    ax: Axes,
    input_image_cropped: Image.Image,
    x_max_adj: float,
    x_min_adj: float,
    y_max_plot: int,
    y_min_plot: int,
) -> None:
    """
    Plot the original cropped image on the provided axes with adjusted axis limits.

    Args:
        ax (Axes): Matplotlib Axes object.
        input_image_cropped (Image.Image): The cropped PIL image.
        x_max_adj (float): Adjusted maximum x-limit.
        x_min_adj (float): Adjusted minimum x-limit.
        y_max_plot (int): Adjusted maximum y-limit.
        y_min_plot (int): Adjusted minimum y-limit.
    """
    ax.imshow(input_image_cropped)
    ax.set_xlim(x_max_adj, x_min_adj)
    ax.set_ylim(y_max_plot, y_min_plot)
    ax.set_aspect("equal")
    ax.set_title("Original Image (Cropped)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()


def plot_silhouette(
    ax: Axes,
    binary_np_cropped: np.ndarray,
    x_max_adj: float,
    x_min_adj: float,
    y_max_plot: int,
    y_min_plot: int,
) -> None:
    """
    Plot the binary silhouette image on the provided axes with adjusted axis limits.

    Args:
        ax (Axes): Matplotlib Axes object.
        binary_np_cropped (np.ndarray): The cropped binary image array.
        x_max_adj (float): Adjusted maximum x-limit.
        x_min_adj (float): Adjusted minimum x-limit.
        y_max_plot (int): Adjusted maximum y-limit.
        y_min_plot (int): Adjusted minimum y-limit.
    """
    ax.imshow(binary_np_cropped, cmap="gray")
    ax.set_xlim(x_max_adj, x_min_adj)
    ax.set_ylim(y_max_plot, y_min_plot)
    ax.set_aspect("equal")
    ax.set_title("Silhouette Image (Cropped)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()


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


def debug_lateral(input_image: Image.Image) -> None:
    """
    Analyze the side profile from a loaded PIL image by processing the image, extracting the sagittal profile,
    computing derivatives, and generating plots for the original image, silhouette, and profile with annotations.

    Args:
        input_image (Image.Image): The loaded PIL image to analyze.
    """
    # Process the image: flip, remove background, threshold, and clean up.
    processed_image, binary_np, width, height = process_image(input_image)

    # Extract the sagittal profile.
    sagittal_x, sagittal_y = extract_sagittal_profile(binary_np)

    # Trim the sagittal profile.
    sagittal_x, sagittal_y = trim_sagittal_profile(sagittal_x, sagittal_y)

    # Calculate crop boundaries and new axis limits.
    x_min_crop, x_max_crop, y_min_crop, y_max_crop, x_min_adj, x_max_adj = (
        calculate_crop_boundaries(sagittal_x, sagittal_y, width, height)
    )

    # Crop images and adjust the sagittal coordinates.
    input_image_cropped, binary_np_cropped, sagittal_x_adj, sagittal_y_adj = (
        crop_and_adjust(
            processed_image,
            binary_np,
            sagittal_x,
            sagittal_y,
            x_min_crop,
            x_max_crop,
            y_min_crop,
            y_max_crop,
        )
    )

    # Compute derivatives on the original trimmed sagittal_x.
    dx, ddx, dx_scaled, ddx_scaled = compute_derivatives(sagittal_x)

    # Create side-by-side subplots.
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 10))

    # Set global y-axis limits (inverted as in the original code).
    y_min_plot: int = y_max_crop  # Lower image coordinate (bottom)
    y_max_plot: int = y_min_crop  # Higher image coordinate (top)

    # Plot original image and silhouette using the correct axis order.
    plot_original(
        ax0, input_image_cropped, x_max_adj, x_min_adj, y_max_plot, y_min_plot
    )
    plot_silhouette(
        ax1, binary_np_cropped, x_max_adj, x_min_adj, y_max_plot, y_min_plot
    )

    # Plot the sagittal profile with derivatives and extrema.
    plot_sagittal_profile(ax2, sagittal_x_adj, sagittal_y_adj, dx_scaled, ddx_scaled)
    plot_and_print_local_extrema(ax2, sagittal_x_adj, sagittal_y_adj)
    if DEBUG:
        plot_quarter_lines(ax2, sagittal_y_adj)

    # Finalize the sagittal profile plot.
    ax2.set_xlim(x_max_adj, x_min_adj)
    ax2.set_ylim(y_max_plot, y_min_plot)
    ax2.set_aspect("equal")
    ax2.set_title("Sagittal Profile & Derivatives (Cropped)")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.legend()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    plt.tight_layout()
    plt.show()

    return util.convert_matplotlib_to_opencv(ax2)


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
    processed_image, binary_np, width, height = process_image(input_image)

    # Extract the sagittal profile.
    sagittal_x, sagittal_y = extract_sagittal_profile(binary_np)

    # Trim the sagittal profile.
    sagittal_x, sagittal_y = trim_sagittal_profile(sagittal_x, sagittal_y)

    # Calculate crop boundaries and new axis limits.
    x_min_crop, x_max_crop, y_min_crop, y_max_crop, x_min_adj, x_max_adj = (
        calculate_crop_boundaries(sagittal_x, sagittal_y, width, height)
    )

    # Crop images and adjust the sagittal coordinates.
    _, _, sagittal_x_adj, sagittal_y_adj = crop_and_adjust(
        processed_image,
        binary_np,
        sagittal_x,
        sagittal_y,
        x_min_crop,
        x_max_crop,
        y_min_crop,
        y_max_crop,
    )

    # Compute derivatives on the original trimmed sagittal_x.
    dx, ddx, dx_scaled, ddx_scaled = compute_derivatives(sagittal_x)

    # Create only the rightmost plot (sagittal profile).
    fig, ax2 = plt.subplots(figsize=(6, 10))

    # Set global y-axis limits (inverted as in the original code).
    y_min_plot = y_max_crop  # Lower image coordinate (bottom)
    y_max_plot = y_min_crop  # Higher image coordinate (top)

    # Plot the sagittal profile with derivatives and extrema.
    plot_sagittal_profile(ax2, sagittal_x_adj, sagittal_y_adj, dx_scaled, ddx_scaled)
    plot_and_print_local_extrema(ax2, sagittal_x_adj, sagittal_y_adj)
    if DEBUG:
        plot_quarter_lines(ax2, sagittal_y_adj)

    # Finalize the sagittal profile plot.
    ax2.set_xlim(x_max_adj, x_min_adj)
    ax2.set_ylim(0, y_min_plot)
    ax2.set_aspect("equal")

    # Hide axes, ticks, labels, and spines
    ax2.axis("off")
    ax2.spines["top"].set_color("none")
    ax2.spines["bottom"].set_color("none")
    ax2.spines["left"].set_color("none")
    ax2.spines["right"].set_color("none")

    # Ensure the legend is inside the plot
    legend = ax2.legend(frameon=True, loc="upper left", bbox_to_anchor=(0, 1))
    legend.get_frame().set_alpha(0.8)  # Slight transparency for better visibility

    plt.tight_layout(pad=0)  # Remove extra margins

    # Convert the plot to OpenCV format
    return util.convert_matplotlib_to_opencv(ax2)
