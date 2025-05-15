import math
from typing import Any, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

Number = Union[int, float]

# Should request.get verify SSL certificates. The strict setting of True can cause this to
# fail when run from corporate networks that intercept HTTPS traffic. Setting to True will
# generally work on home networks. For a controlled server environment you will likely
# predownload any needed files and block network traffic, so this value may not matter.
VERIFY_CERTS: bool = True


def PolyArea(x: NDArray[Any], y: NDArray[Any]) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.

    Parameters:
        x (NDArray[Any]): Array of x-coordinates.
        y (NDArray[Any]): Array of y-coordinates.

    Returns:
        float: The computed area of the polygon.
    """
    # Explicitly cast the result to float to avoid mypy returning Any.
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def safe_clip(
    cv2_image: NDArray[Any],
    x: int,
    y: int,
    width: int,
    height: int,
    background: Tuple[int, int, int],
) -> Tuple[NDArray[Any], int, int]:
    """
    Clips a region from an OpenCV image, adjusting for boundaries and filling missing areas with a specified color.
    """
    img_height, img_width = cv2_image.shape[:2]

    # Adjust start and end points to be within the image boundaries
    x_start = max(x, 0)
    y_start = max(y, 0)
    x_end = min(x + width, img_width)
    y_end = min(y + height, img_height)

    # Calculate the size of the region that will be clipped from the original image
    clipped_width = x_end - x_start
    clipped_height = y_end - y_start

    # Create a new image filled with the background color
    new_image = np.full((height, width, 3), background, dtype=cv2_image.dtype)

    # Calculate where to place the clipped region in the new image
    new_x_start = max(0, -x)
    new_y_start = max(0, -y)

    # Clip the region from the original image and place it in the new image
    if clipped_width > 0 and clipped_height > 0:
        clipped_region = cv2_image[y_start:y_end, x_start:x_end]
        new_image[
            new_y_start : new_y_start + clipped_height,
            new_x_start : new_x_start + clipped_width,
        ] = clipped_region

    return new_image, new_x_start, new_y_start


def scale_crop_points(
    lst: List[Tuple[int, int]], crop_x: int, crop_y: int, scale: float
) -> List[Tuple[int, int]]:
    """
    Scales and crops a list of points.
    """
    lst2: List[Tuple[int, int]] = []
    for pt in lst:
        lst2.append((int((pt[0] * scale) - crop_x), int((pt[1] * scale) - crop_y)))
    return lst2


def rotate_crop_points(
    points: List[Tuple[int, int]], center: Tuple[int, int], angle_degrees: float
) -> List[Tuple[int, int]]:
    """
    Rotate the points around the center by the specified angle.
    """
    angle_radians = -np.deg2rad(angle_degrees)  # Negate for correct rotation direction
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_points: List[Tuple[int, int]] = []
    for point in points:
        vector = np.array(point) - np.array(center)
        rotated_vector = np.dot(rotation_matrix, vector)
        rotated_point = rotated_vector + np.array(center)
        rotated_points.append(
            (int(round(rotated_point[0])), int(round(rotated_point[1])))
        )

    return rotated_points


def calculate_face_rotation(
    pupil_coords: Tuple[Tuple[int, int], Tuple[int, int]],
) -> float:
    """
    Calculate the rotation angle of a face based on the coordinates of the pupils.
    """
    (x1, y1), (x2, y2) = pupil_coords
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle = math.atan2(delta_y, delta_x)
    return angle


def calculate_average_rgb(image: NDArray[Any]) -> Tuple[int, int, int]:
    """
    Calculate the average RGB value of an image.
    """
    average_color_per_row = np.mean(image, axis=0)
    average_color = np.mean(average_color_per_row, axis=0)
    # Unpack exactly three channels to match the declared return type.
    r, g, b = average_color
    return int(r), int(g), int(b)


def straighten(image: NDArray[Any], angle_radians: float) -> NDArray[Any]:
    """
    Rotate the image to align the pupils horizontally, crop to original dimensions,
    and fill dead-space with the average RGB color.
    """
    angle_degrees = angle_radians * (180 / math.pi)

    # Adjust the angle to avoid upside down rotation
    if angle_degrees > 45:
        angle_degrees -= 180
    elif angle_degrees < -45:
        angle_degrees += 180

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    avg_rgb = calculate_average_rgb(image)
    result_image = np.full_like(rotated_image, avg_rgb, dtype=np.uint8)

    result_center = (result_image.shape[1] // 2, result_image.shape[0] // 2)
    top_left_x = result_center[0] - center[0]
    top_left_y = result_center[1] - center[1]
    result_image[top_left_y : top_left_y + h, top_left_x : top_left_x + w] = (
        rotated_image
    )

    cropped_image = result_image[:h, :w]
    return cropped_image


def symmetry_ratio(a: float, b: float) -> float:
    """
    Calculate the symmetry ratio between two numbers.
    """
    if a == 0 and b == 0:
        return 1.0  # Perfect symmetry if both are zero.
    return min(a, b) / max(a, b)


def line_intersection(
    line: Tuple[Tuple[int, int], Tuple[int, int]],
    contour: NDArray[Any],
    tol: float = 1e-7,
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Return a list of (intersection_point, edge_index) for all edges in 'contour'
    that intersect with 'line'. Deduplicate near-identical points.
    """
    intersections: List[Tuple[Tuple[int, int], int]] = []
    for i in range(len(contour)):
        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]
        intersection = compute_intersection(line, (p1, p2))
        if intersection is not None:
            intersections.append(((int(intersection[0]), int(intersection[1])), i))

    unique_intersections: List[Tuple[Tuple[int, int], int]] = []
    for pt, idx in intersections:
        if not any(
            np.linalg.norm(np.array(pt) - np.array(u_pt)) < tol
            for (u_pt, _) in unique_intersections
        ):
            unique_intersections.append(((int(pt[0]), int(pt[1])), idx))

    return unique_intersections


def compute_intersection(
    line1: Tuple[Tuple[int, int], Tuple[int, int]],
    line2: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """
    Compute the intersection point of two lines if they intersect within the bounds of the second line segment.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines do not intersect

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Check if the intersection point is within the bounds of the second line segment
    if min(line2[0][0], line2[1][0]) <= x <= max(line2[0][0], line2[1][0]) and min(
        line2[0][1], line2[1][1]
    ) <= y <= max(line2[0][1], line2[1][1]):
        return int(x), int(y)
    return None


def split_polygon(
    polygon: NDArray[Any], line: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Split a polygon into two parts using a line that bisects it.
    """
    intersections = line_intersection(line, polygon)
    if len(intersections) != 2:
        raise ValueError(
            f"The line does not properly bisect the polygon. line={line}, polygon={polygon}"
        )

    intersections = sorted(intersections, key=lambda x: x[1])
    intersection1, idx1 = intersections[0]
    intersection2, idx2 = intersections[1]

    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
        intersection1, intersection2 = intersection2, intersection1

    poly1 = polygon[: idx1 + 1].tolist()
    poly1.append(intersection1)
    poly1.append(intersection2)
    poly1.extend(polygon[idx2 + 1 :])

    poly2 = polygon[idx1 + 1 : idx2 + 1].tolist()
    poly2.append(intersection2)
    poly2.append(intersection1)

    return np.array(poly1), np.array(poly2)


def bisecting_line_coordinates(
    img_size: int, pupils: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Compute the coordinates of the bisecting line of the eyes in an image.
    """
    (x1, y1), (x2, y2) = pupils

    # Calculate midpoint between the pupils
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Calculate the angle of the line
    if x1 == x2:
        angle = np.pi / 2
    else:
        angle = np.arctan2((y2 - y1), (x2 - x1))

    # Determine the slope of the perpendicular bisecting line
    perp_slope = np.tan(angle + np.pi / 2)

    def get_y(x: float, mid_x: float, mid_y: float, slope: float) -> float:
        return slope * (x - mid_x) + mid_y

    def get_x(y: float, mid_x: float, mid_y: float, slope: float) -> float:
        return (y - mid_y) / slope + mid_x

    # Initialize as floats so that assignments from get_x/get_y are valid.
    x0, x1_img = 0.0, float(img_size)
    y0, y1_img = get_y(x0, mid_x, mid_y, perp_slope), get_y(
        x1_img, mid_x, mid_y, perp_slope
    )

    if y0 < 0:
        y0 = 0.0
        x0 = get_x(y0, mid_x, mid_y, perp_slope)
    elif y0 > img_size:
        y0 = float(img_size)
        x0 = get_x(y0, mid_x, mid_y, perp_slope)

    if y1_img < 0:
        y1_img = 0.0
        x1_img = get_x(y1_img, mid_x, mid_y, perp_slope)
    elif y1_img > img_size:
        y1_img = float(img_size)
        x1_img = get_x(y1_img, mid_x, mid_y, perp_slope)

    return (int(x0), int(y0)), (int(x1_img), int(y1_img))


def line_to_edge(
    img_size: int, start_point: Tuple[int, int], angle: float
) -> Optional[Tuple[int, int]]:
    """
    Compute the intersection point where a line, originating from a given start point and extending at a specified angle,
    meets the boundary of a square image.
    """
    x0, y0 = start_point
    slope = math.tan(angle)
    possible_endpoints: List[Tuple[int, int]] = []

    def safe_append(x: float, y: float) -> None:
        if not math.isfinite(x) or not math.isfinite(y):
            return
        if 0 <= x <= img_size and 0 <= y <= img_size:
            possible_endpoints.append((int(x), int(y)))

    # Right edge (x = img_size)
    if slope != 0:
        x_right = img_size
        y_right = slope * (x_right - x0) + y0
        safe_append(x_right, y_right)

    # Left edge (x = 0)
    if slope != 0:
        x_left = 0
        y_left = slope * (x_left - x0) + y0
        safe_append(x_left, y_left)

    # Top edge (y = 0)
    if slope != 0:
        y_top = 0
        x_top = (y_top - y0) / slope + x0
        safe_append(x_top, y_top)

    # Bottom edge (y = img_size)
    if slope != 0:
        y_bottom = img_size
        x_bottom = (y_bottom - y0) / slope + x0
        safe_append(x_bottom, y_bottom)

    if not possible_endpoints:
        return None

    return possible_endpoints[0]


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [0, 2*pi).
    """
    return angle % (2 * math.pi)


def cv2_to_pil(cv_image: NDArray[Any]) -> Image.Image:
    """
    Convert an OpenCV image (NumPy array) to a PIL image.
    """
    if len(cv_image.shape) == 2:  # Grayscale image
        return Image.fromarray(cv_image)
    elif len(cv_image.shape) == 3 and cv_image.shape[2] == 3:  # Color image
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    else:
        raise ValueError("Unsupported image format")


def convert_matplotlib_to_opencv(ax: Axes) -> NDArray[Any]:
    """
    Convert a Matplotlib axis to an OpenCV image without extra whitespace.
    """
    fig = ax.figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
    # Cast fig to a Figure to satisfy the type for FigureCanvas.
    canvas = FigureCanvas(cast(Figure, fig))
    canvas.draw()
    image = np.array(canvas.renderer.buffer_rgba())
    bbox_obj = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Convert bbox extents to a list of ints.
    bbox = [int(coord * fig.dpi) for coord in bbox_obj.extents]
    # Unpack bbox into integer coordinates for slicing.
    left, bottom, right, top = bbox[0], bbox[1], bbox[2], bbox[3]
    image = image[bottom:top, left:right, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def trim_sides(image: NDArray[Any]) -> NDArray[Any]:
    """
    Trims space from the left and right sides of the image based on the background color sampled
    from the upper-left pixel, while maintaining the original height.
    """
    background_color = image[0, 0]

    if len(image.shape) == 3:
        mask = np.all(image == background_color, axis=-1).astype(np.uint8)
    else:
        mask = (image == background_color).astype(np.uint8)

    col_sum = mask.sum(axis=0)
    left_trim = int(np.argmax(col_sum < image.shape[0]))
    right_trim = len(col_sum) - int(np.argmax(col_sum[::-1] < image.shape[0]))

    cropped = image[:, left_trim:right_trim]
    return cropped


def is_zero_tuple(p: Any, tol: float = 1e-9) -> bool:
    if not isinstance(p, tuple):
        return False

    p_tuple = cast(Tuple[Any, ...], p)

    if not all(isinstance(x, (int, float)) for x in p_tuple):
        return False

    return math.isclose(sum(p_tuple), 0.0, abs_tol=tol)


def to_degrees(r: float) -> float:
    """
    Convert an angle from radians to degrees and adjust it to be in a more intuitive range.

    Args:
        r (float): Angle in radians.

    Returns:
        float: Adjusted angle in degrees.
    """
    tilt = r * (180 / math.pi)

    if tilt > 90:
        tilt -= 180
    elif tilt < -90:
        tilt += 180

    return tilt


def correct_distance_2d_for_yaw(measured_distance: float, yaw_degrees: float) -> float:
    """
    Corrects a 2D facial landmark distance for yaw rotation.

    Args:
        measured_distance (float): The 2D distance between two landmarks.
        yaw_degrees (float): The yaw angle of the head in degrees (0 = frontal).

    Returns:
        float: The corrected distance.
    """
    # Convert yaw from degrees to radians
    yaw_radians = math.radians(yaw_degrees)

    # Avoid division by zero or extreme correction at 90 degrees
    correction_factor = max(math.cos(yaw_radians), 1e-6)

    corrected_distance = measured_distance / correction_factor
    return corrected_distance


def closest_to_zero(values):
    return min(values, key=abs)


def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        pt1 (Tuple[float, float]): The first point (x1, y1).
        pt2 (Tuple[float, float]): The second point (x2, y2).

    Returns:
        float: The Euclidean distance.
    """
    return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))
