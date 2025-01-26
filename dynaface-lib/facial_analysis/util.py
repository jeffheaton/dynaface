import math

import cv2
import numpy as np
from facial_analysis import facial


# Shoelace, https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def safe_clip(cv2_image, x, y, width, height, background):
    """
    Clips a region from an OpenCV image, adjusting for boundaries and filling missing areas with a specified color.

    If the specified region extends beyond the image boundaries, the function adjusts the coordinates and size
    to fit within the image. If the region is partially or completely outside the image, it fills the missing
    areas with the specified background color.

    Args:
    cv2_image (numpy.ndarray): The source image in OpenCV format.
    x (int): The x-coordinate of the top-left corner of the clipping region.
    y (int): The y-coordinate of the top-left corner of the clipping region.
    width (int): The width of the clipping region.
    height (int): The height of the clipping region.
    background (tuple): A tuple (R, G, B) specifying the fill color for missing areas.

    Returns:
    tuple: A tuple containing the clipped image, x offset, and y offset.
           The x and y offsets indicate how much the origin of the clipped image has shifted
           relative to the original image.
    """

    # Image dimensions
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


def scale_crop_points(lst, crop_x, crop_y, scale):
    lst2 = []
    for pt in lst:
        lst2.append((int(((pt[0] * scale) - crop_x)), int((pt[1] * scale) - crop_y)))
    return lst2


def rotate_crop_points(points, center, angle_degrees):
    """
    Rotate the points around the center by the specified angle.

    Parameters:
    points (list): List of (x, y) coordinates to rotate.
    center (tuple): The center of rotation (x, y).
    angle_degrees (float): The rotation angle in degrees.

    Returns:
    list: List of rotated (x, y) coordinates.
    """
    # Convert angle from degrees to radians and adjust the sign
    angle_radians = -np.deg2rad(
        angle_degrees
    )  # Negate the angle for correct rotation direction

    # Calculate the rotation matrix
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Rotate each point
    rotated_points = []
    for point in points:
        vector = np.array(point) - np.array(center)
        rotated_vector = np.dot(rotation_matrix, vector)
        rotated_point = rotated_vector + np.array(center)
        rotated_points.append(rotated_point)

    return rotated_points


def calc_pd(pupils):
    left_pupil, right_pupil = pupils
    left_pupil = np.array(left_pupil)
    right_pupil = np.array(right_pupil)

    # Calculate Euclidean distance between the two pupils
    pupillary_distance = np.linalg.norm(left_pupil - right_pupil)

    # Convert the distance from pixels to millimeters
    pix2mm = facial.AnalyzeFace.pd / pupillary_distance

    return pupillary_distance, pix2mm


def get_pupils(landmarks):
    return landmarks[facial.LM_LEFT_PUPIL], landmarks[facial.LM_RIGHT_PUPIL]


def calculate_face_rotation(pupil_coords):
    """
    Calculate the rotation angle of a face based on the coordinates of the pupils.

    Parameters:
    pupil_coords (tuple): A tuple containing two tuples, each with the (x, y) coordinates of the pupils.
                          Example: ((640, 481), (380, 480))

    Returns:
    float: The rotation angle of the face in radians.
    """
    (x1, y1), (x2, y2) = pupil_coords
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle = math.atan2(delta_y, delta_x)
    return angle


def calculate_average_rgb(image):
    """
    Calculate the average RGB value of an image.

    Parameters:
    image (numpy.ndarray): The input image in BGR format.

    Returns:
    tuple: A tuple containing the average RGB values.
    """
    average_color_per_row = np.mean(image, axis=0)
    average_color = np.mean(average_color_per_row, axis=0)
    return tuple(map(int, average_color))


def straighten(image, angle_radians):
    """
    Rotate the image to align the pupils horizontally, crop to original dimensions, and fill dead-space with the average RGB color.

    Parameters:
    image (numpy.ndarray): The input image in BGR format.
    angle_radians (tuple): The amount that the face is rotated.

    Returns:
    numpy.ndarray: The rotated and cropped image.
    """
    # Calculate the rotation angle
    angle_degrees = angle_radians * (180 / math.pi)

    # Adjust the angle to avoid upside down rotation
    if angle_degrees > 45:
        angle_degrees -= 180
    elif angle_degrees < -45:
        angle_degrees += 180

    # Get image dimensions
    h, w = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # Rotate the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # Calculate the average RGB value
    avg_rgb = calculate_average_rgb(image)

    # Create a new image with the average RGB color
    result_image = np.full_like(rotated_image, avg_rgb, dtype=np.uint8)

    # Calculate the size of the new image
    result_center = (result_image.shape[1] // 2, result_image.shape[0] // 2)

    # Calculate the top-left corner of the region to paste the rotated image
    top_left_x = result_center[0] - center[0]
    top_left_y = result_center[1] - center[1]

    # Paste the rotated image onto the result image
    result_image[top_left_y : top_left_y + h, top_left_x : top_left_x + w] = (
        rotated_image
    )

    # Crop the result image to the original dimensions
    cropped_image = result_image[:h, :w]

    return cropped_image


def symmetry_ratio(a, b):
    """
    Calculate the symmetry ratio between two numbers.

    Parameters:
    a (float): The size of the first object.
    b (float): The size of the second object.

    Returns:
    float: The symmetry ratio, a value between 0 and 1.
    """
    if a == 0 and b == 0:
        return 1.0  # If both sizes are zero, they are perfectly symmetric.
    return min(a, b) / max(a, b)


def line_intersection(line, contour, tol=1e-7):
    """Return a list of (intersection_point, edge_index) for all edges in 'contour'
    that intersect with 'line'. Deduplicate near-identical points."""
    intersections = []
    for i in range(len(contour)):
        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]
        intersection = compute_intersection(line, (p1, p2))
        if intersection is not None:
            intersections.append((intersection, i))

    # Deduplicate intersection points that are effectively the same
    unique_intersections = []
    for pt, idx in intersections:
        if not any(
            np.linalg.norm(np.array(pt) - np.array(u_pt)) < tol
            for (u_pt, _) in unique_intersections
        ):
            unique_intersections.append((pt, idx))

    return unique_intersections


def compute_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines do not intersect

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Check if the intersection point is within the bounds of the polygon edge segment
    if min(line2[0][0], line2[1][0]) <= x <= max(line2[0][0], line2[1][0]) and min(
        line2[0][1], line2[1][1]
    ) <= y <= max(line2[0][1], line2[1][1]):
        return x, y
    return None


def split_polygon(polygon, line):
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


def bisecting_line_coordinates(img_size, pupils):
    # Unpack pupil coordinates
    (x1, y1), (x2, y2) = pupils

    # Calculate midpoint between the pupils
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Calculate the angle of the line
    if x1 == x2:
        # Vertical line case (no tilt)
        angle = np.pi / 2
    else:
        angle = np.arctan2((y2 - y1), (x2 - x1))

    # Determine the slope of the perpendicular bisecting line
    perp_slope = np.tan(angle + np.pi / 2)

    # Function to get y coordinate given x
    def get_y(x, mid_x, mid_y, slope):
        return slope * (x - mid_x) + mid_y

    # Function to get x coordinate given y
    def get_x(y, mid_x, mid_y, slope):
        return (y - mid_y) / slope + mid_x

    # Calculate intersection points with the edges of the image
    x0, x1 = 0, img_size
    y0, y1 = get_y(x0, mid_x, mid_y, perp_slope), get_y(x1, mid_x, mid_y, perp_slope)

    if y0 < 0:
        y0 = 0
        x0 = get_x(y0, mid_x, mid_y, perp_slope)
    elif y0 > img_size:
        y0 = img_size
        x0 = get_x(y0, mid_x, mid_y, perp_slope)

    if y1 < 0:
        y1 = 0
        x1 = get_x(y1, mid_x, mid_y, perp_slope)
    elif y1 > img_size:
        y1 = img_size
        x1 = get_x(y1, mid_x, mid_y, perp_slope)

    return (int(x0), int(y0)), (int(x1), int(y1))


def line_to_edge(img_size, start_point, angle):
    # Unpack start point
    x0, y0 = start_point

    # Calculate the slope of the line
    slope = np.tan(angle)

    # Determine the possible intersections with the image boundaries
    possible_endpoints = []

    # Intersection with the right edge (x = img_size)
    if slope != 0:
        x_right = img_size
        y_right = slope * (x_right - x0) + y0
        if 0 <= y_right <= img_size:
            possible_endpoints.append((x_right, y_right))

    # Intersection with the left edge (x = 0)
    if slope != 0:
        x_left = 0
        y_left = slope * (x_left - x0) + y0
        if 0 <= y_left <= img_size:
            possible_endpoints.append((x_left, y_left))

    # Intersection with the top edge (y = 0)
    if slope != np.inf:
        y_top = 0
        x_top = (y_top - y0) / slope + x0
        if 0 <= x_top <= img_size:
            possible_endpoints.append((x_top, y_top))

    # Intersection with the bottom edge (y = img_size)
    if slope != np.inf:
        y_bottom = img_size
        x_bottom = (y_bottom - y0) / slope + x0
        if 0 <= x_bottom <= img_size:
            possible_endpoints.append((x_bottom, y_bottom))

    if not possible_endpoints:
        raise ValueError("No valid endpoint found on the image boundaries.")

    # Choose the first valid endpoint (the one closest to the starting point)
    endpoint = possible_endpoints[0]

    return (int(endpoint[0]), int(endpoint[1]))


def normalize_angle(angle):
    # Use the modulus operator to normalize the angle
    return angle % (2 * math.pi)
