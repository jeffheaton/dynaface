import numpy as np

import cv2
import numpy as np
import math
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
