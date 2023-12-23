import numpy as np

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


def crop_stylegan(img, pupils, landmarks):
    print(img.shape)
    width, height = img.shape[1], img.shape[0]
    print("**", width, height)

    if pupils:
        left_eye, right_eye = pupils
    else:
        left_eye, right_eye = get_pupils(landmarks=landmarks)

    d = abs(right_eye[0] - left_eye[0])
    ar = width / height
    new_width = int(width * (facial.STYLEGAN_PUPIL_DIST / d))
    new_height = int(new_width / ar)
    scale = new_width / width
    img = cv2.resize(img, (new_width, new_height))

    crop_x = int((landmarks[96][0] * scale) - facial.STYLEGAN_RIGHT_PUPIL[0])
    crop_y = int((landmarks[96][1] * scale) - facial.STYLEGAN_RIGHT_PUPIL[1])
    img2, _, _ = safe_clip(
        img,
        crop_x,
        crop_y,
        facial.STYLEGAN_WIDTH,
        facial.STYLEGAN_WIDTH,
        facial.FILL_COLOR,
    )
    landmarks2 = scale_crop_points(landmarks, crop_x, crop_y, scale)
    return img2, landmarks2


def calc_pd(landmarks):
    pupillary_distance = abs(
        landmarks[facial.LM_LEFT_PUPIL][0] - landmarks[facial.LM_RIGHT_PUPIL][0]
    )
    pix2mm = facial.AnalyzeFace.pd / pupillary_distance
    return pupillary_distance, pix2mm


def get_pupils(landmarks):
    return landmarks[facial.LM_LEFT_PUPIL], landmarks[facial.LM_RIGHT_PUPIL]
