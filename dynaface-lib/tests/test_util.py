import unittest
import math
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from facial_analysis.util import (
    PolyArea,
    safe_clip,
    scale_crop_points,
    rotate_crop_points,
    calc_pd,
    get_pupils,
    calculate_face_rotation,
    calculate_average_rgb,
    straighten,
    symmetry_ratio,
    line_intersection,
    compute_intersection,
    split_polygon,
    bisecting_line_coordinates,
    line_to_edge,
    normalize_angle,
)

# If they are all in the current namespace, you can import them directly:
from facial_analysis import facial


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def safe_clip(cv2_image, x, y, width, height, background):
    img_height, img_width = cv2_image.shape[:2]

    x_start = max(x, 0)
    y_start = max(y, 0)
    x_end = min(x + width, img_width)
    y_end = min(y + height, img_height)

    clipped_width = x_end - x_start
    clipped_height = y_end - y_start

    new_image = np.full((height, width, 3), background, dtype=cv2_image.dtype)

    new_x_start = max(0, -x)
    new_y_start = max(0, -y)

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
    angle_radians = -np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_points = []
    for point in points:
        vector = np.array(point) - np.array(center)
        rotated_vector = np.dot(rotation_matrix, vector)
        rotated_point = rotated_vector + np.array(center)
        rotated_points.append(rotated_point)

    return rotated_points


def calc_pd(pupils):
    # Pupils is a tuple like ((x1, y1), (x2, y2))
    left_pupil, right_pupil = pupils
    left_pupil = np.array(left_pupil)
    right_pupil = np.array(right_pupil)

    # Euclidean distance
    pupillary_distance = np.linalg.norm(left_pupil - right_pupil)

    # Convert the distance from pixels to millimeters
    pix2mm = facial.AnalyzeFace.pd / pupillary_distance

    return pupillary_distance, pix2mm


def get_pupils(landmarks):
    # Just returns the pupil coords from a landmark array using some fixed indices
    return landmarks[facial.LM_LEFT_PUPIL], landmarks[facial.LM_RIGHT_PUPIL]


def calculate_face_rotation(pupil_coords):
    (x1, y1), (x2, y2) = pupil_coords
    delta_y = y2 - y1
    delta_x = x2 - x1
    angle = math.atan2(delta_y, delta_x)
    return angle


def calculate_average_rgb(image):
    average_color_per_row = np.mean(image, axis=0)
    average_color = np.mean(average_color_per_row, axis=0)
    return tuple(map(int, average_color))


def straighten(image, angle_radians):
    angle_degrees = angle_radians * (180 / math.pi)

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


def symmetry_ratio(a, b):
    if a == 0 and b == 0:
        return 1.0
    return min(a, b) / max(a, b)


def line_intersection(line, contour):
    intersections = []
    for i in range(len(contour)):
        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]
        intersection = compute_intersection(line, (p1, p2))
        if intersection is not None:
            intersections.append((intersection, i))
    return intersections


def compute_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Check if intersection is on the segment (line2)
    if min(line2[0][0], line2[1][0]) <= x <= max(line2[0][0], line2[1][0]) and min(
        line2[0][1], line2[1][1]
    ) <= y <= max(line2[0][1], line2[1][1]):
        return (x, y)
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
    (x1, y1), (x2, y2) = pupils
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    if x1 == x2:
        angle = np.pi / 2
    else:
        angle = np.arctan2((y2 - y1), (x2 - x1))

    perp_slope = np.tan(angle + np.pi / 2)

    def get_y(x, mid_x, mid_y, slope):
        return slope * (x - mid_x) + mid_y

    def get_x(y, mid_x, mid_y, slope):
        return (y - mid_y) / slope + mid_x

    x0, x1_ = 0, img_size
    y0 = get_y(x0, mid_x, mid_y, perp_slope)
    y1_ = get_y(x1_, mid_x, mid_y, perp_slope)

    if y0 < 0:
        y0 = 0
        x0 = get_x(y0, mid_x, mid_y, perp_slope)
    elif y0 > img_size:
        y0 = img_size
        x0 = get_x(y0, mid_x, mid_y, perp_slope)

    if y1_ < 0:
        y1_ = 0
        x1_ = get_x(y1_, mid_x, mid_y, perp_slope)
    elif y1_ > img_size:
        y1_ = img_size
        x1_ = get_x(y1_, mid_x, mid_y, perp_slope)

    return (int(x0), int(y0)), (int(x1_), int(y1_))


def line_to_edge(img_size, start_point, angle):
    x0, y0 = start_point
    slope = np.tan(angle)

    possible_endpoints = []

    # Right edge
    if slope != 0:
        x_right = img_size
        y_right = slope * (x_right - x0) + y0
        if 0 <= y_right <= img_size:
            possible_endpoints.append((x_right, y_right))

    # Left edge
    if slope != 0:
        x_left = 0
        y_left = slope * (x_left - x0) + y0
        if 0 <= y_left <= img_size:
            possible_endpoints.append((x_left, y_left))

    # Top edge
    if slope != np.inf:
        y_top = 0
        x_top = (y_top - y0) / slope + x0 if slope != 0 else float("inf")
        if 0 <= x_top <= img_size:
            possible_endpoints.append((x_top, y_top))

    # Bottom edge
    if slope != np.inf:
        y_bottom = img_size
        x_bottom = (y_bottom - y0) / slope + x0 if slope != 0 else float("inf")
        if 0 <= x_bottom <= img_size:
            possible_endpoints.append((x_bottom, y_bottom))

    if not possible_endpoints:
        raise ValueError("No valid endpoint found on the image boundaries.")

    # Choose the endpoint closest to the start
    distances = [
        (np.linalg.norm(np.array(pt) - np.array(start_point)), pt)
        for pt in possible_endpoints
    ]
    distances.sort(key=lambda x: x[0])
    endpoint = distances[0][1]

    return (int(endpoint[0]), int(endpoint[1]))


def normalize_angle(angle):
    return angle % (2 * math.pi)


class TestFunctions(unittest.TestCase):

    def test_PolyArea_triangle(self):
        # A simple right triangle with points (0,0), (4,0), (0,3)
        x_coords = np.array([0, 4, 0])
        y_coords = np.array([0, 0, 3])
        area = PolyArea(x_coords, y_coords)
        self.assertAlmostEqual(area, 6.0, places=5)

    def test_safe_clip_no_clip(self):
        # Create a 10x10 black image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        clipped, offset_x, offset_y = safe_clip(img, 2, 2, 5, 5, (255, 255, 255))
        self.assertEqual(clipped.shape, (5, 5, 3))
        # Since (2,2) to (7,7) is fully within 10x10, we expect no background fill
        self.assertTrue(np.all(clipped == 0))
        self.assertEqual(offset_x, 0)
        self.assertEqual(offset_y, 0)

    def test_safe_clip_out_of_bounds(self):
        # Create a 10x10 black image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        # Request a region partially out of bounds
        clipped, offset_x, offset_y = safe_clip(img, -2, -2, 5, 5, (255, 255, 255))
        self.assertEqual(clipped.shape, (5, 5, 3))
        # The top-left 3x3 portion is background, the bottom-right 2x2 portion is the original
        # Check corners
        # top-left corner should be white
        self.assertTrue(np.all(clipped[0, 0] == [255, 255, 255]))
        # bottom-right corner should be black
        self.assertTrue(np.all(clipped[4, 4] == [0, 0, 0]))
        self.assertEqual(offset_x, 2)  # we shifted by 2 to the right
        self.assertEqual(offset_y, 2)  # we shifted by 2 down

    def test_scale_crop_points(self):
        points = [(10, 10), (20, 20)]
        # scale = 1, crop_x=10, crop_y=10
        scaled = scale_crop_points(points, 10, 10, 1.0)
        self.assertEqual(scaled, [(0, 0), (10, 10)])

        # scale = 2
        scaled2 = scale_crop_points(points, 10, 10, 2.0)
        self.assertEqual(scaled2, [(10, 10), (30, 30)])

    def test_rotate_crop_points(self):
        points = [(0, 0), (10, 0)]
        center = (0, 0)
        rotated = rotate_crop_points(points, center, 90)
        # Rotating (10,0) around (0,0) by +90 deg → (0,10)
        self.assertAlmostEqual(rotated[0][0], 0, places=5)
        self.assertAlmostEqual(rotated[0][1], 0, places=5)
        self.assertAlmostEqual(rotated[1][0], 0, places=5)
        self.assertAlmostEqual(rotated[1][1], 10, places=5)

    @patch.object(facial.AnalyzeFace, "pd", 60.0)
    def test_calc_pd(self):
        # Pupils 10 px apart
        pupils = ((0, 0), (10, 0))
        distance, pix2mm = calc_pd(pupils)
        self.assertEqual(distance, 10.0)
        # pix2mm = 60 / 10 = 6 mm per pixel
        self.assertEqual(pix2mm, 6.0)

    def test_get_pupils(self):
        # Mock some landmarks
        mock_landmarks = {
            facial.LM_LEFT_PUPIL: (100, 100),
            facial.LM_RIGHT_PUPIL: (200, 100),
        }
        left_pupil, right_pupil = get_pupils(mock_landmarks)
        self.assertEqual(left_pupil, (100, 100))
        self.assertEqual(right_pupil, (200, 100))

    def test_calculate_face_rotation(self):
        # Pupils horizontally aligned should yield angle = 0
        angle = calculate_face_rotation(((100, 100), (200, 100)))
        self.assertAlmostEqual(angle, 0.0, places=5)

        # Pupils vertical: (0,0) & (0,10) → angle = pi/2 or -pi/2 (depending on order)
        angle = calculate_face_rotation(((0, 0), (0, 10)))
        # In that case, delta_x = 0, delta_y = 10, atan2(10, 0) = pi/2
        self.assertAlmostEqual(angle, math.pi / 2, places=5)

    def test_calculate_average_rgb(self):
        # 2x2 image with color:
        # [ [ (0,0,255), (0,0,255) ],
        #   [ (255,0,0), (255,0,0) ] ]
        # in BGR format
        img = np.array(
            [[[255, 0, 0], [255, 0, 0]], [[0, 0, 255], [0, 0, 255]]], dtype=np.uint8
        )
        # That means top row is pure blue, bottom row is pure red in BGR space
        # average B = (255+255+0+0)/4 = 128
        # average G = 0
        # average R = (0+0+255+255)/4 = 128
        avg = calculate_average_rgb(img)
        self.assertEqual(avg, (128, 0, 128))

    def test_straighten(self):
        # Create a 2x2 image, rotate a small angle, then straighten it
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        # Let's draw something easy to see
        img[0, 0] = [255, 0, 0]  # top-left pixel is blue in BGR

        # We will rotate by a small angle (e.g., 30 degrees → ~0.523 rad)
        angle = math.radians(30)
        rotated = straighten(img, angle)
        # Because the image is so small, we won't do an exact pixel test here,
        # but let's check shape remains the same
        self.assertEqual(rotated.shape, (2, 2, 3))

    def test_symmetry_ratio(self):
        self.assertEqual(symmetry_ratio(0, 0), 1.0)
        self.assertEqual(symmetry_ratio(5, 5), 1.0)
        self.assertEqual(symmetry_ratio(2, 4), 0.5)
        self.assertEqual(symmetry_ratio(10, 2), 0.2)

    def test_line_intersection(self):
        # Square contour
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        line_ = ((5, -1), (5, 11))  # vertical line x=5
        intersects = line_intersection(line_, square)
        self.assertEqual(len(intersects), 2)  # Should intersect top & bottom
        # The intersection points will be (5,0) and (5,10)

    def test_compute_intersection(self):
        # Horizontal line from (0,0)->(10,0) and vertical line from (5,-5)->(5,5)
        line1 = ((0, 0), (10, 0))
        line2 = ((5, -5), (5, 5))
        intersection = compute_intersection(line1, line2)
        self.assertIsNotNone(intersection)
        self.assertEqual(intersection, (5.0, 0.0))

    def test_split_polygon(self):
        # A simple square
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        # A line that bisects horizontally at y=5
        line_ = ((-1, 5), (11, 5))
        poly1, poly2 = split_polygon(square, line_)
        # poly1 should have the top portion, poly2 the bottom portion
        # Check that each polygon has intersections in it
        self.assertTrue(
            len(poly1) >= 5
        )  # 4 corners + 2 intersection points - 1 or 2 duplicates
        self.assertTrue(len(poly2) >= 4)

    def test_bisecting_line_coordinates(self):
        # Pupils at (3,3) and (7,3), image size = 10
        # Midpoint is (5,3), line is perpendicular => slope is infinite or vertical
        line_start, line_end = bisecting_line_coordinates(10, ((3, 3), (7, 3)))
        # Expect a vertical line crossing the top edge y=0 and bottom edge y=10 at x=5
        self.assertEqual(line_start, (5, 0))
        self.assertEqual(line_end, (5, 10))

    def test_line_to_edge(self):
        # Start at (5,5), angle=0 => horizontal line to the right edge
        endpoint = line_to_edge(10, (5, 5), 0)
        self.assertEqual(endpoint, (10, 5))

        # Start at (2,2), angle=45 deg => line to bottom or right edge
        endpoint_45 = line_to_edge(10, (2, 2), math.radians(45))
        # We expect it to intersect either the right edge (x=10) or bottom edge (y=10),
        # whichever is encountered first from (2,2) at slope=1.
        # The difference in x to right edge: 10 - 2 = 8 => y would be 2+8=10.
        # That point is (10,10) which is a corner.
        self.assertEqual(endpoint_45, (10, 10))

    def test_normalize_angle(self):
        # 2*pi + 0.5 => 0.5
        val = normalize_angle(2 * math.pi + 0.5)
        self.assertAlmostEqual(val, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
