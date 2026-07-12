import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface.util import (
    PolyArea,
    bisecting_line_coordinates,
    calculate_average_rgb,
    calculate_face_rotation,
    compute_intersection,
    line_intersection,
    line_to_edge,
    normalize_angle,
    rotate_crop_points,
    safe_clip,
    scale_crop_points,
    split_polygon,
    straighten,
    symmetry_ratio,
)


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
        self.assertAlmostEqual(rotated[1][1], -10, places=5)

    #    @patch.object(facial.AnalyzeFace, "pd", 60.0)
    #    def test_calc_pd(self):
    #        # Pupils 10 px apart
    #        pupils = ((0, 0), (10, 0))
    #        distance, pix2mm = calc_pd(pupils)
    #        self.assertEqual(distance, 10.0)
    #        # pix2mm = 60 / 10 = 6 mm per pixel
    #        self.assertEqual(pix2mm, 6.0)

    # def test_get_pupils(self):
    #    # Mock some landmarks
    #    mock_landmarks = {
    #        facial.LM_LEFT_PUPIL: (100, 100),
    #        facial.LM_RIGHT_PUPIL: (200, 100),
    #    }
    #    left_pupil, right_pupil = get_pupils(mock_landmarks)
    #    self.assertEqual(left_pupil, (100, 100))
    #    self.assertEqual(right_pupil, (200, 100))

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
        self.assertEqual(avg, (127, 0, 127))

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
            len(poly1) >= 4
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
        self.assertIsNone(endpoint)

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
