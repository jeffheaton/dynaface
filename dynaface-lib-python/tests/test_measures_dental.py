import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# facial must be imported first to break the circular import chain (see
# test_measures_landmarks.py).
from dynaface import facial, util  # noqa: E402
from dynaface.measures_frontal import AnalyzeDentalArea  # noqa: E402

PIX2MM = 0.25


def face_with_mouth(inner_lip, pupils=((380, 480), (640, 480))):
    """A face with no models: blank image, given inner lip (WFLW 88-95) and pupils."""
    face = facial.AnalyzeFace(measures=[AnalyzeDentalArea()])
    face.init_image(np.zeros((1024, 1024, 3), dtype=np.uint8))
    landmarks = [(510, 600)] * 98
    for i, p in enumerate(inner_lip):
        landmarks[88 + i] = p
    landmarks[96], landmarks[97] = pupils
    face.landmarks = landmarks
    face.pix2mm = PIX2MM
    return face


# Closed mouth: the lower inner lip (93-95) retraces the upper (89-91), so the outline
# has no area, and the midline (x=510) passes through its vertices. split_polygon
# raises on this shape; it is how every closed-mouth video frame looks.
CLOSED = [
    (440, 700),
    (480, 710),
    (510, 712),
    (540, 710),
    (580, 700),
    (540, 710),
    (510, 712),
    (480, 710),
]

# Open mouth: a 120 x 30 px rectangle, x 450-570, y 690-720.
OPEN = [
    (450, 690),
    (480, 690),
    (530, 690),
    (570, 690),
    (570, 720),
    (530, 720),
    (480, 720),
    (450, 720),
]


# Shared with dynaface-lib-dotnet's MeasureDentalAreaTests, which must produce the
# same numbers: (inner lip, pupils (WFLW 96, 97), expected values). dental_right is
# the person's right, i.e. the image-left half, like eye.right and oce.r.
PARITY_CASES = {
    "closed": (
        CLOSED,
        ((380, 480), (640, 480)),
        {
            "dental_area": 0.0,
            "dental_right": 0.0,
            "dental_left": 0.0,
            "dental_diff": 0.0,
            "dental_ratio": 1.0,
        },
    ),
    "open": (
        OPEN,
        ((380, 480), (650, 480)),
        {
            "dental_area": 225.0,
            "dental_right": 121.290894,
            "dental_left": 103.709106,
            "dental_diff": 17.581787,
            "dental_ratio": 0.855044,
        },
    ),
    "tilted": (
        OPEN,
        ((380, 480), (640, 506)),
        {
            "dental_area": 225.0,
            "dental_right": 71.412964,
            "dental_left": 153.587036,
            "dental_diff": 82.174072,
            "dental_ratio": 0.464967,
        },
    ),
}


class TestAnalyzeDentalArea(unittest.TestCase):
    def test_closed_mouth_measures_zero_without_an_error(self):
        with self.assertRaises(ValueError):
            util.split_polygon(np.array(CLOSED), ((510, 0), (510, 1024)))
        face = face_with_mouth(CLOSED)
        with self.assertNoLogs("dynaface.measures_frontal", level="ERROR"):
            r = face.analyze()
        self.assertEqual(r["dental_area"], 0.0)
        self.assertEqual(r["dental_right"], 0.0)
        self.assertEqual(r["dental_left"], 0.0)
        self.assertEqual(r["dental_diff"], 0.0)
        self.assertEqual(r["dental_ratio"], 1.0)

    def image_halves_mm2(self, face):
        """Exact areas of OPEN to the image left and right of the library's midline."""
        (x0, y0), (x1, y1) = face.calc_bisect()
        x_mid = x0 + (x1 - x0) * (705 - y0) / (y1 - y0)  # midline x at mid-height
        image_left = 30 * (x_mid - 450) * PIX2MM**2  # straight line across the band
        return image_left, 120 * 30 * PIX2MM**2 - image_left

    def test_open_mouth_is_split_at_the_midline(self):
        # Pupils at x=380 and 650 put the midline near x=515: about 65 px of the
        # mouth lie to its image left (the person's right) and 55 px to its right.
        face = face_with_mouth(OPEN, pupils=((380, 480), (650, 480)))
        r = face.analyze()
        image_left, image_right = self.image_halves_mm2(face)
        self.assertAlmostEqual(image_left, 65 * 30 * PIX2MM**2, delta=1.0)
        self.assertAlmostEqual(r["dental_right"], image_left, places=6)
        self.assertAlmostEqual(r["dental_left"], image_right, places=6)
        self.assertAlmostEqual(r["dental_area"], 120 * 30 * PIX2MM**2, places=6)
        self.assertAlmostEqual(r["dental_diff"], image_left - image_right, places=6)
        self.assertAlmostEqual(r["dental_ratio"], image_right / image_left, places=6)

    def test_tilted_midline(self):
        # A tilted head tilts the midline; each half follows it.
        face = face_with_mouth(OPEN, pupils=((380, 480), (640, 506)))
        r = face.analyze()
        image_left, image_right = self.image_halves_mm2(face)
        self.assertGreater(abs(image_left - 60 * 30 * PIX2MM**2), 1.0)  # tilted
        self.assertAlmostEqual(r["dental_right"], image_left, places=6)
        self.assertAlmostEqual(r["dental_left"], image_right, places=6)

    def test_shared_parity_cases(self):
        for name, (lip, pupils, expected) in PARITY_CASES.items():
            r = face_with_mouth(lip, pupils=pupils).analyze()
            for key, value in expected.items():
                with self.subTest(case=name, key=key):
                    self.assertAlmostEqual(r[key], value, places=4)

    def test_real_failure_still_falls_back_and_logs(self):
        face = face_with_mouth(OPEN)
        with mock.patch.object(
            util, "clip_polygon_to_side", side_effect=ValueError("boom")
        ), self.assertLogs("dynaface.measures_frontal", level="ERROR"):
            r = face.analyze()
        self.assertEqual(r["dental_area"], 0)
        self.assertEqual(r["dental_ratio"], 1)


class TestClipPolygonToSide(unittest.TestCase):
    SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    def test_either_line_direction_keeps_the_same_side(self):
        for line in (((4, -5), (4, 15)), ((4, 15), (4, -5))):
            with self.subTest(line=line):
                left = util.clip_polygon_to_side(self.SQUARE, line, image_left=True)
                right = util.clip_polygon_to_side(self.SQUARE, line, image_left=False)
                self.assertAlmostEqual(util.PolyArea(left[:, 0], left[:, 1]), 40.0)
                self.assertAlmostEqual(util.PolyArea(right[:, 0], right[:, 1]), 60.0)
                self.assertTrue(np.all(left[:, 0] <= 4))
                self.assertTrue(np.all(right[:, 0] >= 4))

    def test_polygon_wholly_on_the_other_side_is_empty(self):
        out = util.clip_polygon_to_side(self.SQUARE, ((20, 0), (20, 10)), False)
        self.assertEqual(out.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
