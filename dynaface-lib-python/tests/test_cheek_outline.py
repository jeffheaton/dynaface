"""
Tests for dynaface.cheek_outline: the far-cheek tracer, the level frame and the
apex and submalar search. No models: the masks are synthetic or built from the
synthetic-head renders' background color.
"""

import json
import math
import os
import sys
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import cheek  # noqa: E402
from dynaface import cheek_outline as co  # noqa: E402

TESTS_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests_data"
)
LANDMARKS_JSON = os.path.join(TESTS_DATA, "cheek_landmarks.json")
RENDERS_JSON = os.path.join(TESTS_DATA, "cheek", "cheek_renders.json")
RENDERS_DIR = os.path.join(TESTS_DATA, "cheek")
OUTLINES_JSON = os.path.join(TESTS_DATA, "cheek_outlines.json")

# The synthetic face: a level head with parallel canthi and alar-base lines, so
# every level is horizontal and h = (600 - row) / 300 at any x.
BASE_CHIN_ROW = 600
BASE_H = 300.0
BASE_X = 540.0  # the straight synthetic outline (the far cheek is the left)


def _base_landmarks() -> np.ndarray:
    """The 98 landmarks of the synthetic face, the far cheek the subject's left."""
    lm = np.zeros((98, 2), dtype=np.float64)
    lm[60], lm[72] = (300, 300), (500, 300)  # lateral canthi
    lm[96], lm[97] = (330, 300), (470, 300)  # pupils
    lm[55], lm[59] = (360, 400), (440, 400)  # alar bases, parallel to the canthi
    lm[16] = (400, BASE_CHIN_ROW)
    lm[90], lm[94] = (400, 470), (400, 474)  # stomion at 472: h 0.4267
    lm[54], lm[57] = (480, 394), (430, 418)  # nose tip h 0.6867, subnasale 0.6067
    lm[53] = (470, 362)  # the lower dorsum point, h 0.7933
    lm[76], lm[82] = (360, 470), (440, 470)  # mouth corners
    for k in range(17):
        t = k / 16.0
        lm[32 - k] = (round(BASE_X - 140 * t), round(290 + 310 * t))
        lm[k] = (round(260 + 140 * t), round(290 + 310 * t))
    return lm


def _row_of(h: float) -> float:
    """The synthetic face's row at height ratio h."""
    return BASE_CHIN_ROW - h * BASE_H


def _outline(bumps=(), first=290, last=BASE_CHIN_ROW):
    """
    A synthetic outline at x = BASE_X with raised-cosine bumps, each (h, amplitude
    as a fraction of H, outward positive, half-width in rows), all mask rows.
    """
    rows = np.arange(first, last + 1, dtype=np.int32)
    xs = np.full(len(rows), BASE_X, dtype=np.float64)
    for h, amp, half in bumps:
        d = (rows - _row_of(h)) / half
        lobe = np.where(np.abs(d) < 1.0, np.cos(0.5 * np.pi * d) ** 2, 0.0)
        xs += amp * BASE_H * lobe
    src = np.full(len(rows), co.OUTLINE_SRC_MASK, dtype=np.uint8)
    return xs.astype(np.float32), rows, src


def _warp(points, k=4e-4):
    """A projective warp (a keystone converging the horizontals to the right)."""
    pts = np.asarray(points, dtype=np.float64)
    w = 1.0 + k * (pts[..., 0] - 400.0)
    return np.stack(
        [(pts[..., 0] - 400.0) / w + 400.0, (pts[..., 1] - 400.0) / w + 400.0],
        axis=-1,
    )


def _warp_outline(xs, rows, src, k=4e-4):
    """
    An outline through _warp(), resampled on integer rows: x by linear
    interpolation, each row's source from the nearest warped row.
    """
    warped = _warp(np.stack([np.asarray(xs, dtype=np.float64), rows], axis=1), k)
    assert np.all(np.diff(warped[:, 1]) > 0)  # np.interp needs increasing rows
    w_rows = np.arange(
        math.ceil(warped[:, 1].min()), math.floor(warped[:, 1].max()) + 1
    )
    w_xs = np.interp(w_rows, warped[:, 1], warped[:, 0])
    nearest = np.abs(warped[:, 1][None, :] - w_rows[:, None]).argmin(axis=1)
    return (
        w_xs.astype(np.float32),
        w_rows.astype(np.int32),
        np.asarray(src, dtype=np.uint8)[nearest],
    )


def _mirror_landmarks(lm: np.ndarray, width: int) -> np.ndarray:
    """The landmarks of the mirrored image: x flipped and the sides swapped."""
    perm = list(range(98))
    for i in range(33):
        perm[i] = 32 - i
    perm[60], perm[72] = 72, 60
    perm[55], perm[59] = 59, 55
    perm[96], perm[97] = 97, 96
    flipped = np.array(lm, dtype=np.float64)
    flipped[:, 0] = width - 1 - flipped[:, 0]
    mirrored = np.empty_like(flipped)
    mirrored[perm] = flipped
    return mirrored


class DiscScene:
    """
    A synthetic head: a disc of radius 200 at (300, 300) in a 600 px image, the
    far cheek the subject's left (the image's right), the far contour inset px
    (2) inside the disc's edge, the chin below the disc. The tracer's span (and
    H: the levels are horizontal here) is 330, so the walk's limit is 16 px and
    NOSE_STEP is 3.3 px. The mask rows' edge (0.87 px past the
    last disc pixel, from the binary alpha) sits about 3 px outside the seed,
    more where the straight contour segments sag inside the arc (up to 6.4 px):
    the near-uniform departure that the band's reference, the median over the
    mask rows between the stomion level (row 432) and the canthus level (200)
    outside the nose band, absorbs, so a nose bump has to rise NOSE_STEP above
    it to be marked.
    """

    SIZE = 600
    CX, CY, R = 300.0, 300.0, 200.0

    def __init__(self, center_x=CX, soft=False, inset=2):
        yy, xx = np.mgrid[0 : self.SIZE, 0 : self.SIZE]
        rr = np.hypot(xx - center_x, yy - self.CY)
        self.foreground = rr < self.R
        if soft:
            # alpha == MASK_ALPHA exactly at r == R: the edge to recover. The
            # mask is the alpha's threshold, as lateral.foreground_mask() makes it.
            ramp = co.MASK_ALPHA + 40.0 * (self.R - rr)
            self.alpha = np.clip(ramp, 0, 255).astype(np.uint8)
            self.foreground = self.alpha > co.MASK_ALPHA
        else:
            self.alpha = np.where(self.foreground, 255, 0).astype(np.uint8)
        self.disc = self.foreground.copy()  # before any extension
        self.center_x = center_x
        lm = np.zeros((98, 2), dtype=np.float64)
        lm[60], lm[72] = (220, 200), (380, 200)
        lm[96], lm[97] = (250, 200), (350, 200)
        lm[55], lm[59] = (270, 320), (330, 320)
        lm[16] = (300, 530)  # below the disc, whose last row is 499
        lm[90], lm[94] = (300, 430), (300, 434)
        lm[54], lm[57] = (390, 330), (360, 360)  # h 0.606 and 0.515
        for k, y in enumerate(np.round(np.linspace(190, 530, 17)).astype(int)):
            if y < 500:
                lm[32 - k] = (self.edge(y) - inset, y)
            elif y < 530:
                lm[32 - k] = (320, y)
        lm[16] = (300, 530)
        for k in range(17):
            lm[k] = (self.SIZE - 1 - lm[32 - k][0], lm[32 - k][1])
        self.landmarks = lm

    def edge(self, y: int) -> int:
        """The disc's rightmost foreground column in row y."""
        return int(np.flatnonzero(self.disc[y]).max())

    def extend(self, rows, start, stop):
        """Foreground from edge + start to edge + stop (exclusive) on these rows."""
        for y in rows:
            e = self.edge(y)
            self.foreground[y, e + start : e + stop] = True
            self.alpha[y, e + start : e + stop] = 255

    def trace(self, far_left=True, **kwargs):
        fg, alpha, lm = self.foreground, self.alpha, self.landmarks
        if not far_left:
            fg, alpha = fg[:, ::-1], alpha[:, ::-1]
            lm = _mirror_landmarks(lm, self.SIZE)
        return co.trace_far_cheek(fg, alpha, lm, far_left, **kwargs)


class TestLevels(unittest.TestCase):
    """The level frame, the levels through it and Kaptein's h."""

    def _landmarks_with_vanishing(self, vx=1000.0):
        lm = _base_landmarks()
        # The alar-base line through (350, 400) and V = (vx, 300).
        lm[55] = (350, 400)
        lm[59] = (480, 400 - 100 * (130 / (vx - 350)))
        return lm

    def test_level_through_known_vanishing_point(self):
        lm = self._landmarks_with_vanishing(1000.0)
        frame = co.oblique_frame(lm)
        self.assertIsNotNone(frame)
        np.testing.assert_allclose(frame.vanishing, [1000.0, 300.0], atol=1e-9)
        np.testing.assert_allclose(frame.down, [0.0, 1.0], atol=1e-12)
        point, direction = co.level_through(frame, lm[16])
        np.testing.assert_array_equal(point, lm[16])
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=12)
        # The direction runs from V through the chin.
        self.assertAlmostEqual(co._cross(lm[16] - frame.vanishing, direction), 0.0)
        # Where that level meets the vertical through x = 700: on the line V-chin.
        met = co.level_at_vertical((point, direction), (700.0, 100.0), frame)
        self.assertAlmostEqual(met[0], 700.0, places=9)
        self.assertAlmostEqual(met[1], 600.0 - 300.0 * (300.0 / 600.0), places=9)

    def test_singular_branch_uses_mean_direction(self):
        lm = _base_landmarks()  # parallel canthi and alar-base lines
        frame = co.oblique_frame(lm)
        self.assertIsNone(frame.vanishing)
        np.testing.assert_allclose(frame.mean_dir, [1.0, 0.0], atol=1e-12)
        point, direction = co.level_through(frame, lm[16])
        np.testing.assert_allclose(direction, frame.mean_dir)
        met = co.level_at_vertical((point, direction), (540.0, 100.0), frame)
        np.testing.assert_allclose(met, [540.0, BASE_CHIN_ROW], atol=1e-9)
        h, scale = co.height_ratio((540.0, 450.0), lm[16], lm[72], frame)
        self.assertAlmostEqual(h, 0.5, places=12)
        self.assertAlmostEqual(scale, BASE_H, places=9)

    def test_frame_needs_distinct_canthi(self):
        lm = _base_landmarks()
        lm[72] = lm[60]
        self.assertIsNone(co.oblique_frame(lm))
        self.assertIsNone(co.oblique_frame(lm[:97]))
        self.assertIsNone(co.observed_points(lm, None, None, None, True))

    def test_down_points_toward_chin(self):
        lm = _base_landmarks()
        lm[:, 1] = 900 - lm[:, 1]  # an upside-down face
        frame = co.oblique_frame(lm)
        np.testing.assert_allclose(frame.down, [0.0, -1.0], atol=1e-12)
        h, _ = co.height_ratio((540.0, 450.0), lm[16], lm[72], frame)
        self.assertAlmostEqual(h, 0.5, places=12)

    def test_height_ratio_exact_on_trapezoid(self):
        # The chin level and the canthus level meet at V = (2000, 300): H is
        # 318.75 on the vertical x = 300 and 281.25 on x = 500.
        lm = self._landmarks_with_vanishing(2000.0)
        frame = co.oblique_frame(lm)
        np.testing.assert_allclose(frame.vanishing, [2000.0, 300.0], atol=1e-6)
        for x, expected in ((300.0, 318.75), (500.0, 281.25)):
            chin_y = 600.0 - 300.0 * (x - 400.0) / 1600.0
            for p, want in (((x, 300.0), 1.0), ((x, chin_y), 0.0)):
                h, scale = co.height_ratio(p, lm[16], lm[72], frame)
                self.assertAlmostEqual(scale, expected, places=9)
                self.assertAlmostEqual(h, want, places=9)
            mid = (x, 0.5 * (300.0 + chin_y))
            self.assertAlmostEqual(
                co.height_ratio(mid, lm[16], lm[72], frame)[0], 0.5, places=12
            )

    def test_continuity_as_vanishing_point_recedes(self):
        lm = _base_landmarks()
        lm[55] = (350.0, 400.0)
        p = (600.0, 450.0)
        hs = {}
        for eps in (2e-6, 5e-7, 1e-7):
            case = lm.copy()
            case[59] = (450.0, 400.0 + 100.0 * math.tan(eps))  # the lines meet at eps
            frame = co.oblique_frame(case)
            if eps > co.VANISH_MIN_SIN:
                self.assertIsNotNone(frame.vanishing)
            else:
                self.assertIsNone(frame.vanishing)
            hs[eps] = co.height_ratio(p, case[16], case[72], frame)[0]
        self.assertLess(abs(hs[2e-6] - hs[5e-7]), 1e-4)
        self.assertLess(abs(hs[5e-7] - hs[1e-7]), 1e-4)
        self.assertLess(abs(hs[2e-6] - 0.5), 1e-4)

    def test_perspective_warp_exercises_vanishing_point(self):
        """
        The synthetic face and its bulge through a keystone warp; the second
        perspective case, on a render, is
        TestRenders.test_perspective_warp_on_render.
        """
        lm = _base_landmarks()
        xs, rows, src = _outline([(0.78, 0.03, 20)])
        plain = co.observed_points(lm, xs, rows, src, True)
        warped_lm = _warp(lm)
        frame = co.oblique_frame(warped_lm)
        self.assertIsNotNone(frame.vanishing)
        self.assertTrue(np.all(np.isfinite(frame.vanishing)))
        # The horizontal through the chin maps onto the level through the warped
        # chin: a projective warp keeps lines.
        point, direction = co.level_through(frame, warped_lm[16])
        for x in (100.0, 300.0, 700.0):
            q = _warp((x, float(BASE_CHIN_ROW)))
            self.assertLess(abs(co._cross(q - point, direction)), 1e-6)
        # The warped outline, resampled on integer rows.
        w_xs, w_rows, w_src = _warp_outline(xs, rows, src)
        found = co.observed_points(warped_lm, w_xs, w_rows, w_src, True)
        self.assertFalse(found.apex_prior)
        self.assertLess(abs(found.apex_h - plain.apex_h), 0.01)
        np.testing.assert_allclose(found.apex, _warp(plain.apex), atol=2.0)


class TestContourSeed(unittest.TestCase):
    """The seed polyline rule: segment by segment, outward-most crossing."""

    def test_reversing_chin_takes_outward_crossing(self):
        cont = [(300, 900), (310, 930), (330, 964), (350, 982), (380, 991)]
        cont += [(420, 997), (440, 991)]
        rows = [991, 995, 997, 980]
        x_980 = 330 + 20 * 16 / 18  # on the segment from (330, 964) to (350, 982)
        right = co.contour_seed(cont, rows, far_left=False)  # outward is -x
        np.testing.assert_allclose(right, [380.0, 380 + 40 * 4 / 6, 420.0, x_980])
        left = co.contour_seed(cont, rows, far_left=True)  # outward is +x
        np.testing.assert_allclose(left, [440.0, 420 + 20 * 2 / 6, 420.0, x_980])
        # Without far_left, the side is read from the polyline: the chin (the
        # last point) is medial of the first point.
        np.testing.assert_allclose(co.contour_seed(cont, rows), right)
        flipped = [(1000 - x, y) for x, y in cont]  # now outward is +x
        np.testing.assert_allclose(
            co.contour_seed(flipped, rows), [1000 - x for x in right]
        )

    def test_duplicate_rows_take_outward_end(self):
        cont = [(300, 100), (310, 120), (320, 120), (330, 140)]
        rows = [100, 110, 120, 130, 140]
        np.testing.assert_allclose(
            co.contour_seed(cont, rows, far_left=True), [300, 305, 320, 325, 330]
        )
        np.testing.assert_allclose(
            co.contour_seed(cont, rows, far_left=False), [300, 305, 310, 325, 330]
        )

    def test_rows_outside_extrapolate_end_segments(self):
        cont = [(200, 300), (210, 400), (215, 500)]
        seed = co.contour_seed(cont, [280, 290, 300, 350, 500, 520], far_left=True)
        np.testing.assert_allclose(seed, [198, 199, 200, 205, 215, 216])

    def test_degenerate_contours(self):
        self.assertEqual(len(co.contour_seed([(1, 2)], [])), 0)
        np.testing.assert_allclose(co.contour_seed([(7, 2)], [1, 2, 3]), [7, 7, 7])
        flat = co.contour_seed([(5, 2), (9, 2), (3, 2)], [1, 2, 3], far_left=True)
        np.testing.assert_allclose(flat, [9, 9, 9])

    def test_contour_outline_starts_at_canthus_above_contour_top(self):
        lm = _base_landmarks()  # the far contour's top (32) is at row 290
        lm[72] = (500, 280)  # the far canthus above it
        xs, rows, src = co.contour_outline(lm, True)
        self.assertEqual(rows[0], 280)
        self.assertEqual(rows[-1], BASE_CHIN_ROW)
        self.assertEqual(xs.dtype, np.float32)
        self.assertEqual(rows.dtype, np.int32)
        self.assertEqual(src.dtype, np.uint8)
        self.assertTrue(np.all(src == co.OUTLINE_SRC_CONTOUR))
        # Rows 280 to 289 extrapolate the first segment (32 -> 31).
        (x0, y0), (x1, y1) = lm[32], lm[31]
        for i in range(10):
            want = x0 + (rows[i] - y0) * (x1 - x0) / (y1 - y0)
            self.assertAlmostEqual(float(xs[i]), want, places=4)
        # The right side mirrors the rule from the other contour.
        xs_r, rows_r, _ = co.contour_outline(lm, False)
        self.assertEqual(rows_r[0], 290)
        self.assertAlmostEqual(float(xs_r[0]), 260.0, places=4)
        # Fewer than 98 landmarks: nothing.
        self.assertEqual(len(co.contour_outline(lm[:50], True)[0]), 0)


class TestTracer(unittest.TestCase):
    """trace_far_cheek() on synthetic masks."""

    def test_disc_head(self):
        scene = DiscScene()
        scene.extend(range(330, 361), 1, 6)  # the nose: 5 px past the edge
        scene.extend(range(240, 261), 1, 61)  # hair merging with the cheek
        scene.extend(range(400, 421), 60, 81)  # a hair blob 60 px outside
        lm = scene.landmarks
        for y in (424, 445):  # contour points 4 px in the background
            k = int(np.flatnonzero(lm[16:33, 1] == y)[0]) + 16
            lm[k] = (scene.edge(y) + 4, y)
        xs, rows, src = scene.trace()
        self.assertEqual(xs.dtype, np.float32)
        self.assertEqual(rows.dtype, np.int32)
        self.assertEqual(src.dtype, np.uint8)
        self.assertEqual(rows[0], 190)  # the contour top, above the canthus row
        self.assertEqual(rows[-1], 530)  # the chin row
        by_row = {int(y): i for i, y in enumerate(rows)}
        seed = co.contour_seed(lm[list(co.LEFT_FAR_CONTOUR)], rows, far_left=True)
        # Disc rows: the sub-pixel edge from the binary alpha, 0.8745 px past
        # the last foreground pixel.
        for y in list(range(200, 240)) + list(range(261, 330)) + list(range(361, 470)):
            i = by_row[y]
            self.assertEqual(src[i], co.OUTLINE_SRC_MASK, y)
            self.assertAlmostEqual(float(xs[i]), scene.edge(y) + 223 / 255, places=3)
        # The seed 4 px in the background (at rows 424 and 445; the straight
        # segments between the points sag back toward the curved edge) walks
        # inward to the same edge.
        for y in list(range(422, 427)) + list(range(443, 448)):
            self.assertGreater(seed[by_row[y]], scene.edge(y) + 2.5)
        # Merged hair: the walk reaches the limit and keeps the seed.
        for y in range(240, 261):
            i = by_row[y]
            self.assertEqual(src[i], co.OUTLINE_SRC_CONTOUR, y)
            self.assertAlmostEqual(float(xs[i]), seed[i], places=4)
        # The nose rows: the edge departed by 7.9 px, 5 px beyond the other mask
        # rows' 3 px (the band's reference; NOSE_STEP is 3.3), so the seed stands
        # in, between the tip's and the subnasale's levels only.
        for y in range(330, 361):
            i = by_row[y]
            self.assertEqual(src[i], co.OUTLINE_SRC_NOSE, y)
            self.assertAlmostEqual(float(xs[i]), seed[i], places=4)
        self.assertEqual(src[by_row[329]], co.OUTLINE_SRC_MASK)
        self.assertEqual(src[by_row[361]], co.OUTLINE_SRC_MASK)
        # Rows beyond the mask (below the disc, whose last row is 499, where the
        # seed is already 22 px outside the edge) keep the seed.
        for y in range(499, 531):
            i = by_row[y]
            self.assertEqual(src[i], co.OUTLINE_SRC_CONTOUR, y)
            self.assertAlmostEqual(float(xs[i]), seed[i], places=4)
        self.assertEqual(src[by_row[498]], co.OUTLINE_SRC_MASK)

    def test_nose_extends_upward_while_departing(self):
        scene = DiscScene()
        scene.extend(range(300, 361), 1, 6)  # the bump reaches above the tip row
        xs, rows, src = scene.trace()
        by_row = {int(y): i for i, y in enumerate(rows)}
        for y in range(300, 361):
            self.assertEqual(src[by_row[y]], co.OUTLINE_SRC_NOSE, y)
        self.assertEqual(src[by_row[299]], co.OUTLINE_SRC_MASK)
        # A departure above the tip that the tip's row does not share stays mask.
        scene = DiscScene()
        scene.extend(range(340, 361), 1, 6)
        scene.extend(range(280, 301), 1, 6)
        xs, rows, src = scene.trace()
        by_row = {int(y): i for i, y in enumerate(rows)}
        for y in range(340, 361):
            self.assertEqual(src[by_row[y]], co.OUTLINE_SRC_NOSE, y)
        for y in range(280, 340):
            self.assertEqual(src[by_row[y]], co.OUTLINE_SRC_MASK, y)

    def test_uniform_offset_is_not_the_nose(self):
        """
        The mask's edge sits a near-uniform few px outside the contour on every
        row (the background model's edge against SPIGA's contour on a photo):
        the band's reference absorbs it however large, and only a departure
        NOSE_STEP beyond it is the nose.
        """
        scene = DiscScene(inset=5)  # 3 px further inside the disc than usual
        lm = scene.landmarks
        frame = co.oblique_frame(lm)
        step_px = co.NOSE_STEP * float((lm[16] - lm[72]) @ frame.down)  # 3.3
        xs, rows, src = scene.trace()
        seed = co.contour_seed(lm[list(co.LEFT_FAR_CONTOUR)], rows, far_left=True)
        mask = src == co.OUTLINE_SRC_MASK
        # Every mask row departs by more than NOSE_STEP: 5 px plus the sub-pixel
        # 0.87, more where the straight contour sags inside the arc (median 6).
        self.assertGreater(np.median((xs - seed)[mask]), 1.5 * step_px)
        self.assertGreater((xs - seed)[mask].max(), 2.5 * step_px)
        self.assertFalse(np.any(src == co.OUTLINE_SRC_NOSE))
        self.assertEqual(int(mask.sum()), 309)
        # A nose bump of 5 px on the band rows is marked, its neighbors are not.
        scene.extend(range(330, 361), 1, 6)
        xs, rows, src = scene.trace()
        self.assertEqual(set(rows[src == co.OUTLINE_SRC_NOSE]), set(range(330, 361)))
        by_row = {int(y): i for i, y in enumerate(rows)}
        self.assertEqual(src[by_row[329]], co.OUTLINE_SRC_MASK)
        self.assertEqual(src[by_row[361]], co.OUTLINE_SRC_MASK)

    def test_contour_and_edge_rows_are_never_nose_rows(self):
        """
        The band rows have no mask edge within the window (hair) and the mask
        rows elsewhere sit 5 px inside the seed, so the reference is negative
        (-4.6 px) and a row kept at the seed itself departs from it by more than
        NOSE_STEP (3 px); a gray step 8 px outside the seed departs further
        still. Neither the contour rows nor the EDGE rows standing in on the
        band are the nose: only mask rows are.
        """
        lm = _base_landmarks()
        size = 700
        rows_all = np.arange(size)
        seed = co.contour_seed(lm[list(co.LEFT_FAR_CONTOUR)], rows_all, far_left=True)
        xx = np.arange(size)[None, :]
        foreground = xx <= (seed[:, None] - 5)  # 5 px inside the seed
        band = (rows_all >= 394) & (rows_all <= 418)  # the tip's to subnasale's
        foreground[band] |= xx <= 600  # no mask edge within the window there
        alpha = np.where(foreground, 255, 0).astype(np.uint8)
        gray = np.where(xx < seed[:, None] + 8, 180, 60).astype(np.uint8)
        step_px = co.NOSE_STEP * BASE_H
        cases = ((None, co.OUTLINE_SRC_CONTOUR), (gray, co.OUTLINE_SRC_EDGE))
        for image, source in cases:
            with self.subTest(source=int(source)):
                xs, rows, src = co.trace_far_cheek(
                    foreground, alpha, lm, True, gray=image
                )
                depart = xs - seed[rows]
                mask = src == co.OUTLINE_SRC_MASK
                self.assertLess(np.median(depart[mask]), -step_px)
                in_band = (rows >= 394) & (rows <= 418)
                self.assertTrue(np.all(src[in_band] == source))
                if source == co.OUTLINE_SRC_EDGE:
                    self.assertTrue(np.all(depart[in_band] > 2 * step_px))
                else:
                    np.testing.assert_allclose(depart[in_band], 0.0, atol=1e-4)
                self.assertFalse(np.any(src == co.OUTLINE_SRC_NOSE))

    def test_reference_excludes_rows_below_stomion_level(self):
        """
        The jaw rows (below the stomion level) depart 8 px more than the cheek
        rows, as on photos, and outnumber them (170 to 106 with the stomion at
        row 430): with the band rows at the cheek rows' departure no row is the
        nose, and with the band 5 px beyond the cheek rows (above NOSE_STEP's 3
        px) the band is marked, which a reference over every mask row (median
        9 px, the jaw's) would have hidden.
        """
        size = 700
        rows_all = np.arange(size)
        xx = np.arange(size)[None, :]
        lm = _base_landmarks()
        lm[90], lm[94] = (400, 428), (400, 432)  # the stomion at row 430, h 0.567
        seed = co.contour_seed(lm[list(co.LEFT_FAR_CONTOUR)], rows_all, far_left=True)
        in_band = (rows_all >= 394) & (rows_all <= 418)
        step_px = co.NOSE_STEP * BASE_H
        for band_extra, want in ((0.0, set()), (5.0, set(range(394, 419)))):
            with self.subTest(band_extra=band_extra):
                offset = np.where(rows_all > 430, 9.0, 1.0)
                offset[in_band] += band_extra
                foreground = xx <= (seed[:, None] + offset[:, None])
                alpha = np.where(foreground, 255, 0).astype(np.uint8)
                xs, rows, src = co.trace_far_cheek(foreground, alpha, lm, True)
                mask = src == co.OUTLINE_SRC_MASK
                depart = xs - seed[rows]
                jaw = mask & (rows > 430)
                cheek = mask & (rows <= 430) & (rows >= 300)
                self.assertGreater(int(jaw.sum()), int(cheek.sum()))
                # The median over every mask row is the jaw's departure.
                self.assertGreater(np.median(depart[mask]), 2 * step_px)
                self.assertEqual(set(rows[src == co.OUTLINE_SRC_NOSE]), want)

    def test_refined_crossing_takes_nearest_pair(self):
        row = np.array([255, 255, 255, 10, 5, 0], dtype=np.uint8)
        # The mask's edge pair (3, 4) doesn't bracket 32 (the closing's shift on
        # the image's right); the pair one pixel inward (2, 3) does.
        self.assertAlmostEqual(co._refined_crossing(row, 3, 1), 2 + 223 / 245)
        # Two pixels outward (the shift on the image's left, mirrored here).
        row = np.array([255, 255, 255, 255, 255, 0], dtype=np.uint8)
        self.assertAlmostEqual(co._refined_crossing(row, 2, 1), 4 + 223 / 255)
        flipped = np.ascontiguousarray(row[::-1])
        self.assertAlmostEqual(co._refined_crossing(flipped, 3, -1), 1 - 223 / 255)
        # Nearest first, inward before outward.
        row = np.array([255, 0, 255, 0], dtype=np.uint8)
        self.assertAlmostEqual(co._refined_crossing(row, 1, 1), 223 / 255)
        # No bracketing pair within ALPHA_SEARCH: the mask's pixel boundary.
        self.assertEqual(co.ALPHA_SEARCH, 2)
        self.assertEqual(co._refined_crossing(np.full(8, 255, np.uint8), 3, 1), 3.5)
        self.assertEqual(co._refined_crossing(np.full(8, 0, np.uint8), 3, -1), 2.5)
        row = np.array([255, 255, 255, 255, 255, 255, 0], dtype=np.uint8)
        self.assertEqual(co._refined_crossing(row, 2, 1), 2.5)
        # Pairs past the border are skipped, never indexed.
        self.assertEqual(co._refined_crossing(np.full(3, 255, np.uint8), 2, 1), 2.5)
        self.assertEqual(co._refined_crossing(np.full(3, 255, np.uint8), 0, -1), -0.5)
        row = np.array([255, 0], dtype=np.uint8)
        self.assertAlmostEqual(co._refined_crossing(row, 0, 1), 223 / 255)

    def test_alpha_crossing_refined_past_shifted_mask(self):
        """
        The closed mask shifted a pixel in +x relative to its alpha (lateral's
        even closing kernel does this: outward on the image's right, inward on
        its left): the traced x is the alpha's own crossing on every row, where
        the mask's edge pair alone would put it a pixel off; shifted 3 px, past
        ALPHA_SEARCH, no pair brackets and the mask's pixel boundary stands in.
        """
        for far_left in (True, False):
            scene = DiscScene(soft=True)
            fg, alpha, lm = scene.foreground, scene.alpha, scene.landmarks
            cx = scene.CX
            if not far_left:
                fg, alpha = fg[:, ::-1], alpha[:, ::-1]
                lm = _mirror_landmarks(lm, scene.SIZE)
                cx = scene.SIZE - 1 - cx
            for shift in (1, 2, 3):  # 2 is ALPHA_SEARCH itself, still refined
                with self.subTest(far_left=far_left, shift=shift):
                    shifted = np.zeros_like(fg)
                    shifted[:, shift:] = fg[:, :-shift]
                    xs, rows, src = co.trace_far_cheek(shifted, alpha, lm, far_left)
                    mask = (src == co.OUTLINE_SRC_MASK) & (rows >= 200) & (rows <= 470)
                    self.assertGreater(mask.sum(), 200)
                    radius = np.hypot(xs[mask] - cx, rows[mask] - scene.CY)
                    on_boundary = np.abs(xs[mask] - np.floor(xs[mask]) - 0.5) < 1e-6
                    if shift <= co.ALPHA_SEARCH:
                        self.assertLess(np.abs(radius - scene.R).max(), 0.25)
                        # A crossing lands on a .5 only by coincidence (5 rows).
                        self.assertLess(on_boundary.mean(), 0.05)
                    else:
                        self.assertTrue(np.all(on_boundary))
                        self.assertGreater(np.abs(radius - scene.R).min(), 1.0)
                    # The mask's own edge pair (its binary alpha) is a pixel off.
                    binary = np.where(shifted, 255, 0).astype(np.uint8)
                    xs_b, _, src_b = co.trace_far_cheek(shifted, binary, lm, far_left)
                    mask_b = (src_b == co.OUTLINE_SRC_MASK) & (rows >= 200)
                    mask_b &= rows <= 470
                    radius_b = np.hypot(xs_b[mask_b] - cx, rows[mask_b] - scene.CY)
                    self.assertGreater(np.median(np.abs(radius_b - scene.R)), 0.5)

    def test_seed_near_border_never_indexes_outside(self):
        scene = DiscScene(center_x=420.0)  # the disc runs past the right border
        lm = scene.landmarks
        for k in range(17, 33):
            y = int(lm[k][1])
            if 213 <= y <= 387:
                lm[k] = (scene.SIZE - 4, y)  # 3 px from the border
        xs, rows, src = scene.trace()
        by_row = {int(y): i for i, y in enumerate(rows)}
        for y in range(220, 380):
            i = by_row[y]
            self.assertEqual(src[i], co.OUTLINE_SRC_CONTOUR, y)
            self.assertLess(float(xs[i]), scene.SIZE)
        self.assertEqual(src[by_row[190]], co.OUTLINE_SRC_MASK)
        # Rows outside the image keep the seed too.
        lm[16] = (300, 650)
        xs, rows, src = scene.trace()
        self.assertEqual(rows[-1], 650)
        self.assertTrue(np.all(src[rows >= scene.SIZE] == co.OUTLINE_SRC_CONTOUR))

    def test_subpixel_edge_of_soft_disc(self):
        scene = DiscScene(soft=True)
        xs, rows, src = scene.trace()
        mask_rows = (src == co.OUTLINE_SRC_MASK) & (rows >= 200) & (rows <= 470)
        self.assertGreater(mask_rows.sum(), 200)
        radius = np.hypot(xs[mask_rows] - scene.CX, rows[mask_rows] - scene.CY)
        self.assertLess(np.abs(radius - scene.R).max(), 0.25)
        # The pair's interpolation (_alpha_crossing): the binary alpha puts the
        # edge 0.8745 px past the last foreground pixel; alphas that don't
        # bracket MASK_ALPHA (a pixel the closing filled in) put it on the pixel
        # boundary. The tracer looks for a bracketing pair near the mask's edge
        # first (_refined_crossing, test_refined_crossing_takes_nearest_pair).
        self.assertAlmostEqual(
            co._alpha_crossing(np.array([255, 0], dtype=np.uint8), 0, 1), 223 / 255
        )
        self.assertAlmostEqual(
            co._alpha_crossing(np.array([10, 5], dtype=np.uint8), 1, 0), 0.5
        )

    def test_both_far_sides_mirror(self):
        scene = DiscScene()
        scene.extend(range(330, 361), 1, 6)
        scene.extend(range(240, 261), 1, 61)
        xs, rows, src = scene.trace(far_left=True)
        xs_m, rows_m, src_m = scene.trace(far_left=False)
        np.testing.assert_array_equal(rows_m, rows)
        np.testing.assert_array_equal(src_m, src)
        np.testing.assert_allclose(xs_m, scene.SIZE - 1 - xs, atol=1e-3)
        self.assertEqual(set(rows[src == co.OUTLINE_SRC_NOSE]), set(range(330, 361)))
        self.assertEqual(
            set(rows[src == co.OUTLINE_SRC_CONTOUR]),
            set(range(240, 261)) | set(range(499, 531)),
        )

    def test_nothing_without_landmarks_or_frame(self):
        scene = DiscScene()
        empty = co.trace_far_cheek(scene.foreground, scene.alpha, [(1, 1)] * 10, True)
        self.assertEqual([len(a) for a in empty], [0, 0, 0])
        lm = scene.landmarks.copy()
        lm[72] = lm[60]
        empty = co.trace_far_cheek(scene.foreground, scene.alpha, lm, True)
        self.assertEqual([len(a) for a in empty], [0, 0, 0])

    def test_non_finite_landmarks_give_nothing(self):
        """
        A NaN chin or an inf canthus: no outline, no frame and no points, where
        the band rows' int(round()) would raise ValueError or OverflowError.
        """
        scene = DiscScene()
        for index, value in ((16, math.nan), (72, math.inf)):
            lm = scene.landmarks.copy()
            lm[index, 1] = value
            with self.subTest(landmark=index):
                empty = co.trace_far_cheek(scene.foreground, scene.alpha, lm, True)
                self.assertEqual([len(a) for a in empty], [0, 0, 0])
                fallback = co.contour_outline(lm, True)
                self.assertEqual([len(a) for a in fallback], [0, 0, 0])
                self.assertIsNone(co.oblique_frame(lm))
                self.assertIsNone(co.observed_points(lm, None, None, None, True))

    def test_edge_source_on_gray_step(self):
        lm = _base_landmarks()
        size = 700
        rows_all = np.arange(size)
        seed = co.contour_seed(lm[list(co.LEFT_FAR_CONTOUR)], rows_all, far_left=True)
        xx = np.arange(size)[None, :]
        foreground = xx <= (seed[:, None] + 1)  # 1 px outside the seed
        hair = ((rows_all >= 300) & (rows_all <= 314)) | (
            (rows_all >= 330) & (rows_all <= 390)
        )
        foreground[hair] |= xx <= 600  # no mask edge within the window there
        alpha = np.where(foreground, 255, 0).astype(np.uint8)
        gray = np.where(xx < seed[:, None] + 8, 180, 60).astype(np.uint8)
        xs, rows, src = co.trace_far_cheek(foreground, alpha, lm, True, gray=gray)
        by_row = {int(y): i for i, y in enumerate(rows)}
        for y in range(330, 391):
            i = by_row[y]
            self.assertEqual(src[i], co.OUTLINE_SRC_EDGE, y)
            self.assertLess(abs(float(xs[i]) - (seed[y] + 8)), 1.0)
        # Off above a plain row ratio of 0.95 (rows above 315).
        for y in range(300, 315):
            self.assertEqual(src[by_row[y]], co.OUTLINE_SRC_CONTOUR, y)
        for y in list(range(315, 330)) + list(range(391, 600)):
            self.assertEqual(src[by_row[y]], co.OUTLINE_SRC_MASK, y)
        # Without the gray image those rows keep the seed.
        _, _, src_plain = co.trace_far_cheek(foreground, alpha, lm, True)
        for y in range(330, 391):
            self.assertEqual(src_plain[by_row[y]], co.OUTLINE_SRC_CONTOUR, y)
        # A weak step (below EDGE_GRAD_MIN, 2.5 times the step) is not an edge.
        faint = np.where(xx < seed[:, None] + 8, 100, 90).astype(np.uint8)
        _, _, src_faint = co.trace_far_cheek(foreground, alpha, lm, True, gray=faint)
        np.testing.assert_array_equal(src_faint, src_plain)
        # observed_points() takes the EDGE rows' x but counts only mask rows.
        found = co.observed_points(lm, xs, rows, src, True)
        band_rows = found.outline[:, 1]
        self.assertEqual(band_rows[0], 300)
        self.assertEqual(band_rows[-1], 472)
        self.assertAlmostEqual(found.mask_fraction, 97 / 173, places=9)
        self.assertEqual(found.window_mask, 0.0)
        edge_rows = found.outline_src == co.OUTLINE_SRC_EDGE
        self.assertEqual(int(edge_rows.sum()), 61)
        edge_seed = seed[found.outline[edge_rows, 1].astype(int)]
        self.assertTrue(np.all(found.outline[edge_rows, 0] > edge_seed + 6))
        # No mask row in the window: the prior stays at Kaptein's height, on
        # an edge row (the nearest mask row, 391, is 0.1 H below), apex.ok's
        # window gate fails and no depth is read below a stand-in row, though
        # 82 of the 111 rows from the apex down came from the mask.
        self.assertTrue(found.apex_prior)
        self.assertEqual(found.apex[1], 362)
        self.assertEqual(found.outline_src[found.outline[:, 1] == 362][0], 3)
        self.assertAlmostEqual(found.apex_h, co.APEX_H_PRIOR, delta=1.0 / BASE_H)
        self.assertAlmostEqual(found.below_mask, 82 / 111, places=9)
        self.assertTrue(math.isnan(found.submalar_depth))
        self.assertTrue(math.isnan(found.submalar_present))
        self.assertTrue(np.all(np.isnan(found.submalar)))

    def test_gray_image_is_the_fixed_point_luma(self):
        # The library's own conversion, so both ports read the same edge rows:
        # OpenCV's integer BT.601 weights over 2 ** 14, rounded. OpenCV's own
        # result is within one level of it (its rounding differs between its
        # versions and code paths, which is why the library defines its own).
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
        gray = co.gray_image(rgb)
        self.assertEqual(gray.shape, (37, 53))
        self.assertEqual(gray.dtype, np.uint8)
        r, g, b = (rgb[..., k].astype(np.int64) for k in range(3))
        expected = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14
        np.testing.assert_array_equal(gray, expected.astype(np.uint8))
        self.assertEqual(
            int(co.gray_image(np.full((2, 3, 3), 255, np.uint8)).min()), 255
        )
        self.assertEqual(int(co.gray_image(np.zeros((2, 3, 3), np.uint8)).max()), 0)
        from_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.assertLessEqual(
            int(np.abs(gray.astype(int) - from_cv.astype(int)).max()), 1
        )
        # The render the tracer runs on reads the same through both.
        render = np.asarray(
            Image.open(os.path.join(RENDERS_DIR, "yaw45.png")).convert("RGB")
        )
        np.testing.assert_array_equal(
            co.gray_image(render), cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
        )

    def test_nose_band_bounded_by_tip_level_on_fixture_landmarks(self):
        # The nose rows run from the subnasale's level to the tip's (54), not
        # from the lower dorsum point 53. SPIGA's tip h on these quarter fixtures
        # is 0.687 to 0.707 and landmark 53's about 0.80 (0.796 to 0.808), so
        # the bound is the window's lower edge, 0.71: the property that matters
        # is that the nose band never reaches the apex window, not the tip's
        # exact h.
        with open(LANDMARKS_JSON, encoding="utf-8") as f:
            images = json.load(f)["images"]
        for name in ("yaw40", "yaw42", "yaw45", "yaw48", "yaw50", "yaw-45"):
            with self.subTest(name):
                entry = images[name]
                lm = np.array(entry["raw"]["landmarks"], dtype=np.float64)
                far_left = entry["cues"]["nose_offset"] > 0
                frame = co.oblique_frame(lm)
                chin, canthus = lm[16], lm[co.FAR_CANTHUS[far_left]]
                h_tip, _ = co.height_ratio(lm[co.NOSE_ROWS[0]], chin, canthus, frame)
                h_sub, _ = co.height_ratio(lm[co.NOSE_ROWS[1]], chin, canthus, frame)
                h_53, _ = co.height_ratio(lm[53], chin, canthus, frame)
                self.assertLess(h_sub, h_tip)
                self.assertLess(h_tip, co.APEX_H_WINDOW[0])
                self.assertGreater(h_53, 0.78)


class TestSearch(unittest.TestCase):
    """observed_points() on synthetic outlines of the synthetic face."""

    def setUp(self):
        self.lm = _base_landmarks()

    def find(self, bumps=(), lm=None, changes=()):
        """
        observed_points() on the synthetic outline with these bumps, after each
        change (first row, last row, x or None to keep it, source) is applied.
        """
        xs, rows, src = _outline(bumps)
        for first, last, x, source in changes:
            sel = (rows >= first) & (rows <= last)
            if x is not None:
                xs[sel] = x
            src[sel] = source
        found = co.observed_points(self.lm if lm is None else lm, xs, rows, src, True)
        self.assertIsNotNone(found)
        return found

    def test_bulge_found_within_two_rows(self):
        found = self.find([(0.78, 0.03, 20)])
        self.assertFalse(found.apex_prior)
        self.assertLessEqual(abs(found.apex[1] - _row_of(0.78)), 2.0)
        self.assertAlmostEqual(found.apex_h, 0.78, delta=0.01)
        self.assertGreater(found.apex_dev, 0.025)
        self.assertLess(found.apex_dev, 0.031)
        self.assertAlmostEqual(found.apex[0], BASE_X + 9.0, delta=0.5)
        self.assertFalse(found.apex_near_nose)
        self.assertFalse(found.nose_on)
        self.assertAlmostEqual(found.scale, BASE_H, places=9)
        self.assertEqual(found.mask_fraction, 1.0)
        self.assertEqual(found.window_mask, 1.0)
        self.assertEqual(found.below_mask, 1.0)
        # The band runs from the canthus level to the stomion level.
        self.assertEqual(found.outline[0, 1], 300)
        self.assertEqual(found.outline[-1, 1], 472)
        np.testing.assert_allclose(found.chord[0], [BASE_X, 300])
        np.testing.assert_allclose(found.chord[1], [BASE_X, 472])
        self.assertEqual(set(found.levels), {"canthus", "stomion", "chin"})

    def test_straight_outline_gives_prior(self):
        found = self.find()
        self.assertTrue(found.apex_prior)
        self.assertAlmostEqual(found.apex_h, co.APEX_H_PRIOR, delta=0.5 / BASE_H)
        self.assertEqual(found.apex[1], 362)
        self.assertAlmostEqual(found.apex_dev, 0.0, places=9)

    def test_bulge_outside_window_gives_prior(self):
        found = self.find([(0.60, 0.03, 20)])
        self.assertTrue(found.apex_prior)
        self.assertAlmostEqual(found.apex_h, co.APEX_H_PRIOR, delta=0.5 / BASE_H)

    def test_small_bulge_gives_prior(self):
        found = self.find([(0.78, 0.005, 20)])
        self.assertTrue(found.apex_prior)
        self.assertAlmostEqual(found.apex_h, co.APEX_H_PRIOR, delta=0.5 / BASE_H)

    def test_two_bulges_pick_nearer_prior(self):
        found = self.find([(0.74, 0.03, 10), (0.81, 0.03, 10)])
        self.assertFalse(found.apex_prior)
        self.assertAlmostEqual(found.apex_h, 0.81, delta=0.01)
        found = self.find([(0.76, 0.03, 10), (0.83, 0.03, 10)])
        self.assertFalse(found.apex_prior)
        self.assertAlmostEqual(found.apex_h, 0.76, delta=0.01)

    def test_step_in_smoothing_halo_is_not_a_candidate(self):
        # A hump whose maximum lies below the window, rising through it, with
        # the nose rows set back 12 px: the box filter's drop at the step makes
        # the last unaffected row a local maximum. It sits SMOOTH_ROWS // 2 + 1
        # rows above the first nose row, so the exclusion covers it.
        nose = (372, 384)
        found = self.find(
            [(0.62, 0.04, 75)], changes=[(*nose, BASE_X - 12.0, co.OUTLINE_SRC_NOSE)]
        )
        self.assertTrue(found.nose_on)
        self.assertTrue(found.apex_prior)
        self.assertEqual(found.apex[1], 362)
        # The same hump without the step has no candidate either, so the step
        # alone would have made one.
        plain = self.find([(0.62, 0.04, 75)])
        self.assertTrue(plain.apex_prior)
        rows = np.arange(290, 601)
        sel = (rows >= nose[0]) & (rows <= nose[1])
        xs, _, _ = _outline([(0.62, 0.04, 75)])
        xs[sel] = BASE_X - 12.0
        band = (rows >= 300) & (rows <= 472)
        dev = co._smooth((xs[band] - BASE_X).astype(np.float64)) / BASE_H
        b_rows = rows[band]
        i = int(np.flatnonzero(b_rows == nose[0] - co.SMOOTH_ROWS // 2 - 1)[0])
        self.assertGreater(dev[i], dev[i - 1])
        self.assertGreater(dev[i], dev[i + 1])
        self.assertGreater(dev[i], co.APEX_MIN_DEV)

    def test_step_at_contour_or_edge_rows_is_not_a_candidate(self):
        # The same hump and the same 12 px set-back as above, on rows the
        # contour or the intensity edge gave instead of the nose: a step
        # between sources is the same step whatever the source, so the shoulder
        # it makes is never a candidate, and the prior stands.
        for source in (co.OUTLINE_SRC_CONTOUR, co.OUTLINE_SRC_EDGE):
            with self.subTest(source=source):
                found = self.find(
                    [(0.62, 0.04, 75)],
                    changes=[(372, 384, BASE_X - 12.0, source)],
                )
                self.assertFalse(found.nose_on)
                self.assertTrue(found.apex_prior)
                self.assertEqual(found.apex[1], 362)
                self.assertLess(found.window_mask, 1.0)
        # The prior never lands on such a row while the window has a mask row:
        # it takes the mask row nearest h = 0.793 instead (365, h 0.7833,
        # against 359's 0.8033), so its x is the mask's, not the stand-in's 2 px
        # set-back; a nose row keeps the prior out of its halo as well.
        for source in (co.OUTLINE_SRC_CONTOUR, co.OUTLINE_SRC_EDGE):
            with self.subTest(prior_off=source):
                found = self.find(changes=[(360, 364, BASE_X - 2.0, source)])
                self.assertTrue(found.apex_prior)
                self.assertEqual(found.apex[1], 365)
                self.assertAlmostEqual(found.apex[0], BASE_X, places=4)
                self.assertEqual(found.apex_h, (BASE_CHIN_ROW - 365) / BASE_H)
                self.assertTrue(math.isfinite(found.submalar_depth))
        found = self.find(changes=[(360, 364, BASE_X - 2.0, co.OUTLINE_SRC_NOSE)])
        self.assertTrue(found.apex_prior)
        self.assertGreater(abs(found.apex[1] - 362), co.SMOOTH_ROWS // 2 + 2)

    def test_submalar_is_never_a_chord_end(self):
        # A convex outline: a broad bulge below the window, so the apex is the
        # prior and every row between it and the stomion level bows outward
        # from the apex-to-stomion chord. The chord's ends deviate by nothing
        # but the filter's edge padding, so the deepest point could be the apex
        # itself; it is a row between the ends, with no hollow present.
        found = self.find([(0.60, 0.02, 60)])
        self.assertTrue(found.apex_prior)
        self.assertEqual(found.apex[1], 362)
        last_row = found.outline[-1, 1]
        self.assertEqual(last_row, 472)
        self.assertTrue(362 < found.submalar[1] < 472, found.submalar)
        self.assertEqual(found.submalar_present, 0.0)
        self.assertLessEqual(found.submalar_depth, 1e-6)
        # And on a straight outline, where every row deviates by nothing.
        found = self.find()
        self.assertTrue(362 < found.submalar[1] < 472, found.submalar)

    def test_notch_present_with_depth(self):
        found = self.find([(0.55, -0.01, 30)])
        self.assertTrue(found.apex_prior)
        self.assertAlmostEqual(found.submalar_depth, 0.01, delta=0.001)
        self.assertEqual(found.submalar_present, 1.0)
        self.assertLessEqual(abs(found.submalar[1] - _row_of(0.55)), 2.0)
        self.assertAlmostEqual(found.submalar_h, 0.55, delta=0.01)
        self.assertEqual(found.submalar_depth_px, found.submalar_depth * found.scale)
        self.assertAlmostEqual(found.submalar_depth_px, 3.0, delta=0.3)

    def test_no_notch_gives_zero_with_point(self):
        found = self.find()
        self.assertAlmostEqual(found.submalar_depth, 0.0, places=9)
        self.assertAlmostEqual(found.submalar_depth_px, 0.0, places=6)
        self.assertEqual(found.submalar_present, 0.0)
        self.assertTrue(np.all(np.isfinite(found.submalar)))
        self.assertAlmostEqual(found.submalar[0], BASE_X, places=4)
        self.assertTrue(362 <= found.submalar[1] <= 472)

    def test_notch_inside_nose_rows_skipped(self):
        found = self.find(
            [(0.643, -0.03, 6)], changes=[(400, 414, None, co.OUTLINE_SRC_NOSE)]
        )
        self.assertTrue(found.nose_on)
        self.assertGreaterEqual(found.below_mask, co.OUTLINE_MIN_MASK)
        self.assertAlmostEqual(found.submalar_depth, 0.0, places=9)
        self.assertEqual(found.submalar_present, 0.0)
        self.assertFalse(395 <= found.submalar[1] <= 419)

    def test_too_few_mask_rows_give_nan_depth(self):
        found = self.find(
            [(0.55, -0.01, 30)], changes=[(430, 472, None, co.OUTLINE_SRC_CONTOUR)]
        )
        self.assertLess(found.below_mask, co.OUTLINE_MIN_MASK)
        self.assertTrue(math.isnan(found.submalar_depth))
        self.assertTrue(math.isnan(found.submalar_present))
        self.assertTrue(np.all(np.isnan(found.submalar)))

    def test_contour_fallback_gives_nan_depth(self):
        found = co.observed_points(self.lm, None, None, None, True)
        self.assertIsNotNone(found)
        self.assertTrue(np.all(found.outline_src == co.OUTLINE_SRC_CONTOUR))
        self.assertEqual(found.mask_fraction, 0.0)
        self.assertEqual(found.window_mask, 0.0)
        self.assertEqual(found.below_mask, 0.0)
        self.assertTrue(math.isnan(found.submalar_depth))
        self.assertTrue(math.isnan(found.submalar_depth_px))
        self.assertTrue(math.isnan(found.submalar_present))
        self.assertTrue(np.all(np.isnan(found.submalar)))
        self.assertTrue(math.isnan(found.submalar_h))
        self.assertTrue(np.all(np.isfinite(found.apex)))
        self.assertTrue(found.apex_prior)
        self.assertFalse(found.nose_on)

    def test_short_band_gives_none(self):
        xs, rows, src = _outline(first=300, last=301)
        self.assertIsNone(co.observed_points(self.lm, xs, rows, src, True))
        xs, rows, src = _outline(first=300, last=302)
        self.assertIsNotNone(co.observed_points(self.lm, xs, rows, src, True))

    def test_commissures_moved_change_nothing(self):
        plain = self.find([(0.78, 0.03, 20)])
        moved = self.lm.copy()
        moved[76] += (0, 40)
        moved[82] += (0, 40)
        found = self.find([(0.78, 0.03, 20)], lm=moved)
        self.assertEqual(found.apex_h, plain.apex_h)
        np.testing.assert_array_equal(found.apex, plain.apex)
        self.assertEqual(found.submalar_depth, plain.submalar_depth)

    def test_alar_bases_moved_tilt_h(self):
        plain = self.find([(0.78, 0.03, 20)])
        tilted = self.lm.copy()
        tilted[59] = (440, 390)  # V at (1160, 300): the levels fan out
        found = self.find([(0.78, 0.03, 20)], lm=tilted)
        self.assertIsNotNone(co.oblique_frame(tilted).vanishing)
        self.assertFalse(found.apex_prior)
        self.assertGreater(abs(found.apex_h - plain.apex_h), 0.02)
        self.assertLess(found.scale, BASE_H)

    def test_nose_tip_on_outline_sets_nose_on(self):
        on = self.lm.copy()
        on[54] = (BASE_X, 394)
        found = self.find(lm=on)
        self.assertLessEqual(found.nose_gap, 0.0)
        self.assertTrue(found.nose_on)
        outside = self.lm.copy()
        outside[54] = (BASE_X + 5, 394)
        found = self.find(lm=outside)
        self.assertLess(found.nose_gap, 0.0)
        self.assertTrue(found.nose_on)
        inside = self.lm.copy()
        inside[54] = (BASE_X - 0.048 * BASE_H, 394)
        found = self.find(lm=inside)
        self.assertAlmostEqual(found.nose_gap, 0.048, places=6)
        self.assertFalse(found.nose_on)
        # Nose rows set the flag whatever the gap.
        found = self.find(lm=inside, changes=[(400, 410, None, co.OUTLINE_SRC_NOSE)])
        self.assertTrue(found.nose_on)

    def test_candidate_near_nose_tip_flagged(self):
        high = self.lm.copy()
        high[54] = (480, 384)  # the tip at h 0.72
        found = self.find([(0.73, 0.03, 10)], lm=high)
        self.assertFalse(found.apex_prior)
        self.assertTrue(found.apex_near_nose)
        found = self.find([(0.78, 0.03, 10)], lm=high)
        self.assertFalse(found.apex_near_nose)


class TestRenders(unittest.TestCase):
    """
    The synthetic-head renders: the mask from the background color, as
    cheek_renders.json's header defines it, the raw SPIGA landmarks of
    cheek_landmarks.json in the same 800 px frame.
    """

    @classmethod
    def setUpClass(cls):
        with open(LANDMARKS_JSON, encoding="utf-8") as f:
            cls.images = json.load(f)["images"]
        with open(RENDERS_JSON, encoding="utf-8") as f:
            cls.renders = json.load(f)
        cls.background = np.array(cls.renders["background"], dtype=np.uint8)

    def load(self, name):
        """The render's entry, its mask from the background color, its raw
        landmarks and far_left."""
        render = self.renders["renders"][name]
        path = os.path.join(RENDERS_DIR, render["file"])
        image = np.asarray(Image.open(path).convert("RGB"))
        foreground = np.any(image != self.background, axis=2)
        lm = np.array(self.images[name]["raw"]["landmarks"], dtype=np.float64)
        return render, foreground, lm, render["far_side"] == "left"

    def analyze(self, name):
        render, foreground, lm, far_left = self.load(name)
        alpha = np.where(foreground, 255, 0).astype(np.uint8)
        xs, rows, src = co.trace_far_cheek(foreground, alpha, lm, far_left)
        found = co.observed_points(lm, xs, rows, src, far_left)
        return render, foreground, far_left, (xs, rows, src), found

    def silhouette(self, foreground, far_left, y):
        cols = np.flatnonzero(foreground[y])
        return int(cols.max() if far_left else cols.min())

    def test_renders(self):
        depths = {}
        for name in ("yaw45", "yaw-45", "yaw45_hollow3"):
            with self.subTest(name):
                render, fg, far_left, (xs, rows, src), found = self.analyze(name)
                self.assertIsNotNone(found)
                mask_rows = src == co.OUTLINE_SRC_MASK
                self.assertGreater(mask_rows.mean(), 0.9)
                for x, y in zip(xs[mask_rows], rows[mask_rows]):
                    self.assertLess(abs(float(x) - self.silhouette(fg, far_left, y)), 2)
                self.assertFalse(np.any(src == co.OUTLINE_SRC_NOSE))
                self.assertFalse(found.nose_on)
                self.assertGreater(found.nose_gap, 0.0)
                self.assertEqual(found.mask_fraction, 1.0)
                oracle = render["apex"]
                self.assertLess(abs(found.apex[0] - oracle["point"][0]), 1.0)
                apex_row = int(found.apex[1])
                self.assertLess(
                    abs(found.apex[0] - self.silhouette(fg, far_left, apex_row)), 1.0
                )
                if found.apex_prior:
                    # No local maximum inside the window: SPIGA's chin on the
                    # +45 render sits 25 px above the render's, which puts the
                    # silhouette's flat plateau (rows 364 to 388) at h 0.61 to
                    # 0.69 in SPIGA's frame, below the window's 0.71. The window
                    # excludes it, not the tracer: those are mask rows within
                    # 2 px of the silhouette, as asserted above.
                    self.assertEqual(name, "yaw45")
                    self.assertAlmostEqual(
                        found.apex_h, co.APEX_H_PRIOR, delta=1.0 / found.scale
                    )
                else:
                    first, last = oracle["plateau_rows"]
                    self.assertTrue(first <= apex_row <= last, (name, apex_row))
                self.assertGreater(found.apex_dev, co.APEX_MIN_DEV)
                self.assertTrue(math.isfinite(found.submalar_depth))
                depths[name] = found.submalar_depth
                if name == "yaw45_hollow3":
                    self.assertEqual(found.submalar_present, 1.0)
                    self.assertAlmostEqual(
                        found.submalar_depth_px,
                        render["submalar"]["depth_px"],
                        delta=0.3 * render["submalar"]["depth_px"],
                    )
                else:
                    self.assertEqual(found.submalar_present, 0.0)
        self.assertGreater(depths["yaw45_hollow3"], depths["yaw45"])
        self.assertGreater(depths["yaw45_hollow3"], 2 * depths["yaw-45"])

    def test_contour_dropout_in_the_window_keeps_the_prior(self):
        """
        Five of the apex window's 34 rows on the yaw45 render (334 to 338) set
        back to the contour seed, 1.4 to 1.8 px inside the mask's edge, as on a
        photo where the mask drops a few rows to hair: a step whose smoothed
        upper shoulder (row 329) is a local maximum of 0.012 H, which read as an
        observed apex at h 0.819 (prior 0, window_mask 0.853 above
        OUTLINE_MIN_MASK, so apex.ok 1) before the halo around every row the
        mask didn't give was excluded. The apex stays the prior, on the mask row
        nearest h = 0.793 outside the dropout (339, h 0.783; the untouched
        outline's row, 336, is a dropout row now) with the mask's own x there,
        and the depth stays the untouched outline's; an intensity-edge run the
        same.
        """
        _, _, far_left, (xs, rows, src), plain = self.analyze("yaw45")
        self.assertTrue(plain.apex_prior)
        self.assertAlmostEqual(plain.apex_h, co.APEX_H_PRIOR, delta=0.002)
        lm = np.array(self.images["yaw45"]["raw"]["landmarks"], dtype=np.float64)
        seed, seed_rows, _ = co.contour_outline(lm, far_left)
        np.testing.assert_array_equal(seed_rows, rows)
        out = 1.0 if far_left else -1.0
        sel = (rows >= 334) & (rows <= 338)
        self.assertEqual(int(sel.sum()), 5)
        step = (xs[sel] - seed[sel]) * out
        self.assertTrue(np.all((step > 1.0) & (step < 2.0)), step)
        for source in (co.OUTLINE_SRC_CONTOUR, co.OUTLINE_SRC_EDGE):
            with self.subTest(source=source):
                xs2, src2 = xs.copy(), src.copy()
                xs2[sel] = seed[sel]
                src2[sel] = source
                found = co.observed_points(lm, xs2, rows, src2, far_left)
                self.assertTrue(found.apex_prior)
                self.assertEqual(plain.apex[1], 336)
                self.assertEqual(found.apex[1], 339)
                self.assertAlmostEqual(found.apex_h, 0.7830, delta=0.001)
                self.assertAlmostEqual(found.apex_h, co.APEX_H_PRIOR, delta=0.02)
                self.assertEqual(src2[rows == 339][0], co.OUTLINE_SRC_MASK)
                self.assertEqual(float(found.apex[0]), float(xs[rows == 339][0]))
                self.assertGreaterEqual(found.window_mask, co.OUTLINE_MIN_MASK)
                self.assertLess(found.window_mask, 1.0)
                self.assertFalse(found.nose_on)
                self.assertTrue(math.isfinite(found.submalar_depth))
                self.assertAlmostEqual(
                    found.submalar_depth, plain.submalar_depth, delta=0.0002
                )
                self.assertEqual(found.submalar_present, plain.submalar_present)
                # The step did make a local maximum of the smoothed deviation
                # on the shoulder above it: only the exclusion keeps it out.
                P = found.outline
                a, b = found.chord
                normal = co._outward_normal(a, b, out)
                dev = co._smooth((P - a) @ normal) / found.scale
                i = int(np.flatnonzero(P[:, 1] == 334 - co.SMOOTH_ROWS // 2 - 1)[0])
                self.assertGreater(dev[i], dev[i - 1])
                self.assertGreaterEqual(dev[i], dev[i + 1])
                self.assertGreater(dev[i], co.APEX_MIN_DEV)

    def test_stand_in_rows_at_the_prior_never_carry_the_apex_or_the_chord(self):
        """
        The same five rows (334 to 338) as an intensity-edge run 20 px outside
        the seed (a hair strand within the walk's reach at Kaptein's height; 8
        px too) and as a contour run 10 px inside the mask (a photo's contour).
        Before the prior preferred the window's mask rows, the 20 px run carried
        the apex 18.4 px outward on its own row, 336, with apex.ok 1 (window_mask
        0.853), and tilted the apex-to-stomion chord so that every row below
        read as a hollow: depth 0.058 H, present 1, sub.ok 1; the contour run
        would tilt it the other way over a real hollow. Now the apex is the
        mask row 339 with the mask's own x, and the depth is the plain
        outline's, 0.0005 H, absent.
        """
        _, _, far_left, (xs, rows, src), plain = self.analyze("yaw45")
        lm = np.array(self.images["yaw45"]["raw"]["landmarks"], dtype=np.float64)
        seed, _, _ = co.contour_outline(lm, far_left)
        out = 1.0 if far_left else -1.0
        sel = (rows >= 334) & (rows <= 338)
        self.assertEqual(plain.apex[1], 336)
        self.assertEqual(plain.submalar_present, 0.0)
        cases = (
            (co.OUTLINE_SRC_EDGE, 20.0),
            (co.OUTLINE_SRC_EDGE, 8.0),
            (co.OUTLINE_SRC_CONTOUR, -10.0),
        )
        for source, shift in cases:
            with self.subTest(source=source, shift=shift):
                xs2, src2 = xs.copy(), src.copy()
                xs2[sel] = (seed[sel] + shift * out).astype(np.float32)
                src2[sel] = source
                found = co.observed_points(lm, xs2, rows, src2, far_left)
                self.assertTrue(found.apex_prior)
                self.assertEqual(found.apex[1], 339)
                self.assertEqual(float(found.apex[0]), float(xs[rows == 339][0]))
                self.assertLess(abs(found.apex[0] - plain.apex[0]), 0.5)
                self.assertGreaterEqual(found.window_mask, co.OUTLINE_MIN_MASK)
                self.assertEqual(found.below_mask, 1.0)
                self.assertAlmostEqual(
                    found.submalar_depth, plain.submalar_depth, delta=0.0002
                )
                self.assertEqual(found.submalar_present, 0.0)
                self.assertEqual(found.submalar[1], plain.submalar[1])
                # The stand-in rows themselves sit off the canthus-to-stomion
                # chord by most of the shift (the chord's tilt and the seed's 1
                # to 2 px inset absorb the rest): what the apex-to-stomion chord
                # read as a hollow when it started on one of them.
                a, b = found.chord
                normal = co._outward_normal(a, b, out)
                in_band = sel[np.isin(rows, found.outline[:, 1])]
                off = (found.outline[in_band] - a) @ normal
                self.assertGreater(float(np.abs(off).min()), 0.5 * abs(shift))

    def test_dilated_masks_read_the_offset_not_the_nose(self):
        """
        The committed masks dilated outward on the far side by 1 to 4 px, the
        alpha with them so the offset is stored (1 to 4 px, 0.0033 to 0.0144
        of H, which is 277 to 301 px on these raw renders): a uniform
        offset is not the nose on the 45s at any dilation, and the 50's nose
        rows (26) stay at every one. With the alpha left as it was, the
        refinement puts the traced x back on the alpha's crossing for a shift
        of up to ALPHA_SEARCH px and on the dilated mask's pixel boundary past
        it.
        """
        cases = (
            ("yaw45", False),
            ("yaw-45", False),
            ("yaw45_hollow3", False),
            ("yaw50", True),
        )
        for name, want_nose in cases:
            _, fg0, lm, far_left = self.load(name)
            out = 1 if far_left else -1
            alpha0 = np.where(fg0, 255, 0).astype(np.uint8)
            xs0, rows0, src0 = co.trace_far_cheek(fg0, alpha0, lm, far_left)
            self.assertEqual(bool(np.any(src0 == co.OUTLINE_SRC_NOSE)), want_nose)
            for k in range(1, 5):
                with self.subTest(name=name, dilation=k):
                    fg = fg0.copy()
                    for j in range(1, k + 1):  # the far edge only: the near one
                        fg |= np.roll(fg0, out * j, axis=1)  # keeps its pixels
                    alpha = np.where(fg, 255, 0).astype(np.uint8)
                    xs, rows, src = co.trace_far_cheek(fg, alpha, lm, far_left)
                    found = co.observed_points(lm, xs, rows, src, far_left)
                    nose = src == co.OUTLINE_SRC_NOSE
                    self.assertEqual(bool(nose.any()), want_nose)
                    self.assertEqual(found.nose_on, want_nose)
                    if want_nose:
                        self.assertEqual(int(nose.sum()), 26)
                    xs_a, rows_a, src_a = co.trace_far_cheek(fg, alpha0, lm, far_left)
                    np.testing.assert_array_equal(rows_a, rows0)
                    both = (src_a == co.OUTLINE_SRC_MASK) & (
                        src0 == co.OUTLINE_SRC_MASK
                    )
                    self.assertGreater(int(both.sum()), 240)
                    on_boundary = np.abs(xs_a - np.floor(xs_a) - 0.5) < 1e-6
                    if k <= co.ALPHA_SEARCH:
                        np.testing.assert_allclose(xs_a[both], xs0[both], atol=1e-5)
                    else:
                        self.assertTrue(np.all(on_boundary[both]))
                        self.assertGreater(np.abs(xs_a[both] - xs0[both]).min(), k - 1)

    def test_perspective_warp_on_render(self):
        """
        The second perspective case (the first is
        TestLevels.test_perspective_warp_exercises_vanishing_point): the yaw-45
        render's raw landmarks and traced outline through the keystone warp,
        the outline resampled on integer rows. Measured: apex h 0.7191 against
        0.7175 on the plain render, the apex within 0.2 px of the warped plain
        apex, and both within APEX_NEAR_NOSE_H of the tip (h 0.7069 plain).
        """
        _, _, far_left, (xs, rows, src), plain = self.analyze("yaw-45")
        self.assertFalse(far_left)
        self.assertFalse(plain.apex_prior)
        self.assertTrue(plain.apex_near_nose)
        lm = np.array(self.images["yaw-45"]["raw"]["landmarks"], dtype=np.float64)
        warped_lm = _warp(lm)
        frame = co.oblique_frame(warped_lm)
        self.assertIsNotNone(frame.vanishing)
        self.assertTrue(np.all(np.isfinite(frame.vanishing)))
        w_xs, w_rows, w_src = _warp_outline(xs, rows, src)
        found = co.observed_points(warped_lm, w_xs, w_rows, w_src, far_left)
        self.assertIsNotNone(found)
        self.assertFalse(found.apex_prior)
        self.assertLess(abs(found.apex_h - plain.apex_h), 0.01)
        np.testing.assert_allclose(found.apex, _warp(plain.apex), atol=2.0)
        self.assertTrue(found.apex_near_nose)
        self.assertEqual(found.mask_fraction, 1.0)
        self.assertLess(abs(found.submalar_h - plain.submalar_h), 0.01)
        self.assertLess(abs(found.submalar_depth - plain.submalar_depth), 0.001)

    def test_mirrored_turns_agree(self):
        _, _, _, _, plus = self.analyze("yaw45")
        _, _, _, _, minus = self.analyze("yaw-45")
        self.assertTrue(plus.far_left)
        self.assertFalse(minus.far_left)
        self.assertLess(abs(plus.nose_gap - minus.nose_gap), 0.01)
        self.assertLess(abs(plus.submalar_depth - minus.submalar_depth), 0.001)


def _rle_image(rows, size):
    """An image from its run-length encoded rows ([value, count, ...] each)."""
    height, width = size
    image = np.empty((height, width), dtype=np.uint8)
    for y, runs in enumerate(rows):
        row = np.repeat(runs[0::2], runs[1::2]).astype(np.uint8)
        if len(row) != width:
            raise ValueError(f"row {y} decodes to {len(row)} px, not {width}")
        image[y] = row
    return image


def _nan_for_null(value):
    """A stored number, where null stands for NaN; lists recursively."""
    if value is None:
        return math.nan
    if isinstance(value, list):
        return [_nan_for_null(v) for v in value]
    return value


class TestSharedOutlines(unittest.TestCase):
    """
    The cases in tests_data/cheek_outlines.json, which dynaface-lib-dotnet's
    CheekOutlineTests will also check once the cheek module is ported, so both
    ports must trace the same outline and find the same points: the tracer's
    outputs (the sub-pixel x, the rows and each row's source), the points, the
    flags and the fractions must equal the Python outputs frozen there, and
    the derived heights, deviations and levels must agree to 1e-9 (they come
    through matrix products, whose last bit may differ between builds).
    scripts/make_cheek_outlines.py writes the file; a port whose walk, median
    or edge search differs by one pixel fails here, where the render tests
    above would pass it within their pixel tolerances.
    """

    @classmethod
    def setUpClass(cls):
        with open(OUTLINES_JSON, encoding="utf-8") as f:
            cls.data = json.load(f)
        cls.background = np.array(cls.data["mask"]["background"], dtype=np.uint8)
        cls.mask_alpha = int(cls.data["mask"]["mask_alpha"])

    def inputs(self, case):
        """The case's alpha, gray image (or None) and foreground mask."""
        if case["kind"] == "synthetic":
            alpha = _rle_image(case["alpha"], case["size"])
            gray = _rle_image(case["gray"], case["size"]) if case["gray"] else None
        else:
            image = np.asarray(
                Image.open(os.path.join(TESTS_DATA, case["file"])).convert("RGB")
            )
            alpha = np.where(np.any(image != self.background, axis=2), 255, 0)
            alpha = alpha.astype(np.uint8)
            gray = None
        self.assertEqual(self.mask_alpha, co.MASK_ALPHA)
        return alpha, gray, alpha > self.mask_alpha

    def assert_point(self, actual, expected):
        np.testing.assert_array_equal(
            np.asarray(actual, dtype=np.float64),
            np.asarray(_nan_for_null(expected), dtype=np.float64),
        )

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(_nan_for_null(expected), dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        )

    def test_cases(self):
        names = [case["name"] for case in self.data["cases"]]
        self.assertEqual(
            names,
            [
                "soft_disc",
                "nose_disc",
                "hair_disc",
                "yaw45",
                "yaw-45",
                "yaw45_hollow3",
                "yaw50",
            ],
        )
        for case in self.data["cases"]:
            with self.subTest(case["name"]):
                alpha, gray, foreground = self.inputs(case)
                lm = np.array(case["landmarks"], dtype=np.float64)
                far_left = case["far_left"]
                xs, rows, src = co.trace_far_cheek(
                    foreground, alpha, lm, far_left, gray=gray
                )
                expected = case["outline"]
                self.assertEqual(xs.dtype, np.float32)
                np.testing.assert_array_equal(
                    xs, np.array(expected["x"], dtype=np.float32)
                )
                self.assertEqual(rows.tolist(), expected["rows"])
                self.assertEqual(src.tolist(), expected["src"])
                found = co.observed_points(lm, xs, rows, src, far_left)
                self.assertIsNotNone(found)
                ob = case["observed"]
                self.assertEqual(found.far_left, ob["far_left"])
                self.assert_point(found.outline, ob["outline"])
                self.assertEqual(found.outline_src.tolist(), ob["outline_src"])
                self.assert_point(found.chord, ob["chord"])
                self.assertEqual(set(found.levels), set(ob["levels"]))
                for key, level in found.levels.items():
                    self.assert_close(level, ob["levels"][key])
                self.assert_point(found.apex, ob["apex"])
                self.assert_point(found.submalar, ob["submalar"])
                for name in ("apex_prior", "apex_near_nose", "nose_on"):
                    self.assertIs(getattr(found, name), ob[name], name)
                for name in (
                    "submalar_present",
                    "mask_fraction",
                    "window_mask",
                    "below_mask",
                ):
                    self.assert_point([getattr(found, name)], [ob[name]])
                for name in (
                    "apex_h",
                    "apex_dev",
                    "submalar_h",
                    "submalar_depth",
                    "submalar_depth_px",
                    "nose_gap",
                    "scale",
                ):
                    self.assert_close([getattr(found, name)], [ob[name]])

    def test_sources_exercised(self):
        """The synthetic cases cover every row source and the renders both far
        sides, so a port is checked on all of them."""
        sources = {}
        sides = set()
        for case in self.data["cases"]:
            sources[case["name"]] = set(case["outline"]["src"])
            sides.add(case["far_left"])
        self.assertEqual(sides, {True, False})
        self.assertEqual(
            sources["soft_disc"], {co.OUTLINE_SRC_CONTOUR, co.OUTLINE_SRC_MASK}
        )
        self.assertIn(co.OUTLINE_SRC_NOSE, sources["nose_disc"])
        self.assertIn(co.OUTLINE_SRC_EDGE, sources["hair_disc"])
        self.assertIn(co.OUTLINE_SRC_NOSE, sources["yaw50"])
        present = [c["observed"]["submalar_present"] for c in self.data["cases"]]
        self.assertIn(1.0, present)
        self.assertIn(0.0, present)
        self.assertIn(None, present)

    def test_frontal_constructions(self):
        for case in self.data["frontal"]:
            with self.subTest(case["name"]):
                lm = np.array(case["landmarks"], dtype=np.float64)
                fc = cheek.frontal_constructions(lm)
                self.assertIsNotNone(fc)
                expected = case["constructions"]
                frame = expected["frame"]
                self.assert_close(fc.frame.origin, frame["origin"])
                self.assert_close(fc.frame.across, frame["across"])
                self.assert_close(fc.frame.down, frame["down"])
                self.assert_close([fc.frame.ipd], [frame["ipd"]])
                for name, value in expected.items():
                    if name == "frame":
                        continue
                    actual = getattr(fc, name)
                    if name.endswith("_clip"):
                        self.assertEqual(actual, value, name)
                    else:
                        self.assert_close(np.atleast_1d(actual), np.atleast_1d(value))


if __name__ == "__main__":
    unittest.main()
