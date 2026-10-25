"""
Undefined geometry must come back as NaN, never as a plausible-looking number:
lateral angles and tip projection when a landmark is missing or a side has zero
length, and the mm scale when the pupils coincide. Mirrored by
dynaface-lib-dotnet's DegenerateGeometryTests.
"""

import math
import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import facial  # noqa: E402
from dynaface.const import Pose  # noqa: E402
from dynaface.measures_frontal import AnalyzePosition  # noqa: E402
from dynaface.measures_lateral import AnalyzeLateral  # noqa: E402

# A missing lateral landmark: (-1, -1) with x shifted into image space (shift 100).
MISSING = (99, -1)

# Glabella, nasion, nasal tip, subnasal, mentolabial, pogonion. NFA at the nasion is
# 135 degrees, NLA at the subnasal point arccos(-2/sqrt(5)) = 153.435 degrees, and the
# tip-projection triangle has AT = NT = 50.
PROFILE = [(300, 300), (300, 350), (350, 400), (320, 460), (310, 540), (320, 600)]
NLA = math.degrees(math.acos(-2 / math.sqrt(5)))


def lateral(points):
    face = facial.AnalyzeFace(measures=[])
    face.lateral = True
    face.lateral_landmarks = np.array(points, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero RuntimeWarnings either
        return AnalyzeLateral().calc(face, render=False)


def with_points(**changes):
    names = ["glabella", "nasion", "tip", "subnasal", "mentolabial", "pogonion"]
    points = list(PROFILE)
    for name, value in changes.items():
        points[names.index(name)] = value
    return points


class TestLateralDegenerateGeometry(unittest.TestCase):
    def test_complete_profile(self):
        r = lateral(PROFILE)
        self.assertAlmostEqual(r["nfa"], 135.0, places=4)
        self.assertAlmostEqual(r["nla"], NLA, places=4)
        self.assertAlmostEqual(r["at"], 50.0)
        self.assertAlmostEqual(r["nt"], 50.0)
        self.assertAlmostEqual(r["at_nt"], 1.0)

    def test_missing_glabella_only_affects_nfa(self):
        r = lateral(with_points(glabella=MISSING))
        self.assertTrue(math.isnan(r["nfa"]))
        self.assertAlmostEqual(r["nla"], NLA, places=4)
        self.assertAlmostEqual(r["at_nt"], 1.0)

    def test_missing_nasal_tip_makes_everything_using_it_nan(self):
        r = lateral(with_points(tip=MISSING))
        for key in ("nfa", "nla", "at", "nt", "at_nt"):
            self.assertTrue(math.isnan(r[key]), key)

    def test_zero_length_side_is_nan(self):
        r = lateral(with_points(glabella=PROFILE[1]))  # glabella on the nasion
        self.assertTrue(math.isnan(r["nfa"]))

    def test_two_missing_points_are_nan_not_zero(self):
        r = lateral(with_points(glabella=MISSING, nasion=MISSING))
        self.assertTrue(math.isnan(r["nfa"]))
        self.assertTrue(math.isnan(r["at_nt"]))


class TestCoincidentPupils(unittest.TestCase):
    PUPILS = {96: (500, 480), 97: (500, 480)}

    def landmarks(self):
        lm = [(510, 600)] * 98
        for i, p in self.PUPILS.items():
            lm[i] = p
        return lm

    def test_calc_pd_gives_nan_scale(self):
        pd, pix2mm = facial.util_calc_pd(((500, 480), (500, 480)))
        self.assertEqual(pd, 0.0)
        self.assertTrue(math.isnan(pix2mm))

    def test_position_reports_nan_px2mm(self):
        face = facial.AnalyzeFace(measures=[])
        face.landmarks = self.landmarks()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = AnalyzePosition().calc(face, render=False)
        self.assertEqual(r["pd"], 0.0)
        self.assertTrue(math.isnan(r["px2mm"]))

    def test_uncropped_load_gives_nan_scale(self):
        face = facial.AnalyzeFace(measures=[])
        img = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        found = (self.landmarks(), np.zeros(6))
        with mock.patch.object(
            facial.AnalyzeFace, "_find_landmarks", return_value=found
        ):
            face.load_image(img, crop=False, force_pose=Pose.FRONTAL)
        self.assertTrue(math.isnan(face.pix2mm))


if __name__ == "__main__":
    unittest.main()
