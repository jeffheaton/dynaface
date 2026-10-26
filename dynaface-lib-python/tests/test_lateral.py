"""Tests for the side-profile analysis in dynaface.lateral."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

from dynaface import facial, lateral, models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROFILE = "./tests_data/img2-1024-right-lateral.jpg"


class TestLateral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models.init_models(models.download_models(), models.detect_device())

    def test_debug_plots(self):
        # With DEBUG on, analyze_lateral() also marks the outline's extrema,
        # corners and quarter lines on its chart. That changes the drawing, but
        # not the landmarks it finds.
        plain = facial.load_face_image(PROFILE, measures=[])
        with mock.patch.object(lateral, "DEBUG", True):
            debug = facial.load_face_image(PROFILE, measures=[])

        self.assertTrue(debug.lateral)
        np.testing.assert_array_equal(debug.lateral_landmarks, plain.lateral_landmarks)
        self.assertFalse(np.array_equal(debug.render_img, plain.render_img))
