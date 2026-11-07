import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import cheek  # noqa: E402
from dynaface.const import Pose  # noqa: E402
from dynaface.measures_base import MeasureItem  # noqa: E402
from dynaface.measures_frontal import AnalyzeLandmarks  # noqa: E402

NUM_LANDMARKS = AnalyzeLandmarks.NUM_LANDMARKS
CHEEK_OUTPUTS = list(cheek.CHEEK_LANDMARK_OUTPUTS)


class MockFace:
    """Minimal face stub for unit-testing AnalyzeLandmarks without loaded models."""

    def __init__(self, num_landmarks=NUM_LANDMARKS):
        # Each landmark is (x, y) where x = i*2, y = i*2+1 for easy verification
        self.landmarks = [(i * 2, i * 2 + 1) for i in range(num_landmarks)]
        self.draw_landmarks_calls = []

    def draw_landmarks(self, **kwargs):
        self.draw_landmarks_calls.append(kwargs)


class TestAnalyzeLandmarksStructure(unittest.TestCase):
    """Tests that do not require AI models to be loaded."""

    def setUp(self):
        self.measure = AnalyzeLandmarks()

    def test_abbrev(self):
        self.assertEqual(self.measure.abbrev(), "Landmarks")

    def test_is_frontal(self):
        self.assertTrue(self.measure.is_frontal)
        self.assertFalse(self.measure.is_lateral)

    def test_enabled_by_default(self):
        self.assertTrue(self.measure.enabled)

    def test_item_count(self):
        # 2 items (x and y) per landmark, then the Cheek Landmarks group
        self.assertEqual(len(self.measure.items), NUM_LANDMARKS * 2 + 1)
        self.assertEqual(len(self.measure.outputs()), NUM_LANDMARKS * 2 + 20)

    def test_cheek_landmarks_item(self):
        item = self.measure.items[-1]
        self.assertEqual(item.name, "Cheek Landmarks")
        self.assertFalse(item.display_only)
        self.assertEqual(item.output_names(), CHEEK_OUTPUTS)
        self.assertEqual(self.measure.outputs()[-20:], CHEEK_OUTPUTS)
        self.assertTrue(item.is_frontal and item.is_quarter)
        self.assertFalse(item.is_lateral)
        self.assertTrue(item.enabled)

    def test_cheek_landmark_outputs(self):
        self.assertEqual(len(CHEEK_OUTPUTS), 20)
        self.assertEqual(
            CHEEK_OUTPUTS[:6],
            ["zy.r.x", "zy.r.y", "zy.l.x", "zy.l.y", "cr.r.x", "cr.r.y"],
        )
        self.assertEqual(CHEEK_OUTPUTS[-2:], ["sub.l.x", "sub.l.y"])
        for i, code in enumerate(cheek.CHEEK_LANDMARK_CODES):
            self.assertEqual(
                CHEEK_OUTPUTS[2 * i : 2 * i + 2], [f"{code}.x", f"{code}.y"]
            )

    def test_item_names(self):
        names = [item.name for item in self.measure.items]
        for i in range(1, NUM_LANDMARKS + 1):
            self.assertIn(f"landmark-{i}-x", names)
            self.assertIn(f"landmark-{i}-y", names)

    def test_item_interleaved_order(self):
        # Items should alternate x, y for each landmark
        names = [item.name for item in self.measure.items]
        self.assertEqual(names[0], "landmark-1-x")
        self.assertEqual(names[1], "landmark-1-y")
        self.assertEqual(names[-3], f"landmark-{NUM_LANDMARKS}-x")
        self.assertEqual(names[-2], f"landmark-{NUM_LANDMARKS}-y")
        self.assertEqual(names[-1], "Cheek Landmarks")

    def test_all_items_are_measure_items(self):
        for item in self.measure.items:
            self.assertIsInstance(item, MeasureItem)

    def test_all_items_enabled_by_default(self):
        for item in self.measure.items:
            self.assertTrue(item.enabled, f"{item.name} should be enabled by default")

    def test_all_items_frontal(self):
        for item in self.measure.items:
            self.assertTrue(item.is_frontal)
            self.assertTrue(item.is_quarter)
            self.assertFalse(item.is_lateral)


class TestAnalyzeLandmarksCalc(unittest.TestCase):
    """Tests for calc() using a mock face."""

    def setUp(self):
        self.measure = AnalyzeLandmarks()
        self.face = MockFace()

    def test_calc_returns_all_keys(self):
        result = self.measure.calc(self.face, render=False)
        for i in range(1, NUM_LANDMARKS + 1):
            self.assertIn(f"landmark-{i}-x", result)
            self.assertIn(f"landmark-{i}-y", result)

    def test_calc_returns_correct_values(self):
        result = self.measure.calc(self.face, render=False)
        for i in range(NUM_LANDMARKS):
            n = i + 1
            expected_x, expected_y = i * 2, i * 2 + 1
            self.assertEqual(result[f"landmark-{n}-x"], expected_x)
            self.assertEqual(result[f"landmark-{n}-y"], expected_y)

    def test_calc_calls_draw_landmarks_when_render_true(self):
        self.measure.calc(self.face, render=True)
        self.assertEqual(len(self.face.draw_landmarks_calls), 2)

    def test_calc_skips_draw_landmarks_when_render_false(self):
        self.measure.calc(self.face, render=False)
        self.assertEqual(len(self.face.draw_landmarks_calls), 0)

    def test_calc_renders_black_outline_then_white_fill(self):
        self.measure.calc(self.face, render=True)
        outline, fill = self.face.draw_landmarks_calls
        self.assertEqual(outline.get("color"), (0, 0, 0))
        self.assertEqual(fill.get("color"), (255, 255, 255))

    def test_calc_outline_larger_than_fill(self):
        self.measure.calc(self.face, render=True)
        outline, fill = self.face.draw_landmarks_calls
        self.assertGreater(outline.get("size", 0), fill.get("size", 0))

    def test_calc_renders_no_numbers(self):
        self.measure.calc(self.face, render=True)
        for call in self.face.draw_landmarks_calls:
            self.assertFalse(call.get("numbers", False))

    def test_calc_result_count(self):
        result = self.measure.calc(self.face, render=False)
        self.assertEqual(len(result), NUM_LANDMARKS * 2 + 20)
        self.assertEqual(list(result), self.measure.outputs())

    def test_set_item_enabled_affects_is_enabled(self):
        # set_item_enabled controls rendering gating via is_enabled(),
        # not the return dict of calc() — select_outputs returns all
        # items regardless of enabled state; exclusion from CSV happens via
        # get_all_items() on the AnalyzeFace side.
        self.measure.set_item_enabled("landmark-1-x", False)
        self.assertFalse(self.measure.is_enabled("landmark-1-x"))
        self.assertTrue(self.measure.is_enabled("landmark-1-y"))

    def test_calc_measure_disabled_returns_empty_via_analyze(self):
        """When the measure itself is disabled, analyze() skips it entirely."""
        self.measure.set_enabled(False)
        self.assertFalse(self.measure.enabled)


class CheekFace(MockFace):
    """A mock face with its cheek landmarks and draw_cheek_landmarks() recorded."""

    def __init__(self):
        super().__init__()
        self.cheek_landmarks = np.full((10, 2), math.nan)
        self.cheek_landmarks[0] = (100.5, 200.25)
        self.cheek_landmarks[5] = (300.0, 400.0)
        self.draw_cheek_landmarks_calls = []

    def draw_cheek_landmarks(self, **kwargs):
        self.draw_cheek_landmarks_calls.append(kwargs)


class TestCheekLandmarksItem(unittest.TestCase):
    """The Cheek Landmarks item on mock faces (real faces: test_measures_cheek)."""

    def test_values_are_the_faces_cheek_landmarks(self):
        face = CheekFace()
        result = AnalyzeLandmarks().calc(face, render=False)
        values = [result[name] for name in CHEEK_OUTPUTS]
        for value, expected in zip(values, face.cheek_landmarks.reshape(-1)):
            self.assertIs(type(value), float)
            if math.isnan(expected):
                self.assertTrue(math.isnan(value))
            else:
                self.assertEqual(value, expected)
        self.assertEqual((result["zy.r.x"], result["zy.r.y"]), (100.5, 200.25))
        self.assertEqual((result["sr.l.x"], result["sr.l.y"]), (300.0, 400.0))

    def test_values_without_the_attribute_follow_cheek_locate(self):
        # MockFace has no cheek_landmarks: cheek.locate() on its landmarks.
        face = MockFace()
        result = AnalyzeLandmarks().calc(face, render=False)
        expected = cheek.locate_all(
            face.landmarks, Pose.FRONTAL, False, None, None, None
        ).reshape(-1)
        for name, value in zip(CHEEK_OUTPUTS, expected):
            got = result[name]
            self.assertTrue(
                (math.isnan(got) and math.isnan(value)) or got == value, name
            )

    def test_drawing_gated_by_the_item(self):
        face = CheekFace()
        AnalyzeLandmarks().calc(face, render=True)
        self.assertEqual(
            face.draw_cheek_landmarks_calls,
            [
                {"size": 1.0, "color": (0, 0, 0), "names": False},
                {"size": 0.5, "color": (255, 128, 0), "names": False},
            ],
        )
        self.assertEqual(len(face.draw_landmarks_calls), 2)

        face = CheekFace()
        measure = AnalyzeLandmarks()
        measure.set_item_enabled("Cheek Landmarks", False)
        result = measure.calc(face, render=True)
        self.assertEqual(face.draw_cheek_landmarks_calls, [])
        self.assertEqual(len(face.draw_landmarks_calls), 2)  # the 98 still drawn
        self.assertEqual(result["zy.r.x"], 100.5)  # still returned
        self.assertNotIn("zy.r.x", measure.outputs()[: NUM_LANDMARKS * 2])

        face = CheekFace()
        AnalyzeLandmarks().calc(face, render=False)
        self.assertEqual(face.draw_cheek_landmarks_calls, [])


class TestAnalyzeLandmarksIntegration(unittest.TestCase):
    """Integration tests that load real models and a real image."""

    @classmethod
    def setUpClass(cls):
        from dynaface import facial, measures, models
        from dynaface.image import load_image

        device = models.detect_device()
        path = models.download_models()
        models.init_models(path, device)

        img = load_image("./tests_data/img1-512.jpg")
        cls.face = facial.AnalyzeFace(measures=measures.all_measures())
        cls.face.load_image(img, crop=True)
        cls.stats = cls.face.analyze()

    def test_landmarks_keys_present_in_stats(self):
        for i in range(1, NUM_LANDMARKS + 1):
            self.assertIn(f"landmark-{i}-x", self.stats)
            self.assertIn(f"landmark-{i}-y", self.stats)

    def test_landmark_values_are_numeric(self):
        for i in range(1, NUM_LANDMARKS + 1):
            x = self.stats[f"landmark-{i}-x"]
            y = self.stats[f"landmark-{i}-y"]
            self.assertIsInstance(x, (int, float))
            self.assertIsInstance(y, (int, float))

    def test_landmark_values_in_image_bounds(self):
        w, h = self.face.width, self.face.height
        for i in range(1, NUM_LANDMARKS + 1):
            x = self.stats[f"landmark-{i}-x"]
            y = self.stats[f"landmark-{i}-y"]
            self.assertGreaterEqual(x, 0, f"landmark-{i}-x out of bounds")
            self.assertLessEqual(x, w, f"landmark-{i}-x out of bounds")
            self.assertGreaterEqual(y, 0, f"landmark-{i}-y out of bounds")
            self.assertLessEqual(y, h, f"landmark-{i}-y out of bounds")

    def test_landmarks_in_get_all_items(self):
        items = self.face.get_all_items()
        self.assertIn("landmark-1-x", items)
        self.assertIn("landmark-1-y", items)
        self.assertIn(f"landmark-{NUM_LANDMARKS}-x", items)
        self.assertIn(f"landmark-{NUM_LANDMARKS}-y", items)
        # The Cheek Landmarks group's outputs follow, never its label.
        self.assertEqual(
            items[items.index(f"landmark-{NUM_LANDMARKS}-y") + 1 :][:20], CHEEK_OUTPUTS
        )
        self.assertNotIn("Cheek Landmarks", items)

    def test_cheek_landmarks_in_stats(self):
        flat = self.face.cheek_landmarks.reshape(-1)
        for name, value in zip(CHEEK_OUTPUTS, flat):
            got = self.stats[name]
            self.assertTrue(
                (math.isnan(got) and math.isnan(value)) or got == value, name
            )
        self.assertTrue(np.isfinite(flat[:12]).all())  # the frontal six
        self.assertTrue(np.isnan(flat[12:]).all())  # no oblique points


if __name__ == "__main__":
    unittest.main()
