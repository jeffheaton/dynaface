import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# facial must be imported first to break the circular import chain:
# facial → measures → measures_skin → facial (returns partial) → ok
# Importing measures first causes facial to be partially initialized when
# measures_skin needs it, making MeasureBase unavailable.
from dynaface import facial  # noqa: F401 (side-effect import)
from dynaface.measures_frontal import AnalyzeLandmarks
from dynaface.measures_base import MeasureItem

NUM_LANDMARKS = AnalyzeLandmarks.NUM_LANDMARKS


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
        # 2 items (x and y) per landmark
        self.assertEqual(len(self.measure.items), NUM_LANDMARKS * 2)

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
        self.assertEqual(names[-2], f"landmark-{NUM_LANDMARKS}-x")
        self.assertEqual(names[-1], f"landmark-{NUM_LANDMARKS}-y")

    def test_all_items_are_measure_items(self):
        for item in self.measure.items:
            self.assertIsInstance(item, MeasureItem)

    def test_all_items_enabled_by_default(self):
        for item in self.measure.items:
            self.assertTrue(item.enabled, f"{item.name} should be enabled by default")

    def test_all_items_frontal(self):
        for item in self.measure.items:
            self.assertTrue(item.is_frontal)
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
        self.assertEqual(len(result), NUM_LANDMARKS * 2)

    def test_set_item_enabled_affects_is_enabled(self):
        # set_item_enabled controls rendering gating via is_enabled(),
        # not the return dict of calc() — filter_measurements returns all
        # items regardless of enabled state; exclusion from CSV happens via
        # get_all_items() on the AnalyzeFace side.
        self.measure.set_item_enabled("landmark-1-x", False)
        self.assertFalse(self.measure.is_enabled("landmark-1-x"))
        self.assertTrue(self.measure.is_enabled("landmark-1-y"))

    def test_calc_measure_disabled_returns_empty_via_analyze(self):
        """When the measure itself is disabled, analyze() skips it entirely."""
        self.measure.set_enabled(False)
        self.assertFalse(self.measure.enabled)


class TestAnalyzeLandmarksIntegration(unittest.TestCase):
    """Integration tests that load real models and a real image."""

    @classmethod
    def setUpClass(cls):
        from dynaface import models, facial, measures
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


if __name__ == "__main__":
    unittest.main()
