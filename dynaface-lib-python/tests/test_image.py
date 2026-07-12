import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from dynaface.image import ImageAnalysis, load_image
from dynaface.util import safe_clip

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestImage(unittest.TestCase):
    #
    # load_image() Tests
    #
    def test_load_image(self):
        img = load_image("./tests_data/img1-512.jpg")
        assert img.shape[0] == 512
        assert img.shape[1] == 512

    def test_load_image_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_image("./tests_data/non_existent.jpg")

    def test_load_image_invalid_format(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            temp_file.write(b"This is not an image")  # Write as bytes
            temp_file.flush()  # Ensure content is written to disk

            with self.assertRaises(cv2.error):
                load_image(temp_file.name)

    #
    # ImageAnalysis.load_image() Tests
    #
    def test_load_image_none(self):
        analysis = ImageAnalysis()
        with self.assertRaises(Exception):
            analysis.load_image(None)  # type: ignore

    def test_load_image_too_small(self):
        analysis = ImageAnalysis()
        small_img = np.zeros((4, 4, 3), dtype=np.uint8)  # Too small
        with self.assertRaises(ValueError):
            analysis.load_image(small_img)

    #
    # safe_clip() Tests
    #
    def test_crop_image(self):
        test_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        clipped_image, x_offset, y_offset = safe_clip(
            test_image, -100, -100, 1024, 1024, (255, 0, 0)
        )
        assert clipped_image.shape[0] == 1024
        assert clipped_image.shape[1] == 1024
        assert clipped_image.shape[2] == 3
        assert x_offset == 100
        assert y_offset == 100

    def test_crop_image_normal_case(self):
        test_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        clipped_image, x_offset, y_offset = safe_clip(
            test_image, 100, 100, 200, 200, (0, 0, 0)
        )
        assert clipped_image.shape == (200, 200, 3)
        assert x_offset == 0
        assert y_offset == 0

    def test_crop_image_fully_outside(self):
        test_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        clipped_image, x_offset, y_offset = safe_clip(
            test_image, 600, 600, 100, 100, (0, 0, 0)
        )
        assert clipped_image.shape == (100, 100, 3)
        assert np.all(clipped_image == [0, 0, 0])
        assert x_offset == 0
        assert y_offset == 0

    #
    # write_text() and write_text_sq() Tests
    #
    def test_write_text_default_color(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.write_text((10, 10), "Test")

    def test_write_text_sq_default_color(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.write_text_sq((10, 10), "Test", mark="*")

    def test_write_text_empty_string(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.write_text((10, 10), "")

    #
    # hline() and vline() Tests
    #
    def test_hline_default_bounds(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.hline(50)

    def test_vline_default_bounds(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.vline(50)

    def test_hline_out_of_bounds(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.hline(-10)
        analysis.hline(150)

    def test_vline_out_of_bounds(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.vline(-10)
        analysis.vline(150)

    #
    # arrow() and arrow_head() Tests
    #
    def test_arrow_one_side(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.arrow((10, 10), (90, 90), apt1=True, apt2=False)

    def test_arrow_head_various_params(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.arrow_head((50, 50), (10, 10), par=10)

    def test_arrow_both_sides(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.arrow((10, 10), (90, 90), apt1=True, apt2=True)

    #
    # render_reset() Test
    #
    def test_render_reset(self):
        analysis = ImageAnalysis()
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.hline(50)
        analysis.render_reset()
        assert np.array_equal(analysis.render_img, test_img)

    #
    # line(), circle(), and save() Tests
    #
    def test_line(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.line((10, 10), (90, 90))

    def test_circle(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.circle((50, 50))

    def test_save_image(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name  # Store the name
        analysis.save(temp_path)
        assert os.path.exists(temp_path)
        os.remove(temp_path)  

    #
    # measure_polygon() Tests
    #
    def test_measure_polygon(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        contours = [(10, 10), (40, 10), (40, 40), (10, 40)]
        area = analysis.measure_polygon(contours, pix2mm=1.0)
        assert round(area) == 900

    def test_measure_polygon_no_render(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        contours = [(10, 10), (40, 10), (40, 40), (10, 40)]
        area = analysis.measure_polygon(contours, pix2mm=1.0, render=False)
        assert round(area) == 900

    #
    # extract_horiz() and extract_horiz_hsv() Tests
    #
    def test_extract_horiz(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        section = analysis.extract_horiz(50)
        assert section.shape[0] == 100

    def test_extract_horiz_hsv(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis.load_image(test_img)
        section = analysis.extract_horiz_hsv(50)
        assert section.shape[0] == 100
