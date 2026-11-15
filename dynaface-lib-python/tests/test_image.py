import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

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
        # Closed before it's read: Windows can't open a NamedTemporaryFile again
        # while it is open.
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "not_an_image.txt")
            with open(path, "wb") as f:
                f.write(b"This is not an image")
            with self.assertRaises(cv2.error):
                load_image(path)

    def test_load_image_empty_file(self):
        # cv2.imdecode raises its own assertion on no bytes; load_image says why.
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "empty.jpg")
            open(path, "wb").close()
            with self.assertRaisesRegex(cv2.error, "Could not decode image"):
                load_image(path)

    def test_load_image_non_ascii_path(self):
        # cv2.imread can't open a path with a character outside Windows' ANSI code
        # page, so load_image() reads the file itself; with imread failing, as it
        # does there, the photo still opens, the same pixels as from its own name.
        expected = load_image("./tests_data/img1-512.jpg")
        with tempfile.TemporaryDirectory(suffix="_José") as folder:
            path = os.path.join(folder, "Universität 測試.jpg")
            with open("./tests_data/img1-512.jpg", "rb") as src, open(path, "wb") as f:
                f.write(src.read())
            with mock.patch("cv2.imread", return_value=None):
                np.testing.assert_array_equal(load_image(path), expected)

    def test_load_image_applies_exif_orientation(self):
        # As cv2.imread does: a photo whose EXIF says "rotate 90 degrees clockwise"
        # loads turned, so a phone's portrait photo comes out upright.
        pixels = np.zeros((40, 80, 3), dtype=np.uint8)
        pixels[:, :40] = (255, 0, 0)  # red on the left half
        photo = Image.fromarray(pixels)
        exif = photo.getexif()
        exif[0x0112] = 6  # Orientation: rotate 90 CW to display
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "portrait.jpg")
            photo.save(path, exif=exif.tobytes(), quality=95)
            img = load_image(path)
            self.assertEqual(img.shape, (80, 40, 3))
            # The left half turned to the top: red there, black below.
            self.assertGreater(img[10, 20, 0], 200)
            self.assertLess(img[70, 20, 0], 50)
            np.testing.assert_array_equal(
                img, cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            )

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

    def test_write_text_outline_shows_on_white(self):
        # OpenCV 5 draws text at the same weight for any thickness of 2 or more,
        # which once hid the outline: white text vanished on light backgrounds.
        analysis = ImageAnalysis()
        test_img = np.full((100, 300, 3), 255, dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.write_text((10, 60), "dental=577 mm")
        dark = analysis.render_img.max(axis=2) < 64
        self.assertGreater(int(dark.sum()), 100)

    def test_write_text_uses_color(self):
        analysis = ImageAnalysis()
        test_img = np.full((100, 300, 3), 128, dtype=np.uint8)
        analysis.load_image(test_img)
        analysis.write_text((10, 60), "Test", color=(255, 0, 0))
        img = analysis.render_img
        red = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] < 60)
        self.assertTrue(red.any())

    def test_write_text_off_the_edges(self):
        analysis = ImageAnalysis()
        test_img = np.full((100, 100, 3), 255, dtype=np.uint8)
        analysis.load_image(test_img)
        for pos in [(-30, 50), (80, 50), (10, 5), (10, 110), (-500, -500)]:
            analysis.write_text(pos, "Edge")
            analysis.write_text_sq(pos, "Edge")

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

    def test_save_image_non_ascii_path(self):
        # As for load_image(): cv2.imwrite can't open such a path on Windows, and
        # returned False, so the image wasn't written and nothing said so.
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[20:60, 10:50] = (10, 200, 30)
        analysis.load_image(test_img)
        with tempfile.TemporaryDirectory(suffix="_José") as folder:
            path = os.path.join(folder, "Universität 測試.png")
            with mock.patch("cv2.imwrite", return_value=False):
                analysis.save(path)
            # PNG is lossless, so the file holds exactly the rendered image.
            np.testing.assert_array_equal(load_image(path), analysis.render_img)

    def test_save_image_writes_what_imwrite_did(self):
        analysis = ImageAnalysis()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[20:60, 10:50] = (10, 200, 30)
        analysis.load_image(test_img)
        bgr = cv2.cvtColor(analysis.render_img, cv2.COLOR_RGB2BGR)
        with tempfile.TemporaryDirectory() as folder:
            for name in ("a.jpg", "b.JPEG", "c.png"):
                path = os.path.join(folder, name)
                reference = os.path.join(folder, "imwrite_" + name)
                analysis.save(path)
                cv2.imwrite(reference, bgr)
                with open(path, "rb") as f, open(reference, "rb") as g:
                    self.assertEqual(f.read(), g.read(), name)

    def test_save_image_failure_raises(self):
        analysis = ImageAnalysis()
        analysis.load_image(np.zeros((100, 100, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(OSError):
                analysis.save(os.path.join(folder, "missing", "out.png"))
            with self.assertRaises(cv2.error):
                analysis.save(os.path.join(folder, "out.not_an_image_type"))

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
