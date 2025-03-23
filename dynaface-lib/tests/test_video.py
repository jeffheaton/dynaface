import os
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import cv2
from dynaface.video import ProcessVideoFFMPEG, ProcessVideoOpenCV, VideoToVideo


class TestProcessVideoFFMPEG(unittest.TestCase):

    @patch("dynaface.video.subprocess.call", return_value=0)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Stream #0:0 Video: h264, yuv420p(tv), 1280x720, 30 fps",
    )
    def test_extract_success(self, mock_file, mock_call):
        pv = ProcessVideoFFMPEG()
        input_file = "fake_video.mp4"

        with patch("os.path.exists", return_value=False):
            result = pv.extract(input_file)

        self.assertTrue(result)
        self.assertEqual(pv.frame_rate, 30)

    @patch("dynaface.video.subprocess.call", return_value=1)
    @patch("builtins.open", new_callable=mock_open)
    def test_extract_ffmpeg_fail(self, mock_file, mock_call):
        pv = ProcessVideoFFMPEG()
        input_file = "fake_video.mp4"

        result = pv.extract(input_file)
        self.assertFalse(result)


class TestProcessVideoOpenCV(unittest.TestCase):

    @patch("cv2.VideoCapture")
    @patch("cv2.imwrite")
    def test_extract_success(self, mock_imwrite, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, MagicMock()) for _ in range(5)] + [
            (False, None)
        ]
        mock_cap.get.side_effect = lambda x: 30 if x == cv2.CAP_PROP_FPS else 150
        mock_video_capture.return_value = mock_cap

        pvo = ProcessVideoOpenCV()
        result = pvo.extract("dummy.mp4")
        self.assertTrue(result)
        self.assertEqual(pvo.frame_rate, 30)
        self.assertEqual(pvo.frames, 150)


class TestVideoToVideo(unittest.TestCase):
    @patch("dynaface.video.ProcessVideoOpenCV")
    @patch("dynaface.video.AnalyzeFace")
    @patch("cv2.cvtColor")
    @patch("cv2.imread")
    def test_process_video(
        self, mock_imread, mock_cvt, mock_analyzeface, mock_process_video
    ):
        # Mock image with shape attribute
        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)
        mock_imread.return_value = mock_image
        mock_cvt.return_value = mock_image

        # Mock ProcessVideoOpenCV instance
        mock_proc = MagicMock()
        mock_proc.extract.return_value = True
        mock_proc.temp_path = "/tmp/fake"
        mock_proc.frame_rate = 30
        mock_proc.build.return_value = None
        mock_proc.cleanup.return_value = None
        mock_process_video.return_value = mock_proc

        # Simulate two image frames exist
        def exists_side_effect(path):
            return "input-1.jpg" in path or "input-2.jpg" in path

        with patch("os.path.exists", side_effect=exists_side_effect):
            # Mock AnalyzeFace behavior
            mock_af_instance = MagicMock()
            mock_af_instance.get_all_items.return_value = ["eye_distance"]
            mock_af_instance.get_pupils.return_value = (1, 2)
            mock_af_instance.analyze.return_value = {"eye_distance": 42}
            mock_analyzeface.return_value = mock_af_instance

            vtv = VideoToVideo(points=True, crop=(0, 0, 100, 100))
            result = vtv.process("input.mp4", "output.mp4", stats=["eye_distance"])

            self.assertTrue(result)
            self.assertIn("eye_distance", vtv.data)
            self.assertEqual(vtv.data["eye_distance"], [42, 42])
