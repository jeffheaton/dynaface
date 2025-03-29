import os
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import cv2
import numpy as np
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

    @patch("shutil.rmtree")
    def test_cleanup(self, mock_rmtree):
        pv = ProcessVideoFFMPEG()
        pv.cleanup()
        mock_rmtree.assert_called_once_with(pv.temp_path)

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch(
        "dynaface.video.ProcessVideoFFMPEG.execute_command", return_value=["success"]
    )
    def test_build_existing_output(self, mock_exec, mock_remove, mock_exists):
        pv = ProcessVideoFFMPEG()
        pv.audio_file = "/tmp/fake/audio.wav"  # Ensure audio_file is set
        output_file = "output.mp4"
        pv.build(output_file, 30)
        mock_remove.assert_called_once_with(output_file)
        mock_exec.assert_called_once()


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

    @patch("cv2.VideoCapture")
    def test_extract_fail(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        pvo = ProcessVideoOpenCV()
        result = pvo.extract("bad_video.mp4")

        self.assertFalse(result)

    @patch("shutil.rmtree")
    def test_cleanup_opencv(self, mock_rmtree):
        pvo = ProcessVideoOpenCV()
        pvo.cleanup()
        mock_rmtree.assert_called_once_with(pvo.temp_path)

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch("cv2.VideoWriter")
    @patch(
        "cv2.imread",
        side_effect=lambda path: (
            np.ones((100, 100, 3), dtype=np.uint8)
            if "output" in path
            else np.ones((100, 100, 3), dtype=np.uint8)
        ),
    )
    @patch("os.listdir", return_value=["output-1.jpg", "output-2.jpg"])
    def test_build_opencv(
        self, mock_listdir, mock_imread, mock_vw, mock_remove, mock_exists
    ):
        mock_vw_instance = MagicMock()
        mock_vw.return_value = mock_vw_instance

        pvo = ProcessVideoOpenCV()
        result = pvo.build("output.mp4", 30)

        self.assertTrue(result)
        mock_vw_instance.write.assert_called()
        mock_vw_instance.release.assert_called()


class TestVideoToVideo(unittest.TestCase):
    @patch("dynaface.video.ProcessVideoOpenCV")
    @patch("dynaface.video.AnalyzeFace")
    @patch("cv2.cvtColor")
    @patch("cv2.imread")
    def test_process_video(
        self, mock_imread, mock_cvt, mock_analyzeface, mock_process_video
    ):
        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)
        mock_imread.return_value = mock_image
        mock_cvt.return_value = mock_image

        mock_proc = MagicMock()
        mock_proc.extract.return_value = True
        mock_proc.temp_path = "/tmp/fake"
        mock_proc.frame_rate = 30
        mock_proc.build.return_value = None
        mock_proc.cleanup.return_value = None
        mock_process_video.return_value = mock_proc

        def exists_side_effect(path):
            return "input-1.jpg" in path or "input-2.jpg" in path

        with patch("os.path.exists", side_effect=exists_side_effect):
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

    @patch("dynaface.video.ProcessVideoOpenCV")
    @patch("dynaface.video.AnalyzeFace")
    @patch(
        "cv2.imread",
        side_effect=lambda path: (
            np.ones((100, 100, 3), dtype=np.uint8) if "input-" in path else None
        ),
    )
    @patch(
        "os.path.exists",
        side_effect=lambda path: path
        in ["/tmp/fake/input-1.jpg", "/tmp/fake/input-2.jpg"],
    )
    def test_process_no_frames(
        self, mock_analyzeface, mock_exists, mock_imread, mock_process_video
    ):
        mock_proc = MagicMock()
        mock_proc.extract.return_value = True
        mock_proc.temp_path = "/tmp/fake"
        mock_proc.frame_rate = 30
        mock_process_video.return_value = mock_proc

        # Mock AnalyzeFace instance to return an empty dictionary for analyze()
        mock_af_instance = MagicMock()
        mock_af_instance.analyze.return_value = {}  # <-- Important fix!
        mock_analyzeface.return_value = mock_af_instance

        vtv = VideoToVideo(points=True, crop=(0, 0, 100, 100))
        result = vtv.process("./tests_data/video1-512.mp4", "output.mp4")

        self.assertTrue(result)
        self.assertEqual(vtv.data, {})

    @patch("builtins.open", new_callable=mock_open)
    def test_dump_data(self, mock_file):
        vtv = VideoToVideo(points=True, crop=(0, 0, 100, 100))
        vtv.data = {"eye_distance": [10, 20, 30]}
        vtv.stats = ["eye_distance"]

        vtv.dump_data("fake.csv")
        mock_file.assert_called_with("fake.csv", "w")

    def test_plot_chart(self):
        vtv = VideoToVideo(points=True, crop=(0, 0, 100, 100))
        try:
            vtv.plot_chart("chart.png")
        except Exception as e:
            self.fail(f"plot_chart raised {e} unexpectedly!")
