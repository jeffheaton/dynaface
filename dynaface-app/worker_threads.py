import logging
import time

import cv2
from facial_analysis import facial
from facial_analysis.facial import load_face_image
from PyQt6.QtCore import QThread, pyqtSignal
from jth_ui import utl_etc
from facial_analysis import models, util

logger = logging.getLogger(__name__)


class WorkerExport(QThread):
    """Export loaded video frames to an annotated video."""

    _update_signal = pyqtSignal(str)

    def __init__(self, dlg, output_file):
        super().__init__()
        self._dialog = dlg
        self._output_file = output_file

    def run(self):
        if self._dialog._window.loading:
            self._update_signal.emit("Waiting for load to complete.")
            while self._dialog._window.loading:
                time.sleep(1)

        self._update_signal.emit("Exporting...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID'
        width = self._dialog._window._face.width
        height = self._dialog._window._face.height
        out = cv2.VideoWriter(
            self._output_file, fourcc, self._dialog._window.frame_rate, (width, height)
        )

        face = facial.AnalyzeFace(self._dialog._window._face.measures)

        c = len(self._dialog._window._frames)

        t = self._dialog._window
        for i in range(t._frame_begin, t._frame_end):
            frame = t._frames[i]
            self._update_signal.emit(f"Exporting frame {i:,}/{c:,}...")
            face.load_state(frame)
            face.render_reset()
            face.analyze()
            image = cv2.cvtColor(face.render_img, cv2.COLOR_BGR2RGB)
            out.write(image)

        out.release()
        self._update_signal.emit("*")


class WorkerLoad(QThread):
    """Load a video in the background."""

    _update_signal = pyqtSignal(str)

    def __init__(self, target):
        super().__init__()
        self._target = target
        self._total = self._target.frame_count
        self.running = True

    def run(self):
        start_time = time.time()
        logger.debug("Running background thread")
        self._target.loading = True
        self._loading_etc = utl_etc.CalcETC(self._total)
        self._face = facial.AnalyzeFace([])
        try:
            i = 0
            while self.running:
                i += 1
                logger.debug(f"Begin frame {i}")
                ret, frame = self._target.video_stream.read()

                if not ret:
                    logger.debug("Thread done")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Find face bounding box
                bbox, prob = models.mtcnn_model.detect(frame)
                bbox = bbox[0]

                bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

                # Find the facial features
                landmarks = models.spiga_model.inference(frame, [bbox])
                landmarks = models.convert_landmarks(landmarks)

                # Crop to the eyes
                frame, landmarks = util.crop_stylegan(
                    img=frame, pupils=None, landmarks=landmarks
                )

                # Extract
                pupillary_distance, pix2mm = util.calc_pd(landmarks)

                if not self.running:
                    break

                # Build frame-state data
                frame_state = [
                    frame,
                    None,
                    landmarks,
                    pupillary_distance,
                    pix2mm,
                ]
                self._target.add_frame(frame_state)

                if self.running:
                    self._update_signal.emit(self._loading_etc.cycle())
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Video processing time: {duration}")

        except Exception as e:
            logger.error("Error loading video", exc_info=True)
        finally:
            self._update_signal.emit(None)
            self._target.loading = False


class WorkerWaitLoad(QThread):
    _update_signal = pyqtSignal(str)

    def __init__(self, dlg):
        super().__init__()
        self._dialog = dlg
        self._total = self._dialog._window.frame_count
        self._etc = utl_etc.CalcETC(self._total)

    def run(self):
        while self._dialog._window.loading:
            cur = len(self._dialog._window._frames)
            time_left = self._etc.cycle()
            self._update_signal.emit(
                f"Loading frame {cur:,} of {self._total:,}; time left: {time_left}."
            )
            time.sleep(1)
        self._update_signal.emit("*")
