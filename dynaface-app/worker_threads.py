import logging
import time
from typing import Callable

import cv2
from facial_analysis import facial, models, util
from facial_analysis.facial import load_face_image
from jth_ui import utl_etc
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

BATCH_SIZE = 10


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
        t = self._dialog._window
        c = t._frame_end - t._frame_begin
        for i in range(t._frame_begin, t._frame_end):
            frame = t._frames[i]
            self._update_signal.emit(f"Exporting frame {i-t._frame_begin:,}/{c:,}...")
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
        self.frame_number = 1
        self.accepted_frame_number = 1

    def detect_faces(self, frames_pass1, frames_pass2):
        """Find the rectangle outlines of faces in the image."""
        logger.debug("Detecting faces")
        bbox_list, prob = models.mtcnn_model.detect(frames_pass1)
        for i in range(len(prob)):
            if (
                (prob[i] is not None)
                and (prob[i][0] is not None)
                and (prob[i][0] > 0.98)
            ):
                bbox = bbox_list[i][0]
                bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                frames_pass2.append((frames_pass1[i], bbox))
                logger.debug(
                    f"Actual frame {self.frame_number} -> analyze frame {self.accepted_frame_number}"
                )
                self.accepted_frame_number += 1

            self.frame_number += 1

        frames_pass1.clear()

    def detect_landmarks(self, frames_pass2):
        logger.debug("Detecting landmarks")
        top = min(BATCH_SIZE, len(frames_pass2))

        for i in range(top):
            item = frames_pass2[i]
            bbox = item[1]
            frame = item[0]
            landmarks_batch = models.spiga_model.inference(frame, [bbox])
            landmarks_batch = models.convert_landmarks(landmarks_batch)
            self.dispatch_frames(landmarks_batch=landmarks_batch, frames=[frame])

        del frames_pass2[0:top]

    def dispatch_frames(self, landmarks_batch, frames):
        for landmarks, frame in zip(landmarks_batch, frames):
            # Crop to the eyes
            frame, landmarks = util.crop_stylegan(
                img=frame, pupils=None, landmarks=landmarks
            )
            # Extract
            pupillary_distance, pix2mm = util.calc_pd(landmarks)
            # Build frame-state data
            frame_state = [
                frame,
                None,
                landmarks,
                pupillary_distance,
                pix2mm,
            ]
            self._target.add_frame(frame_state)

    def run(self):
        start_time = time.time()
        logger.debug("Running background thread")
        self._target.loading = True
        self._loading_etc = utl_etc.CalcETC(self._total)
        self._face = facial.AnalyzeFace([])
        frames_available = True
        frames_pass1 = []
        frames_pass2 = []
        try:
            i = 0
            while self.running:
                if frames_available:
                    i += 1
                    # left = self._loading_etc.cycle()
                    msg = self._loading_etc.cycle()
                    self._update_signal.emit(msg)
                    logger.debug(f"Read frame {i} of {self._total}")
                    ret, frame = self._target.video_stream.read()
                    if not ret:
                        frames_available = False
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frames_pass1.append(frame)

                if (len(frames_pass1) >= BATCH_SIZE) or (
                    (not frames_available) and (len(frames_pass1) > 0)
                ):
                    self.detect_faces(frames_pass1, frames_pass2)

                if (len(frames_pass2) >= BATCH_SIZE) or (
                    (not frames_available) and (len(frames_pass2) > 0)
                ):
                    self.detect_landmarks(frames_pass2)

                if (
                    (not frames_available)
                    and (len(frames_pass1) < 1)
                    and (len(frames_pass2) < 1)
                ):
                    self.running = False

            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Video processing time: {duration}")
            self._update_signal.emit("****")
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


class WorkerPleaseWait(QThread):
    update_signal = pyqtSignal()

    def __init__(self, proc: Callable):
        super().__init__()
        self._proc = proc

    def run(self):
        self._proc()
        self.update_signal.emit()
