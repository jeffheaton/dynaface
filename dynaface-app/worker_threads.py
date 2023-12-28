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

    def process_batch(self, frames, x1, y1, x2, y2):
        frames_cropped = [x[y1:y2, x1:x2] for x in frames]
        bbox = [0, 0, x2 - x1, y2 - y1]
        landmarks_batch = models.spiga_model.inference_batch(frames_cropped, bbox)
        landmarks_batch = models.convert_landmarks(landmarks_batch)

        for landmarks, frame in zip(landmarks_batch, frames):
            landmarks = util.scale_crop_points(
                lst=landmarks, crop_x=-x1, crop_y=-y1, scale=1.0
            )
            # Crop to the eyes
            frame, landmarks = util.crop_stylegan(
                img=frames[0], pupils=None, landmarks=landmarks
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

    def detect_faces(self, frames_pass1, frames_pass2):
        print("Detecting faces")
        bbox, prob = models.mtcnn_model.detect(frames_pass1)
        for i in range(len(prob)):
            if prob[i][0] > 0.98:
                frames_pass2.append((frames_pass1[i], bbox[i][0]))

        frames_pass1.clear()

    def detect_landmarks(self, frames_pass2):
        print("Detecting landmarks")
        top = min(BATCH_SIZE, len(frames_pass2))
        lst_bbox = []
        lst_crop = []
        lst_frames = []
        for i in range(top):
            item = frames_pass2.pop(0)
            bbox = item[1]
            x1 = int(bbox[0])
            x2 = int(bbox[2])
            y1 = int(bbox[1])
            y2 = int(bbox[3])
            frame = item[0]
            lst_frames.append(frame)
            frame = frame[y1:y2, x1:x2]
            bbox = [0, 0, x2 - x1, y2 - y1]
            lst_bbox.append(bbox)
            lst_crop.append(frame)

        landmarks_batch = models.spiga_model.inference_batch(lst_crop, lst_bbox[0])
        landmarks_batch = models.convert_landmarks(landmarks_batch)
        self.dispatch_frames(landmarks_batch, lst_frames, x1, y1)
        frames_pass2.clear()

    def dispatch_frames(self, landmarks_batch, frames, x1, y1):
        for landmarks, frame in zip(landmarks_batch, frames):
            landmarks = util.scale_crop_points(
                lst=landmarks, crop_x=-x1, crop_y=-y1, scale=1.0
            )
            # Crop to the eyes
            frame, landmarks = util.crop_stylegan(
                img=frames[0], pupils=None, landmarks=landmarks
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
        last_bbox = 100
        frames_available = True
        frames_pass1 = []
        frames_pass2 = []
        try:
            i = 0
            self._update_signal.emit(self._loading_etc.cycle())
            while self.running:
                if frames_available:
                    i += 1
                    # left = self._loading_etc.cycle()
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

            print("done")
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


class WorkerPleaseWait(QThread):
    update_signal = pyqtSignal()

    def __init__(self, proc: Callable):
        super().__init__()
        self._proc = proc

    def run(self):
        self._proc()
        self.update_signal.emit()
