import logging
import time

import cv2
from facial_analysis import facial
from facial_analysis.facial import load_face_image
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class WorkerExport(QThread):
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

        face = facial.AnalyzeFace(self._dialog._window._calcs, data_path=None)

        c = len(self._dialog._window._frames)
        for i, frame in enumerate(self._dialog._window._frames):
            self._update_signal.emit(f"Exporting frame {i:,}/{c:,}...")
            face.load_state(frame)
            face.render_reset()
            face.analyze()
            image = cv2.cvtColor(face.render_img, cv2.COLOR_BGR2RGB)
            out.write(image)

        out.release()
        self._update_signal.emit("*")


class WorkerLoad(QThread):
    _update_signal = pyqtSignal(str)

    def __init__(self, target):
        super().__init__()
        self._target = target

    def run(self):
        logger.info("Running background thread")
        self._target.loading = True
        self._face = facial.AnalyzeFace([], data_path=None)
        try:
            i = 0
            while True:
                ret, frame = self._target.video_stream.read()

                if not ret:
                    logger.info("Thread done")
                    break

                i += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self._face.load_image(img=frame, crop=True)
                self._target.add_frame(self._face)
                self._update_signal.emit(None)
        finally:
            self._update_signal.emit(None)
            self._target.loading = False