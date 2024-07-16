import logging
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from facial_analysis import facial, models, util
from facial_analysis.facial import AnalyzeFace
from jth_ui import utl_etc

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


def mean_pupils(data):
    # Determine the length of the inner lists
    if not data or not data[0]:
        return []

    inner_list_length = len(data[0])

    # Initialize sums for each tuple position
    sums = [(0, 0) for _ in range(inner_list_length)]

    # Calculate the number of outer lists
    n_outer_lists = len(data)

    # Iterate through the outer list
    for outer_list in data:
        for i, (x, y) in enumerate(outer_list):
            sums[i] = (sums[i][0] + x, sums[i][1] + y)

    # Calculate the averages
    averages = [
        (sum_x / n_outer_lists, sum_y / n_outer_lists) for (sum_x, sum_y) in sums
    ]

    return averages


class WorkerLoad(QThread):
    """Load a video in the background."""

    _update_signal = pyqtSignal(str)

    def __init__(self, target, avg_len=5):
        super().__init__()
        self._target = target
        self._total = self._target.frame_count
        self.running = True
        self._avg_len = avg_len

    def run(self):
        logger.debug("Running background thread")
        logger.debug(f"Averaging buffer size: {self._avg_len}")
        self._target.loading = True
        self._loading_etc = utl_etc.CalcETC(self._total)
        self._face = facial.AnalyzeFace([])

        pupils = None
        pupil_queue = deque(
            maxlen=self._avg_len
        )  # Queue to store the last 5 pupil positions

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

                # Make sure we did not get a request to stop during each of these:
                if self.running:
                    self._face.load_image(img=frame, crop=True, pupils=pupils)

                    pupil_queue.append(self._face.orig_pupils)
                    pupils = mean_pupils(pupil_queue)

                if self.running:
                    # Extract
                    landmarks = self._face.landmarks
                    pupillary_distance, pix2mm = util.calc_pd(
                        util.get_pupils(landmarks)
                    )
                    # Build frame-state data
                    frame_state = [
                        self._face.original_img,
                        None,
                        landmarks,
                        int(pupillary_distance),
                        pix2mm,
                    ]
                    self._target.add_frame(frame_state)

                if self.running:
                    self._update_signal.emit(self._loading_etc.cycle())

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
