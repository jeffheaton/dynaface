import logging
import time
from collections import deque
from typing import Callable

import cv2
from jth_ui import utl_etc
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from dynaface import facial, measures, util

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
        app = QApplication.instance()

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
        tilt_threshold = app.tilt_threshold
        face = facial.AnalyzeFace(
            self._dialog._window._face.measures, tilt_threshold=tilt_threshold
        )
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


def mean_landmarks(data):
    """
    Calculate the average of corresponding tuples across multiple lists of tuples.

    Args:
        data (list of list of tuples): A list where each element is a list of tuples.
            Each tuple contains two numeric values. For example:
            data = [ [(1, 2), (3, 4)], [(1, 2), (3, 4)] ]

    Returns:
        list of tuples: A list of tuples where each tuple contains the average of the corresponding
            tuples from the input lists. For example:
            output = [(1.0, 2.0), (3.0, 4.0)]

    Example:
        data = [ [(1, 2), (3, 4)], [(1, 2), (3, 4)] ]
        result = average_tuples(data)
        print(result)  # Output: [(1.0, 2.0), (3.0, 4.0)]
    """
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
        (int(sum_x / n_outer_lists), int(sum_y / n_outer_lists))
        for (sum_x, sum_y) in sums
    ]

    return averages


class WorkerLoad(QThread):
    """Load a video in the background."""

    _update_signal = pyqtSignal(str)

    def __init__(self, target):
        super().__init__()
        self._target = target
        self._total = self._target.frame_count
        self.running = True

    def preprocess_frame(self, frame, pupils, color=True):
        if self._target.base_rotation is not None:
            frame = cv2.rotate(frame, self._target.base_rotation)
        if color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        success = self._face.load_image(img=frame, crop=True, pupils=pupils)
        if not success:
            return None

        return frame

    def run(self):
        app = QApplication.instance()

        dynamic_adjust = app.dynamic_adjust
        data_smoothing = app.data_smoothing
        tilt_threshold = app.tilt_threshold

        logger.debug("Running background thread")
        logger.debug(f"Smoothing crop buffer size: {dynamic_adjust}")
        logger.debug(f"Smoothing landmarks buffer size: {data_smoothing}")
        logger.debug(f"Head tilt correct threshold: {tilt_threshold}")

        self._target.loading = True
        self._loading_etc = utl_etc.CalcETC(self._total)
        self._face = facial.AnalyzeFace([], tilt_threshold=tilt_threshold)

        pupils = None
        pupil_queue = deque(
            maxlen=dynamic_adjust
        )  # Queue to store the last 5 pupil positions

        landmarks_queue = deque(
            maxlen=data_smoothing
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

                # Make sure we did not get a request to stop during each of these:
                if not self.running:
                    break

                # Handle frames where we fail to find a face
                frame2 = frame
                frame = self.preprocess_frame(frame, pupils=pupils)
                if frame is None:
                    # See if a rotation fixes this
                    self._target.base_rotation = cv2.ROTATE_90_CLOCKWISE
                    frame = self.preprocess_frame(frame2, pupils=pupils)
                    if frame is None:
                        self._target.base_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
                        frame = self.preprocess_frame(frame2, pupils=pupils)
                        if frame is None:
                            self._target.base_rotation = None
                            logger.info(f"No face found on frame {i}")
                            continue

                # See if we have a considerable tilt, and try to resolve
                # This is likely due to an iphone held vertically but video stored horizontally
                tilt = self._face.calculate_face_rotation()
                if abs(tilt) > 70 and self._target.base_rotation is None:
                    if tilt < 0:
                        self._target.base_rotation = cv2.ROTATE_90_CLOCKWISE
                    else:
                        self._target.base_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE

                    frame = self.preprocess_frame(frame, pupils=pupils, color=False)

                #
                pupil_queue.append(self._face.orig_pupils)
                pupils = mean_landmarks(pupil_queue)

                # Extract
                landmarks = self._face.landmarks
                landmarks_queue.append(landmarks)
                if len(landmarks_queue) > 1:
                    landmarks = mean_landmarks(landmarks_queue)

                pupillary_distance, pix2mm = facial.util_calc_pd(
                    facial.util_get_pupils(landmarks)
                )

                # Update remaining face properties
                self._face.landmarks = landmarks
                self._face.pupillary_distance = pupillary_distance
                self._face.pix2mm = pix2mm

                # Build frame-state data
                frame_state = self._face.dump_state()
                self._target.add_frame(frame_state)

                # Update UI
                self._update_signal.emit(self._loading_etc.cycle())

        except Exception as e:
            logger.error("Error loading video", exc_info=True)
        finally:
            self._update_signal.emit("*")
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
