import logging
import time
from collections import deque
from typing import Callable

import cv2
from jth_ui import utl_etc
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from dynaface import facial, measures

logger = logging.getLogger(__name__)

BATCH_SIZE = 10


class WorkerExport(QThread):
    """Export loaded video frames to an annotated video."""

    _update_signal = pyqtSignal(str)

    def __init__(self, dlg, output_file):
        super().__init__()
        self.setStackSize(16 * 1024 * 1024)  # 16 MB — same OpenBLAS requirement
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
        t = self._dialog._window
        # The export's own measures, so this thread never touches the flags the
        # tab draws with, and the tab's choices applied to each frame's view, as
        # the tab does for the frame it shows: a frontal frame of a mixed-view
        # video gets the chosen millimeter measures, a turned or profile frame
        # never does, whichever frame the slider sits on.
        selection = t._selection
        face = facial.AnalyzeFace(
            measures.all_measures(), tilt_threshold=tilt_threshold
        )
        c = t._frame_end - t._frame_begin
        for i in range(t._frame_begin, t._frame_end):
            frame = t._frames[i]
            self._update_signal.emit(f"Exporting frame {i-t._frame_begin:,}/{c:,}...")
            face.load_state(frame)
            selection.apply(face.measures, face.lateral, quarter=face.quarter)
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
        # OpenBLAS (used by numpy linalg inside SPIGA) needs a large stack.
        # The default QThread stack (~512KB) overflows during dgetrf_parallel.
        self.setStackSize(16 * 1024 * 1024)  # 16 MB
        self._target = target
        self._total = self._target.frame_count
        self.running = True

    def preprocess_frame(self, frame, previous_pose, color=True):
        """
        Load one frame into the worker's face: rotate it as the import settled
        on, convert its colors, and run the detection with the pose the tab
        asked for (its force_pose, Pose.DETECT to classify each frame) and the
        previous kept frame's pose, so the three-quarter classification keeps a
        frame in its view a little past the yaw window's edge (hysteresis), as
        VideoAnalyzer.add() does.

        Returns the frame as loaded, or None when it has no face.
        """
        if self._target.base_rotation is not None:
            frame = cv2.rotate(frame, self._target.base_rotation)
        if color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self._face.load_image(
            img=frame,
            crop=True,
            force_pose=self._force_pose,
            previous_pose=previous_pose,
        )
        if not self._face.landmarks:
            # No face. load_image() returns True either way, so this used to go on
            # to fail on the missing landmarks, which stopped the whole import.
            return None

        return frame

    def run(self):
        app = QApplication.instance()

        data_smoothing = app.data_smoothing
        tilt_threshold = app.tilt_threshold

        logger.debug("Running background thread")
        logger.debug(f"Smoothing landmarks buffer size: {data_smoothing}")
        logger.debug(f"Head tilt correct threshold: {tilt_threshold}")

        # The tab sets loading before it starts this thread, so its buttons are
        # disabled from the first frame on; set here as well for completeness.
        self._target.loading = True
        self._loading_etc = utl_etc.CalcETC(self._total)
        self._face = facial.AnalyzeFace([], tilt_threshold=tilt_threshold)
        # The pose the tab asked for, read once: Pose.DETECT classifies each
        # frame; a pose chosen with the Pose button is used for all of them.
        self._force_pose = self._target.force_pose
        logger.debug(f"Force pose: {self._force_pose}")

        # The last few frames' landmarks, averaged into each kept frame's. (The
        # crop-smoothing size, app.dynamic_adjust, isn't used here: the
        # library's load_image() always places the crop on the pupils it finds
        # in the frame itself and ignores any pupils passed in.)
        landmarks_queue = deque(maxlen=data_smoothing)

        # The last kept frame's view, (pose, flipped, far_left), and its pose:
        # the landmark average restarts when the view changes, and the pose
        # goes to the next frame's load_image(), as VideoAnalyzer.add() does.
        previous_view = None
        previous_pose = None

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
                frame = self.preprocess_frame(frame, previous_pose)
                if frame is None:
                    # See if a rotation fixes this
                    self._target.base_rotation = cv2.ROTATE_90_CLOCKWISE
                    frame = self.preprocess_frame(frame2, previous_pose)
                    if frame is None:
                        self._target.base_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
                        frame = self.preprocess_frame(frame2, previous_pose)
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

                    frame = self.preprocess_frame(frame, previous_pose, color=False)
                    if frame is None:
                        # The rotation that was meant to fix the tilt lost the
                        # face; skip the frame rather than go on without one.
                        self._target.base_rotation = None
                        logger.info(f"No face found on frame {i} after rotating it")
                        continue

                # One view's landmarks don't average with another's: a profile's
                # with a frontal face's, a quarter face's band crop with the
                # frontal crop, a profile facing the other way (flipped), or a
                # quarter face whose far cheek is the other one (its band crop
                # is mirrored). far_left is False off a quarter face, so frontal
                # and profile frames key on the pose and the flip alone.
                view = (self._face.pose, self._face.flipped, self._face.far_left)
                if view != previous_view:
                    if previous_view is not None:
                        logger.info(
                            f"View changed on frame {i}: {previous_view} to "
                            f"{view}; landmark smoothing restarts"
                        )
                    landmarks_queue.clear()
                    previous_view = view
                previous_pose = self._face.pose

                # Average the landmarks with the frames before this one. The
                # averaged landmarks replace the frame's own, so what
                # load_image() measured on them is measured again, as
                # VideoAnalyzer.add() does: the pupil scale for a frontal or
                # quarter face, a quarter face's far-cheek outline (from the
                # frame's own mask, no second model pass) and the cheek
                # landmarks. A frame that wasn't averaged (the first of its
                # view, or a smoothing size of 1) keeps what load_image()
                # measured. A profile keeps the library's fixed scale: the
                # import used to overwrite every frame's scale with a pupil
                # scale, which on a lateral frame replaced that fixed profile
                # scale, a defect.
                landmarks_queue.append(self._face.landmarks)
                if len(landmarks_queue) > 1:
                    self._face.landmarks = mean_landmarks(landmarks_queue)
                    if not self._face.lateral:
                        self._face.calc_pd()
                    if self._face.quarter:
                        self._face.retrace_far_cheek()
                    self._face.find_cheek_landmarks()

                # Build frame-state data
                frame_state = self._face.dump_state()
                self._target.add_frame(frame_state)

                # Update UI
                self._update_signal.emit(self._loading_etc.cycle())

        except Exception as e:
            logger.error("Error loading video", exc_info=True)
            # Reported when loading ends, rather than as no faces found.
            self._target.load_failure = e
        finally:
            # Cleared before the end is signaled, so the tab's handler of "*"
            # sees the import as finished when it enables its buttons.
            self._target.loading = False
            self._update_signal.emit("*")


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
        self.aborted = False

    def run(self):
        try:
            self._proc()
            self.update_signal.emit()
        except Exception as e:
            logger.error("Error in worker thread", exc_info=True)
            self.update_signal.emit()
            self.aborted = True
