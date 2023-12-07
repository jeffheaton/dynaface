import logging
import time

from facial_analysis import facial
from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QPushButton, QVBoxLayout, QLabel
import cv2

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


class ChoiceDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Save Options")
        self.user_choice = None  # Attribute to store user's choice

        # Buttons for video and image
        self.save_video_button = QPushButton("Save Video", self)
        self.save_image_button = QPushButton("Save Image", self)

        # Connect buttons to functions
        self.save_video_button.clicked.connect(self.save_video)
        self.save_image_button.clicked.connect(self.save_image)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.save_video_button)
        layout.addWidget(self.save_image_button)

    def save_video(self):
        self.user_choice = "video"
        self.accept()

    def save_image(self):
        self.user_choice = "image"
        self.accept()


class VideoExportDialog(QDialog):
    def __init__(self, window, output_file):
        super().__init__()
        self.setWindowTitle("Export Video")
        self._window = window
        self.user_choice = None  # Attribute to store user's choice

        self._update_status = QLabel("Starting export")

        # Buttons for video and image
        self._cancel_button = QPushButton("Cancel", self)

        # Connect buttons to functions
        self._cancel_button.clicked.connect(self.accept)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self._update_status)
        layout.addWidget(self._cancel_button)

        # Thread
        self.thread = WorkerExport(self, output_file=output_file)
        self.thread._update_signal.connect(self.update_export_progress)
        self.thread.start()

    def update_export_progress(self, status):
        if status == "*":
            self._update_status.setText("Export complete")
            self._cancel_button.setText("Ok")
        else:
            self._update_status.setText(status)
