import logging
import time

from facial_analysis import facial
from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QPushButton, QVBoxLayout, QLabel
import cv2
import worker_threads


class ChoiceDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Save Options")
        self.user_choice = None  # Attribute to store user's choice

        # Buttons for video and image
        self.save_video_button = QPushButton("Save Video", self)
        self.save_image_button = QPushButton("Save Image", self)
        self.save_data_button = QPushButton("Save Data (CSV)", self)

        # Connect buttons to functions
        self.save_video_button.clicked.connect(self.save_video)
        self.save_image_button.clicked.connect(self.save_image)
        self.save_data_button.clicked.connect(self.save_data)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.save_video_button)
        layout.addWidget(self.save_image_button)
        layout.addWidget(self.save_data_button)

    def save_video(self):
        self.user_choice = "video"
        self.accept()

    def save_image(self):
        self.user_choice = "image"
        self.accept()

    def save_data(self):
        self.user_choice = "data"
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
        self.thread = worker_threads.WorkerExport(self, output_file=output_file)
        self.thread._update_signal.connect(self.update_export_progress)
        self.thread.start()

    def update_export_progress(self, status):
        if status == "*":
            self._update_status.setText("Export complete")
            self._cancel_button.setText("Ok")
        else:
            self._update_status.setText(status)

    def obtain_data(self):
        face = facial.AnalyzeFace(self._calcs, data_path=None)
        c = len(self._dialog._window._frames)
        for i, frame in enumerate(self._dialog._window._frames):
            self._update_signal.emit(f"Exporting frame {i:,}/{c:,}...")
            face.load_state(frame)
            face.analyze()


class WaitLoadingDialog(QDialog):
    def __init__(self, window):
        super().__init__()
        self.setWindowTitle("Waiting")
        self._window = window
        self.did_cancel = False

        self._update_status = QLabel("Waiting")

        # Buttons for video and image
        self._cancel_button = QPushButton("Stop Waiting", self)

        # Connect buttons to functions
        self._cancel_button.clicked.connect(self.cancel_action)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self._update_status)
        layout.addWidget(self._cancel_button)

        # Thread
        self.thread = worker_threads.WorkerWaitLoad(self)
        self.thread._update_signal.connect(self.update_load_progress)
        self.thread.start()

    def cancel_action(self):
        self.did_cancel = True
        self.accept()

    def update_load_progress(self, status):
        if status == "*":
            self.accept()
        else:
            self._update_status.setText(status)
