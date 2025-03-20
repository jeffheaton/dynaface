from typing import Callable

import worker_threads
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dynaface import facial


class SaveVideoDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Save Options")
        self.user_choice = None  # Attribute to store user's choice

        # Buttons for video and image
        self.save_document_button = QPushButton("Save Document", self)
        self.save_video_button = QPushButton("Save Video", self)
        self.save_image_button = QPushButton("Save Image", self)
        self.save_data_button = QPushButton("Save Data (CSV)", self)

        # Connect buttons to functions
        self.save_document_button.clicked.connect(self.save_document)
        self.save_video_button.clicked.connect(self.save_video)
        self.save_image_button.clicked.connect(self.save_image)
        self.save_data_button.clicked.connect(self.save_data)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.save_document_button)
        layout.addWidget(self.save_video_button)
        layout.addWidget(self.save_image_button)
        layout.addWidget(self.save_data_button)

    def save_document(self):
        self.user_choice = "document"
        self.accept()

    def save_video(self):
        self.user_choice = "video"
        self.accept()

    def save_image(self):
        self.user_choice = "image"
        self.accept()

    def save_data(self):
        self.user_choice = "data"
        self.accept()


class SaveImageDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Save Options")
        self.user_choice = None  # Attribute to store user's choice

        # Buttons for video and image
        self.save_document_button = QPushButton("Save Document", self)
        self.save_image_button = QPushButton("Save Image", self)
        self.save_data_button = QPushButton("Save Data (CSV)", self)

        # Connect buttons to functions
        self.save_document_button.clicked.connect(self.save_document)
        self.save_image_button.clicked.connect(self.save_image)
        self.save_data_button.clicked.connect(self.save_data)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.save_document_button)
        layout.addWidget(self.save_image_button)
        layout.addWidget(self.save_data_button)

    def save_document(self):
        self.user_choice = "document"
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
        app = QApplication.instance()

        tilt_threshold = app.tilt_threshold
        face = facial.AnalyzeFace(
            self._calcs, data_path=None, tilt_threshold=tilt_threshold
        )
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


class PleaseWaitDialog(QDialog):
    def __init__(self, window, f: Callable, message: str = "Waiting"):
        super().__init__()
        self.setWindowTitle("Waiting")
        self._window = window
        self.did_cancel = False

        self._update_status = QLabel(message)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self._update_status)

        # Set window flags to disable maximize and close buttons
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        # Set fixed size for the dialog
        self.setFixedSize(300, 100)  # You can adjust the size as needed

        # Thread
        self.thread = worker_threads.WorkerPleaseWait(f)
        self.thread.finished.connect(self.close)  # Close dialog when thread finishes
        self.thread.start()
        self.thread.update_signal.connect(self.thread_done)

    def thread_done(self):
        self.accept()


def display_please_wait(window: QWidget, f: Callable, message: str = "Waiting") -> None:
    dlog = PleaseWaitDialog(window=window, f=f, message=message)
    dlog.exec()


def prompt_save_changes():
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Warning)
    msg_box.setText("You have unsaved changes")
    msg_box.setInformativeText("Do you want to save your changes?")
    msg_box.setStandardButtons(
        QMessageBox.StandardButton.Yes
        | QMessageBox.StandardButton.No
        # | QMessageBox.StandardButton.Cancel
    )
    msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return msg_box.exec()


def save_as_document(
    window: QWidget,
    caption: str,
    defaultSuffix: str,
    filter: str,
    initialFilter: str,
    required_ext: list,
    directory: str = "",  # Move the default parameter to the end
) -> str:
    dialog = QFileDialog(window, caption)
    dialog.setFileMode(QFileDialog.FileMode.AnyFile)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    dialog.setNameFilters([filter])
    if initialFilter:
        dialog.selectNameFilter(initialFilter)
    dialog.setDirectory(directory)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, False)
    dialog.setDefaultSuffix(defaultSuffix)

    if dialog.exec() == QFileDialog.DialogCode.Accepted:
        filenames = dialog.selectedFiles()
        if filenames:
            filename = filenames[0]
            # Check the file extension
            extension = filename.split(".")[-1]
            if extension not in required_ext:
                window.display_message_box(
                    "Invalid File Extension",
                    f"Filename must end in one of {required_ext}.",
                )
                return None
            return filename
    return None
