from typing import Callable, Optional

import worker_threads
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dynaface import facial


class SaveVideoDialog(QDialog):
    def __init__(self, is_video: bool):
        super().__init__()
        self.setWindowTitle("Save Options")
        self.user_choice = None  # Attribute to store user's choice

        # Buttons common to both video and image
        self.save_document_button = QPushButton("Save Document", self)
        self.save_image_button = QPushButton("Save Image", self)
        self.save_data_button = QPushButton("Save Data (CSV)", self)

        # Layout setup
        layout = QVBoxLayout(self)
        layout.addWidget(self.save_document_button)

        # Conditionally add the video option
        if is_video:
            self.save_video_button = QPushButton("Save Video", self)
            self.save_video_button.clicked.connect(self.save_video)
            layout.addWidget(self.save_video_button)

        layout.addWidget(self.save_image_button)
        layout.addWidget(self.save_data_button)

        # Connect buttons to functions
        self.save_document_button.clicked.connect(self.save_document)
        self.save_image_button.clicked.connect(self.save_image)
        self.save_data_button.clicked.connect(self.save_data)

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
    if dlog.thread.aborted:
        return False
    return True


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


class SelectPoseDialog(QDialog):
    """
    Modal dialog for choosing a face pose. Shows three images (Frontal, Profile, 3/4),
    one of which is selected by default.  Call get_choice() after exec() to retrieve
    'frontal', 'profile', or 'quarter'.
    """

    def __init__(self, default_pose: str = "frontal"):
        super().__init__()
        self.setWindowTitle("Select Pose")
        self.user_choice: Optional[str] = None

        # Store buttons for manual exclusive logic
        self._buttons: list[QPushButton] = []

        # Main vertical layout
        main_layout = QVBoxLayout(self)

        # Row for pose options
        row = QHBoxLayout()
        main_layout.addLayout(row)

        # Define poses: (label text, image filename, key)
        poses = [
            ("Frontal", "pose-frontal.png", "frontal"),
            ("Lateral", "pose-profile.png", "profile"),
            ("3/4", "pose-quarter.png", "quarter"),
        ]

        # Create each button manually
        for label_text, img_file, key in poses:
            col = QVBoxLayout()
            # Label above the image
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(label)

            # Image button
            btn = QPushButton(self)
            btn.setCheckable(True)
            btn.setIcon(QIcon(f"data/{img_file}"))
            btn.setIconSize(QSize(120, 120))
            btn.setFixedSize(130, 150)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

            # Connect click to manual handler
            btn.clicked.connect(lambda checked, b=btn, k=key: self._select(b, k))

            # Add to list
            self._buttons.append(btn)
            col.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
            row.addLayout(col)

            # Initialize default
            if key == default_pose:
                btn.setChecked(True)
                self.user_choice = key

        # OK and Cancel buttons
        ctrl_layout = QHBoxLayout()
        main_layout.addLayout(ctrl_layout)
        ctrl_layout.addStretch()
        ok_btn = QPushButton("OK", self)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        ctrl_layout.addWidget(ok_btn)
        ctrl_layout.addWidget(cancel_btn)

    def _select(self, button: QPushButton, key: str) -> None:
        """
        Manually enforce single selection: uncheck all, then check chosen button.
        Update user_choice to the provided key.
        """
        # Uncheck everyone
        for btn in self._buttons:
            btn.setChecked(False)
        # Check the one clicked
        button.setChecked(True)
        # Record choice
        self.user_choice = key

    def get_choice(self) -> Optional[str]:
        """
        Returns the key ('frontal', 'profile', or 'quarter'), or None if cancelled.
        """
        return self.user_choice
