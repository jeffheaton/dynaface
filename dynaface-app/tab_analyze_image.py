import csv
import logging
from functools import partial

import dlg_modal
import dynaface_document
import numpy as np
import utl_gfx
import utl_print
from facial_analysis.facial import STATS, AnalyzeFace, load_face_image
from jth_ui import utl_etc
from jth_ui.tab_graphic import TabGraphic
from PIL import Image
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGestureEvent,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPinchGesture,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class AnalyzeImageTab(TabGraphic):
    def __init__(self, window, path):
        super().__init__(window)

        self._auto_update = False
        self.unsaved_changes = False

        # Load the face
        if path.lower().endswith(".dyfc"):
            self.load_document(path)
        else:
            self.load_image(path)

        # Horiz toolbar
        tab_layout = QVBoxLayout(self)
        self.init_horizontal_toolbar(tab_layout)

        # Create a horizontal layout for the content of the tab
        content_layout = QHBoxLayout()
        tab_layout.addLayout(content_layout)
        self.init_vertical_toolbar(content_layout)
        self.init_graphics(content_layout)

        # Prepare to display face
        self.create_graphic(buffer=self._face.render_img)
        self.update_face()

        # Scale the view as desired
        self._view.scale(1, 1)

        # Allow touch zoom
        self.grabGesture(Qt.GestureType.PinchGesture)
        self._auto_update = True

        # Auto fit
        QTimer.singleShot(1, self.fit)

    def load_image(self, path):
        if path.lower().endswith(".heic"):
            pil_image = Image.open(path)
            image_np = np.array(pil_image)
            # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            self._face = AnalyzeFace(STATS)
            self._face.load_image(image_np, crop=True)
        else:
            self._face = load_face_image(path, crop=True)

        # We did not load a document, so this forces a "save as" if a save is requested
        self.filename = None

    def init_horizontal_toolbar(self, layout):
        self._toolbar = QToolBar()
        layout.addWidget(self._toolbar)  # Add the toolbar to the layout first

        self._chk_landmarks = QCheckBox("Landmarks")
        self._toolbar.addWidget(self._chk_landmarks)
        self._chk_landmarks.stateChanged.connect(self.action_landmarks)
        self._toolbar.addSeparator()

        self._chk_measures = QCheckBox("Measures")
        self._toolbar.addWidget(self._chk_measures)
        self._chk_measures.setChecked(True)
        self._chk_measures.stateChanged.connect(self.action_measures)
        self._toolbar.addSeparator()

        self._toolbar.addWidget(QLabel("Zoom(%): ", self._toolbar))
        self._spin_zoom = QSpinBox()
        self._toolbar.addWidget(self._spin_zoom)
        self._spin_zoom.setMinimum(1)
        self._spin_zoom.setMaximum(200)
        self._spin_zoom.setSingleStep(5)
        self._spin_zoom.setValue(100)  # Starting value
        self._spin_zoom.setFixedWidth(60)  # Adjust the width as needed
        self._spin_zoom.valueChanged.connect(self.action_zoom)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(self.fit)
        self._toolbar.addWidget(btn_fit)
        self._toolbar.addSeparator()

        # btn_test = QPushButton("Test")
        # self._toolbar.addWidget(btn_test)
        # btn_test.clicked.connect(self.test_action)

    def init_vertical_toolbar(self, layout):
        # Add a vertical toolbar (left side of the tab)
        self.left_toolbar = QToolBar("Left Toolbar", self)
        self.left_toolbar.setOrientation(Qt.Orientation.Vertical)
        layout.addWidget(self.left_toolbar)

        # Create a horizontal layout for "All" and "None" buttons
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(0)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Add "All" and "None" buttons to the horizontal layout
        self.all_button = QPushButton("All", self)
        self.none_button = QPushButton("None", self)
        button_width = 50
        self.all_button.setFixedWidth(button_width)
        self.none_button.setFixedWidth(button_width)
        self.buttons_layout.addWidget(self.all_button)
        self.buttons_layout.addWidget(self.none_button)

        # Add the buttons layout to the left toolbar as a widget
        self.buttons_widget = QWidget()
        self.buttons_widget.setLayout(self.buttons_layout)
        self.left_toolbar.addWidget(self.buttons_widget)

        # Create a scrollable area for checkboxes
        self.scroll_area = QScrollArea(self)
        self.scroll_area_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)
        self.left_toolbar.addWidget(self.scroll_area)

        # Store checkboxes in a list for easy access
        self.checkboxes = []

        for stat in self._face.calcs:
            checkbox = QCheckBox(stat.abbrev())
            checkbox.stateChanged.connect(
                partial(self.checkbox_clicked, checkbox, stat)
            )
            self.scroll_area_layout.addWidget(checkbox)
            checkbox.setChecked(stat.enabled)
            self.checkboxes.append(checkbox)

        # Connect buttons to slot functions
        self.all_button.clicked.connect(self.check_all)
        self.none_button.clicked.connect(self.uncheck_all)

        # Since we just finished updating all the check boxes, which likely triggeded it
        self.unsaved_changes = False

    def action_landmarks(self, state):
        self.update_face()

    def action_measures(self, state):
        self.update_face()

    def update_face(self):
        self._face.render_reset()
        if self._chk_measures.isChecked():
            self._face.analyze()
        if self._chk_landmarks.isChecked():
            self._face.draw_landmarks(numbers=True)
        self.update_graphic(resize=False)

    def gestureEvent(self, event: QGestureEvent):
        pinch = event.gesture(Qt.GestureType.PinchGesture)
        if isinstance(pinch, QPinchGesture):
            scaleFactor = pinch.scaleFactor()
            if scaleFactor > 1:
                new_value = self._spin_zoom.value() + 2
            else:
                new_value = self._spin_zoom.value() - 2
            self._spin_zoom.setValue(new_value)
            return True
        return super().event(event)

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self.gestureEvent(event)
        return super().event(event)

    def on_close(self):
        # Check for unsaved changes here
        if self.unsaved_changes:
            response = dlg_modal.prompt_save_changes()

            if response == QMessageBox.StandardButton.Yes:
                self.on_save()
            elif response == QMessageBox.StandardButton.Cancel:
                # Cancel the tab closing
                return

    def on_resize(self):
        pass

    def on_copy(self):
        logging.info(f"Copy image: {self._face.render_img.shape}")
        utl_gfx.copy_image_to_clipboard(self._face.render_img)

    def check_all(self):
        self._auto_update = False
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
        self._auto_update = True
        self.update_face()

    def uncheck_all(self):
        self._auto_update = False
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
        self._auto_update = True
        self.update_face()

    def checkbox_clicked(self, checkbox, stat):
        stat.enabled = checkbox.isChecked()
        self.unsaved_changes = True
        if self._auto_update:
            self.update_face()

    def action_zoom(self, value):
        z = value / 100
        self._view.resetTransform()
        self._view.scale(z, z)

    def fit(self):
        view_size = self._view.size()
        scene_rect = self._scene.sceneRect()
        x_scale = view_size.width() / scene_rect.width()
        y_scale = view_size.height() / scene_rect.height()
        scale_factor = (
            min(x_scale, y_scale) * 100
        )  # Scale factor adjusted for action_zoom
        self.action_zoom(int(scale_factor))
        self._spin_zoom.setValue(int(scale_factor))

    def on_save_as(self):
        try:
            # Create and show the dialog
            dialog = dlg_modal.SaveImageDialog()
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if dialog.user_choice == "image":
                    self._save_as_image()
                elif dialog.user_choice == "data":
                    self._save_as_data()
                elif dialog.user_choice == "document":
                    self._save_as_document()
        except FileNotFoundError as e:
            logger.error("Error during save (FileNotFoundError)", exc_info=True)
            self._window.display_message_box("Unable to save file. (FileNotFoundError)")
        except PermissionError as e:
            logger.error("Error during save (PermissionError)", exc_info=True)
            self._window.display_message_box("Unable to save file. (PermissionError)")
        except IsADirectoryError as e:
            logger.error("Error during save (IsADirectoryError)", exc_info=True)
            self._window.display_message_box("Unable to save file. (IsADirectoryError)")
        except FileExistsError as e:
            logger.error("Error during save (FileExistsError)", exc_info=True)
            self._window.display_message_box("Unable to save file. (FileExistsError)")
        except OSError as e:
            logger.error("Error during save (OSError)", exc_info=True)
            self._window.display_message_box("Unable to save file. (OSError)")
        except Exception as e:
            logger.error("Error during save", exc_info=True)
            self._window.display_message_box("Unable to save file.")

    def _save_as_image(self):
        options = QFileDialog.Option.DontUseNativeDialog

        # Show the dialog and get the selected file name and format
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Images (*.png *.jpeg *.jpg)", options=options
        )

        filename = utl_etc.default_extension(filename, ".jpeg")

        if filename:
            if not (
                filename.lower().endswith(".png")
                or filename.lower().endswith(".jpg")
                or filename.lower().endswith(".jpeg")
            ):
                self._window.display_message_box(
                    "Filename must end in .png or .jpeg/.jpg"
                )
            else:
                self._face.save(filename)

    def _save_as_data(self):
        options = QFileDialog.Option.DontUseNativeDialog

        # Show the dialog and get the selected file name and format
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV Data",
            "",
            "CSV Data (*.csv)",
            options=options,
        )

        filename = utl_etc.default_extension(filename, ".csv")

        if filename:
            if not filename.lower().endswith(".csv"):
                self._window.display_message_box("Filename must end in .csv")
            else:
                self.save_csv(filename)

    def _save_as_document(self):
        options = QFileDialog.Option.DontUseNativeDialog

        # Show the dialog and get the selected file name and format
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Dynaface Document",
            "",
            "Dynaface (*.dyfc)",
            options=options,
        )

        filename = utl_etc.default_extension(filename, ".dyfc")

        if filename:
            if not filename.lower().endswith(".dyfc"):
                self._window.display_message_box("Filename must end in .dyfc")
            else:
                self.save_document(filename)

    def on_print(self):
        pixmap = QPixmap.fromImage(self._display_buffer)
        utl_print.print_pixmap(self._window, pixmap)

    def save_csv(self, filename):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            rec = self._face.analyze()
            writer.writerow(list(rec.keys()))
            writer.writerow(list(rec.values()))

    def save_document(self, filename):
        doc = dynaface_document.DynafaceDocument(dynaface_document.DOC_TYPE_IMAGE)
        doc.face = self._face
        doc.save(filename)
        self.unsaved_changes = False

    def load_document(self, filename):
        doc = dynaface_document.DynafaceDocument(dynaface_document.DOC_TYPE_IMAGE)
        doc.load(filename)
        self._face = doc.face
        self.filename = filename

    def on_save(self):
        if self.filename is None:
            self.on_save_as()
        else:
            self.save_document(self.filename)

    def update_enabled(self) -> None:
        pass
