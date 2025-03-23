import csv
import io
import logging

import cmds
import custom_control
import cv2
import dlg_modal
import dynaface_document
import numpy as np
import utl_gfx
import utl_print
import worker_threads
from dynaface.facial import AnalyzeFace
from dynaface.measures import AnalyzeDentalArea, AnalyzeEyeArea, all_measures
from jth_ui import app_jth, utl_etc
from jth_ui.tab_graphic import TabGraphic
from matplotlib.figure import Figure
from PIL import Image
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QPixmap, QUndoStack
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGestureEvent,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPinchGesture,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

MAX_FRAMES = 5000
GRAPH_MAX = 100


class AnalyzeVideoTab(TabGraphic):
    def __init__(self, window, path):
        super().__init__(window)
        self.unsaved_changes = False

        self._auto_update = False

        # Load the face
        self._frames = []
        self._frame_begin = 0
        self._frame_end = 0
        self.frame_rate = 30
        self._frame_step = 1  # The scale of the video graph

        if path.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".heic")):
            self.load_image(path)
        elif path.lower().endswith(".dyfc"):
            self.filename = path
            self.load_document(path)
        else:
            self.begin_load_video(path)
            self.filename = None
        self._chart_view = None

        # Horiz toolbar
        tab_layout = QVBoxLayout(self)
        self.init_top_horizontal_toolbar(tab_layout)

        self._tab_content_layout = QHBoxLayout()

        # Create a horizontal layout for the content of the tab
        self._content_layout = QVBoxLayout()
        self._tab_content_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        tab_layout.addLayout(self._tab_content_layout)
        self.init_vertical_toolbar(self._tab_content_layout)
        self._tab_content_layout.addLayout(self._content_layout)

        # self._content_layout.removeWidget(self._view)
        self._graph_splitter = QSplitter(Qt.Orientation.Vertical)

        self._graph_splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._content_layout.addWidget(self._graph_splitter)

        # self.init_graphics(self._content_layout)
        self.init_graphics(self._graph_splitter)

        self.loading = False
        # Video bar
        self.init_bottom_horizontal_toolbar(tab_layout)

        # Allow touch zoom
        # self.grabGesture(Qt.GestureType.PinchGesture)
        self._auto_update = True

        # Undo stack
        self._undo_stack = QUndoStack(self)

        # If the filename is set, we loaded a document, that is already decoded
        # If there are frames already defined (1), we loaded a single image.
        self._last_etc = ""
        if self.filename is None and len(self._frames) == 0:
            self.thread = worker_threads.WorkerLoad(self)
            self.thread._update_signal.connect(self.update_load_progress)
            self.thread.start()
        else:
            self.thread = None
            self.load_first_frame()
            self.lbl_status.setText(self.status())
            self._video_slider.setRange(0, len(self._frames) - 1)

        self.unsaved_changes = False
        self._window.update_enabled()

    def begin_load_video(self, path):
        app = QApplication.instance()

        # Open the video file
        self.base_rotation = None
        self.video_stream = cv2.VideoCapture(path)

        # Check if video file opened successfully
        if not self.video_stream.isOpened():
            raise ValueError("Unknown video format")

        # Get video width and height
        width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video dimensions: width={width}, height={height}")

        # Get the frame rate of the video
        self.frame_rate = int(self.video_stream.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_length = self.frame_count / self.frame_rate

        if self.frame_count > MAX_FRAMES:
            self._window.display_message_box(
                f"""This program can open at most {MAX_FRAMES} video frames, you have {self.frame_count} frames.
and is designed mainly to analyze short video clips. Please cut this video down to just the
gesture you wish to analyze."""
            )
            self.video_stream.release()
            self.video_stream = None
            self.frame_count = 0
            self.frame_rate = 1
            self.video_length = 0
            raise ValueError("Video too long")

        logger.info(f"Frame rate: {self.frame_rate}")
        logger.info(f"Frame count: {self.frame_count}")
        logger.info(f"Video length: {self.video_length}")

        # Prepare facial analysis
        tilt_threshold = app.tilt_threshold
        self._face = AnalyzeFace(all_measures(), tilt_threshold=tilt_threshold)
        self._face.measures = self.get_init_measures()

    def get_init_measures(self):
        measures = all_measures()
        for measure in measures:
            measure.set_enabled(False)
        return measures

    def init_bottom_horizontal_toolbar(self, layout):
        toolbar = QToolBar()
        layout.addWidget(toolbar)  # Add the toolbar to the layout first

        # Back Button
        self._btn_backward = QPushButton()
        self._btn_backward.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward)
        )
        self._btn_backward.clicked.connect(self.backward_action)
        toolbar.addWidget(self._btn_backward)

        # Start Button
        self._btn_play = QPushButton()
        self._btn_play.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self._btn_play.clicked.connect(self.action_play_pause)
        toolbar.addWidget(self._btn_play)

        # Forward Button
        self._btn_forward = QPushButton()
        self._btn_forward.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
        )
        self._btn_forward.clicked.connect(self.forward_action)
        toolbar.addWidget(self._btn_forward)
        toolbar.addSeparator()

        self.lbl_status = QLabel("0 / 0")
        toolbar.addWidget(self.lbl_status)

        self._video_slider = QSlider(Qt.Orientation.Horizontal)
        self._video_slider.setRange(0, 0)
        self._video_slider.valueChanged.connect(self.action_video_seek)
        toolbar.addWidget(self._video_slider)

    def init_top_horizontal_toolbar(self, layout):
        toolbar = QToolBar()
        layout.addWidget(toolbar)  # Add the toolbar to the layout first

        self._chk_landmarks = QCheckBox("Landmarks")
        toolbar.addWidget(self._chk_landmarks)
        self._chk_landmarks.stateChanged.connect(self.action_landmarks)

        self._chk_measures = QCheckBox("Measures")
        toolbar.addWidget(self._chk_measures)
        self._chk_measures.setChecked(True)
        self._chk_measures.stateChanged.connect(self.action_measures)

        self._chk_graph = custom_control.CheckingCheckBox(title="Chart", parent=self)
        toolbar.addWidget(self._chk_graph)
        self._chk_graph.setChecked(False)
        self._chk_graph.stateChanged.connect(self.action_graph)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Zoom(%): ", toolbar))
        self._spin_zoom = QSpinBox()
        toolbar.addWidget(self._spin_zoom)
        self._spin_zoom.setMinimum(1)
        self._spin_zoom.setMaximum(200)
        self._spin_zoom.setSingleStep(5)
        self._spin_zoom.setValue(100)  # Starting value
        self._spin_zoom.setFixedWidth(60)  # Adjust the width as needed
        self._spin_zoom.valueChanged.connect(self.action_zoom)

        self._spin_zoom_chart = QSpinBox()
        toolbar.addWidget(self._spin_zoom_chart)
        self._spin_zoom_chart.setMinimum(1)
        self._spin_zoom_chart.setMaximum(200)
        self._spin_zoom_chart.setSingleStep(5)
        self._spin_zoom_chart.setValue(100)  # Starting value
        self._spin_zoom_chart.setFixedWidth(60)  # Adjust the width as needed
        self._spin_zoom_chart.valueChanged.connect(self.action_zoom_chart)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(self.fit)
        toolbar.addWidget(btn_fit)
        toolbar.addSeparator()

        self._btn_cut_left = QPushButton("Cut <")
        toolbar.addWidget(self._btn_cut_left)
        self._btn_cut_left.clicked.connect(self.action_cut_left)

        self._btn_cut_right = QPushButton("Cut >")
        toolbar.addWidget(self._btn_cut_right)
        self._btn_cut_right.clicked.connect(self.action_cut_right)

        self._btn_restore = QPushButton("Restore")
        toolbar.addWidget(self._btn_restore)
        self._btn_restore.clicked.connect(self.action_restore)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Text: ", toolbar))
        self._spin_text_zoom = QSpinBox()
        toolbar.addWidget(self._spin_text_zoom)
        self._spin_text_zoom.setMinimum(1)
        self._spin_text_zoom.setMaximum(5)
        self._spin_text_zoom.setSingleStep(1)
        self._spin_text_zoom.setValue(1)  # Starting value
        self._spin_text_zoom.setFixedWidth(60)  # Adjust the width as needed
        self._spin_text_zoom.valueChanged.connect(self.action_text_zoom)

        # Create a combo box and add items
        self._jump_to = QComboBox()
        self._jump_to.addItem("Jump To")
        self._jump_to.addItem("Max Dental")
        self._jump_to.addItem("Max Ocular")
        toolbar.addWidget(self._jump_to)
        self._jump_to.currentIndexChanged.connect(self.jump_option_selected)

        toolbar.addSeparator()
        self._btn_eval = QPushButton("Eval")
        self._btn_eval.clicked.connect(self.action_eval)
        toolbar.addWidget(self._btn_eval)

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

        # Store checkboxes in a list for easy access
        self._tree = QTreeWidget()
        self._tree.header().hide()
        # self._tree.itemChanged.connect(on_item_changed)

        for measure in self._face.measures:
            parent = QTreeWidgetItem(self._tree)
            parent.setText(0, measure.abbrev())
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            parent.setCheckState(
                0, Qt.CheckState.Checked if measure.enabled else Qt.CheckState.Unchecked
            )
            parent.setData(0, Qt.ItemDataRole.UserRole, measure)

            for item in measure.items:
                child = QTreeWidgetItem(parent)
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setText(0, item.name)
                child.setCheckState(
                    0,
                    (
                        Qt.CheckState.Checked
                        if (measure.enabled and item.enabled)
                        else Qt.CheckState.Unchecked
                    ),
                )
                child.setData(0, Qt.ItemDataRole.UserRole, item)

        self._tree.itemChanged.connect(self.on_tree_item_changed)
        self.left_toolbar.addWidget(self._tree)
        # Connect buttons to slot functions
        self.all_button.clicked.connect(self.check_all)
        self.none_button.clicked.connect(self.uncheck_all)

    def check_measure(self, item):
        measure = item.data(0, Qt.ItemDataRole.UserRole)
        if self._face.lateral and not measure.is_lateral:
            self._window.display_message_box("Measure requires frontal view of face.")
            return False
        if not self._face.lateral and not measure.is_frontal:
            self._window.display_message_box("Measure requires lateral view of face.")
            return False
        return True

    def on_tree_item_changed(self, item, column):
        """Handle changes to the checkboxes on the measures."""
        try:
            self._tree.blockSignals(True)
            if not self.check_measure(item):
                item.setCheckState(0, Qt.CheckState.Unchecked)
                QApplication.processEvents()
            else:
                # Check if the item is a parent
                if item.childCount() > 0 and column == 0:
                    # Parent item
                    # Update all children based on the parent's state
                    for i in range(item.childCount()):
                        child = item.child(i)
                        child.setCheckState(0, item.checkState(0))
                else:
                    # Child item
                    # Update parent based on children's state
                    parent = item.parent()
                    if parent is not None:
                        all_unchecked = all(
                            parent.child(i).checkState(0) == Qt.CheckState.Unchecked
                            for i in range(parent.childCount())
                        )
                        all_checked = all(
                            parent.child(i).checkState(0) == Qt.CheckState.Checked
                            for i in range(parent.childCount())
                        )

                        if all_unchecked:
                            parent.setCheckState(0, Qt.CheckState.Unchecked)
                        else:
                            parent.setCheckState(0, Qt.CheckState.Checked)
        except Exception as e:
            logger.error("Error updating measures", exc_info=True)
        finally:
            self._tree.blockSignals(False)

        if self._auto_update:
            self.update_items()
            self.update_face()
            if self._chk_graph.isChecked():
                logger.debug("Update chart, because measures changed")
                self.update_chart()
                self.render_chart()

    def update_items(self):
        """Sync the tree of measures/sub-measures with the face analysis."""
        root = self._tree.invisibleRootItem()
        child_count = root.childCount()

        for i in range(child_count):
            parent_item = root.child(i)
            self._update_item(parent_item)
            self._update_children(parent_item)

    def _update_item(self, item):
        if item is not None:
            # Assuming the checkbox is in the first column
            is_checked = item.checkState(0) == Qt.CheckState.Checked

            # Accessing the first user data item
            user_data = item.data(0, Qt.ItemDataRole.UserRole)
            if user_data is not None:
                # Update the 'enabled' property based on the checkbox state
                user_data.enabled = is_checked

    def _update_children(self, parent_item):
        child_count = parent_item.childCount()
        for i in range(child_count):
            child_item = parent_item.child(i)
            self._update_item(child_item)
            # If the child item has its own children, call _update_children recursively
            self._update_children(child_item)

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
        self._face.draw_static()

        if self._face.render_img is not None and self._render_buffer is not None:
            self._render_buffer[:, :] = self._face.render_img
            self.update_graphic(resize=False)

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self.gestureEvent(event)
        return super().event(event)

    def on_close(self):
        if self.loading:
            logger.info("Closed analyze video tab (during load)")
        else:
            logger.info("Closed analyze video tab")

        if self.thread is not None:
            self.loading = False
            self.thread.running = False

    def on_resize(self):
        pass

    def on_copy(self):
        logging.info(f"Copy image: {self._face.render_img.shape}")
        utl_gfx.copy_image_to_clipboard(self._face.render_img)

    def check_all(self):
        """Check all checkboxes in top-level tree items."""

        self._auto_update = False
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            item.setCheckState(0, Qt.CheckState.Checked)
        self._auto_update = True
        self.update_items()
        self.update_face()

        if self._chart_view is not None:
            logger.debug("Update chart, because measures changed")
            self.update_chart()
            self.render_chart()

    def uncheck_all(self):
        self._auto_update = False
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self._auto_update = True
        self.update_items()
        self.update_face()

        if self._chart_view is not None:
            logger.debug("Update chart, because measures changed")
            self.update_chart()
            self.render_chart()

    def update_load_progress(self, status):
        if len(status) > 0 and status[0] == "*":
            if len(self._frames) < 1:
                self._window.display_message_box("No faces found in that video")
                logger.info("No faces found in video")
                self._window.close_action()
            # Loading complete
            if self._video_slider.value() == self._video_slider.maximum():
                # If at the end, then move to beginning
                self._video_slider.setValue(0)
            self.update_top_message("")
        else:
            # self.lbl_status.setText(status)
            self.lbl_status.setText(self.status(status))

            if self._view is None:
                self.load_first_frame()

            self._window.update_enabled()

    def load_first_frame(self):
        if len(self._frames) > 0:
            logger.debug("Display first video frame on load")
            self._face.load_state(self._frames[0])
            self.create_graphic(buffer=self._face.render_img, msg_overlay=True)
            self._view.grabGesture(Qt.GestureType.PinchGesture)
            self._view.installEventFilter(self)
            self.update_face()
            logger.debug("Done, display first video frame on load")
            # Auto fit
            QTimer.singleShot(1, self.fit)

    def add_frame(self, state):
        if self._video_slider.value() == self._video_slider.maximum():
            at_end = True
        else:
            at_end = False
        self._frames.append(state)
        self._video_slider.setRange(0, len(self._frames) - 1)
        self._frame_end = len(self._frames)

        # If we were at the end, move back to the end
        if at_end:
            self._video_slider.setValue(self._video_slider.maximum())

    def forward_action(self):
        i = self._video_slider.sliderPosition()

        # Auto move back to the beginning if at last frame, if not running
        if (i == self._video_slider.maximum()) and not self._running:
            self._video_slider.setSliderPosition(0)
        elif i < (self._video_slider.maximum()):
            self._video_slider.setSliderPosition(i + 1)
        elif self._running:
            self.stop_animate()
            self._btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )

    def backward_action(self):
        i = self._video_slider.sliderPosition()
        if i > 0:
            self._video_slider.setSliderPosition(i - 1)

    def status(self, etc=None):
        i = self._video_slider.sliderPosition() - self._frame_begin
        mx = self._video_slider.maximum()
        frame_count = self._frame_end - self._frame_begin
        if self.loading == False and len(self._frames) == 0:
            return "(0/0)"
        elif self.loading:
            if not etc:
                etc = self._last_etc
            else:
                self._last_etc = etc
            self.update_top_message("Loading... " + etc)
            return f"({mx:,} / {self.frame_count:,}, loading... time: {etc})"
        else:
            if self._face.landmarks is None:
                self.update_top_message("No face detected")
            else:
                self.update_top_message("")
            return f"({i+1:,} / {frame_count:,})"

    def update_top_message(self, message):
        if self._view:
            self._view.message = message
            self._view.update()
            self._scene.update()

    def open_frame(self, num=None):
        if num is None:
            num = self._video_slider.sliderPosition()
        frame = self._frames[num]
        self._face.load_state(frame)
        self.update_face()

    def action_play_pause(self):
        if not self._running:
            # Auto move back to the beginning if at last frame
            if self._video_slider.value() == self._video_slider.maximum():
                self._video_slider.setValue(0)

            self._btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            )
            self.start_game()
            self.init_animate(30)
        else:
            self._btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )
            self.stop_animate()

    def running_step(self):
        self.forward_action()

    def action_video_seek(self, _):
        try:
            self.open_frame()
            self.lbl_status.setText(self.status())
            if self._chart_view:
                # self.update_chart()
                current_frame = self._video_slider.value() - self._frame_begin
                self._frame_line.set_xdata([current_frame])
                self.render_chart()
        except Exception as e:
            current_frame = self._video_slider.value() - self._frame_begin
            logger.error(
                f"Error moving to new video frame: {current_frame}", exc_info=True
            )

    def action_zoom(self, value):
        z = value / 100
        self._view.resetTransform()
        self._view.scale(z, z)

    def action_zoom_chart(self, value):
        if self.loading:
            return
        z = value / 100
        self._chart_view.resetTransform()
        self._chart_view.scale(z, z)

    def _fit_face(self):
        view_size = self._view.size()
        scene_rect = self._scene.sceneRect()
        x_scale = view_size.width() / scene_rect.width()
        y_scale = view_size.height() / scene_rect.height()
        scale_factor = (
            min(x_scale, y_scale) * 100
        )  # Scale factor adjusted for action_zoom
        self.action_zoom(int(scale_factor))
        self._spin_zoom.setValue(int(scale_factor))

    def _fit_chart(self):
        view_size = self._chart_view.size()
        scene_rect = self._chart_scene.sceneRect()
        x_scale = view_size.width() / scene_rect.width()
        y_scale = view_size.height() / scene_rect.height()
        scale_factor = (
            min(x_scale, y_scale) * 100
        )  # Scale factor adjusted for action_zoom
        self.zoom_chart(int(scale_factor))
        self._spin_zoom_chart.setValue(int(scale_factor))

    def fit(self):
        self._fit_face()
        if self._chk_graph.isChecked():
            self._fit_chart()

    def _save_as_image(self):
        filename = dlg_modal.save_as_document(
            window=self._window,
            caption="Save Image",
            defaultSuffix="jpeg",
            filter="Images (*.png *.jpeg *.jpg)",
            initialFilter=None,
            required_ext=["png", "jpeg", "jpg"],
            directory=self._window.app.state[app_jth.STATE_LAST_FOLDER],
        )

        if not filename:
            return

        self._face.save(filename)

    def _save_as_video(self):
        filename = dlg_modal.save_as_document(
            window=self._window,
            caption="Save Video",
            defaultSuffix="jpeg",
            filter="Videos (*.mp4)",
            initialFilter=None,
            required_ext=["mp4"],
            directory=self._window.app.state[app_jth.STATE_LAST_FOLDER],
        )

        if not filename:
            return

        if not self.wait_load_complete():
            return
        dialog = dlg_modal.VideoExportDialog(self, filename)
        dialog.exec()

    def _save_as_data(self):
        filename = dlg_modal.save_as_document(
            window=self._window,
            caption="Save CSV Data",
            defaultSuffix="csv",
            filter="CSV Data (*.csv)",
            initialFilter=None,
            required_ext=["csv"],
            directory=self._window.app.state[app_jth.STATE_LAST_FOLDER],
        )

        if not filename:
            return

        self.save_csv(filename)

    def on_save_as(self):
        try:
            # Create and show the dialog
            dialog = dlg_modal.SaveVideoDialog()
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if dialog.user_choice == "image":
                    self._save_as_image()
                elif dialog.user_choice == "video":
                    self._save_as_video()
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

    def collect_data(self, step_size=1):
        app = QApplication.instance()

        tilt_threshold = app.tilt_threshold
        face = AnalyzeFace(self._face.measures, tilt_threshold=tilt_threshold)
        stats = face.get_all_items()
        data = {stat: [] for stat in stats}

        cols = list(data.keys())
        cols = ["frame", "time"] + cols

        frames = range(self._frame_begin, self._frame_end, step_size)

        for i in frames:
            frame = self._frames[i]
            face.load_state(frame)
            rec = face.analyze()
            for stat in rec.keys():
                if stat in data:
                    data[stat].append(rec[stat])
        frame_count = self._frame_end - self._frame_begin
        data["frame"] = list(range(0, frame_count, step_size))
        return data

    def save_csv(self, filename):
        data = self.collect_data()

        with open(filename, "w") as f:
            writer = csv.writer(f)
            cols = list(data.keys())
            if "frame" in cols:
                cols.remove("frame")
            if "time" in cols:
                cols.remove("time")
            writer.writerow(["frame", "time"] + cols)
            all_stats = self._face.get_all_items()
            l = len(data[all_stats[0]])
            rt = 1.0 / self.frame_rate
            lst_time = [round(x * rt, 2) for x in range(l)]

            for i in range(l):
                row = [str(i), lst_time[i]]
                for col in cols:
                    row.append(data[col][i])
                writer.writerow(row)

    def update_chart(self):
        """Create the chart object, or update it if already there."""

        # Do we need to scale?
        count = self._frame_end - self._frame_begin
        if count > GRAPH_MAX:
            self._frame_step = int(count / GRAPH_MAX)
        else:
            self._frame_step = 1

        # Create a Matplotlib figure
        self.chart_fig = Figure(figsize=(12, 2.5), dpi=100)
        ax = self.chart_fig.add_subplot(111)
        self.chart_fig.subplots_adjust(right=0.75)  # Adjust this value as needed

        data = self.collect_data(self._frame_step)
        plot_stats = data.keys()
        lst_time = data["frame"]  # list(range(l))

        for stat in data.keys():
            if stat in plot_stats and stat != "frame":
                ax.plot(lst_time, data[stat], label=stat)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        # ax.legend()
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1.04))

        # Add the red vertical bar at current_frame
        current_frame = self._video_slider.value() - self._frame_begin
        self._frame_line = ax.axvline(x=current_frame, color="red", linewidth=2)

    def render_chart(self):
        """Now that the chart has been created, render it."""
        # Render figure to a buffer, going in and out of PNG is not ideal, but seems fast enough
        # will find more direct route later.
        buf = io.BytesIO()
        self.chart_fig.savefig(buf, format="png")
        buf.seek(0)

        # Create QPixmap from buffer
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), format="png")
        pixmap = utl_gfx.crop_pixmap(pixmap, 5)

        if self._chart_view is None:
            logger.debug("New chart created")
            self._chart_scene = QGraphicsScene()
            self._chart_scene.setBackgroundBrush(QColor("white"))
            self._chart_pixmap_item = self._chart_scene.addPixmap(pixmap)

            # Create and configure QGraphicsView
            self._chart_view = QGraphicsView(self._chart_scene)
            self._chart_view.setBackgroundBrush(QColor("white"))
            self._chart_view.grabGesture(Qt.GestureType.PinchGesture)
            self._chart_view.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
            )
            self._chart_view.installEventFilter(self)
            self._chart_view.setTransformationAnchor(
                QGraphicsView.ViewportAnchor.AnchorUnderMouse
            )
            self._chart_view.setResizeAnchor(
                QGraphicsView.ViewportAnchor.AnchorUnderMouse
            )
            self._chart_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

            self._chart_view.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._chart_view.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._chart_view.setContentsMargins(0, 0, 0, 0)

            self._graph_splitter.addWidget(self._chart_view)

            # size the splitter (just the first time to 3/4)
            height = self.height()
            self._graph_splitter.setSizes([height // 4, 3 * height // 4])
        else:
            logger.debug("Update existing chart")
            # Update the scene with the new pixmap
            # self._chart_scene.clear()
            # self._chart_scene.addPixmap(pixmap)
            # self._chart_view.update()

            self._chart_pixmap_item.setPixmap(pixmap)

    def wait_load_complete(self):
        if self.loading:
            dialog = dlg_modal.WaitLoadingDialog(self)
            dialog.exec()
            return not dialog.did_cancel

        return True

    def action_graph(self):
        if self._chk_graph.isChecked():
            if self._chart_view is not None:
                # Redisplay graph
                self._graph_splitter.addWidget(self._chart_view)
                self._chart_view.show()
            else:
                # Show graph for the first time
                self.update_chart()
                self.render_chart()
                self._adjust_chart()

        else:
            # Hide the graph
            self._chart_view.setParent(None)
            self._chart_view.hide()
            # Adjust the sizes of the remaining widgets to fill the space
            remaining_size = sum(self._graph_splitter.sizes())
            self._graph_splitter.setSizes([remaining_size])

        self._window.update_enabled()

    def set_video_range(self, frame_begin: int, frame_end: int):
        self._frame_begin = frame_begin
        self._frame_end = frame_end
        self._video_slider.setRange(self._frame_begin, self._frame_end - 1)
        self.lbl_status.setText(self.status())
        if self._chart_view is not None:
            self.update_chart()
            self.render_chart()

    def action_cut_left(self):
        if self.loading:
            self._window.display_message_box("Can't cut while loading.")
        elif (
            len(self._frames) > 1
            and self._video_slider.sliderPosition() < self._frame_end
        ):
            cmd = cmds.CommandClip(
                self, self._video_slider.sliderPosition(), self._frame_end - 1
            )
            self._undo_stack.push(cmd)
            self._window.update_enabled()

    def action_cut_right(self):
        if self.loading:
            self._window.display_message_box("Can't cut while loading.")
        elif (
            len(self._frames) > 1
            and self._video_slider.sliderPosition() > self._frame_begin
        ):
            cmd = cmds.CommandClip(
                self, self._frame_begin, self._video_slider.sliderPosition()
            )
            self._undo_stack.push(cmd)
            self._window.update_enabled()

    def _adjust_chart(self):
        # Resize the QGraphicsView to fit the pixmap
        self._chart_view.fitInView(
            self._chart_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

        # Get the size of the graphics view
        view_size = self._chart_view.sizeHint()

        # Set the splitter sizes
        self._graph_splitter.setSizes(
            [(self.height() - view_size.height()), view_size.height()]
        )

        # adjust video area
        self.fit()

    def on_print(self):
        pixmap = QPixmap.fromImage(self._display_buffer)
        utl_print.print_pixmap(self._window, pixmap)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.Gesture:
            return self.handleGestureEvent(source, event)
        return super().eventFilter(source, event)

    def handleGestureEvent(self, source, event):
        if isinstance(event, QGestureEvent):
            pinch = event.gesture(Qt.GestureType.PinchGesture)
            if isinstance(pinch, QPinchGesture):
                # Check if the gesture is over the top widget or the bottom widget
                if source == self._view:
                    self.gestureEvent(pinch)
                elif (self._chart_view is not None) and (source == self._chart_view):
                    self.zoom_chart(pinch)
                return True
        return super().handleGestureEvent(event)

    def gestureEvent(self, pinch):
        # pinch = event.gesture(Qt.GestureType.PinchGesture)
        if isinstance(pinch, QPinchGesture):
            scaleFactor = pinch.scaleFactor()
            if scaleFactor > 1:
                new_value = self._spin_zoom.value() + 2
            else:
                new_value = self._spin_zoom.value() - 2
            self._spin_zoom.setValue(new_value)
            return True
        return False

    def zoom_chart(self, pinch):
        if isinstance(pinch, QPinchGesture):
            scaleFactor = pinch.scaleFactor()
            if scaleFactor > 1:
                new_value = self._spin_zoom_chart.value() + 2
            else:
                new_value = self._spin_zoom_chart.value() - 2
            self._spin_zoom_chart.setValue(new_value)
            return True
        return False

    def action_restore(self):
        cmd = cmds.CommandClip(self, 0, len(self._frames))
        self._undo_stack.push(cmd)
        self._window.update_enabled()

    def on_redo(self):
        self._undo_stack.redo()
        self._window.update_enabled()

    def on_undo(self):
        self._undo_stack.undo()
        self._window.update_enabled()

    def _save_as_document(self):
        filename = dlg_modal.save_as_document(
            window=self._window,
            caption="Save Dynaface Document",
            defaultSuffix="dyfc",
            filter="Dynaface (*.dyfc)",
            initialFilter=None,
            required_ext=["dyfc"],
            directory=self._window.app.state[app_jth.STATE_LAST_FOLDER],
        )

        if not filename:
            return

        self.save_document(filename)

    def save_document(self, filename):
        doc = dynaface_document.DynafaceDocument()
        doc.measures = self._face.measures
        doc.frames = self._frames[self._frame_begin : self._frame_end]
        doc.fps = self.frame_rate
        f = lambda: doc.save(filename)
        dlg_modal.display_please_wait(window=self, f=f, message="Saving document")
        self.unsaved_changes = False

    def load_document(self, filename):
        app = QApplication.instance()

        doc = dynaface_document.DynafaceDocument()
        f = lambda: doc.load(filename)
        dlg_modal.display_please_wait(window=self, f=f, message="Loading document")
        tilt_threshold = app.tilt_threshold
        self._face = AnalyzeFace(doc.measures, tilt_threshold=tilt_threshold)
        self._frames = doc.frames
        self.filename = filename
        self.frame_count = len(self._frames)
        self._frame_begin = 0
        self._frame_end = len(self._frames)
        self.frame_rate = doc.fps

    def on_save(self):
        if self.filename is None:
            self.on_save_as()
        else:
            self.save_document(self.filename)

    def can_undo(self) -> bool:
        """
        Check if an undo operation is available.
        Returns:
            bool: True if undo operation can be performed, False otherwise.
        """
        return self._undo_stack.canUndo()

    def can_redo(self) -> bool:
        """
        Check if a redo operation is available.
        Returns:
            bool: True if redo operation can be performed, False otherwise.
        """
        return self._undo_stack.canRedo()

    def update_enabled(self) -> None:
        if (self._frame_begin == 0) and (self._frame_end == len(self._frames)):
            self._btn_restore.setEnabled(False)
        else:
            self._btn_restore.setEnabled(True)

        if abs(self._frame_begin - self._frame_end) < 2:
            self._btn_cut_left.setEnabled(False)
            self._btn_cut_right.setEnabled(False)
        else:
            self._btn_cut_left.setEnabled(True)
            self._btn_cut_right.setEnabled(True)

        if self._chk_graph.isChecked():
            self._spin_zoom_chart.setEnabled(True)
        else:
            self._spin_zoom_chart.setEnabled(False)

        # Single image mode?
        if (self._frame_end - self._frame_begin) <= 1:
            self._btn_backward.setEnabled(False)
            self._btn_play.setEnabled(False)
            self._btn_forward.setEnabled(False)
            self._chk_graph.setEnabled(False)
            self._btn_eval.setEnabled(False)
            self._jump_to.setEnabled(False)
        else:
            self._btn_backward.setEnabled(True)
            self._btn_play.setEnabled(True)
            self._btn_forward.setEnabled(True)
            self._chk_graph.setEnabled(True)
            self._btn_eval.setEnabled(not self.loading)
            self._jump_to.setEnabled(not self.loading)

    def load_image(self, path):
        app = QApplication.instance()
        tilt_threshold = app.tilt_threshold
        if path.lower().endswith(".heic"):
            pil_image = Image.open(path)
            image_np = np.array(pil_image)
            self._face = AnalyzeFace(all_measures(), tilt_threshold=tilt_threshold)
            self._face.measures = self.get_init_measures()
            self._face.load_image(image_np, crop=True)
        else:
            self._face = utl_gfx.load_face_image(
                path,
                crop=True,
                tilt_threshold=tilt_threshold,
                stats=all_measures(),
            )
            self._face.measures = self.get_init_measures()

        self._frame_begin = 0
        self._frame_end = 1
        self._frames.append(self._face.dump_state())
        self.filename = None
        self.frame_count = 30

    def action_text_zoom(self, value):
        self._face.text_size = 0.75 + (0.25 * value)
        self.update_face()

    def checkCheckboxEvent(self, target):
        """Check to see if the graph checkbox can be selected."""
        if not self.loading:
            return True
        if not target.isChecked():
            if not self.wait_load_complete():
                return False

            self._chk_graph.setChecked(True)
        return True

    def find_max_dental(self):
        app = QApplication.instance()

        measures = [AnalyzeDentalArea()]
        tilt_threshold = app.tilt_threshold
        face = AnalyzeFace(measures, tilt_threshold=tilt_threshold)
        frames = range(self._frame_begin, self._frame_end, 1)

        max_smile_idx = max_smile = -1
        for i in frames:
            frame = self._frames[i]
            face.load_state(frame)
            rec = face.analyze()
            da = rec["dental_area"]
            if max_smile == -1 or da > max_smile:
                max_smile_idx = i
                max_smile = da

        return max_smile_idx

    def find_max_ocular(self):
        app = QApplication.instance()

        measures = [AnalyzeEyeArea()]
        tilt_threshold = app.tilt_threshold
        face = AnalyzeFace(measures, tilt_threshold=tilt_threshold)
        frames = range(self._frame_begin, self._frame_end, 1)

        max_eye_idx = max_eye = -1
        for i in frames:
            frame = self._frames[i]
            face.load_state(frame)
            rec = face.analyze()
            if "eye.left" not in rec or "eye.right" not in rec:
                logger.info("Jump to max ocular, can't find ocular information")
                return -1
            ea = rec["eye.left"] + rec["eye.right"]
            if max_eye == -1 or ea > max_eye:
                max_eye_idx = i
                max_eye = ea

        return max_eye_idx

    def exec_max_dental(self):
        idx = self.find_max_dental()
        self._video_slider.setValue(idx)

    def exec_max_ocular(self):
        idx = self.find_max_ocular()
        if idx > 0:
            self._video_slider.setValue(idx)

    def jump_max_dental(self):
        logger.info("Jump max dental")
        f = lambda: self.exec_max_dental()
        dlg_modal.display_please_wait(
            window=self, f=f, message="Finding max dental display"
        )

    def jump_max_ocular(self):
        logger.info("Jump max ocular")
        f = lambda: self.exec_max_ocular()
        dlg_modal.display_please_wait(
            window=self, f=f, message="Finding max ocular area"
        )

    def jump_option_selected(self, index):
        if index == 0:
            pass
        elif index == 1:
            self.jump_max_dental()
        elif index == 2:
            self.jump_max_ocular()

        # Reset the combo box to the first item
        self._jump_to.setCurrentIndex(0)

    def action_eval(self):
        self._window.show_eval()
        self._window._eval.generate(self)
