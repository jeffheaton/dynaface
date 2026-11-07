import copy
import csv
import io
import logging
import os

import cmds
import custom_control
import cv2
import dlg_modal
import dynaface_document
import numpy as np
import utl_general
import utl_gfx
import utl_print
import worker_threads
from dynaface.const import Pose
from dynaface.facial import AnalyzeFace
from dynaface.lateral import LATERAL_LANDMARK_NAMES
from dynaface.measures import (
    AnalyzeDentalArea,
    AnalyzeEyeArea,
    MeasureSelection,
    all_measures,
)
from dynaface.video import FrameSeries
from jth_ui import app_jth, utl_etc
from jth_ui.tab_graphic import TabGraphic
from matplotlib.figure import Figure
from PIL import Image
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QColor, QPixmap, QUndoStack
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QGestureEvent,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
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

# Landmarks to show for lateral/profile view
LATERAL_LANDMARKS = [13, 14, 15, 16, 51, 52, 53, 54, 57, 79, 85, 90, 94]


class AnalyzeVideoTab(TabGraphic):
    def __init__(self, window, path, force_pose=Pose.DETECT):
        super().__init__(window)
        self.unsaved_changes = False
        self.load_error = False
        # An error that stopped the video import, set by WorkerLoad.
        self.load_failure = None
        self._auto_update = False
        self._path = path
        # The pose the whole file is analyzed as; Pose.DETECT classifies each
        # frame. WorkerLoad reads it for a video.
        self.force_pose = force_pose
        # True once the tab's frames came from a video file (not a photo or a
        # document), even before the import has kept more than one of them.
        self._video_source = False
        # The user's choice of measures and items, kept apart from the measures'
        # flags, which _apply_view() sets for the frame's view (a millimeter
        # measure is off on a turned or profile head). A new tab chooses nothing;
        # a document's stored flags are its choices.
        self._selection = MeasureSelection()
        # The view the measure tree's rows were last enabled for.
        self._tree_view = None

        # Load the face
        self._frames = []
        self._frame_begin = 0
        self._frame_end = 0
        self.frame_rate = 30
        self._frame_step = 1  # The scale of the video graph

        if path.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".heic")):
            self.load_image(path, force_pose=force_pose)
        elif path.lower().endswith(".dyfc"):
            self.filename = path
            if not self.load_document(path):
                self.load_error = True
                return
        else:
            self.begin_load_video(path, force_pose=force_pose)
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
            # Set here, on the main thread, so update_enabled() below (and the
            # window's, on the tab change) sees the import as running before
            # the worker has kept a frame; the worker clears it when it ends.
            self.loading = True
            self.thread.start()
        else:
            self.thread = None
            self.load_first_frame()
            self.lbl_status.setText(self.status())
            self._video_slider.setRange(0, len(self._frames) - 1)

        self.unsaved_changes = False
        self._window.update_enabled()

    def begin_load_video(self, path, force_pose=Pose.DETECT):
        app = QApplication.instance()

        # WorkerLoad analyzes every frame as this pose; set before it starts.
        self.force_pose = force_pose
        self._video_source = True

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

        self._chk_landmarks = QCheckBox("Marks")
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
        self._btn_pose = QPushButton("Pose")
        self._btn_pose.clicked.connect(self.action_pose)
        toolbar.addWidget(self._btn_pose)

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
        # The column as wide as its longest shown name, with a horizontal scroll
        # bar (the tree's default, as needed) when that is wider than the
        # toolbar, so a long item name is never cut off.
        self._tree.header().setStretchLastSection(False)
        self._tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        # self._tree.itemChanged.connect(on_item_changed)

        for measure in self._face.measures:
            parent = QTreeWidgetItem(self._tree)
            parent.setText(0, measure.abbrev())
            # Checkable but not user tristate: a click on a partly ticked row
            # ticks it (and so all its items); the partial state is set here
            # alone, from the items' rows (_measure_state()).
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            parent.setCheckState(
                0, Qt.CheckState.Checked if measure.enabled else Qt.CheckState.Unchecked
            )
            parent.setData(0, Qt.ItemDataRole.UserRole, measure)

            for item in measure.items:
                child = QTreeWidgetItem(parent)
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # A display-only item is a drawing switch with no value of its
                # own (never exported), so it reads as one.
                name = f"{item.name} (drawing)" if item.display_only else item.name
                child.setText(0, name)
                child.setCheckState(
                    0,
                    (
                        Qt.CheckState.Checked
                        if (measure.enabled and item.enabled)
                        else Qt.CheckState.Unchecked
                    ),
                )
                child.setData(0, Qt.ItemDataRole.UserRole, item)
            # A measure with some of its items off (a reopened document's, say)
            # shows a partly ticked row.
            if parent.childCount() > 0:
                parent.setCheckState(0, self._measure_state(parent))

        self._tree.itemChanged.connect(self.on_tree_item_changed)
        self.left_toolbar.addWidget(self._tree)
        # Connect buttons to slot functions
        self.all_button.clicked.connect(self.check_all)
        self.none_button.clicked.connect(self.uncheck_all)

    def on_tree_item_changed(self, item, column):
        """Handle changes to the checkboxes on the measures.

        A measure that doesn't suit the frame's view has a disabled row, which
        can't be toggled, so no choice needs refusing here. Ticking or unticking
        a measure's row ticks or unticks all of its items (a click on a partly
        ticked row ticks it). A measure's row follows its items' rows: ticked
        when all are, partly ticked when some are, unticked when none are,
        counting the disabled rows of items for another view (such as Cheek
        Measures' quarter items on a frontal frame), whose check states are
        choices too; so unticking every item a view reaches leaves the measure
        partly ticked, and on, when another view's items are still chosen.
        """
        try:
            self._tree.blockSignals(True)
            # Check if the item is a parent
            if item.childCount() > 0 and column == 0:
                # Parent item: tick or untick all its children. A partly
                # ticked state comes from the children alone, so it changes
                # none of them.
                if item.checkState(0) == Qt.CheckState.PartiallyChecked:
                    return
                for i in range(item.childCount()):
                    child = item.child(i)
                    child.setCheckState(0, item.checkState(0))
            else:
                # Child item
                # Update parent based on children's state
                parent = item.parent()
                if parent is not None:
                    parent.setCheckState(0, self._measure_state(parent))
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
        """Take the tree's check states as the user's choices, then set the
        measures' flags for the current frame's view."""
        root = self._tree.invisibleRootItem()
        child_count = root.childCount()

        for i in range(child_count):
            parent_item = root.child(i)
            self._update_item(parent_item)
            self._update_children(parent_item)

        # The tree holds the choices, a disabled (unsuited) row's included; the
        # flags hold only what the view allows.
        self._selection = MeasureSelection.of(self._face.measures)
        self._apply_view()

    def _apply_view(self):
        """Enable the chosen measures that suit the current frame's view, and
        disable the tree rows of the others, keeping every check state: a choice
        made on a frontal frame comes back when a frontal frame shows again."""
        self._selection.apply(
            self._face.measures, self._face.lateral, quarter=self._face.quarter
        )
        self._refresh_tree_views()

    def _chosen_measures(self):
        """A copy of the face's measures with their flags set to the user's
        choices, whatever the view, the way a document stores them: what a
        document and the series are built from. The face's own measures keep the
        view's flags, so playback or a frame kept by an import, which apply the
        view while a save runs, never change what is being written."""
        measures = copy.deepcopy(self._face.measures)
        for measure in measures:
            measure.enabled = self._selection.is_chosen(measure)
            for item in measure.items:
                item.enabled = measure.enabled and self._selection.is_item_chosen(
                    measure, item.name
                )
        return measures

    def _refresh_tree_views(self):
        """Disable the rows of the measures that don't suit the current view,
        and the rows of the items that don't (MeasureItem.suits(), e.g. Cheek
        Measures' frontal items on a quarter face), with a tooltip saying so,
        and enable the rest; check states stay."""
        view = (self._face.lateral, self._face.quarter)
        if view == self._tree_view:
            return
        self._tree_view = view
        # The view's name alone, the view suits() reads: which cheek is the far
        # one has no bearing on which measures suit it.
        if self._face.lateral:
            pose = Pose.LATERAL
        elif self._face.quarter:
            pose = Pose.QUARTER
        else:
            pose = Pose.FRONTAL
        tip = f"Not defined on a {utl_general.POSE_LABELS[pose]} view"
        # Disabling a row and changing its tooltip both emit itemChanged.
        self._tree.blockSignals(True)
        try:
            for i in range(self._tree.topLevelItemCount()):
                parent = self._tree.topLevelItem(i)
                measure = parent.data(0, Qt.ItemDataRole.UserRole)
                suited = measure.suits(self._face.lateral, self._face.quarter)
                parent.setDisabled(not suited)
                parent.setToolTip(0, "" if suited else tip)
                for j in range(parent.childCount()):
                    row = parent.child(j)
                    item = row.data(0, Qt.ItemDataRole.UserRole)
                    item_suited = suited and item.suits(
                        self._face.lateral, self._face.quarter
                    )
                    row.setDisabled(not item_suited)
                    row.setToolTip(0, "" if item_suited else tip)
        finally:
            self._tree.blockSignals(False)

    @staticmethod
    def _measure_state(parent):
        """A measure row's check state from its items' rows, the disabled ones
        included: Checked when every one is ticked, PartiallyChecked when some
        are, Unchecked when none are."""
        ticked = sum(
            parent.child(i).checkState(0) == Qt.CheckState.Checked
            for i in range(parent.childCount())
        )
        if ticked == 0:
            return Qt.CheckState.Unchecked
        if ticked == parent.childCount():
            return Qt.CheckState.Checked
        return Qt.CheckState.PartiallyChecked

    def _update_item(self, item):
        if item is not None:
            # Assuming the checkbox is in the first column. A partly ticked
            # measure row is on: some of its items are chosen.
            is_checked = item.checkState(0) != Qt.CheckState.Unchecked

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
            only = LATERAL_LANDMARKS if self._face.lateral else None
            self._face.draw_landmarks(
                numbers=True,
                size=0.75 * self._spin_text_zoom.value(),
                only=only,
            )
            if not self._face.lateral:
                # The cheek landmarks of a frontal or three-quarter face, with
                # their codes; a profile has none.
                self._face.draw_cheek_landmarks(
                    size=0.75 * self._spin_text_zoom.value()
                )

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
        """Check the top-level rows the current view allows (the enabled ones),
        and so all of their items, a partly ticked row's included; a disabled
        row keeps the choice it holds."""

        self._auto_update = False
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if not item.isDisabled():
                item.setCheckState(0, Qt.CheckState.Checked)
        self._auto_update = True
        self.update_items()
        self.update_face()

        if self._chart_view is not None:
            logger.debug("Update chart, because measures changed")
            self.update_chart()
            self.render_chart()

    def uncheck_all(self):
        """Uncheck every top-level row, the disabled ones included."""
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

    def enable_measure(self, name: str) -> None:
        """Enable a specific measure and all of its child items.

        The ``name`` parameter is matched against the text shown in the tree
        view, which corresponds to ``MeasureBase.abbrev()``.
        """

        self._auto_update = False
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.text(0) == name:
                item.setCheckState(0, Qt.CheckState.Checked)
                for j in range(item.childCount()):
                    child = item.child(j)
                    child.setCheckState(0, Qt.CheckState.Checked)
                break
        self._auto_update = True
        self.update_items()
        self.update_face()

        if self._chart_view is not None:
            logger.debug("Update chart, because measures changed")
            self.update_chart()
            self.render_chart()

    def update_load_progress(self, status):
        if len(status) > 0 and status[0] == "*":
            closed = False
            if self.load_failure is not None:
                message = utl_general.describe_analysis_error(self.load_failure)
                if len(self._frames) > 0:
                    message += (
                        f"\n\nLoading stopped after {len(self._frames):,} frames, "
                        "which are kept."
                    )
                self._window.display_message_box(f"Unable to load the video. {message}")
                if len(self._frames) < 1:
                    self._window.close_action()
                    closed = True
            elif len(self._frames) < 1:
                self._window.display_message_box("No faces found in that video")
                logger.info("No faces found in video")
                self._window.close_action()
                closed = True
            if closed:
                return
            # Loading complete
            if self._video_slider.value() == self._video_slider.maximum():
                # If at the end, then move to beginning
                self._video_slider.setValue(0)
            self.update_top_message("")
            # The worker cleared loading before signaling the end, so the
            # buttons an import disables (Pose, Eval, Jump To) come back now,
            # after a stopped import too.
            self.lbl_status.setText(self.status())
            self._window.update_enabled()
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

            self._face.width = self._face.render_img.shape[1]
            self._face.height = self._face.render_img.shape[0]
            # The choices (a document's stored flags, or nothing yet) for the
            # first frame's view, then, for a photo or video, the view's default
            # measure; a document keeps the choices it stored.
            self._apply_view()
            view = utl_general.view_label(self._face.pose, self._face.far_left)
            if self.filename is not None:
                logger.info(f"Keep the document's measures on a {view} view")
            elif self._face.quarter:
                logger.info(f"Enable Cheek Measures for a {view} view")
                self.enable_measure("Cheek Measures")
            elif not self._face.lateral:
                logger.info(f"Enable FAI for a {view} view")
                self.enable_measure("FAI")
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
            text = f"({i+1:,} / {frame_count:,})"
            if self._face.pose in (Pose.LATERAL, Pose.QUARTER):
                # A turned or profile frame says so beside its number.
                text += " " + utl_general.view_label(
                    self._face.pose, self._face.far_left
                )
            return text

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
        # The frame's view may differ from the last one's (a mixed-view video).
        self._apply_view()
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

        # A video's CSV is the series of every kept frame, whichever frame the
        # slider is on (a profile frame's lateral measures are in it); only a
        # single profile photo writes its sagittal outline instead.
        is_video = self._video_source or len(self._frames) > 1
        if self._face.lateral and not is_video:
            self.save_csv_lateral(filename)
        else:
            self.save_csv_frontal(filename)

    def on_save_as(self):
        try:
            # Create and show the dialog
            is_video = (self._frame_end - self._frame_begin) > 1
            dialog = dlg_modal.SaveVideoDialog(is_video=is_video)
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
        """The chosen measures' outputs for every step_size-th kept frame. Frames
        keep their numbers in the whole video, so a trimmed clip's start at its
        first kept frame.

        The series is built from the user's choices, not from the flags the
        current frame's view allows, so the CSV and the chart hold the same columns
        whichever frame the slider is on: a measure's outputs are NaN on the frames
        whose view it doesn't suit (FrameSeries.from_states()), never dropped."""
        return FrameSeries.from_states(
            self._frames,
            self._chosen_measures(),
            start=self._frame_begin,
            stop=self._frame_end,
            step=step_size,
            fps=self.frame_rate,
        )

    def save_csv_frontal(self, filename):
        self.collect_data().write_csv(filename)

    def save_csv_lateral(self, filename):
        landmarks = self._face.lateral_landmarks  # shape: (N, 2), format (x, y)
        sagittal_y = self._face.sagittal_y
        sagittal_x = self._face.sagittal_x

        combined = np.column_stack((sagittal_y, sagittal_x))  # (y, x) order

        labels = []
        for pt in combined:
            pt_xy = np.array([pt[1], pt[0]])  # convert (y, x) to (x, y)
            label = ""
            for idx, l in enumerate(landmarks):
                if np.array_equal(pt_xy, l):
                    label = LATERAL_LANDMARK_NAMES[idx]
                    break
            labels.append(label)

        combined_with_labels = np.column_stack((sagittal_y, sagittal_x, labels))

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["y", "x", "landmark"])  # header
            writer.writerows(combined_with_labels)

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

        series = self.collect_data(self._frame_step)
        # The chart counts frames from the start of the kept range, like the red
        # current-frame bar below.
        x = [frame - self._frame_begin for frame in series.frames]
        # Only the outputs the kept frames have: a measure's outputs are NaN on
        # every frame whose view it doesn't suit, and an all-NaN line would only
        # crowd the legend.
        names = utl_general.named_series(series)
        for name in names:
            ax.plot(x, series.values[name], label=name)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        if names:
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
        # Every kept frame goes in the document, so an import finishes first.
        if not self.wait_load_complete():
            return
        doc = dynaface_document.DynafaceDocument()
        # The document stores the user's choices, not what the current frame's
        # view allows, so a frame of another view loads them whole.
        doc.measures = self._chosen_measures()
        doc.frames = self._frames[self._frame_begin : self._frame_end]
        doc.fps = self.frame_rate
        doc.pose = self._face.pose
        doc.lateral = self._face.lateral
        if self._face.lateral:
            doc.lateral_landmarks = self._face.lateral_landmarks
            doc.sagittal_x = self._face.sagittal_x
            doc.sagittal_y = self._face.sagittal_y

        f = lambda: doc.save(filename)
        dlg_modal.display_please_wait(window=self, f=f, message="Saving document")
        self.unsaved_changes = False

    def load_document(self, filename):
        app = QApplication.instance()

        doc = dynaface_document.DynafaceDocument()
        f = lambda: doc.load(filename)
        if dlg_modal.display_please_wait(window=self, f=f, message="Loading document"):
            tilt_threshold = app.tilt_threshold
            self._face = AnalyzeFace(doc.measures, tilt_threshold=tilt_threshold)
            # A profile document saved by Dynaface 2.0.4 or earlier holds its
            # lateral data in the header alone: give its frames the view, so
            # the series, the Evaluation and the video export, which restore
            # each frame on a face of their own, measure them as the profile.
            doc.add_missing_views()
            self._frames = doc.frames
            self.filename = filename
            self.frame_count = len(self._frames)
            self._frame_begin = 0
            self._frame_end = len(self._frames)
            self.frame_rate = doc.fps
            self._face.lateral = doc.lateral
            self._face.lateral_landmarks = doc.lateral_landmarks
            self._face.sagittal_x = doc.sagittal_x
            self._face.sagittal_y = doc.sagittal_y
            self._face.pose = doc.pose
            # The stored flags are the choices; the first frame's load_state()
            # then applies its view (load_first_frame(), open_frame()).
            self._selection = MeasureSelection.of(self._face.measures)
            return True
        else:
            self._window.close_action()
            return False

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

        # The pose can be forced on a photo or video file, which is analyzed
        # again, but not on a document, whose frames are already analyzed.
        is_document = self._path.lower().endswith(".dyfc")
        self._btn_pose.setEnabled(not is_document and not self.loading)

    def load_image(self, path, force_pose=Pose.DETECT):
        app = QApplication.instance()
        tilt_threshold = app.tilt_threshold
        logger.info(f"Loading image: {path}, force pose: {force_pose}")
        if path.lower().endswith(".heic"):
            pil_image = Image.open(path)
            image_np = np.array(pil_image)
            self._face = AnalyzeFace(all_measures(), tilt_threshold=tilt_threshold)
            self._face.measures = self.get_init_measures()
            self._face.load_image(image_np, crop=True, force_pose=force_pose)
        else:
            self._face = utl_gfx.load_face_image(
                path,
                crop=True,
                tilt_threshold=tilt_threshold,
                stats=all_measures(),
                force_pose=force_pose,
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

    def _kept_frames(self, measures):
        """These measures' outputs for every kept frame."""
        return FrameSeries.from_states(
            self._frames,
            measures,
            start=self._frame_begin,
            stop=self._frame_end,
            fps=self.frame_rate,
        )

    def find_max_dental(self):
        """The kept frame with the widest smile, or -1 if none has a value."""
        frame = self._kept_frames([AnalyzeDentalArea()]).peak("dental_area")
        return -1 if frame is None else frame

    def find_max_ocular(self):
        """The kept frame with the widest eye opening, or -1 if none has a value."""
        frame = self._kept_frames([AnalyzeEyeArea()]).peak("eye.left", "eye.right")
        if frame is None:
            logger.info("Jump to max ocular, can't find ocular information")
            return -1
        return frame

    def exec_max_dental(self):
        idx = self.find_max_dental()
        if idx >= 0:
            self._video_slider.setValue(idx)

    def exec_max_ocular(self):
        idx = self.find_max_ocular()
        if idx >= 0:  # frame 0 can be the widest, too
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

    def _reload_with_pose(self, force_pose: Pose) -> None:
        """
        Close this tab and reopen its file analyzed as the given pose (frontal,
        lateral or three-quarter).
        """
        label = utl_general.POSE_LABELS[force_pose]
        logger.info(f"Forced {label} view")

        try:
            basename = os.path.basename(self._path)
            tab_name = f"Analyze: {basename}"
            new_tab = AnalyzeVideoTab(self._window, self._path, force_pose=force_pose)
            if new_tab.load_error:
                self._window.display_message_box(
                    f"Unable to force the {label} view for {self._path}; "
                    "invalid format."
                )
                return

            self._window.add_tab(new_tab, tab_name)
            # close the old one
            idx = self._window._tab_widget.indexOf(self)
            if idx != -1:
                self._window.close_tab(idx)

        except Exception as e:
            logger.error("Error reloading with new pose", exc_info=True)
            self._window.display_message_box(f"Unable to open file: {e}")

    def action_pose(self):
        # 1) The current pose (frontal, lateral or three-quarter) is preselected
        current = self._face.pose
        logger.info(f"Adjust pose, current pose is {utl_general.POSE_LABELS[current]}")

        # 2) Show the dialog; a video says it will be analyzed again
        is_video = self._video_source or len(self._frames) > 1
        dialog = dlg_modal.SelectPoseDialog(
            self._window, default_pose=current, video=is_video
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        chosen = dialog.get_choice()
        if chosen is None:
            return

        # 3) A video is analyzed again unless it was already forced to that
        # pose: under detection its frames may hold several views, and the
        # one under the slider says nothing about the rest. A photo whose pose
        # is already the chosen one stays as it is.
        if is_video:
            if chosen != self.force_pose:
                self._reload_with_pose(chosen)
        elif chosen != current:
            self._reload_with_pose(chosen)
