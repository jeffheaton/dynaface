import csv
import io
import logging
from functools import partial

import custom_control
import cv2
import dlg_modal
import plotly.graph_objects as go
import utl_gfx
import utl_print
import worker_threads
from facial_analysis import facial
from jth_ui.tab_graphic import TabGraphic
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QColor, QPixmap, QUndoStack
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGestureEvent,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPinchGesture,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
import cmds
import dynaface_document

logger = logging.getLogger(__name__)

GRAPH_HORIZ = False


class AnalyzeVideoTab(TabGraphic):
    def __init__(self, window, path):
        super().__init__(window)
        self.unsaved_changes = False

        self._auto_update = False

        self._calcs = [
            facial.AnalyzeFAI(),
            facial.AnalyzeOralCommissureExcursion(),
            facial.AnalyzeBrows(),
            facial.AnalyzeDentalArea(),
            facial.AnalyzeEyeArea(),
        ]

        # Load the face
        self._frames = []
        self._frame_begin = 0
        self._frame_end = 0
        if path.lower().endswith(".dyfc"):
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
        if GRAPH_HORIZ:
            self._content_layout = QHBoxLayout()
        else:
            self._content_layout = QVBoxLayout()
        self._tab_content_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        tab_layout.addLayout(self._tab_content_layout)
        self.init_vertical_toolbar(self._tab_content_layout)
        self._tab_content_layout.addLayout(self._content_layout)

        # self._content_layout.removeWidget(self._view)
        if GRAPH_HORIZ:
            self._splitter = QSplitter(Qt.Orientation.Horizontal)
        else:
            self._splitter = QSplitter(Qt.Orientation.Vertical)

        self._splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._content_layout.addWidget(self._splitter)

        # self.init_graphics(self._content_layout)
        self.init_graphics(self._splitter)

        self.loading = False
        # Video bar
        self.init_bottom_horizontal_toolbar(tab_layout)

        # Allow touch zoom
        # self.grabGesture(Qt.GestureType.PinchGesture)
        self._auto_update = True

        # Undo stack
        self._undo_stack = QUndoStack(self)

        if self.filename is None:
            self.thread = worker_threads.WorkerLoad(self)
            self.thread._update_signal.connect(self.update_load_progress)
            self.thread.start()
        else:
            self.thread = None
            self.load_first_frame()
            self.lbl_status.setText(self.status())
            self._video_slider.setRange(0, len(self._frames) - 2)

        self.unsaved_changes = False

    def begin_load_video(self, path):
        # Open the video file
        self.video_stream = cv2.VideoCapture(path)

        # Check if video file opened successfully
        if not self.video_stream.isOpened():
            raise ValueError("Unknown video format")

        # Get the frame rate of the video
        self.frame_rate = int(self.video_stream.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_length = self.frame_count / self.frame_rate

        logger.info(f"Frame rate: {self.frame_rate}")
        logger.info(f"Frame count: {self.frame_count}")
        logger.info(f"Video length: {self.video_length}")

        # Prepare facial analysis
        self._face = facial.AnalyzeFace(self._calcs)

    def init_bottom_horizontal_toolbar(self, layout):
        toolbar = QToolBar()
        layout.addWidget(toolbar)  # Add the toolbar to the layout first

        # Back Button
        btn_backward = QPushButton()
        btn_backward.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward)
        )
        btn_backward.clicked.connect(self.backward_action)
        toolbar.addWidget(btn_backward)

        # Start Button
        self._btn_play = QPushButton()
        self._btn_play.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self._btn_play.clicked.connect(self.action_play_pause)
        toolbar.addWidget(self._btn_play)

        # Forward Button
        btn_forward = QPushButton()
        btn_forward.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
        )
        btn_forward.clicked.connect(self.forward_action)
        toolbar.addWidget(btn_forward)
        toolbar.addSeparator()

        self.lbl_status = QLabel("0/0")
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

        self._chk_graph = custom_control.CheckingCheckBox(title="Graph", parent=self)
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

        btn_cut_left = QPushButton("Cut <")
        toolbar.addWidget(btn_cut_left)
        btn_cut_left.clicked.connect(self.action_cut_left)

        btn_cut_right = QPushButton("Cut >")
        toolbar.addWidget(btn_cut_right)
        btn_cut_right.clicked.connect(self.action_cut_right)

        btn_cut_restore = QPushButton("Restore")
        toolbar.addWidget(btn_cut_restore)
        btn_cut_restore.clicked.connect(self.action_restore)

        # btn_test = QPushButton("Test")
        # toolbar.addWidget(btn_test)
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

        if self._chart_view is not None:
            logger.debug("Update chart, because measures changed")
            self.update_chart()
            self.render_chart()

    def checkbox_clicked(self, checkbox, stat):
        stat.enabled = checkbox.isChecked()
        self.unsaved_changes = True
        if self._auto_update:
            self.update_face()
            if self._chart_view is not None:
                logger.debug("Update chart, because measures changed")
                self.update_chart()
                self.render_chart()

    def update_load_progress(self, status):
        # self.lbl_status.setText(status)
        self.lbl_status.setText(self.status(status))

        if self._view is None:
            self.load_first_frame()

    def load_first_frame(self):
        logger.debug("Display first video frame on load")
        self._face.load_state(self._frames[0])
        self.create_graphic(buffer=self._face.render_img)
        self._view.grabGesture(Qt.GestureType.PinchGesture)
        self._view.installEventFilter(self)
        self.update_face()
        logger.debug("Done, display first video frame on load")
        # Auto fit
        QTimer.singleShot(1, self.fit)

    def add_frame(self, face):
        self._frames.append(face.dump_state())
        self._video_slider.setRange(0, len(self._frames) - 2)
        self._frame_end = len(self._frames)

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
            return f"({mx:,}/{self.frame_count:,}, loading... time: {etc})"
        else:
            return f"({i+1:,}/{frame_count:,})"

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
        self.open_frame()
        self.lbl_status.setText(self.status())
        if self._chart_view:
            # self.update_chart()
            current_frame = self._video_slider.value() - self._frame_begin
            self._frame_line.set_xdata(current_frame)
            self.render_chart()

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

    def _save_as_image(self):
        options = QFileDialog.Option.DontUseNativeDialog

        # Show the dialog and get the selected file name and format
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "Images (*.png *.jpeg *.jpg)",
            options=options,
        )

        if filename:
            if not (
                filename.lower().endswith(".png")
                or filename.lower().endswith(".jpg")
                or filename.lower().endswith(".jpeg")
            ):
                self._window.display_message_box("Filename must end in .jpg or .png")
            else:
                self._face.save(filename)

    def _save_as_video(self):
        options = QFileDialog.Option.DontUseNativeDialog

        # Show the dialog and get the selected file name and format
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video",
            "",
            "Videos (*.mp4)",
            options=options,
        )

        if filename:
            if not filename.lower().endswith(".mp4"):
                self._window.display_message_box("Filename must end in .mp4")
            else:
                if not self.wait_load_complete():
                    return
                dialog = dlg_modal.VideoExportDialog(self, filename)
                dialog.exec()

    def _save_as_data(self):
        options = QFileDialog.Option.DontUseNativeDialog

        # Show the dialog and get the selected file name and format
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV Data",
            "",
            "Videos (*.csv)",
            options=options,
        )

        if filename:
            if not filename.lower().endswith(".csv"):
                self._window.display_message_box("Filename must end in .csv")
            else:
                self.save_csv(filename)

    def on_save_as(self):
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

    def collect_data(self):
        face = facial.AnalyzeFace(self._calcs)
        stats = face.get_all_stats()
        data = {stat: [] for stat in stats}

        cols = list(data.keys())
        cols = ["frame", "time"] + cols

        for i in range(self._frame_begin, self._frame_end):
            frame = self._frames[i]
            face.load_state(frame)
            rec = face.analyze()
            for stat in rec.keys():
                data[stat].append(rec[stat])

        return data

    def save_csv(self, filename):
        data = self.collect_data()

        with open(filename, "w") as f:
            writer = csv.writer(f)
            cols = list(data.keys())
            writer.writerow(["frame", "time"] + cols)
            all_stats = self._face.get_all_stats()
            l = len(data[all_stats[0]])
            lst_time = [x * self.frame_rate for x in range(l)]

            for i in range(l):
                row = [str(i), lst_time[i]]
                for col in cols:
                    row.append(data[col][i])
                writer.writerow(row)

    def update_chart(self):
        """Create the chart object, or update it if already there."""
        all_stats = self._face.get_all_stats()
        if len(all_stats) < 1:
            return

        # Create a Matplotlib figure
        self.chart_fig = Figure(figsize=(12, 2.5), dpi=100)
        ax = self.chart_fig.add_subplot(111)
        self.chart_fig.subplots_adjust(right=0.75)  # Adjust this value as needed

        data = self.collect_data()
        plot_stats = data.keys()
        l = len(data[all_stats[0]])
        # lst_time = [x * self.frame_rate for x in range(l)]
        lst_time = list(range(l))

        for stat in data.keys():
            if stat in plot_stats:
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

            self._splitter.addWidget(self._chart_view)
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

    def checkCheckboxEvent(self, target):
        if not self.loading:
            return True
        if not target.isChecked():
            if not self.wait_load_complete():
                return False

            self._chk_graph.setChecked(True)
        return True

    def action_graph(self):
        if self._chk_graph.isChecked():
            if self._chart_view is not None:
                # Redisplay graph
                self._splitter.addWidget(self._chart_view)
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
            remaining_size = sum(self._splitter.sizes())
            self._splitter.setSizes([remaining_size])

    def set_video_range(self, frame_begin: int, frame_end: int):
        self._frame_begin = frame_begin
        self._frame_end = frame_end
        self._video_slider.setRange(self._frame_begin, self._frame_end - 1)
        self.lbl_status.setText(self.status())
        if self._chart_view is not None:
            self.update_chart()
            self.render_chart()

    def action_cut_left(self):
        cmd = cmds.CommandClip(
            self, self._video_slider.sliderPosition(), self._frame_end - 1
        )
        self._undo_stack.push(cmd)

    def action_cut_right(self):
        cmd = cmds.CommandClip(
            self, self._frame_begin, self._video_slider.sliderPosition()
        )
        self._undo_stack.push(cmd)

    def _adjust_chart(self):
        # Resize the QGraphicsView to fit the pixmap
        self._chart_view.fitInView(
            self._chart_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

        # Get the size of the graphics view
        view_size = self._chart_view.sizeHint()

        # Set the splitter sizes
        self._splitter.setSizes(
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
        cmd = cmds.CommandClip(self, 0, len(self._frames) - 1)
        self._undo_stack.push(cmd)

    def on_redo(self):
        self._undo_stack.redo()

    def on_undo(self):
        self._undo_stack.undo()

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

        if filename:
            if not filename.lower().endswith(".dyfc"):
                self._window.display_message_box("Filename must end in .dyfc")
            else:
                self.save_document(filename)

    def save_document(self, filename):
        doc = dynaface_document.DynafaceDocument(dynaface_document.DOC_TYPE_VIDEO)
        doc.calcs = self._face.calcs
        doc.frames = self._frames[self._frame_begin : self._frame_end]
        doc.fps = self.frame_rate
        doc.save(filename)
        self.unsaved_changes = False

    def load_document(self, filename):
        doc = dynaface_document.DynafaceDocument(dynaface_document.DOC_TYPE_VIDEO)
        doc.load(filename)
        print("**", doc.calcs)
        self._face = facial.AnalyzeFace(doc.calcs)
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
