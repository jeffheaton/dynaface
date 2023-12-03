import logging
import sys
import time
from functools import partial

import cv2
import utl_gfx
from facial_analysis import facial
from facial_analysis.facial import load_face_image
from jth_ui.tab_graphic import TabGraphic
from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QStyle,
    QCheckBox,
    QGestureEvent,
    QHBoxLayout,
    QLabel,
    QPinchGesture,
    QPushButton,
    QSlider,
    QScrollArea,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class AnalyzeVideoTab(TabGraphic):
    def __init__(self, window, path):
        super().__init__(window)

        self._auto_update = False

        # Load the face
        self.begin_load_video(path)

        # Horiz toolbar
        tab_layout = QVBoxLayout(self)
        self.init_top_horizontal_toolbar(tab_layout)

        # Create a horizontal layout for the content of the tab
        content_layout = QHBoxLayout()
        tab_layout.addLayout(content_layout)
        self.init_vertical_toolbar(content_layout)
        self.init_graphics(content_layout)

        # Setup video buffer
        self._video_frames = []
        self._current_frame = 0

        # Advance to first frame
        self.advance_frame()

        # Prepare to display face
        self.create_graphic(buffer=self._face.render_img)
        self.update_face()

        # Scale the view as desired
        self._view.scale(1, 1)

        # Allow touch zoom
        self.grabGesture(Qt.GestureType.PinchGesture)
        self._auto_update = True

        self.init_bottom_horizontal_toolbar(tab_layout)

    def begin_load_video(self, path):
        # Open the video file
        self.video_stream = cv2.VideoCapture(path)

        # Check if video file opened successfully
        if not self.video_stream.isOpened():
            raise ValueError("Unknown video format")

        # Get the frame rate of the video
        self.frame_rate = int(self.video_stream.get(cv2.CAP_PROP_FPS))
        self.frames = self.video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_length = self.frames / self.frame_rate, 2

        # Prepare facial analysis
        self._face = facial.AnalyzeFace(facial.STATS, data_path=None)

    def advance_frame(self):
        start_time = time.time()
        ret, frame = self.video_stream.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not ret:
            return None

        end_time = time.time()

        elapsed_time = end_time - start_time
        logger.info(elapsed_time)
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        self._face.load_image(img=frame, crop=True)
        logger.info(
            f"Frame load time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        )
        self.update_face()

    def init_bottom_horizontal_toolbar(self, layout):
        toolbar = QToolBar()
        layout.addWidget(toolbar)  # Add the toolbar to the layout first

        # Start Button
        btn_start = QPushButton()
        btn_start.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        # btn_start.clicked.connect(self.advance_frame)
        toolbar.addWidget(btn_start)

        btn_backward = QPushButton()
        btn_backward.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward)
        )
        toolbar.addWidget(btn_backward)

        btn_forward = QPushButton()
        btn_forward.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
        )
        toolbar.addWidget(btn_forward)
        toolbar.addSeparator()

        lbl_status = QLabel("0/0")
        toolbar.addWidget(lbl_status)

        video_slider = QSlider(Qt.Orientation.Horizontal)
        video_slider.setRange(0, 10)
        toolbar.addWidget(video_slider)
        # self.positionSlider.sliderMoved.connect(self.setPosition)

    def init_top_horizontal_toolbar(self, layout):
        toolbar = QToolBar()
        layout.addWidget(toolbar)  # Add the toolbar to the layout first

        # Start Button
        self._btn_start = QPushButton("Reset")
        self._btn_start.clicked.connect(self.advance_frame)
        toolbar.addWidget(self._btn_start)
        toolbar.addSeparator()

        self._chk_landmarks = QCheckBox("Landmarks")
        toolbar.addWidget(self._chk_landmarks)
        self._chk_landmarks.stateChanged.connect(self.action_landmarks)
        toolbar.addSeparator()

        self._chk_measures = QCheckBox("Measures")
        toolbar.addWidget(self._chk_measures)
        self._chk_measures.setChecked(True)
        self._chk_measures.stateChanged.connect(self.action_measures)
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
        toolbar.addSeparator()

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
            checkbox.setChecked(True)
            self.checkboxes.append(checkbox)

        # Connect buttons to slot functions
        self.all_button.clicked.connect(self.check_all)
        self.none_button.clicked.connect(self.uncheck_all)

    def action_landmarks(self, state):
        self.update_face()

    def action_measures(self, state):
        self.update_face()

    def action_zoom(self, value):
        z = value / 100
        self._view.resetTransform()
        self._view.scale(value / 100, value / 100)

    def update_face(self):
        self._face.render_reset()
        if self._chk_measures.isChecked():
            self._face.analyze()
        if self._chk_landmarks.isChecked():
            self._face.draw_landmarks(numbers=True)

        if self._face.render_img is not None and self._render_buffer is not None:
            self._render_buffer[:, :] = self._face.render_img
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
        pass

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
        if self._auto_update:
            self.update_face()
