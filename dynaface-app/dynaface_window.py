import logging
import logging.config
import logging.handlers
import webbrowser

import tab_settings
import tab_splash
from jth_ui.window_jth import MainWindowJTH
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenu, QMenuBar, QTabWidget
from tab_about import AboutTab
from tab_analyze_image import AnalyzeImageTab

logger = logging.getLogger(__name__)


class DynafaceWindow(MainWindowJTH):
    def __init__(self, app, app_name):
        super().__init__(app)
        self.running = False
        self.setWindowTitle(app_name)
        self.setGeometry(100, 100, 1000, 500)

        self.render_buffer = None
        self.display_buffer = None

        self._drop_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

        if self.app.get_system_name() == "osx":
            self.setup_mac_menu()
        else:
            self.setup_menu()
        self.initUI()

    def setup_menu(self):
        # Create the File menu
        file_menu = self.menuBar().addMenu("File")

        # Create a "Exit" action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create the Edit menu
        edit_menu = self.menuBar().addMenu("Edit")

        # Create an "About" action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        edit_menu.addAction(about_action)

    def setup_mac_menu(self):
        # Create a main menu bar
        # if platform.uname().system.startswith('Darw') :
        #    self.menubar = QMenuBar() # parentless menu bar for Mac OS
        # else:
        #    self.menubar = self.menuBar() # refer to the default one

        self.menubar = QMenuBar()  # self.menuBar()

        # Create the app menu and add it to the menu bar
        app_menu = QMenu(self.app.APP_NAME, self)

        # Add items to the app menu
        about_action = QAction(f"About {self.app.APP_NAME}", self)
        app_menu.addAction(about_action)
        self.about_menu = QMenu("About", self)
        about_action.triggered.connect(self.show_about)

        preferences_action = QAction("Settings...", self)
        app_menu.addAction(preferences_action)
        preferences_action.triggered.connect(self.show_properties)

        exit_action = QAction("Quit", self)
        exit_action.triggered.connect(self.close)
        app_menu.addAction(exit_action)

        # File menu
        self._file_menu = QMenu("File", self)

        # Add open action
        openAction = QAction("Open...", self)
        openAction.triggered.connect(self.open_action)
        self._file_menu.addAction(openAction)

        # Close Window action
        closeAction = QAction("Close Window", self)
        closeAction.setShortcut(QKeySequence(QKeySequence.StandardKey.Close))
        closeAction.triggered.connect(self.close)
        self._file_menu.addAction(closeAction)

        # Edit menu
        self._edit_menu = QMenu("Edit", self)
        cutAction = QAction("Cut", self)
        cutAction.setShortcut(QKeySequence(QKeySequence.StandardKey.Cut))
        self._edit_menu.addAction(cutAction)

        copyAction = QAction("Copy", self)
        copyAction.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        self._edit_menu.addAction(copyAction)
        copyAction.triggered.connect(self.perform_edit_copy)

        pasteAction = QAction("Paste", self)
        pasteAction.setShortcut(QKeySequence(QKeySequence.StandardKey.Paste))
        self._edit_menu.addAction(pasteAction)

        # Help menu
        self._help_menu = QMenu("Help", self)
        tutorial_action = QAction("Tutorial", self)
        tutorial_action.triggered.connect(self.open_tutorial)
        self._help_menu.addAction(tutorial_action)

        #
        self.menubar.addMenu(app_menu)
        self.menubar.addMenu(self._file_menu)
        self.menubar.addMenu(self._edit_menu)
        self.menubar.addMenu(self._help_menu)

    def initUI(self):
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self._tab_widget)

        # Configure the resize timer
        self.resize_timer = QTimer(self)
        self.resize_timer.timeout.connect(self.finished_resizing)
        self.resize_timer.setInterval(300)  # 300 milliseconds

        # Configure the resize timer
        self._background_timer = QTimer(self)
        self._background_timer.timeout.connect(self.background_timer)
        self._background_timer.setInterval(1000)  # 1 second
        self._background_timer.start()

    def background_timer(self):
        if self._tab_widget.count() == 0:
            self.add_tab(tab_splash.SplashTab(self), "Welcome to Dynaface")

    def show_about(self):
        try:
            if not self.is_tab_open("About"):
                self.add_tab(AboutTab(), "About Dynaface")
        except Exception as e:
            logger.error("Error during about open", exc_info=True)

    def show_analyze_image(self, filename):
        try:
            self.close_analyze_tabs()
            self.add_tab(AnalyzeImageTab(self, filename), "Analyze Image")
        except Exception as e:
            logger.error("Error during image open", exc_info=True)

    def close_analyze_tabs(self):
        try:
            logger.info("Closing any analyze tabs due to config change")
            index = 0
            while index < self._tab_widget.count():
                if self._tab_widget.tabText(index).startswith("Analyze"):
                    self.close_tab(index)
                    # Since we've removed a tab, the indices shift, so we don't increase the index in this case
                    continue
                index += 1
        except Exception as e:
            logger.error("Error forcing analyze tab close", exc_info=True)

    def show_rule(self, rule):
        try:
            if not self.is_tab_open("Rule"):
                self.add_tab(RuleTab(rule), "Rule")
        except Exception as e:
            logger.error("Error during show rule", exc_info=True)

    def show_properties(self):
        try:
            if not self.is_tab_open("Preferences"):
                self.add_tab(tab_settings.SettingsTab(self), "Preferences")
        except Exception as e:
            logger.error("Error during show properties", exc_info=True)

    def open_tutorial(self):
        webbrowser.open("https://www.heatonresearch.com/mergelife/heaton-ca.html")

    def open_file(self, file_path):
        super().open_file(file_path)
        logger.info(f"Open File: {file_path}")

        if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
            self.show_analyze_image(file_path)
        elif file_path.lower().endswith((".mp4", ".mov")):
            self.displayMessageBox("Video not supported yet.")

    def perform_edit_copy(self):
        current_tab = self._tab_widget.currentWidget()

        # Check if there is a current tab
        if current_tab is not None:
            # Check if the current tab has the 'on_copy' method
            if hasattr(current_tab, "on_copy"):
                current_tab.on_copy()
