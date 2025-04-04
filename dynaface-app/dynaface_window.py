import logging
import logging.config
import logging.handlers
import os
import webbrowser

import dynaface_document
import tab_settings
import tab_splash
from jth_ui import app_jth
from jth_ui.window_jth import MainWindowJTH
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenu, QMenuBar, QMessageBox, QTabWidget
from tab_about import AboutTab
from tab_analyze_video import AnalyzeVideoTab
from tab_eval import TabEval

logger = logging.getLogger(__name__)

ALLOW_ONE_ANALYZE_TAB = False


class DynafaceWindow(MainWindowJTH):
    def __init__(self, app, app_name):
        super().__init__(app)
        self.running = False
        self.setWindowTitle(app_name)
        self.setGeometry(100, 100, 1100, 500)

        self.render_buffer = None
        self.display_buffer = None

        self._drop_ext = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".mp4",
            ".mov",
            ".heic",
            ".dyfc",
        )

        self.open_extensions = "All Files (*.mp4 *.mov *.jpg *.jpeg *.png *.tiff *.heic *.dyfc);;Images (*.jpg *.jpeg *.png *.tiff *.heic);;Videos (*.mp4 *.mov);;Dynaface Documents (*.dyfc)"

        self.setup_menu()
        self.initUI()
        self.update_recent_menu()

    def setup_menu(self):
        self.menubar = QMenuBar(self)
        self.setMenuBar(self.menubar)

        # Create the app menu and add it to the menu bar
        app_menu = QMenu(self.app.APP_NAME, self)

        # Add items to the app menu
        if self.app.get_system_name() == "osx":
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
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.open_action)
        self._file_menu.addAction(openAction)

        # Open recent submenu
        self._recent_menu = QMenu("Open &Recent", self)
        self._file_menu.addMenu(self._recent_menu)
        self._file_menu.addSeparator()

        # Add save action
        self._close_menu = QAction("Close", self)
        self._close_menu.setShortcut(QKeySequence.StandardKey.Close)
        self._close_menu.triggered.connect(self.close_action)
        self._file_menu.addAction(self._close_menu)

        self._save_menu = QAction("Save", self)
        self._save_menu.setShortcut(QKeySequence.StandardKey.Save)
        self._save_menu.triggered.connect(self.save_action)
        self._file_menu.addAction(self._save_menu)

        # Add save as... action
        self._save_as_menu = QAction("Save As...", self)
        self._save_as_menu.setShortcut(QKeySequence.StandardKey.SaveAs)
        self._save_as_menu.triggered.connect(self.save_as_action)
        self._file_menu.addAction(self._save_as_menu)

        self._file_menu.addSeparator()

        # Add print... action
        self._print_menu = QAction("Print...", self)
        self._print_menu.setShortcut("Ctrl+P")
        self._print_menu.triggered.connect(self.print_action)
        self._file_menu.addAction(self._print_menu)

        if self.app.get_system_name() == "windows":
            preferences_action = QAction("Settings...", self)
            self._file_menu.addAction(preferences_action)
            preferences_action.triggered.connect(self.show_properties)

            exit_action = QAction("Exit", self)
            exit_action.triggered.connect(self.close)
            self._file_menu.addAction(exit_action)

        # Edit menu
        self._edit_menu = QMenu("Edit", self)

        # Create the Undo action with a standard shortcut
        self._undo_menu = QAction("Undo", self)
        self._undo_menu.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_menu.triggered.connect(self.undo_action)
        self._edit_menu.addAction(self._undo_menu)

        # Create the Redo action with a standard shortcut
        self._redo_menu = QAction("Redo", self)
        self._redo_menu.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_menu.triggered.connect(self.redo_action)
        self._edit_menu.addAction(self._redo_menu)

        self._edit_menu.addSeparator()

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

        if self.app.get_system_name() == "windows":
            about_action = QAction(f"About {self.app.APP_NAME}", self)
            self._help_menu.addAction(about_action)
            about_action.triggered.connect(self.show_about)

        tutorial_action = QAction("Tutorial", self)
        tutorial_action.triggered.connect(self.open_tutorial)
        self._help_menu.addAction(tutorial_action)

        logs_action = QAction("Open Support Logs...", self)
        logs_action.triggered.connect(self.open_logs)
        self._help_menu.addAction(logs_action)

        #
        self.menubar.addMenu(app_menu)
        self.menubar.addMenu(self._file_menu)
        self.menubar.addMenu(self._edit_menu)
        self.menubar.addMenu(self._help_menu)

    def initUI(self):
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.tabCloseRequested.connect(self.close_tab)
        self._tab_widget.currentChanged.connect(self.onTabChanged)
        self.setCentralWidget(self._tab_widget)

        # Configure the resize timer
        self.resize_timer = QTimer(self)
        self.resize_timer.timeout.connect(self.finished_resizing)
        self.resize_timer.setInterval(300)  # 300 milliseconds

        # Configure the background timer
        self._background_timer = QTimer(self)
        self._background_timer.timeout.connect(self.background_timer)
        self._background_timer.setInterval(1000)  # 1 second
        self._background_timer.start()
        self._background_queue = []

    def background_timer(self):
        try:
            if self._tab_widget.count() == 0:
                self.add_tab(tab_splash.SplashTab(self), "Welcome to Dynaface")

            if self.app.file_open_request:
                filename = self.app.file_open_request
                self.app.file_open_request = None
                self.open_file(filename)

            if len(self._background_queue) > 0:
                item = self._background_queue.pop()
                item()
        except Exception as e:
            logger.error("Error during background timer", exc_info=True)

    def show_about(self):
        try:
            if not self.is_tab_open("About"):
                self.add_tab(AboutTab(self), "About Dynaface")
        except Exception as e:
            logger.error("Error during about open", exc_info=True)

    def show_eval(self):
        try:
            if not self.is_tab_open("Evaluation"):
                self._eval = TabEval(self)
                self.add_tab(self._eval, "Evaluation")
        except Exception as e:
            logger.error("Error during eval open", exc_info=True)

    def show_analyze_video(self, filename):
        try:
            if ALLOW_ONE_ANALYZE_TAB:
                self.close_analyze_tabs()
            basename = os.path.basename(filename)
            tab_name = f"Analyze: {basename}"
            self.add_tab(AnalyzeVideoTab(self, filename), tab_name)
        except Exception as e:
            logger.error("Error during video open", exc_info=True)
            self.display_message_box(f"Unable to open file. {e}")

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
        webbrowser.open(
            "https://github.com/jeffheaton/present/blob/master/facial/manual.md"
        )

    def open_file(self, file_path):
        try:
            super().open_file(file_path)
            logger.info(f"Open File: {file_path}")

            if file_path.lower().endswith(
                (".jpg", ".jpeg", ".png", ".tiff", ".heic", ".mp4", ".mov")
            ):
                self.show_analyze_video(file_path)
                self.update_recent_files(file_path)
            elif file_path.lower().endswith((".dyfc")):
                self.show_analyze_video(file_path)
        except Exception as ex:
            logger.error("Error during open file", exc_info=True)
            self.display_message_box(f"Error opening file: {ex}")

    def perform_edit_copy(self):
        current_tab = self._tab_widget.currentWidget()

        # Check if there is a current tab
        if current_tab is not None:
            # Check if the current tab has the 'on_copy' method
            if hasattr(current_tab, "on_copy"):
                current_tab.on_copy()

    def save_as_action(self):
        current_tab = self._tab_widget.currentWidget()

        # Check if there is a current tab
        if current_tab is not None:
            # Check if the current tab has the 'on_save_as' method
            if hasattr(current_tab, "on_save_as"):
                current_tab.on_save_as()

    def save_action(self):
        current_tab = self._tab_widget.currentWidget()

        # Check if there is a current tab
        if current_tab is not None:
            # Check if the current tab has the 'on_save' method
            if hasattr(current_tab, "on_save"):
                current_tab.on_save()

    def print_action(self):
        try:
            current_tab = self._tab_widget.currentWidget()

            # Check if there is a current tab
            if current_tab is not None:
                # Check if the current tab has the 'on_print' method
                if hasattr(current_tab, "on_print"):
                    current_tab.on_print()
        except:
            logger.error("Error during print", exc_info=True)
            self.display_message_box("Unable to print image.")

    def get_recent_file_list(self):
        recent_files = self.app.state.get(app_jth.STATE_LAST_FILES, [])
        self.app.state[app_jth.STATE_LAST_FILES] = recent_files
        return recent_files

    def update_recent_files(self, path):
        recent_files = self.get_recent_file_list()
        if path not in recent_files:
            recent_files.append(path)
            if len(recent_files) > 5:
                recent_files.pop(0)
        self.update_recent_menu()

    def update_recent_menu(self):
        recent_files = self.get_recent_file_list()
        self._recent_menu.clear()
        for path in reversed(recent_files):
            action = QAction(path, self)
            action.triggered.connect(lambda _, p=path: self.open_file(p))
            self._recent_menu.addAction(action)

    def undo_action(self):
        try:
            current_tab = self._tab_widget.currentWidget()

            # Check if there is a current tab
            if current_tab is not None:
                # Check if the current tab has the 'on_undo' method
                if hasattr(current_tab, "on_undo"):
                    current_tab.on_undo()
        except:
            logger.error("Error during undo", exc_info=True)
            self.display_message_box("Unable to undo.")

    def redo_action(self):
        try:
            current_tab = self._tab_widget.currentWidget()

            # Check if there is a current tab
            if current_tab is not None:
                # Check if the current tab has the 'on_undo' method
                if hasattr(current_tab, "on_redo"):
                    current_tab.on_redo()
        except:
            logger.error("Error during redo", exc_info=True)
            self.display_message_box("Unable to redo.")

    def close_action(self):
        try:
            current_index = self._tab_widget.currentIndex()
            if current_index != -1:
                self.close_tab(current_index)
        except:
            logger.error("Error during close tab", exc_info=True)
            self.display_message_box("Unable to close tab.")

    def open_logs(self):
        self.app.open_logs()

    def has_method(self, name):
        current_tab = self._tab_widget.currentWidget()
        if current_tab is None:
            return False
        return hasattr(current_tab, name)

    def update_enabled(self) -> None:
        current_tab = self._tab_widget.currentWidget()

        self._print_menu.setEnabled(self.has_method("on_print"))
        self._save_menu.setEnabled(self.has_method("on_save"))
        self._save_as_menu.setEnabled(self.has_method("on_save_as"))

        if self.has_method("can_redo"):
            self._redo_menu.setEnabled(current_tab.can_redo())
        else:
            self._redo_menu.setEnabled(False)

        if self.has_method("can_undo"):
            self._undo_menu.setEnabled(current_tab.can_undo())
        else:
            self._undo_menu.setEnabled(False)

        # Pass on to the tabs
        for index in range(self._tab_widget.count()):
            # Get the widget of the current tab
            tab = self._tab_widget.widget(index)

            # Check if the 'update_enabled' method exists for the tab
            if hasattr(tab, "update_enabled") and callable(
                getattr(tab, "update_enabled")
            ):
                # Call the 'update_enabled' method
                tab.update_enabled()

    def onTabChanged(self):
        self.update_enabled()
        logger.info("Updated enabled/disabled")
