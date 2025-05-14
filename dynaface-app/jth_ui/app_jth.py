import datetime
import glob
import json
import logging
import logging.handlers
import os
import platform
import plistlib
import sys

import appdirs
import pkg_resources
from PyQt6.QtCore import QEvent, Qt, QtMsgType, QUrl, qInstallMessageHandler
from PyQt6.QtGui import QDesktopServices, QFileOpenEvent
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from jth_ui import app_const, utl_env


logger = logging.getLogger(__name__)

STATE_LAST_FOLDER = "last_folder"
STATE_LAST_FILES = "recent"


def get_library_version(library_name):
    try:
        library = __import__(library_name)
        version = library.__version__
    except (ImportError, AttributeError):
        try:
            version = pkg_resources.get_distribution(library_name).version
        except pkg_resources.DistributionNotFound:
            version = None
    return version


class AppJTH(QApplication):
    def __init__(self):
        super().__init__(sys.argv)
        self.file_open_request = None

        self.settings = {}
        if utl_env.get_system_name() == "osx":
            if utl_env.is_sandboxed():
                self.SETTING_DIR = os.path.expanduser(f"~/preferences")
            else:
                self.SETTING_DIR = os.path.expanduser(
                    f"~/Library/Application Support/{app_const.APP_ID}/"
                )
            self.SETTING_FILE = os.path.join(
                self.SETTING_DIR, f"{app_const.APP_ID}.plist"
            )
            self.STATE_FILE = os.path.join(self.SETTING_DIR, "state.json")
        elif utl_env.get_system_name() == "windows":
            base_dir = appdirs.user_config_dir(
                app_const.APP_NAME, app_const.APP_AUTHOR, roaming=False
            )
            self.SETTING_DIR = os.path.join(base_dir, "preferences")
            self.SETTING_FILE = os.path.join(
                self.SETTING_DIR, f"{app_const.APP_ID}.json"
            )
            self.STATE_FILE = os.path.join(self.SETTING_DIR, "state.json")
        else:
            home_dir = os.path.expanduser("~")
            base_dir = os.path.join(home_dir, app_const.APP_ID)
            os.makedirs(base_dir, exist_ok=True)
            self.LOG_DIR = os.path.join(base_dir, "logs")
            self.SETTING_DIR = os.path.join(base_dir, "preferences")
            self.SETTING_FILE = os.path.join(
                self.SETTING_DIR, f"{app_const.APP_ID}.json"
            )
            self.STATE_FILE = os.path.join(self.SETTING_DIR, "state.json")

        print(f"Settings path: {self.SETTING_DIR}")
        print(f"Settings file: {self.SETTING_FILE}")

        self.load_settings()

        logging.info("Application starting up")
        s = utl_env.get_system_name()
        logging.info(f"System: {s}")
        logging.info(f"Pyinstaller: {utl_env.is_pyinstaller_bundle()}")
        z = os.path.expanduser("~")
        logging.info(f"User: {z}")
        if s == "osx":
            logging.info(f"Sandbox mode: {utl_env.is_sandboxed()}")

        self.setApplicationName(app_const.APP_NAME)

        self.load_state()

    def show_main_window(self, main_window):
        self.main_window = main_window
        self.main_window.show()
        if sys.platform == "win32":
            # Windows-specific activation logic:
            self.main_window.raise_()
            self.main_window.activateWindow()

            # Splash screen close (only relevant on Windows)
            try:
                import pyi_splash

                pyi_splash.close()
                logger.info("Splash screen closed.")
            except ImportError:
                logger.info("No splash screen to close.")

        elif sys.platform == "darwin":
            # macOS minimal, clean startup logic:
            # NSApp.activateIgnoringOtherApps_(True)
            # No timers, no additional raise_, no redundant activateWindow()
            # You can optionally add the AppDelegate here if needed
            # but start simple first to confirm stability
            pass
        else:
            # Linux/other platforms (simple approach)
            self.main_window.raise_()
            self.main_window.activateWindow()

    def exec(self):
        try:
            logger.info("Starting app main loop")
            super().exec()
            logger.info("Exited app main loop")
        except Exception as e:
            logger.error("Error running app", exc_info=True)

    def get_resource_path(self, relative_path, base_path):
        """Get the path to a resource, supporting both normal and bundled (PyInstaller) modes."""
        if getattr(sys, "frozen", False):
            # If the application is run as a bundle (via PyInstaller)
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(base_path)

        return os.path.join(base_path, relative_path)

    def shutdown(self):
        self.save_state()
        self.save_settings()
        logging.info("Application shutting down")

    def load_state(self):
        try:
            with open(self.STATE_FILE, "r") as fp:
                self.state = json.load(fp)
        except Exception:
            self.init_state()
            logger.info("Failed to read state file, using defaults.")

    def save_state(self):
        try:
            with open(self.STATE_FILE, "w") as fp:
                json.dump(self.state, fp)
        except Exception as e:
            logger.error("Failed to write state file", exc_info=True)

    def init_state(self):
        home_directory = os.path.expanduser("~")
        documents_path = os.path.join(home_directory, "Documents")
        self.state = {STATE_LAST_FOLDER: documents_path, STATE_LAST_FILES: []}

    def init_settings(self):
        self.settings = {}

    # Save settings to a JSON file
    def save_settings(self):
        try:
            if utl_env.get_system_name() == "osx":
                logger.info("Saved MacOS settings")
                with open(self.SETTING_FILE, "wb") as fp:
                    plistlib.dump(self.settings, fp)
            else:
                logger.info("Saved Windows settings")
                with open(self.SETTING_FILE, "w") as fp:
                    json.dump(self.settings, fp)
        except Exception as e:
            logging.error("Caught an exception saving settings", exc_info=True)

    def load_settings(self):
        try:
            os.makedirs(self.SETTING_DIR, exist_ok=True)

            if not os.path.exists(self.SETTING_FILE):
                logger.info("Resetting to default settings")
                self.init_settings()
            else:
                if utl_env.get_system_name() == "osx":
                    with open(self.SETTING_FILE, "rb") as fp:
                        self.settings = plistlib.load(fp)
                    logger.info("Loaded MacOS settings")
                else:
                    with open(self.SETTING_FILE, "r") as fp:
                        self.settings = json.load(fp)
                        logger.info("Loaded Windows settings")
        except Exception as e:
            logging.error("Caught an exception loading settings", exc_info=True)

    def event(self, event):
        try:
            if event.type() == QEvent.Type.FileOpen:
                file_path = event.file()
                self.file_open_request = file_path
                logger.info(f"MacOS open file request: {file_path}")
                return True
            return super().event(event)
        except Exception as e:
            logger.error("Error during application event", exc_info=True)
