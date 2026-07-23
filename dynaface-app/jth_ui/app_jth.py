import datetime
import glob
import json
import logging
import logging.handlers
import os
import platform
import plistlib
import shutil
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


def settings_paths_for(app_id):
    """Settings dir and settings file for a given app id, per platform."""
    if utl_env.get_system_name() == "osx":
        if utl_env.is_sandboxed():
            setting_dir = os.path.expanduser("~/preferences")
        else:
            setting_dir = os.path.expanduser(f"~/Library/Application Support/{app_id}/")
        setting_file = os.path.join(setting_dir, f"{app_id}.plist")
    elif utl_env.get_system_name() == "windows":
        base_dir = appdirs.user_config_dir(
            app_const.APP_NAME, app_const.APP_AUTHOR, roaming=False
        )
        setting_dir = os.path.join(base_dir, "preferences")
        setting_file = os.path.join(setting_dir, f"{app_id}.json")
    else:
        setting_dir = os.path.join(os.path.expanduser("~"), app_id, "preferences")
        setting_file = os.path.join(setting_dir, f"{app_id}.json")
    return setting_dir, setting_file


def migrate_legacy_settings(legacy_app_id="testapp"):
    """One-time adoption of settings written while APP_ID was stuck at the
    framework placeholder id. Safe to call every launch: does nothing once
    settings exist under the current id. Must run before anything (e.g.
    logging setup) creates directories under the new id, or the whole-dir
    move degrades to the file-by-file fallback. Returns messages describing
    what was migrated, for the caller to log once logging is configured."""
    actions = []
    try:
        if app_const.APP_ID == legacy_app_id:
            return actions
        new_dir, new_file = settings_paths_for(app_const.APP_ID)
        if os.path.exists(new_file):
            return actions
        legacy_dir, legacy_file = settings_paths_for(legacy_app_id)
        if (
            legacy_dir != new_dir
            and os.path.isdir(legacy_dir)
            and not os.path.exists(new_dir)
        ):
            os.makedirs(os.path.dirname(os.path.normpath(new_dir)), exist_ok=True)
            shutil.move(legacy_dir, new_dir)
            actions.append(f"Migrated legacy settings dir {legacy_dir} -> {new_dir}")
        carried = os.path.join(new_dir, os.path.basename(legacy_file))
        if os.path.isfile(carried):
            os.replace(carried, new_file)
            actions.append(f"Renamed legacy settings file {carried} -> {new_file}")
        elif os.path.isfile(legacy_file):
            # The new dir already existed, so the dir was not moved; bring the
            # settings and window state over individually instead.
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy2(legacy_file, new_file)
            actions.append(f"Copied legacy settings file {legacy_file} -> {new_file}")
            legacy_state = os.path.join(legacy_dir, "state.json")
            new_state = os.path.join(new_dir, "state.json")
            if os.path.isfile(legacy_state) and not os.path.exists(new_state):
                shutil.copy2(legacy_state, new_state)
                actions.append(
                    f"Copied legacy state file {legacy_state} -> {new_state}"
                )
    except Exception:
        logger.error("Failed to migrate legacy settings", exc_info=True)
    return actions


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
        self.SETTING_DIR, self.SETTING_FILE = settings_paths_for(app_const.APP_ID)
        self.STATE_FILE = os.path.join(self.SETTING_DIR, "state.json")
        if utl_env.get_system_name() not in ("osx", "windows"):
            base_dir = os.path.dirname(self.SETTING_DIR)
            os.makedirs(base_dir, exist_ok=True)
            self.LOG_DIR = os.path.join(base_dir, "logs")

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
