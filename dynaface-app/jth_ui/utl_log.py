import datetime
import glob
import logging
import logging.handlers
import os
import sys

import appdirs
from jth_ui import app_const, utl_env
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

_console_handler = None
_file_handler = None

logger = logging.getLogger(__name__)
logger.propagate = True


def get_log_dir():
    if utl_env.get_system_name() == "osx":
        if utl_env.is_sandboxed():
            log_dir = os.path.join(os.path.expanduser("~"), "logs")
        else:
            log_dir = os.path.expanduser(
                f"~/Library/Application Support/{app_const.APP_ID}/logs/"
            )
    elif utl_env.get_system_name() == "windows":
        base_dir = appdirs.user_config_dir(
            app_const.APP_NAME, app_const.APP_AUTHOR, roaming=False
        )
        log_dir = os.path.join(base_dir, "logs")
    else:
        home_dir = os.path.expanduser("~")
        base_dir = os.path.join(home_dir, app_const.APP_ID)
        log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


# Define a function to handle deletion of old log files
def delete_old_logs():
    log_dir = get_log_dir()
    retention_period = 7  # days
    current_time = datetime.datetime.now()
    log_files = glob.glob(os.path.join(log_dir, "*.log"))

    for file in log_files:
        creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file))
        if (current_time - creation_time).days > retention_period:
            os.remove(file)


def setup_logging(level=logging.DEBUG):
    global _console_handler, _file_handler

    log_dir = get_log_dir()

    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(
        log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
    )

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to prevent duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler
    _file_handler = logging.handlers.TimedRotatingFileHandler(
        log_filename, when="midnight", interval=1, backupCount=7
    )
    _file_handler.setLevel(level)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    
    _file_handler.flush()

    # Console Handler (if stderr is available)
    if sys.stderr and hasattr(sys.stderr, 'write'):
        _console_handler = logging.StreamHandler()
        _console_handler.setLevel(level)
        _console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(_console_handler)

    # Attach handlers to the root logger
    root_logger.addHandler(_file_handler)
    logger.info(f"Logging initialized. Output: {log_dir}")


def change_log_level(level):
    """Change the log level of the logger and its handlers."""

    if not isinstance(level, int):
        raise ValueError("Log level must be an integer (e.g., logging.INFO)")

    _console_handler.setLevel(level)
    _file_handler.setLevel(level)

    logging.info(f"Log level changed to {logging.getLevelName(level)} ({level})")


def open_logs():
    log_dir = get_log_dir()
    # Convert the file path to a QUrl object
    file_url = QUrl.fromLocalFile(log_dir)

    # Open the file location in the default file explorer
    QDesktopServices.openUrl(file_url)
