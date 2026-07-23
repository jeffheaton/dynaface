# Load ONYX before anything else, to prevent DLL errors
import onnxruntime as ort

# Begin logging ASAP
from jth_ui import app_const, utl_log

app_const.BUNDLE_ID = "com.heatonresearch.dynaface"
app_const.APP_NAME = "Dynaface"
app_const.APP_AUTHOR = "Jeff Heaton"
app_const.COPYRIGHT = "Copyright 2026 by Jeff Heaton, released under the <a href='https://www.apache.org/licenses/LICENSE-2.0'>Apache 2.0 License</a>"
app_const.APP_ID = app_const.BUNDLE_ID.split(".")[-1]

# Adopt settings saved while APP_ID was stuck at the framework placeholder
# ("testapp"); must run before logging creates directories under the new id.
from jth_ui.app_jth import migrate_legacy_settings

_migration_actions = migrate_legacy_settings()

utl_log.delete_old_logs()
utl_log.setup_logging()

# Setup the application information
import version as v

app_const.VERSION = v.VERSION

# Need the above thread setting because of this issue:
# https://github.com/numpy/numpy/issues/654
# See note in spgia.augmentors.utils
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["MKL_THREADING_LAYER"] = "GNU"


import logging
import os
import sys

logger = logging.getLogger(__name__)

for _msg in _migration_actions:
    logger.info(_msg)

import jth_ui.utl_settings as utl_settings
from dynaface.const import DEFAULT_TILT_THRESHOLD, STD_PUPIL_DIST
from dynaface_window import DynafaceWindow
from jth_ui import app_const, utl_log
from jth_ui.app_jth import AppJTH, get_library_version
from pillow_heif import register_heif_opener
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

import dynaface

# Constants for settings keys
SETTING_PD = "pd"
SETTING_LOG_LEVEL = "log_level"
SETTING_ACC = "accelerator"
SETTING_TILT_THRESHOLD = "tilt"
SETTING_DYNAMIC_ADJUST = "dynamic"
SETTING_SMOOTH = "smooth"
SETTING_LATERAL = "lateral"

DEFAULT_DYNAMIC_ADJUST = 2
DEFAULT_SMOOTH = 2

register_heif_opener()

# https://stackoverflow.com/questions/75746637/how-to-suppress-qt-pointer-dispatch-warning


class _ModelLoaderWorker(QObject):
    """Loads the ONNX AI models off the GUI thread.

    ``init_models()`` builds several ONNX Runtime inference sessions, which is
    heavy native work (~20s). ONNX Runtime releases the GIL during that work,
    so running it here keeps the main window responsive during startup instead
    of freezing it right after the splash closes (which looked like a crash).
    See ``AppDynaface._start_model_loader``.
    """

    # Emits the device actually used (may be "cpu" if we fell back).
    finished = pyqtSignal(str)
    # Emits an error message if the models could not be initialized at all.
    failed = pyqtSignal(str)

    def __init__(self, model_path: str, device: str):
        super().__init__()
        self._model_path = model_path
        self._device = device

    def run(self):
        import dynaface.models

        try:
            dynaface.models.init_models(model_path=self._model_path, device=self._device)
            self.finished.emit(self._device)
            return
        except Exception:
            logger.error(
                f"Error starting AI models on device {self._device}", exc_info=True
            )

        # Fall back to CPU if a hardware accelerator failed to initialize.
        if self._device != "cpu":
            logger.info("Trying CPU as AI device.")
            try:
                dynaface.models.init_models(model_path=self._model_path, device="cpu")
                self.finished.emit("cpu")
                return
            except Exception as e:
                logger.error("Error starting AI models on CPU", exc_info=True)
                self.failed.emit(str(e))
                return

        self.failed.emit("Unable to initialize AI models.")


class AppDynaface(AppJTH):
    def __init__(self):
        try:
            super().__init__()
            self.models_ready = False
            self.dynamic_adjust = DEFAULT_DYNAMIC_ADJUST
            self.data_smoothing = DEFAULT_SMOOTH
            self.tilt_threshold = DEFAULT_TILT_THRESHOLD

            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.DATA_DIR = os.path.join(self.BASE_DIR, "data")

            main_window = DynafaceWindow(app=self, app_name=app_const.APP_NAME)
            self.show_main_window(main_window)

            # Defer heavy initialization until the event loop is running so the
            # window can appear immediately.
            QTimer.singleShot(0, self.load_dynaface_settings)

        except Exception as e:
            logger.error("Error running app", exc_info=True)

    def load_dynaface_settings(self):
        try:
            self._load_dynaface_settings_impl()
        except Exception as e:
            logger.error("Error in load_dynaface_settings", exc_info=True)

    def _load_dynaface_settings_impl(self):
        import dynaface.facial
        import dynaface.models
        import dynaface.config

        # Set logging level
        level = utl_settings.get_str(
            self.settings, key=SETTING_LOG_LEVEL, default="INFO"
        )
        logging_level = getattr(logging, level)
        utl_log.change_log_level(logging_level)

        # Set pupillary distance (PD)
        dynaface.facial.AnalyzeFace.pd = utl_settings.get_int(
            self.settings, key=SETTING_PD, default=STD_PUPIL_DIST
        )

        # Set the tilt threshold
        self.tilt_threshold = utl_settings.get_int(
            self.settings, key=SETTING_TILT_THRESHOLD, default=DEFAULT_TILT_THRESHOLD
        )
        # Set the dynamic adjust
        self.dynamic_adjust = utl_settings.get_int(
            self.settings, key=SETTING_DYNAMIC_ADJUST, default=DEFAULT_DYNAMIC_ADJUST
        )

        # Set the data smoothing
        self.data_smoothing = utl_settings.get_int(
            self.settings, key=SETTING_SMOOTH, default=DEFAULT_SMOOTH
        )

        # Set the lateral
        lateral = utl_settings.get_bool(
            self.settings, key=SETTING_LATERAL, default=True
        )
        dynaface.config.AUTO_LATERAL = lateral

        # accelerator device
        acc = utl_settings.get_bool(self.settings, key=SETTING_ACC, default=True)

        if acc:
            # Detect CUDA, MPS (CoreML), or failing that, CPU
            self.device = dynaface.models.detect_device()
            logger.info(f"ONNX Runtime Device: {self.device}")
        else:
            self.device = "cpu"

        logging.info(f"Using device: {self.device}")
        v = get_library_version("onnxruntime")
        logging.info(f"ONNX Runtime version: {v}")

        # Initialize AI models on a worker thread so the GUI stays responsive.
        # Building the ONNX Runtime sessions is ~20s of native work; doing it on
        # the main thread froze the window right after the splash closed, which
        # users mistook for a crash. models_ready is flipped in the completion
        # slot on the main thread.
        self._start_model_loader(self.device)

    def _start_model_loader(self, device: str):
        self._model_thread = QThread()
        self._model_worker = _ModelLoaderWorker(model_path=self.DATA_DIR, device=device)
        self._model_worker.moveToThread(self._model_thread)
        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.finished.connect(self._on_models_loaded)
        self._model_worker.failed.connect(self._on_models_failed)
        self._model_thread.start()

    def _stop_model_thread(self):
        thread = getattr(self, "_model_thread", None)
        if thread is None:
            return
        thread.quit()
        thread.wait()
        self._model_worker.deleteLater()
        thread.deleteLater()
        self._model_thread = None
        self._model_worker = None

    def _on_models_loaded(self, device: str):
        try:
            if device != self.device:
                # A hardware accelerator failed and we fell back to CPU. Persist
                # the choice so we don't retry the failing accelerator on the
                # next launch.
                logger.info(f"AI device fell back to {device}.")
                self.device = device
                self.settings[SETTING_ACC] = False
            self.models_ready = True
            logger.info("AI models ready.")
            if self.main_window is not None:
                self.main_window.on_models_ready()
        finally:
            self._stop_model_thread()

    def _on_models_failed(self, message: str):
        logger.error(f"AI models failed to initialize: {message}")
        try:
            # Drop any file the user queued while waiting; it can't be analyzed
            # without models.
            self.file_open_request = None
            if self.main_window is not None:
                self.main_window.on_models_error(message)
        finally:
            self._stop_model_thread()

    def shutdown(self):
        try:
            # Ensure the model-loader thread is stopped before we tear down, or
            # Qt aborts on a QThread destroyed while still running.
            self._stop_model_thread()
            super().shutdown()
            sys.exit(0)
        except Exception as e:
            logger.error("Error shutting down app", exc_info=True)

    def init_settings(self):
        self.settings = {
            SETTING_PD: STD_PUPIL_DIST,
            SETTING_LOG_LEVEL: "INFO",
            SETTING_ACC: True,
            SETTING_TILT_THRESHOLD: DEFAULT_TILT_THRESHOLD,
            SETTING_DYNAMIC_ADJUST: DEFAULT_DYNAMIC_ADJUST,
            SETTING_SMOOTH: DEFAULT_SMOOTH,
            SETTING_LATERAL: True,
        }


if __name__ == "__main__":
    app = AppDynaface()
    app.exec()
    app.shutdown()
