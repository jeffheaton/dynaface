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
from dynaface.const import (
    DEFAULT_TILT_THRESHOLD,
    MAX_PUPIL_DIST,
    MIN_PUPIL_DIST,
    STD_PUPIL_DIST,
)
from dynaface_window import DynafaceWindow
from jth_ui import app_const, utl_errors, utl_log
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
SETTING_QUARTER = "quarter"

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
            dynaface.models.init_models(
                model_path=self._model_path, device=self._device
            )
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
    # What the user is told when the AI models couldn't start; saving Settings
    # loads them again (load_dynaface_settings()).
    MODELS_FAILED_MESSAGE = (
        "Dynaface was unable to start its AI models, so it can't analyze photos "
        "or videos. To try again, choose Settings and click Save."
    )

    def __init__(self):
        try:
            super().__init__()
            # An exception nothing caught is logged and reported, and Dynaface
            # goes on, instead of PyQt6 ending it with the user's work unsaved.
            utl_errors.install(self._report_unexpected_error)
            self.models_ready = False
            # Set when the last load failed, until another starts.
            self.models_failed = False
            # The device the AI models were last asked to load on, the device the
            # running load is for (None when no load runs) and that load's thread.
            self._models_device = None
            self._loading_device = None
            self._model_thread = None
            self._model_worker = None
            # The device the loaded models run on, once a load has finished.
            self.device = None
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

        # Set pupillary distance (PD). The library refuses one outside its range
        # when it loads an image, so a stored value outside it, as versions before
        # 2.1 allowed, is reset rather than failing every image and video.
        pd = utl_settings.get_int(self.settings, key=SETTING_PD, default=STD_PUPIL_DIST)
        try:
            dynaface.facial.check_pupil_distance(pd)
        except ValueError:
            logger.warning(f"Stored pupillary distance {pd} mm is out of range; reset")
            self.settings[SETTING_PD] = STD_PUPIL_DIST
            self.save_settings()
            QTimer.singleShot(0, lambda bad=pd: self._report_pd_reset(bad))
            pd = STD_PUPIL_DIST
        dynaface.facial.AnalyzeFace.pd = pd

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

        # Pose detection: a profile face is detected unless switched off; a
        # three-quarter (turned) face only when the preference is on, since it
        # slows video import and the library ships the detection off.
        lateral = utl_settings.get_bool(
            self.settings, key=SETTING_LATERAL, default=True
        )
        dynaface.config.AUTO_LATERAL = lateral
        dynaface.config.AUTO_QUARTER = utl_settings.get_bool(
            self.settings, key=SETTING_QUARTER, default=False
        )
        logger.info(f"Detect lateral view: {dynaface.config.AUTO_LATERAL}")
        logger.info(f"Detect three-quarter view: {dynaface.config.AUTO_QUARTER}")

        # accelerator device
        acc = utl_settings.get_bool(self.settings, key=SETTING_ACC, default=True)
        device = self.device_for(acc)

        # The models are built again only for a device they aren't on, or
        # loading on: a rebuild takes them away from anything that analyzes
        # meanwhile, and Settings is saved for many other reasons.
        if device == self._models_device:
            return
        self._models_device = device
        logging.info(f"Using device: {device}")
        v = get_library_version("onnxruntime")
        logging.info(f"ONNX Runtime version: {v}")
        self._start_model_loader()

    @staticmethod
    def device_for(accelerator: bool) -> str:
        """The device the AI models run on with the accelerator switch on or off:
        CUDA, MPS (CoreML) or, failing those or with the switch off, the CPU."""
        import dynaface.models

        if not accelerator:
            return "cpu"
        device = dynaface.models.detect_device()
        logger.info(f"ONNX Runtime Device: {device}")
        return device

    def would_reload_models(self, accelerator: bool) -> bool:
        """Whether saving this accelerator switch would load the AI models again,
        on another device."""
        return self.device_for(accelerator) != self._models_device

    def _report_pd_reset(self, pd):
        if self.main_window is not None:
            self.main_window.display_message_box(
                f"Your pupillary distance setting, {pd} mm, is outside the "
                f"{MIN_PUPIL_DIST} to {MAX_PUPIL_DIST} mm that Dynaface accepts, so it "
                f"has been reset to {STD_PUPIL_DIST} mm. You can change it in Settings."
            )

    def _start_model_loader(self):
        """
        Load the AI models on _models_device on a worker thread, so the GUI stays
        responsive: building the ONNX Runtime sessions is seconds of native work
        (CoreML's about ten), and doing it on the main thread froze the window
        right after the splash closed, which users mistook for a crash.

        While a load runs, this starts nothing: that load starts the next one
        when it ends (_on_models_loaded(), _on_models_failed()), so its thread is
        never replaced while it runs, which made Qt abort the app. models_ready is
        False until a load finishes, so files opened meanwhile wait for it
        (DynafaceWindow.open_file()).
        """
        if self._model_thread is not None:
            return
        self.models_ready = False
        self.models_failed = False
        device = self._models_device
        self._loading_device = device
        if self.main_window is not None:
            self.main_window.on_models_loading()
        self._model_thread = QThread()
        self._model_worker = _ModelLoaderWorker(model_path=self.DATA_DIR, device=device)
        self._model_worker.moveToThread(self._model_thread)
        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.finished.connect(self._on_models_loaded)
        self._model_worker.failed.connect(self._on_models_failed)
        self._model_thread.start()

    def _stop_model_thread(self):
        thread = self._model_thread
        if thread is None:
            return
        thread.quit()
        thread.wait()
        self._model_worker.deleteLater()
        thread.deleteLater()
        self._model_thread = None
        self._model_worker = None

    def _on_models_loaded(self, device: str):
        requested = self._loading_device
        self._loading_device = None
        self._stop_model_thread()
        if device != requested:
            # A hardware accelerator failed and we fell back to CPU. Persist
            # the choice so we don't retry the failing accelerator on the
            # next launch.
            logger.info(f"AI device fell back to {device}.")
            self.settings[SETTING_ACC] = False
            if self._models_device == requested:
                self._models_device = device
        self.device = device
        if self._models_device != device:
            # Settings chose another device while these loaded.
            self._start_model_loader()
            return
        self.models_ready = True
        logger.info("AI models ready.")
        if self.main_window is not None:
            self.main_window.on_models_ready()

    def _on_models_failed(self, message: str):
        logger.error(f"AI models failed to initialize: {message}")
        requested = self._loading_device
        self._loading_device = None
        self._stop_model_thread()
        if self._models_device != requested:
            # Settings chose another device while this load failed; try it.
            self._start_model_loader()
            return
        # A later save of Settings tries again, whatever device it chooses.
        self._models_device = None
        # Opening a photo or video, or choosing a pose, now says so rather
        # than wait for models no load will bring.
        self.models_failed = True
        # A document queued while waiting still opens (DynafaceWindow's
        # background_timer passes it on); a photo or video can't be analyzed
        # without the models, as the message below says.
        if not (self.file_open_request or "").lower().endswith(".dyfc"):
            self.file_open_request = None
        if self.main_window is not None:
            self.main_window.on_models_error(message)

    def _report_unexpected_error(self, summary: str) -> None:
        """Tell the user about an exception nothing caught (see utl_errors)."""
        window = getattr(self, "main_window", None)
        if window is not None:
            window.display_message_box(
                "Dynaface ran into an unexpected error, so something may not work "
                f"as it should: {summary}\n\nSave your work under a new name, then "
                "restart Dynaface. The details are in the support logs (Help > "
                "Open Support Logs...)."
            )

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
            SETTING_QUARTER: False,
        }


if __name__ == "__main__":
    app = AppDynaface()
    app.exec()
    app.shutdown()
