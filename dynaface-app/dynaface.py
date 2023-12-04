import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import logging
import logging.config
import logging.handlers

from dynaface_app import AppDynaface

# Need this setting because of this issue:
# https://github.com/numpy/numpy/issues/654
# See note in spgia.augmentors.utils


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    app = AppDynaface()
    app.exec()
    app.shutdown()
