import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import logging
import logging.config
import logging.handlers

from dynaface_app import AppDynaface

# Need the above thread setting because of this issue:
# https://github.com/numpy/numpy/issues/654
# See note in spgia.augmentors.utils


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    app = AppDynaface()
    app.exec()
    app.shutdown()
