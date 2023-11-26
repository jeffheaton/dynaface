import logging
import logging.config
import logging.handlers

from dynaface_app import AppDynaface

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    app = AppDynaface()
    app.exec()
    app.shutdown()
