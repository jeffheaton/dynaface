import importlib
import os
import sys
import unittest
from unittest.mock import patch

# Make sure dynaface can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dynaface


class TestDynaface(unittest.TestCase):

    def test_version_present(self):
        self.assertNotEqual(dynaface.__version__, "unknown")

    def test_version_fallback(self):
        # Remove dynaface from sys.modules to force __init__.py to re-run
        sys.modules.pop("dynaface", None)

        with patch("importlib.metadata.version", side_effect=Exception("fail")):
            import dynaface  # Triggers re-import with patched version()

            importlib.reload(dynaface)
            self.assertEqual(dynaface.__version__, "unknown")
