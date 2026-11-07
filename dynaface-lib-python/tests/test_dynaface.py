import importlib
import os
import sys
import unittest
from unittest.mock import patch

# Make sure dynaface can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dynaface  # noqa: E402


class TestDynaface(unittest.TestCase):

    def test_version_present(self):
        self.assertNotEqual(dynaface.__version__, "unknown")

    def test_version_fallback(self):
        # Run __init__.py again in a fresh package object, then put the original back.
        # Left in sys.modules, the fresh one would lack the submodule attributes
        # (dynaface.lateral and the rest) that Python 3.10's mock.patch resolves a
        # target such as "dynaface.lateral.analyze_lateral" through.
        original = sys.modules.pop("dynaface")
        self.addCleanup(sys.modules.__setitem__, "dynaface", original)

        with patch("importlib.metadata.version", side_effect=Exception("fail")):
            fresh = importlib.import_module("dynaface")
            self.assertEqual(fresh.__version__, "unknown")
        self.assertIsNot(fresh, original)
