"""
Every module of the library imports in a fresh interpreter, whichever is imported
first. The measures and the face analysis use each other, so a module that imports
another at its top can fail when it is the first one imported: up to Dynaface 2.0.4,
importing dynaface.measures, dynaface.video, dynaface.measures_frontal or
dynaface.measures_skin before dynaface.facial failed.
"""

import os
import pkgutil
import subprocess  # nosec B404: runs this interpreter on a fixed import
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import dynaface  # noqa: E402


def import_first(name: str) -> subprocess.CompletedProcess:
    path = os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))
    return subprocess.run(  # nosec B603: no shell, a fixed command
        [sys.executable, "-c", f"import {name}"],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=path),
        timeout=300,
    )


class TestImports(unittest.TestCase):
    def test_each_module_imports_first(self):
        names = [
            f"dynaface.{module.name}"
            for module in pkgutil.iter_modules(dynaface.__path__)
            if not module.ispkg
        ]
        self.assertIn("dynaface.measures", names)
        with ThreadPoolExecutor(max_workers=4) as pool:
            runs = dict(zip(names, pool.map(import_first, names)))
        for name, run in runs.items():
            with self.subTest(module=name):
                self.assertEqual(run.returncode, 0, run.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
