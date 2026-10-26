"""Tests that Dynaface documents (.dyfc files) round-trip and refuse unsafe content.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import gzip
import io
import json
import math
import os
import pickle
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from dynaface import facial, measures  # noqa: F401 (facial first avoids a cycle)
from dynaface.const import Pose

from dynaface_document import DOC_BODY, DOC_BODY_FRAMES, DOC_HEADER, DynafaceDocument

_calls = []


def _record_call(*args):
    _calls.append(args)


class _RunsCode:
    """Pickles to a call of _record_call, the way a malicious file would run code."""

    def __reduce__(self):
        return (_record_call, ("ran",))


def _sample_document():
    doc = DynafaceDocument()
    doc.measures = measures.all_measures()
    doc.measures[0].enabled = False
    doc.measures[1].set_item_enabled(doc.measures[1].items[0].name, False)
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    headpose = np.array([1.5, -2.0, 0.25])
    doc.frames = [[image, headpose, [(10, 20), (30, 40)], 63.0, 0.25, 1.5]]
    doc.fps = 30
    doc.pose = Pose.LATERAL
    doc.lateral = True
    doc.lateral_landmarks = np.arange(12, dtype=np.int32).reshape(6, 2)
    doc.sagittal_x = np.arange(5, dtype=np.int32)
    doc.sagittal_y = np.arange(5, 10, dtype=np.int32)
    return doc


def _version_1(frames, **header):
    """A document as earlier versions of the app saved it, before pickling."""
    return {
        DOC_HEADER: {"version": 1, "fps": 24, **header},
        DOC_BODY: {"measures": [], DOC_BODY_FRAMES: frames},
    }


class DynafaceDocumentTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "test.dyfc")
        _calls.clear()

    def tearDown(self):
        self._dir.cleanup()

    def load(self):
        doc = DynafaceDocument()
        doc.load(self.path)
        return doc

    def write_gzip(self, data):
        with gzip.open(self.path, "wb") as f:
            f.write(data)

    def rewrite(self, change_json=None, replace=None):
        """Edits a saved document the way someone crafting a file could."""
        with zipfile.ZipFile(self.path) as zf:
            entries = {name: zf.read(name) for name in zf.namelist()}
        if change_json:
            doc = json.loads(entries["document.json"])
            change_json(doc)
            entries["document.json"] = json.dumps(doc).encode()
        entries.update(replace or {})
        with zipfile.ZipFile(self.path, "w") as zf:
            for name, data in entries.items():
                zf.writestr(name, data)

    def assert_same(self, expected, actual):
        """Asserts equal values of exactly the same types, arrays included."""
        self.assertIs(type(actual), type(expected))
        if isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            for e, a in zip(expected.flat, actual.flat):
                self.assert_same(e, a)
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for e, a in zip(expected, actual):
                self.assert_same(e, a)
        elif isinstance(expected, dict):
            self.assertEqual(list(actual), list(expected))
            for key in expected:
                self.assert_same(expected[key], actual[key])
        elif isinstance(expected, float) and math.isnan(expected):
            self.assertTrue(math.isnan(actual))
        else:
            self.assertEqual(actual, expected)

    def test_round_trip(self):
        saved = _sample_document()
        saved.save(self.path)
        loaded = self.load()

        self.assertEqual(loaded.fps, 30)
        self.assertEqual(loaded.pose, Pose.LATERAL)
        self.assertTrue(loaded.lateral)
        self.assert_same(saved.frames, loaded.frames)
        self.assert_same(saved.lateral_landmarks, loaded.lateral_landmarks)
        self.assert_same(saved.sagittal_x, loaded.sagittal_x)
        self.assert_same(saved.sagittal_y, loaded.sagittal_y)
        self.assertEqual(
            [type(m) for m in loaded.measures], [type(m) for m in saved.measures]
        )
        for s, l in zip(saved.measures, loaded.measures):
            self.assertEqual(l.enabled, s.enabled)
            self.assertEqual([i.enabled for i in l.items], [i.enabled for i in s.items])

    def test_file_holds_only_json_and_npy(self):
        _sample_document().save(self.path)
        with zipfile.ZipFile(self.path) as zf:
            json.loads(zf.read("document.json"))
            arrays = [name for name in zf.namelist() if name != "document.json"]
            self.assertTrue(arrays)
            for name in arrays:
                self.assertRegex(name, r"^arrays/\d+\.npy$")
                np.load(io.BytesIO(zf.read(name)), allow_pickle=False)

    def test_other_standard_types_round_trip(self):
        objects = np.empty((2, 2), dtype=object)
        objects[0, 0], objects[0, 1] = 1.5, None
        objects[1, 0], objects[1, 1] = "x", (1, 2)
        values = [
            None,
            True,
            7,
            2.5,
            float("nan"),
            "text",
            (1, (2, 3)),
            {4, 5},
            frozenset({6}),
            1 + 2j,
            b"\x00\xff",
            bytearray(b"ab"),
            {1: "int key", (2, 3): "tuple key"},
            {"$type": "a key that looks like a type tag"},
            objects,
            np.zeros((0, 3), dtype=np.float32),
        ]
        doc = _sample_document()
        doc.frames = [values]
        doc.save(self.path)
        self.assert_same([values], self.load().frames)

    def test_opens_version_1_files(self):
        image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        frames = [[image, np.array([1.5, -2.0]), [(10, 20)], 63.0, np.float64(0.25)]]
        # Documents saved before May 2025 had no lateral data in them.
        self.write_gzip(pickle.dumps(_version_1(frames)))
        loaded = self.load()
        self.assertEqual(loaded.fps, 24)
        self.assertFalse(loaded.lateral)
        self.assert_same(frames, loaded.frames)

    def test_opens_version_1_files_written_with_numpy_1(self):
        frames = [[np.arange(6, dtype=np.int32).reshape(2, 3)]]
        # Protocol 3 spells module names out as text, so the numpy 2 names can be
        # swapped for the ones numpy 1 wrote.
        data = pickle.dumps(_version_1(frames), protocol=3)
        self.assertIn(b"numpy._core.multiarray\n", data)
        data = data.replace(b"numpy._core.multiarray\n", b"numpy.core.multiarray\n")
        self.write_gzip(data)
        self.assert_same(frames, self.load().frames)

    def test_refuses_version_1_files_that_run_code(self):
        self.write_gzip(pickle.dumps(_version_1([_RunsCode()])))
        with self.assertRaisesRegex(TypeError, "contains code"):
            self.load()
        self.assertEqual(_calls, [])

    def test_refuses_arrays_that_need_pickle(self):
        _sample_document().save(self.path)
        crafted = io.BytesIO()
        np.save(crafted, np.array([_RunsCode()], dtype=object), allow_pickle=True)
        self.rewrite(replace={"arrays/0.npy": crafted.getvalue()})
        with self.assertRaises(TypeError):
            self.load()
        self.assertEqual(_calls, [])

    def test_ignores_measure_classes_dynaface_does_not_have(self):
        _sample_document().save(self.path)
        # Importing "this" prints the Zen of Python, so finding it in sys.modules
        # would mean loading the document imported a module that the file named.
        sys.modules.pop("this", None)

        def add_foreign_measure(doc):
            foreign = {"name": "this.Zen", "enabled": True, "items": []}
            doc["body"]["measures"].append(foreign)

        self.rewrite(change_json=add_foreign_measure)
        loaded = self.load()
        self.assertNotIn("this", sys.modules)
        self.assertEqual(len(loaded.measures), len(measures.all_measures()))

    def test_refuses_newer_versions(self):
        _sample_document().save(self.path)
        self.rewrite(change_json=lambda doc: doc["header"].update(version=3))
        with self.assertRaisesRegex(TypeError, "newer version"):
            self.load()

    def test_refuses_unknown_value_types(self):
        _sample_document().save(self.path)
        crafted = {"$type": "eval", "items": []}
        self.rewrite(change_json=lambda doc: doc["body"].update(frames=crafted))
        with self.assertRaises(TypeError):
            self.load()

    def test_refuses_files_that_are_not_documents(self):
        with open(self.path, "wb") as f:
            f.write(b"not a document")
        with self.assertRaises(TypeError):
            self.load()


if __name__ == "__main__":
    unittest.main()
