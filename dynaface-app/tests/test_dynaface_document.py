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
from dynaface import facial, measures
from dynaface.const import Pose
from dynaface.measures_cheek import AnalyzeCheekMeasures
from dynaface.measures_lateral import AnalyzeLateral

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

    def test_profile_saved_before_2_1(self):
        # Its frames end with the face rotation and hold no view; the document keeps
        # the profile's lateral data, which the app sets before restoring a frame.
        _sample_document().save(self.path)
        loaded = self.load()
        face = facial.AnalyzeFace(measures=[AnalyzeLateral()])
        face.lateral = loaded.lateral
        face.lateral_landmarks = loaded.lateral_landmarks
        face.load_state(loaded.frames[0])
        self.assertTrue(face.lateral)
        results = AnalyzeLateral().calc(face, render=False)
        self.assertEqual(set(results), {"nfa", "nla", "at", "nt", "at_nt"})

    def test_frames_keep_their_view(self):
        # From 2.1 each frame's state ends with its view, including a profile's
        # lateral landmarks and outline as arrays.
        doc = _sample_document()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        view = {
            "pose": "lateral",
            "flipped": True,
            "lateral": True,
            "lateral_landmarks": np.arange(12).reshape(6, 2),
            "sagittal_x": np.arange(5, dtype=np.int32),
            "sagittal_y": np.arange(5, 10, dtype=np.int32),
        }
        doc.frames = [[image, [10.0, -2.0, 1.0], [(10, 20)], 0.0, 0.24, None, view]]
        doc.save(self.path)
        loaded = self.load()
        self.assert_same(doc.frames, loaded.frames)

        face = facial.AnalyzeFace(measures=[AnalyzeLateral()])
        face.load_state(loaded.frames[0])
        self.assertTrue(face.lateral and face.flipped)
        self.assertEqual(face.pose, Pose.LATERAL)

    def test_quarter_frames_keep_their_far_cheek(self):
        # A three-quarter frame's view holds which cheek is the far one and that
        # cheek's traced outline as three numeric arrays, which restore with their
        # dtypes so the cheek measure reads them as it did the original.
        doc = _sample_document()
        doc.pose = Pose.QUARTER
        doc.lateral = False
        doc.lateral_landmarks = None
        doc.sagittal_x = None
        doc.sagittal_y = None
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        view = {
            "pose": "quarter",
            "flipped": False,
            "lateral": False,
            "far_left": True,
            "cheek_outline_x": np.array([12.5, 13.0, 13.5], dtype=np.float32),
            "cheek_outline_y": np.array([20, 21, 22], dtype=np.int32),
            "cheek_outline_src": np.array([0, 1, 2], dtype=np.uint8),
        }
        doc.frames = [[image, [45.0, -2.0, 1.0], [(10, 20)], 0.0, 0.24, None, view]]
        doc.save(self.path)
        loaded = self.load()
        self.assertEqual(loaded.pose, Pose.QUARTER)
        self.assertFalse(loaded.lateral)
        self.assert_same(doc.frames, loaded.frames)

        face = facial.AnalyzeFace(measures=[AnalyzeCheekMeasures()])
        face.load_state(loaded.frames[0])
        self.assertEqual(face.pose, Pose.QUARTER)
        self.assertTrue(face.quarter and face.far_left)
        self.assertFalse(face.lateral or face.flipped)
        for name in ("cheek_outline_x", "cheek_outline_y", "cheek_outline_src"):
            self.assert_same(view[name], getattr(face, name))

    def test_documents_without_a_lateral_key_save_again(self):
        # A file without a lateral or pose key (saved before 1.2.0) is loaded as
        # frontal and saves as a version 2 file that opens as one. (A numpy
        # scalar in such a frame comes back as a Python float from the version 2
        # file, so the frames hold none.)
        image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        frames = [[image, np.array([1.5, -2.0]), [(10, 20)], 63.0, 0.25]]
        self.write_gzip(pickle.dumps(_version_1(frames)))
        loaded = self.load()
        self.assertEqual(loaded.pose, Pose.FRONTAL)
        loaded.save(self.path)
        with zipfile.ZipFile(self.path) as zf:
            header = json.loads(zf.read("document.json"))[DOC_HEADER]
        self.assertEqual(header["pose"], "frontal")
        self.assertFalse(header["lateral"])
        saved_again = self.load()
        self.assertEqual(saved_again.pose, Pose.FRONTAL)
        self.assertFalse(saved_again.lateral)
        self.assert_same(frames, saved_again.frames)

    def test_a_lateral_document_without_a_pose_key_is_lateral(self):
        # Dynaface 1.2.0 wrote the lateral key but not the pose: the lateral flag
        # says which view the document had, so its labels and a re-save are right.
        image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        frames = [[image, np.array([1.5, -2.0]), [(10, 20)], 63.0, 0.25]]
        doc = _version_1(frames, lateral=True)
        doc[DOC_BODY]["lateral_landmarks"] = np.arange(4, dtype=np.int32).reshape(2, 2)
        doc[DOC_BODY]["sagittal_x"] = np.arange(3, dtype=np.int32)
        doc[DOC_BODY]["sagittal_y"] = np.arange(3, 6, dtype=np.int32)
        self.write_gzip(pickle.dumps(doc))
        loaded = self.load()
        self.assertTrue(loaded.lateral)
        self.assertEqual(loaded.pose, Pose.LATERAL)
        doc = _version_1(frames, lateral=False)
        doc[DOC_BODY]["lateral_landmarks"] = None
        doc[DOC_BODY]["sagittal_x"] = None
        doc[DOC_BODY]["sagittal_y"] = None
        self.write_gzip(pickle.dumps(doc))
        self.assertEqual(self.load().pose, Pose.FRONTAL)

    def test_add_missing_views_gives_a_profile_document_frames_its_view(self):
        # A profile document saved by 2.0.4 or earlier holds its lateral data in
        # the header alone; with the view on each frame, a face that restores
        # the frame on its own (the series, the Evaluation, the video export)
        # measures it as the profile, and a re-save keeps the view.
        saved = _sample_document()
        saved.frames.append(saved.frames[0] + [{"pose": "lateral", "lateral": True}])
        saved.save(self.path)
        loaded = self.load()
        loaded.add_missing_views()
        self.assertEqual([len(frame) for frame in loaded.frames], [7, 7])
        view = loaded.frames[0][6]
        self.assertEqual(view["pose"], "lateral")
        self.assertTrue(view["lateral"])
        self.assertFalse(view["flipped"])
        self.assert_same(saved.lateral_landmarks, view["lateral_landmarks"])
        self.assert_same(saved.sagittal_x, view["sagittal_x"])
        self.assert_same(saved.sagittal_y, view["sagittal_y"])
        # the frame that had a view keeps it
        self.assertEqual(loaded.frames[1][6], {"pose": "lateral", "lateral": True})
        face = facial.AnalyzeFace(measures=[AnalyzeLateral()])
        face.load_state(loaded.frames[0])
        self.assertTrue(face.lateral)
        self.assertEqual(face.pose, Pose.LATERAL)
        self.assert_same(saved.lateral_landmarks, face.lateral_landmarks)
        loaded.save(self.path)
        self.assertEqual([len(frame) for frame in self.load().frames], [7, 7])
        # a frontal document's frames restore as frontal already; left alone
        frontal = _sample_document()
        frontal.pose, frontal.lateral = Pose.FRONTAL, False
        frontal.lateral_landmarks = frontal.sagittal_x = frontal.sagittal_y = None
        frontal.save(self.path)
        loaded = self.load()
        loaded.add_missing_views()
        self.assertEqual([len(frame) for frame in loaded.frames], [6])

    def test_new_documents_save_as_frontal(self):
        # A document whose pose was never set saves as frontal rather than failing.
        doc = _sample_document()
        doc.pose = None
        doc.save(self.path)
        self.assertEqual(self.load().pose, Pose.FRONTAL)

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
