"""Tests that Dynaface documents (.dyfc files) round-trip and refuse unsafe content.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import errno
import gc
import gzip
import io
import json
import math
import os
import pickle
import struct
import sys
import tempfile
import tracemalloc
import unittest
import weakref
import zipfile
from unittest import mock

import numpy as np
from dynaface import facial, measures
from numpy._core.multiarray import _reconstruct as _np_reconstruct
from numpy._core.multiarray import scalar as _np_scalar
from numpy._core.numeric import _frombuffer as _np_frombuffer
from dynaface.const import Pose
from dynaface.measures_cheek import AnalyzeCheekMeasures
from dynaface.measures_lateral import AnalyzeLateral

import dynaface_document
from dynaface_document import (
    DOC_BODY,
    DOC_BODY_FRAMES,
    DOC_HEADER,
    DocumentError,
    DynafaceDocument,
)

_calls = []


def _record_call(*args):
    _calls.append(args)


class _RunsCode:
    """Pickles to a call of _record_call, the way a malicious file would run code."""

    def __reduce__(self):
        return (_record_call, ("ran",))


class _Witness:
    """Stands for memory a crafted file points at: any use of it is recorded."""

    def __getitem__(self, key):
        _record_call("witness", key)
        raise KeyError(key)


# Kept alive for the whole run: a crafted array holds its address as raw bytes.
_WITNESS = _Witness()


class _ForgedDtype:
    """Pickles a dtype whose state says it holds no Python objects, as numpy would
    pickle the dtype `real` with its flags set to 0."""

    def __init__(self, real, built_as=None):
        self._real = np.dtype(real)
        self._built_as = built_as

    def __reduce__(self):
        func, args, state = self._real.__reduce__()
        if self._built_as is not None:
            args = (self._built_as,) + args[1:]
        return (func, args, state[:-1] + (0,))


class _RawArray:
    """Pickles an array made by numpy's _frombuffer from raw bytes, as protocol 5
    pickles an array."""

    def __init__(self, raw, dtype, shape):
        self._args = (raw, dtype, shape, "C")

    def __reduce__(self):
        return (_np_frombuffer, self._args)


class _Reduce:
    """Pickles as a call of func with args, and a BUILD of state when given."""

    def __init__(self, func, args, state=None):
        self._value = (func, args) if state is None else (func, args, state)

    def __reduce__(self):
        return self._value


# Pickle opcodes (protocol 3) for numpy's placeholder for an array, which a BUILD
# would then fill in.
_PLACEHOLDER = (
    pickle.GLOBAL
    + b"numpy._core.multiarray\n_reconstruct\n"
    + pickle.GLOBAL
    + b"numpy\nndarray\n"
    + pickle.BININT1
    + b"\x00"
    + pickle.TUPLE1
    + pickle.SHORT_BINBYTES
    + b"\x01b"
    + pickle.TUPLE3
    + pickle.REDUCE
)


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

    def test_a_failed_save_leaves_the_document_as_it_was(self):
        _sample_document().save(self.path)
        with open(self.path, "rb") as f:
            before = f.read()
        full = OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(dynaface_document.np, "save", side_effect=full):
            with self.assertRaises(OSError):
                _sample_document().save(self.path)
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(os.listdir(self._dir.name), ["test.dyfc"])
        self.load()

    def test_save_replaces_the_document_and_leaves_nothing_else(self):
        _sample_document().save(self.path)
        doc = _sample_document()
        doc.fps = 12
        doc.save(self.path)
        self.assertEqual(self.load().fps, 12)
        self.assertEqual(os.listdir(self._dir.name), ["test.dyfc"])

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_save_keeps_the_document_permissions(self):
        _sample_document().save(self.path)
        os.chmod(self.path, 0o640)
        _sample_document().save(self.path)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o640)

    @unittest.skipIf(
        hasattr(os, "geteuid") and os.geteuid() == 0, "root can write anything"
    )
    def test_save_refuses_a_read_only_document(self):
        _sample_document().save(self.path)
        with open(self.path, "rb") as f:
            before = f.read()
        os.chmod(self.path, 0o444)
        try:
            with self.assertRaises(PermissionError):
                _sample_document().save(self.path)
            with open(self.path, "rb") as f:
                self.assertEqual(f.read(), before)
        finally:
            os.chmod(self.path, 0o644)

    @unittest.skipIf(os.name == "nt", "symbolic links need privileges on Windows")
    def test_save_through_a_link_saves_where_it_points(self):
        _sample_document().save(self.path)
        link = os.path.join(self._dir.name, "link.dyfc")
        os.symlink(self.path, link)
        doc = _sample_document()
        doc.fps = 12
        doc.save(link)
        self.assertTrue(os.path.islink(link))
        self.assertEqual(self.load().fps, 12)

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

    def test_opens_version_1_files_of_each_protocol(self):
        objects = np.empty(2, dtype=object)
        objects[0], objects[1] = (1, 2), None
        frames = [
            [
                np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
                # Fortran order, then neither order (protocol 5 writes these
                # with an axis order).
                np.arange(6.0).reshape(2, 3).T,
                np.arange(24, dtype=np.int32).reshape(2, 3, 4).transpose(2, 0, 1),
                np.zeros((0, 3), dtype=np.float32),
                np.array(7, dtype=np.int16),
                np.array(["ab", "c"]),
                objects,
                [(np.int64(10), np.int64(20))],
                np.float32(0.5),
                np.bool_(True),
                np.complex128(1 + 2j),
                {1, 2},
                frozenset({3}),
                1 + 2j,
                bytearray(b"ab"),
                b"\x00\xff",
            ]
        ]
        for protocol in (3, 4, 5):
            with self.subTest(protocol=protocol):
                self.write_gzip(pickle.dumps(_version_1(frames), protocol=protocol))
                self.assert_same(frames, self.load().frames)

    def test_refuses_version_1_files_that_run_code(self):
        self.write_gzip(pickle.dumps(_version_1([_RunsCode()])))
        with self.assertRaisesRegex(TypeError, "contains code"):
            self.load()
        self.assertEqual(_calls, [])

    def assert_refused(self, crafted, protocol=3):
        """Asserts that a version 1 file whose measures are crafted is refused as
        data a document never holds, before anything touches it. Returns the
        refusal."""
        doc = _version_1([])
        doc[DOC_BODY]["measures"] = crafted
        self.write_gzip(pickle.dumps(doc, protocol=protocol))
        with self.assertRaisesRegex(DocumentError, "contains data") as raised:
            self.load()
        self.assertEqual(_calls, [])
        return raised.exception

    def test_a_malformed_body_is_not_a_document(self):
        doc = _version_1([])
        doc[DOC_BODY]["measures"] = [{"enabled": True}]
        self.write_gzip(pickle.dumps(doc))
        with self.assertRaisesRegex(DocumentError, "valid Dynaface document"):
            self.load()

    def test_refuses_dtypes_whose_state_hides_their_objects(self):
        # Raw bytes holding the address of a live object: an object array made
        # from them would read the address as a reference to that object.
        pointer = struct.pack("<Q", id(_WITNESS))
        forged = {
            "object": (_ForgedDtype("O"), pointer),
            "structured": (
                _ForgedDtype([("a", "O"), ("b", "u8")]),
                pointer + bytes(8),
            ),
            "structured code": (
                _ForgedDtype([("a", "O"), ("b", "u8")], built_as="O,u8"),
                pointer + bytes(8),
            ),
        }
        for name, (dtype, raw) in forged.items():
            for protocol in (3, 5):
                with self.subTest(name, protocol=protocol):
                    self.assert_refused(_RawArray(raw, dtype, (1,)), protocol)

    def test_refuses_arrays_made_by_calling_the_array_class(self):
        call = _Reduce(np.ndarray, ((2,), np.dtype("f8"), bytes(16)))
        self.assert_refused(call)

    def test_refuses_array_states_that_do_not_fit(self):
        pointer = struct.pack("<Q", id(_WITNESS))
        states = {
            "too few bytes": (1, (1000,), np.dtype("f8"), False, bytes(8)),
            "bytes for objects": (1, (1,), np.dtype("O"), False, pointer),
            "too few objects": (1, (3,), np.dtype("O"), False, [1]),
            "negative size": (1, (-1,), np.dtype("f8"), False, b""),
            "four entries": ((1,), np.dtype("f8"), False, bytes(8)),
            "65 dimensions": (1, (1,) * 65, np.dtype("f8"), False, bytes(8)),
            "a size past numpy's": (1, (2**63,), np.dtype("u1"), False, b""),
            "elements of no size": (1, (2**62,), np.dtype("V"), False, b""),
        }
        for name, state in states.items():
            with self.subTest(name):
                placeholder = (np.ndarray, (0,), b"b")
                self.assert_refused(_Reduce(_np_reconstruct, placeholder, state))

    def test_refuses_dtype_states_numpy_did_not_build(self):
        states = {
            "short": (3, "<"),
            "other byte order": (3, ">", None, None, None, -1, -1, 0),
            "float flags": (3, "<", None, None, None, -1, -1, 0.0),
        }
        for name, state in states.items():
            with self.subTest(name):
                self.assert_refused(_Reduce(np.dtype, ("f8", False, True), state))

    def test_refuses_states_for_anything_but_dtypes_and_arrays(self):
        self.assert_refused(_Reduce(set, ([1],), {"x": 1}))
        self.assert_refused(_Reduce(_np_scalar, (np.dtype("f8"), bytes(8)), (1,)))

    def test_refuses_raw_arrays_of_elements_of_no_size(self):
        # Copying an array of 2**62 elements of no size never ends.
        for code in ("V", "S", "U"):
            with self.subTest(code):
                self.assert_refused(_RawArray(b"", np.dtype(code), (2**62,)))

    def assert_stream_refused(self, stream):
        """Asserts that a version 1 file holding this pickle stream is refused as
        data a document never holds."""
        self.write_gzip(stream)
        with self.assertRaisesRegex(DocumentError, "contains data"):
            self.load()

    def test_refuses_out_of_band_buffers(self):
        # READONLY_BUFFER makes a view of an array's data, which a later state
        # for that array frees while the view still reads it.
        data = pickle.BYTEARRAY8 + struct.pack("<Q", 3) + b"abc"
        streams = {
            "read-only buffer": data + pickle.READONLY_BUFFER,
            "next buffer": pickle.NEXT_BUFFER,
        }
        for name, ops in streams.items():
            with self.subTest(name):
                self.assert_stream_refused(b"\x80\x05" + ops + pickle.STOP)

    def test_refuses_a_state_for_an_array_that_has_its_data(self):
        # A state frees the data an array had, which a view of it may still
        # read: only an empty placeholder takes one, and only once.
        state = pickle.dumps((1, (1,), np.dtype("f8"), False, bytes(8)), protocol=3)
        arrays = {
            "a second state": np.arange(3.0),
            "a state for an array from raw bytes": _RawArray(
                bytes(8), np.dtype("f8"), (1,)
            ),
        }
        for name, array in arrays.items():
            with self.subTest(name):
                stream = pickle.dumps(array, protocol=3)
                self.assertTrue(stream.endswith(pickle.STOP))
                self.assert_stream_refused(
                    stream[:-1] + state[2:-1] + pickle.BUILD + pickle.STOP
                )
        with self.subTest("an array where a dropped placeholder was"):
            # The placeholder goes, and the array made next may well take its
            # place in memory, and so its id.
            raw = pickle.dumps(_RawArray(bytes(8), np.dtype("f8"), (1,)), protocol=3)
            self.assert_stream_refused(
                b"\x80\x03"
                + _PLACEHOLDER
                + pickle.POP
                + raw[2:-1]
                + state[2:-1]
                + pickle.BUILD
                + pickle.STOP
            )

    def test_does_not_keep_placeholders_the_file_drops(self):
        # A file can make numpy's placeholder for an array over and over and drop
        # each one; waiting for their states mustn't keep them all alive. The
        # callable and its arguments go in the memo, so each placeholder takes
        # four opcodes.
        callable_and_args = (
            b"\x80\x03"
            + pickle.GLOBAL
            + b"numpy._core.multiarray\n_reconstruct\n"
            + pickle.BINPUT
            + b"\x00"
            + pickle.POP
            + pickle.GLOBAL
            + b"numpy\nndarray\n"
            + pickle.BININT1
            + b"\x00"
            + pickle.TUPLE1
            + pickle.SHORT_BINBYTES
            + b"\x01b"
            + pickle.TUPLE3
            + pickle.BINPUT
            + b"\x01"
            + pickle.POP
        )
        each = pickle.BINGET + b"\x00" + pickle.BINGET + b"\x01" + pickle.REDUCE
        document = pickle.dumps(_version_1([]), protocol=3)
        self.write_gzip(
            callable_and_args + (each + pickle.POP) * 100_000 + document[2:]
        )
        tracemalloc.start()
        try:
            loaded = self.load()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(loaded.fps, 24)
        self.assertLess(peak, 8 << 20)

    def test_refusals_quote_the_file_in_brief(self):
        # The file names what it refers to, any text at any length; the message
        # quotes it on one line, cut short.
        module = "<a href='https://example.com'>update</a>\n" + "x" * 1000
        raw = module.encode()
        stream = (
            b"\x80\x04"
            + pickle.BINUNICODE
            + struct.pack("<I", len(raw))
            + raw
            + pickle.SHORT_BINUNICODE
            + b"\x01y"
            + pickle.STACK_GLOBAL
            + pickle.STOP
        )
        self.write_gzip(stream)
        with self.assertRaises(DocumentError) as raised:
            self.load()
        message = str(raised.exception)
        self.assertIn(
            "contains code (<a href='https://example.com'>update</a>?x", message
        )
        self.assertIn("…) that would run", message)
        self.assertNotIn("\n", message)
        self.assertLess(len(message), len(self.path) + 200)

    def test_refusals_describe_what_the_file_holds_in_brief(self):
        # A list of the same list twice, 16 levels deep, pickles to a few hundred
        # bytes and formats to a quarter of a million characters (64 levels, to
        # more than memory holds), so a refusal describes it in a few words.
        nested = []
        for _ in range(16):
            nested = [nested, nested]
        u1 = np.dtype("u1")
        placeholder = (np.ndarray, (0,), b"b")
        crafted = {
            "shape": _RawArray(b"", u1, nested),
            "shape in a state": _Reduce(
                _np_reconstruct, placeholder, (1, nested, u1, False, b"")
            ),
            "dtype": _RawArray(b"", nested, (0,)),
            "dtype code": _Reduce(np.dtype, (nested, False, True)),
            "array class": _Reduce(_np_reconstruct, (nested, (0,), b"b")),
            "axis order": _Reduce(_np_frombuffer, (b"", u1, (0,), "K", nested)),
            "order": _Reduce(_np_frombuffer, (b"", u1, (0,), nested)),
        }
        for name, value in crafted.items():
            with self.subTest(name):
                refusal = self.assert_refused(value)
                self.assertLess(len(str(refusal.__cause__)), 300)

    def test_refuses_axis_orders_of_anything_but_ints_before_sorting_them(self):
        # Sorting compares whatever the file put there, which can take any time
        # (or, for complex numbers, fail).
        axes = (b"", np.dtype("u1"), (0, 0), "K", (1j, 0j))
        self.assert_refused(_Reduce(_np_frombuffer, axes))

    def test_refuses_a_bytearray_longer_than_its_file_without_making_it(self):
        # A damaged length (one flipped bit can make terabytes) is refused when
        # the file ends, before a bytearray that long is made.
        length = struct.pack("<Q", 1 << 28)
        self.write_gzip(
            b"\x80\x05" + pickle.BYTEARRAY8 + length + bytes(16) + pickle.STOP
        )
        tracemalloc.start()
        try:
            with self.assertRaisesRegex(DocumentError, "valid Dynaface document"):
                self.load()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 64 << 20)

    def test_a_version_1_document_goes_when_it_is_dropped(self):
        # Every protocol 2-4 file memoizes numpy's _reconstruct, which the
        # unpickler hands out as its own method: the memo held the unpickler, and
        # with it every array the file memoized, until the garbage collector ran.
        frames = [[np.arange(12.0)]]
        self.write_gzip(pickle.dumps(_version_1(frames), protocol=4))
        gc.disable()
        try:
            doc = self.load()
            array = weakref.ref(doc.frames[0][0])
            del doc
            self.assertIsNone(array())
        finally:
            gc.enable()

    def test_opens_version_1_arrays_too_big_to_read_at_once(self):
        # Protocol 5 writes an array's data as one bytearray, read in parts.
        frames = [[np.arange(300_000, dtype=np.float64)]]
        data = pickle.dumps(_version_1(frames), protocol=5)
        self.assertIn(pickle.BYTEARRAY8, data)
        self.write_gzip(data)
        self.assert_same(frames, self.load().frames)

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
