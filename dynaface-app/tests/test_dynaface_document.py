"""Tests that Dynaface documents (.dyfc files) round-trip and refuse unsafe content.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import errno
import faulthandler
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
import threading
import time
import tracemalloc
import unittest
import weakref
import zipfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
from dynaface import facial, measures
from numpy._core.multiarray import _reconstruct as _np_reconstruct
from numpy._core.multiarray import scalar as _np_scalar
from numpy._core.numeric import _frombuffer as _np_frombuffer
from dynaface.const import Pose
from dynaface.measures_cheek import AnalyzeCheekMeasures
from dynaface.measures_lateral import AnalyzeLateral

import dlg_modal
import dynaface_document
from dynaface_document import (
    DOC_BODY,
    DOC_BODY_FRAMES,
    DOC_HEADER,
    DocumentError,
    DynafaceDocument,
)
from tab_analyze_video import AnalyzeVideoTab

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


# A face's 98 landmarks, as every version saved them: pairs of Python ints, the
# pupils (the last two) level and 20 px apart.
_LANDMARKS = [(10, 20)] * 96 + [(30, 40), (50, 40)]

# A frame's view, as Dynaface 2.1 saves a frontal face's.
_FRONTAL = {"pose": "frontal", "flipped": False, "lateral": False}


def _frame(view=None, image=None):
    """A frame state as Dynaface 1.2 to 2.0.4 saved one: the image, a [0, 0, 0]
    placeholder for the head pose, the landmarks, the pupillary distance, the
    scale and no face rotation; with a view, as 2.1 saves one, which holds the
    head pose."""
    if image is None:
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    frame = [image, [0, 0, 0], list(_LANDMARKS), 20.0, 3.15, None]
    if view is not None:
        frame[1] = [1.5, -2.0, 0.25]
        frame.append(view)
    return frame


def _document(frames, version=2, **header):
    """A document, before it is written, as Dynaface saves one holding these
    frames: a frontal one, unless header says otherwise."""
    return {
        DOC_HEADER: {
            "version": version,
            "fps": 30,
            "lateral": False,
            "pose": "frontal",
            **header,
        },
        DOC_BODY: {
            "measures": [],
            DOC_BODY_FRAMES: frames,
            "lateral_landmarks": None,
            "sagittal_x": None,
            "sagittal_y": None,
        },
    }


def _sample_document():
    doc = DynafaceDocument()
    doc.measures = measures.all_measures()
    doc.measures[0].enabled = False
    doc.measures[1].set_item_enabled(doc.measures[1].items[0].name, False)
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    headpose = np.array([1.5, -2.0, 0.25])
    doc.frames = [[image, headpose, list(_LANDMARKS), 63.0, 0.25, 1.5]]
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


# A memo index that pickle.dumps() never reaches, for opcodes put in its output.
_SPARE = struct.pack("<I", 1 << 30)
_PUT = pickle.LONG_BINPUT + _SPARE
_GET = pickle.LONG_BINGET + _SPARE


def _deep_tuple(levels):
    """Opcodes for the empty tuple in a tuple, in a tuple, levels deep: a byte a
    level, so that a million levels gzip to a kilobyte."""
    return pickle.EMPTY_TUPLE + pickle.TUPLE1 * levels


def _shared_tuple(levels):
    """Opcodes for a tuple of the tuple below it twice, levels deep: 2**levels
    tuples to hash or walk, in 12 bytes a level."""
    each = _PUT + pickle.POP + _GET + _GET + pickle.TUPLE2
    return pickle.EMPTY_TUPLE + each * levels


def _shared_list(levels):
    """Opcodes for a list of the list below it twice, levels deep."""
    each = (
        _PUT + pickle.POP + pickle.EMPTY_LIST + pickle.MARK + _GET + _GET
    ) + pickle.APPENDS
    return pickle.EMPTY_LIST + each * levels


def _nested_object_arrays(levels):
    """A pickle (protocol 2) of an object array holding an object array, holding
    another, levels deep, as numpy pickles each of them."""
    dtype = pickle.dumps(np.dtype("O"), protocol=2)[2:-1]
    state = (
        pickle.MARK
        + pickle.BININT1
        + b"\x01"
        + pickle.BININT1
        + b"\x01"
        + pickle.TUPLE1
        + dtype
        + pickle.NEWFALSE
        + pickle.EMPTY_LIST
        + _GET
        + pickle.APPEND
        + pickle.TUPLE
    )
    each = _PLACEHOLDER + state + pickle.BUILD + _PUT + pickle.POP
    start = b"\x80\x02" + pickle.NONE + _PUT + pickle.POP
    return start + each * levels + _GET + pickle.STOP


def _object_array_links(links, arrays, holder):
    """A pickle (protocol 2) of links of object arrays, each link the given number
    of arrays deep, the innermost holding the link before through holder (opcodes
    that push a container of the spare memo slot's value), and the outermost
    left in the spare memo slot."""
    dtype = pickle.dumps(np.dtype("O"), protocol=2)[2:-1]

    def holding(item):
        state = (
            pickle.MARK
            + pickle.BININT1
            + b"\x01"
            + pickle.BININT1
            + b"\x01"
            + pickle.TUPLE1
            + dtype
            + pickle.NEWFALSE
            + pickle.EMPTY_LIST
            + item
            + pickle.APPEND
            + pickle.TUPLE
        )
        return _PLACEHOLDER + state + pickle.BUILD + _PUT + pickle.POP

    stream = b"\x80\x02" + pickle.NONE + _PUT + pickle.POP
    for _ in range(links):
        stream += holding(holder) + holding(_GET) * (arrays - 1)
    return stream


def _crafted(doc, opcodes, marker="crafted"):
    """The pickle (protocol 3) of doc, with the opcodes, which push one value,
    where doc holds the string marker."""
    stream = pickle.dumps(doc, protocol=3)
    text = pickle.BINUNICODE + struct.pack("<I", len(marker)) + marker.encode()
    assert stream.count(text) == 1
    return stream.replace(text, opcodes)


def _profile(pose="lateral", **body):
    """A version 1 profile document, as Dynaface 1.2.0 to 2.0.4 saved one."""
    doc = _version_1([], lateral=True, pose=pose)
    doc[DOC_BODY].update(lateral_landmarks=None, sagittal_x=None, sagittal_y=None)
    doc[DOC_BODY].update(body)
    return doc


def _json_nested(levels, opening, closing, inner="null"):
    """JSON text of inner within levels of opening and closing, built as text:
    json.dumps() recurses for each level."""
    return opening * levels + inner + closing * levels


# Changes to a document from _document() that put a value of a type, or a size, no
# document holds where it holds a genuine one: in its header, or in its second
# frame, as the tab opens on the first and reads the others when a frame is drawn,
# in a chart or in a video export.


def _header(**values):
    return lambda doc: doc[DOC_HEADER].update(values)


def _frames(value):
    return lambda doc: doc[DOC_BODY].update({DOC_BODY_FRAMES: value})


def _second_frame(entry, value):
    def change(doc):
        doc[DOC_BODY][DOC_BODY_FRAMES][1][entry] = value

    return change


def _second_view(**values):
    return lambda doc: doc[DOC_BODY][DOC_BODY_FRAMES][1][6].update(values)


def _profile_header(**body):
    """A profile document's header, with these values for its profile."""

    def change(doc):
        doc[DOC_HEADER].update(lateral=True, pose="lateral")
        doc[DOC_BODY].update(body)

    return change


_PROFILE_VIEW = {
    "pose": "lateral",
    "lateral": True,
    "lateral_landmarks": np.arange(12).reshape(6, 2) + 100,
}
_QUARTER_VIEW = {
    "pose": "quarter",
    "far_left": True,
    "cheek_outline_x": np.array([12.5, 13.0, 13.5], dtype=np.float32),
    "cheek_outline_y": np.array([20, 21, 22], dtype=np.int32),
    "cheek_outline_src": np.array([0, 1, 2], dtype=np.uint8),
}

# Most of these opened before, and then drawing that frame, a chart or a video
# export raised, which ended Dynaface: PyQt6 ends the app on an exception that
# leaves a slot or a thread.
_WRONG = {
    "a frame rate of text": _header(fps="30"),
    "a frame rate in a list": _header(fps=[30]),
    "no frame rate": _header(fps=None),
    "a frame rate of True": _header(fps=True),
    "a lateral flag of text": _header(lateral="no"),
    "a lateral flag in an array": _header(lateral=np.array([True, False])),
    "frames of text": _frames("frames"),
    "frames in a dict": _frames({"0": 0}),
    "a frame of text": lambda doc: doc[DOC_BODY][DOC_BODY_FRAMES].append("frame"),
    "a frame of four entries": lambda doc: doc[DOC_BODY][DOC_BODY_FRAMES].append(
        _frame()[:4]
    ),
    "an image of text": _second_frame(0, "image"),
    "no image": _second_frame(0, None),
    "an image of floats": _second_frame(0, np.zeros((4, 4, 3))),
    "a gray image": _second_frame(0, np.zeros((4, 4), dtype=np.uint8)),
    "an image with alpha": _second_frame(0, np.zeros((4, 4, 4), dtype=np.uint8)),
    "an image of 16 bits": _second_frame(0, np.zeros((4, 4, 3), dtype=np.uint16)),
    "an image of no pixels": _second_frame(0, np.zeros((0, 4, 3), dtype=np.uint8)),
    "an image of another size": _second_frame(0, np.zeros((8, 4, 3), dtype=np.uint8)),
    "a head pose of text": _second_frame(1, "level"),
    "a head pose of 1e308": _second_frame(1, [1e308, 0.0, 0.0]),
    "landmarks of text": _second_frame(2, "landmarks"),
    "no landmarks": _second_frame(2, None),
    "landmarks of floats": _second_frame(2, [(x + 0.5, y) for x, y in _LANDMARKS]),
    "landmarks of text pairs": _second_frame(2, [("x", "y")] * 98),
    "landmarks of bools": _second_frame(2, [(True, False)] * 98),
    "landmarks of three numbers": _second_frame(2, [(1, 2, 3)] * 98),
    "landmarks in pairs of pairs": _second_frame(2, [((1, 2), (3, 4))] * 98),
    "five landmarks": _second_frame(2, _LANDMARKS[:5]),
    "landmarks in an array": _second_frame(2, np.array(_LANDMARKS)),
    "landmarks past OpenCV's ints": _second_frame(
        2, [(x + 2**31, y) for x, y in _LANDMARKS]
    ),
    "a pupillary distance of text": _second_frame(3, "20"),
    "a scale of text": _second_frame(4, "3.15"),
    "no scale": _second_frame(4, None),
    "a face rotation in a list": _second_frame(5, [0.1]),
    "a view of text": _second_frame(6, "frontal"),
    "a view's pose unknown": _second_view(pose="sideways"),
    "a view's pose in a list": _second_view(pose=["frontal"]),
    "a view's flag in an array": _second_view(flipped=np.array([True, False])),
    "a view's lateral flag of text": _second_view(lateral="no"),
    "a profile's view without lateral landmarks": _second_view(
        pose="lateral", lateral=True
    ),
    "a view's lateral landmarks of floats": _second_view(
        **dict(_PROFILE_VIEW, lateral_landmarks=np.arange(12.0).reshape(6, 2))
    ),
    "a view's lateral landmarks of one number each": _second_view(
        **dict(_PROFILE_VIEW, lateral_landmarks=np.arange(12))
    ),
    "a view's sagittal outline of text": _second_view(
        **dict(_PROFILE_VIEW, sagittal_x="outline")
    ),
    "a cheek outline of text": _second_view(
        **dict(_QUARTER_VIEW, cheek_outline_x="outline")
    ),
    "a cheek outline without its rows": _second_view(
        **dict(_QUARTER_VIEW, cheek_outline_y=None)
    ),
    "a cheek outline with rows it lacks": _second_view(
        **dict(_QUARTER_VIEW, cheek_outline_y=np.arange(5, dtype=np.int32))
    ),
    "a cheek outline past OpenCV's ints": _second_view(
        **dict(_QUARTER_VIEW, cheek_outline_x=np.full(3, 1e12))
    ),
    "a profile's lateral landmarks of text": _profile_header(lateral_landmarks="x"),
    "a profile's lateral landmarks in a list": _profile_header(
        lateral_landmarks=[[1, 2]] * 6
    ),
    "a profile's sagittal outline of text": _profile_header(sagittal_y="outline"),
}

# Values a version 2 file can't hold: ints past what JSON numbers it reads.
_WRONG_IN_VERSION_1 = {
    "a frame rate past a float": _header(fps=10**400),
    "a head pose past a float": _second_frame(1, [2**2000, 0, 0]),
    "landmarks past 64 bits": _second_frame(2, [(2**70, 0)] * 98),
    "a face rotation past a float": _second_frame(5, 10**400),
}


def _run_and_wait(window, f, message):
    """display_please_wait(), without the dialog: runs f and returns its error,
    or None."""
    try:
        f()
    except Exception as e:  # noqa: BLE001 - the dialog returns any error
        return e
    return None


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

    def read_frames(self):
        """The frames the file holds, as read before load() checks that they are
        frames: what the tests of the reader itself read, which put values of
        every type a file can hold where a document holds its frames."""
        return dynaface_document._read_document(self.path)[DOC_BODY][DOC_BODY_FRAMES]

    def write_gzip(self, data):
        with gzip.open(self.path, "wb") as f:
            f.write(data)

    def write(self, doc, version):
        """Writes doc, made by _document(), as a version 1 file (a gzip pickle)
        or a version 2 one (a zip of JSON and arrays), whatever it holds, as
        someone crafting a file could."""
        if version == 1:
            doc = dict(doc, **{DOC_HEADER: dict(doc[DOC_HEADER], version=1)})
            self.write_gzip(pickle.dumps(doc, protocol=4))
        else:
            with open(self.path, "wb") as f:
                DynafaceDocument._write_zip(f, doc)

    def rewrite(self, change_json=None, replace=None):
        """Edits a saved document the way someone crafting a file could."""
        with zipfile.ZipFile(self.path) as zf:
            entries = {name: zf.read(name) for name in zf.namelist()}
        if change_json:
            doc = json.loads(entries["document.json"])
            change_json(doc)
            entries["document.json"] = json.dumps(doc).encode()
        entries.update(replace or {})
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as zf:
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
        landmarks = list(_LANDMARKS)
        doc.frames = [[image, [10.0, -2.0, 1.0], landmarks, 0.0, 0.24, None, view]]
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
        landmarks = list(_LANDMARKS)
        doc.frames = [[image, [45.0, -2.0, 1.0], landmarks, 0.0, 0.24, None, view]]
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
        frames = [[image, np.array([1.5, -2.0]), list(_LANDMARKS), 63.0, 0.25]]
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
        frames = [[image, np.array([1.5, -2.0]), list(_LANDMARKS), 63.0, 0.25]]
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
        kept = {
            "pose": "lateral",
            "lateral": True,
            "lateral_landmarks": np.arange(12).reshape(6, 2) + 100,
        }
        saved.frames.append(saved.frames[0] + [kept])
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
        self.assert_same(kept, loaded.frames[1][6])
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
        self.assert_same([values], self.read_frames())

    def test_opens_version_1_files(self):
        image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        scale = np.float64(0.25)
        frames = [[image, np.array([1.5, -2.0]), list(_LANDMARKS), 63.0, scale]]
        # Documents saved before May 2025 had no lateral data in them.
        self.write_gzip(pickle.dumps(_version_1(frames)))
        loaded = self.load()
        self.assertEqual(loaded.fps, 24)
        self.assertFalse(loaded.lateral)
        self.assert_same(frames, loaded.frames)

    def test_opens_version_1_files_written_with_numpy_1(self):
        frames = [_frame()]
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
                self.assert_same(frames, self.read_frames())

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
        frames = [_frame()]
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
        # Protocol 5 writes an array's data as one bytearray, read in parts: a
        # 1024-pixel crop is 3 MB.
        crop = (np.arange(1024 * 1024 * 3) % 251).astype(np.uint8)
        frames = [_frame(image=crop.reshape(1024, 1024, 3))]
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

    # Files made to crash Dynaface or keep it busy. Each is refused within a few
    # seconds, as any file that isn't a document is, on a thread with the 512 KB
    # stack of the thread the app loads documents on (Qt's default on macOS).
    # Should a limit stop working, a file that overflows the stack crashes this
    # test run, as it did the app, and one that hashes without end holds the GIL,
    # so no thread can time it out: a watchdog of faulthandler's ends the run
    # then, with every thread's traceback.

    def assert_unreadable(self, seconds=10.0):
        """Asserts that loading self.path fails as for any file that isn't a
        document, within seconds, on a thread with a 512 KB stack. Returns the
        error."""
        error = self.load_on_small_stack(seconds)
        self.assertIsInstance(error, DocumentError)
        self.assertIn("does not appear to be a valid Dynaface document", str(error))
        return error

    def load_on_small_stack(self, seconds=10.0):
        """Loads self.path within seconds on a thread with a 512 KB stack, and
        returns the document, or the error loading it raised."""
        outcome = []

        def load():
            try:
                outcome.append(self.load())
            except BaseException as e:
                outcome.append(e)

        faulthandler.dump_traceback_later(seconds + 20, exit=True)
        try:
            default = threading.stack_size(512 << 10)
            try:
                thread = threading.Thread(target=load, daemon=True)
                thread.start()
            finally:
                threading.stack_size(default)
            thread.join(seconds)
        finally:
            faulthandler.cancel_dump_traceback_later()
        self.assertFalse(thread.is_alive(), f"still loading after {seconds} s")
        return outcome[0]

    def rewrite_text(self, old, new):
        """Replaces the text old, which document.json holds once, with new."""
        with zipfile.ZipFile(self.path) as zf:
            text = zf.read("document.json").decode()
        self.assertEqual(text.count(old), 1)
        self.rewrite(replace={"document.json": text.replace(old, new).encode()})

    def save_frames_json(self, text):
        """Saves a version 2 document whose frames are the JSON text (values,
        without the list's brackets)."""
        doc = _sample_document()
        doc.frames = ["crafted"]
        doc.save(self.path)
        self.rewrite_text('"crafted"', text)

    def test_refuses_keys_nested_deep_enough_to_crash(self):
        # Hashing a tuple recurses in C for each level, about 60 bytes of stack
        # a level: a tuple nested 9,000 deep, 86 bytes of gzip, crashed the app
        # with no message.
        key = _deep_tuple(9_000)
        streams = {
            "SETITEM": pickle.EMPTY_DICT + key + pickle.NONE + pickle.SETITEM,
            "SETITEMS": pickle.EMPTY_DICT
            + pickle.MARK
            + key
            + pickle.NONE
            + pickle.SETITEMS,
            "DICT": pickle.MARK + key + pickle.NONE + pickle.DICT,
            "ADDITEMS": pickle.EMPTY_SET + pickle.MARK + key + pickle.ADDITEMS,
            "FROZENSET": pickle.MARK + key + pickle.FROZENSET,
            "set()": pickle.GLOBAL
            + b"builtins\nset\n"
            + pickle.EMPTY_LIST
            + key
            + pickle.APPEND
            + pickle.TUPLE1
            + pickle.REDUCE,
            "frozenset()": pickle.GLOBAL
            + b"builtins\nfrozenset\n"
            + pickle.EMPTY_LIST
            + key
            + pickle.APPEND
            + pickle.TUPLE1
            + pickle.REDUCE,
        }
        streams["a million deep"] = (
            pickle.EMPTY_DICT + _deep_tuple(1_000_000) + pickle.NONE + pickle.SETITEM
        )
        for name, ops in streams.items():
            with self.subTest(name):
                self.write_gzip(b"\x80\x04" + ops + pickle.STOP)
                self.assert_unreadable()
        with self.subTest("version 2"):
            tuples = _json_nested(450, '{"$type": "tuple", "items": [', "]}")
            self.save_frames_json('{"$type": "set", "items": [%s]}' % tuples)
            self.assert_unreadable()

    def test_refuses_object_arrays_nested_deep_enough_to_crash(self):
        # Freeing an object array frees what it holds, in C, about 1.6 KB of
        # stack an array: a few hundred levels crashed the app as the document
        # was dropped, even one that had been refused.
        self.write_gzip(_nested_object_arrays(1000))
        self.assert_unreadable()
        self.save_frames_json(
            _json_nested(450, '{"$type": "ndarray", "shape": [1], "items": [', "]}")
        )
        self.assert_unreadable()
        # Three object arrays deep, held by a document.
        document = _crafted(_version_1(["crafted"]), _GET)[2:]
        self.write_gzip(_object_array_links(1, 3, _GET) + document)
        self.assert_unreadable()

    def test_frees_object_arrays_linked_through_containers(self):
        # Python puts off freeing a container 50 deep, but not an array, so 50
        # links of arrays free at once: links of 32, the most there were, freed
        # as the load failed crashed the app. Links of two, the most there are
        # now, free safely, whether the load then fails or goes on.
        holders = {
            "tuple": _GET + pickle.TUPLE1,
            "list": pickle.EMPTY_LIST + _GET + pickle.APPEND,
            "dict": pickle.EMPTY_DICT
            + pickle.BININT1
            + b"\x00"
            + _GET
            + pickle.SETITEM,
        }
        document = pickle.dumps(_version_1([]), protocol=2)[2:]
        for name, holder in holders.items():
            with self.subTest(name):
                self.write_gzip(_object_array_links(60, 32, holder) + b"\xff")
                self.assert_unreadable()
                links = _object_array_links(60, 2, holder)
                self.write_gzip(links + b"\xff")
                self.assert_unreadable()
                self.write_gzip(links + pickle.NONE + _PUT + pickle.POP + document)
                self.assertIsInstance(self.load_on_small_stack(), DynafaceDocument)

    def test_opens_an_object_array_held_by_another(self):
        inner = np.empty(1, dtype=object)
        inner[0] = (1, 2)
        outer = np.empty(1, dtype=object)
        outer[0] = inner
        inner = outer
        for protocol in (3, 5):
            with self.subTest(protocol=protocol):
                self.write_gzip(pickle.dumps(_version_1([[inner]]), protocol=protocol))
                self.assert_same([[inner]], self.read_frames())
        doc = _sample_document()
        doc.frames = [[inner]]
        doc.save(self.path)
        self.assert_same([[inner]], self.read_frames())

    def test_refuses_keys_that_share_their_members(self):
        # A tuple of a tuple twice, itself of a tuple twice, 100 levels deep is
        # 1,200 bytes and 2**100 tuples to hash: loading never ended.
        key = _shared_tuple(100)
        streams = {
            "dict key": pickle.EMPTY_DICT + key + pickle.NONE + pickle.SETITEM,
            "set member": pickle.EMPTY_SET + pickle.MARK + key + pickle.ADDITEMS,
            "frozenset member": pickle.MARK + key + pickle.FROZENSET,
        }
        for name, ops in streams.items():
            with self.subTest(name):
                self.write_gzip(b"\x80\x04" + ops + pickle.STOP)
                self.assert_unreadable()

    def test_refuses_int_keys_chosen_to_hash_alike(self):
        # An int hashes to itself modulo 2**61 - 1, so these all hash alike, and
        # each one added to a dict is compared with all the others.
        keys = [k * (2**61 - 1) for k in range(1, 6001)]
        with self.subTest("version 1"):
            self.write_gzip(pickle.dumps(_version_1([dict.fromkeys(keys)])))
            self.assert_unreadable()
        with self.subTest("version 2"):
            items = json.dumps([[key, 0] for key in keys])
            self.save_frames_json('{"$type": "dict", "items": %s}' % items)
            self.assert_unreadable()
        with self.subTest("memo"):
            # Protocol 0's PUT names its memo index in digits, any int at all
            # (and is refused, as protocol 0's text opcodes are).
            puts = b"".join(b"p%d\n" % key for key in keys)
            document = pickle.dumps(_version_1([]), protocol=0)
            self.write_gzip(pickle.NONE + puts + pickle.POP + document)
            self.assert_unreadable()

    def test_refuses_crafted_poses(self):
        # Pose() hashes the value and quotes it when it isn't a pose: a tuple
        # 100,000 deep crashed the app, a shared one never finished hashing,
        # and a list of a list twice, 26 levels deep, quoted to 2 GB of text.
        poses = {
            "deep tuple": _deep_tuple(100_000),
            "shared tuple": _shared_tuple(100),
            "shared list": _shared_list(40),
            "tuple": pickle.BINUNICODE
            + struct.pack("<I", 7)
            + b"lateral"
            + pickle.TUPLE1,
            "unknown": pickle.BINUNICODE + struct.pack("<I", 8) + b"sideways",
        }
        for name, ops in poses.items():
            with self.subTest(name):
                self.write_gzip(_crafted(_profile(pose="crafted"), ops))
                self.assert_unreadable()
        with self.subTest("version 2"):
            _sample_document().save(self.path)
            self.rewrite_text(
                '"pose": "lateral"', '"pose": ' + _json_nested(900, "[", "]")
            )
            self.assert_unreadable()
            _sample_document().save(self.path)
            self.rewrite_text(
                '"pose": "lateral"', '"pose": {"$type": "tuple", "items": ["lateral"]}'
            )
            self.assert_unreadable()

    def test_crafted_measure_names_name_no_measure(self):
        # Looking a name up hashed it, so a shared tuple never finished; one
        # this size is refused, and a name that isn't a str names nothing.
        doc = _version_1([])
        doc[DOC_BODY]["measures"] = [{"name": "crafted", "enabled": True, "items": []}]
        self.write_gzip(_crafted(doc, _shared_tuple(100)))
        self.assert_unreadable()
        self.write_gzip(_crafted(doc, pickle.EMPTY_TUPLE))
        self.assertEqual(len(self.load().measures), len(measures.all_measures()))
        _sample_document().save(self.path)

        def add_foreign_measure(saved):
            name = {"$type": "tuple", "items": ["dynaface.measures_frontal"]}
            foreign = {"name": name, "enabled": True, "items": []}
            saved["body"]["measures"].append(foreign)

        self.rewrite(change_json=add_foreign_measure)
        self.assertEqual(len(self.load().measures), len(measures.all_measures()))

    def test_refuses_values_a_pickle_shares_many_times_over(self):
        # A profile's landmarks of a list of a list twice, 40 levels deep: to
        # np.array() and saving, which the app runs on a document it opens,
        # 2**40 lists.
        self.write_gzip(
            _crafted(_profile(lateral_landmarks="crafted"), _shared_list(40))
        )
        self.assert_unreadable()
        with self.subTest("too deep"):
            frames = _deep_tuple(100)
            self.write_gzip(_crafted(_version_1(["crafted"]), frames))
            self.assert_unreadable()

    def test_refuses_builtins_called_with_what_pickle_never_passes(self):
        # bytearray() of an int is that many zeros, made before anything else;
        # complex() of a str and set() of one go through all of it.
        text = pickle.BINUNICODE + struct.pack("<I", 3) + b"1+j"
        calls = {
            "bytearray": pickle.BININT + struct.pack("<i", 1 << 28),
            "complex": text,
            "set": text,
        }
        for name, argument in calls.items():
            with self.subTest(name):
                call = (
                    pickle.GLOBAL
                    + f"builtins\n{name}\n".encode()
                    + argument
                    + pickle.TUPLE1
                    + pickle.REDUCE
                )
                self.write_gzip(_crafted(_version_1(["crafted"]), call))
                tracemalloc.start()
                try:
                    self.assert_unreadable()
                    peak = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 64 << 20)

    def test_refuses_files_of_many_cheap_opcodes_quickly(self):
        # 100 MB of opcodes that do almost nothing gzip to 100 KB; a file of a
        # billion, 1 MB, took over three minutes to read.
        chunk = (pickle.NONE + pickle.POP) * 1_000_000
        with gzip.open(self.path, "wb") as f:
            f.write(b"\x80\x02")
            for _ in range(50):
                f.write(chunk)
            f.write(pickle.NONE + pickle.STOP)
        start = time.perf_counter()
        self.assert_unreadable()
        self.assertLess(time.perf_counter() - start, 5)

    def test_refuses_version_2_values_beyond_what_its_size_allows(self):
        # json.loads() makes all of a file's values at once: 15 MB of "{}," make
        # five million dicts, and the file is a few kilobytes.
        self.save_frames_json("[" + "{}," * 5_000_000 + "{}]")
        self.assert_unreadable()
        with self.subTest("an array named again and again"):
            array = '{"$type": "ndarray", "file": "arrays/0.npy"}'
            self.save_frames_json("[" + ", ".join([array] * 1000) + "]")
            self.assert_unreadable()

    def test_refuses_version_2_arrays_of_elements_of_no_size(self):
        # 2**62 elements of no size take no data, and copying them never ends.
        _sample_document().save(self.path)
        header = {"descr": "|V0", "fortran_order": False, "shape": (2**62,)}
        crafted = io.BytesIO()
        np.lib.format.write_array_header_1_0(crafted, header)
        self.rewrite(replace={"arrays/0.npy": crafted.getvalue()})
        self.assert_unreadable()

    def test_opens_documents_of_5000_frames(self):
        # The most frames an import keeps, 98 landmarks each, all alike, which
        # compress best and so leave the least room under the limits. Each of
        # the two loads took half a second here.
        rng = np.random.default_rng(0)
        landmarks = [(int(x), int(y)) for x, y in rng.integers(0, 1024, (98, 2))]
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

        def frame(version):
            # As Dynaface 2.0.4 saved a frame, then as 2.1 saves one.
            if version == 1:
                return [image.copy(), [0, 0, 0], list(landmarks), 63.0, 0.25, None]
            view = {"pose": "frontal", "flipped": False, "lateral": False}
            pose = [1.5, -2.0, 0.25]
            return [image.copy(), pose, list(landmarks), 63.0, 0.25, None, view]

        self.write_gzip(pickle.dumps(_version_1([frame(1) for _ in range(5000)])))
        start = time.perf_counter()
        self.assertEqual(len(self.load().frames), 5000)
        self.assertLess(time.perf_counter() - start, 3)
        doc = _sample_document()
        doc.frames = [frame(2) for _ in range(5000)]
        doc.save(self.path)
        start = time.perf_counter()
        self.assertEqual(len(self.load().frames), 5000)
        self.assertLess(time.perf_counter() - start, 3)

    def test_measures_named_again_are_made_once(self):
        # Each measure has items, Landmarks 197 of them, and a file can name one
        # over and over in a few hundred bytes: 100,000 times took 4.6 GB.
        name = "dynaface.measures_frontal.AnalyzeLandmarks"
        entry = {"name": name, "enabled": True, "items": [{"name": "x", "enabled": 1}]}
        doc = _version_1([])
        doc[DOC_BODY]["measures"] = [entry] * 100_000
        self.write_gzip(pickle.dumps(doc, protocol=3))
        tracemalloc.start()
        try:
            loaded = self.load_on_small_stack()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(len(loaded.measures), len(measures.all_measures()))
        self.assertLess(peak, 64 << 20)
        _sample_document().save(self.path)
        entries = ", ".join([json.dumps(entry)] * 100_000)
        self.rewrite_text('"measures": [', f'"measures": [{entries}, ')
        loaded = self.load_on_small_stack()
        self.assertEqual(len(loaded.measures), len(measures.all_measures()))

    def test_refuses_flags_that_are_not_bools(self):
        doc = _version_1([])
        name = "dynaface.measures_frontal.AnalyzeFAI"
        doc[DOC_BODY]["measures"] = [{"name": name, "enabled": [1], "items": []}]
        self.write_gzip(pickle.dumps(doc))
        self.assert_unreadable()

    def test_refuses_strings_with_lone_surrogates(self):
        # Decoding lets them through, a handler call each, so a megabyte of gzip
        # took 23 seconds; no document holds one.
        surrogate = b"\xed\xa0\x80"
        text = pickle.BINUNICODE + struct.pack("<I", 3) + surrogate
        self.write_gzip(_crafted(_version_1(["crafted"]), text))
        self.assert_unreadable()
        _sample_document().save(self.path)
        with zipfile.ZipFile(self.path) as zf:
            data = zf.read("document.json")
        data = data.replace(b'"lateral"', b'"' + surrogate + b'"', 1)
        self.rewrite(replace={"document.json": data})
        self.assert_unreadable()

    def test_refuses_protocol_0_text_opcodes(self):
        # Pickle writes them from protocol 2 on only for GLOBAL, and documents
        # were protocol 3 or 4: an int of 4,300 digits is a line that gzips to a
        # few bytes and took 60 microseconds to read.
        streams = {
            "INT": pickle.INT + b"4" * 4300 + b"\n",
            "LONG": pickle.LONG + b"4" * 4300 + b"L\n",
            "FLOAT": pickle.FLOAT + b"1.5\n",
            "STRING": pickle.STRING + b"'x'\n",
            "UNICODE": pickle.UNICODE + b"x\n",
            "PUT": pickle.NONE + pickle.PUT + b"0\n",
            "GET": pickle.NONE + pickle.BINPUT + b"\x00" + pickle.GET + b"0\n",
            "INST": pickle.MARK + pickle.INST + b"builtins\nset\n",
            "OBJ": pickle.MARK + pickle.GLOBAL + b"builtins\nset\n" + pickle.OBJ,
            "a long GLOBAL": pickle.GLOBAL + b"x" * 300 + b"\nset\n",
        }
        for name, ops in streams.items():
            with self.subTest(name):
                self.write_gzip(_crafted(_version_1(["crafted"]), ops))
                self.assert_unreadable()
        with self.subTest("version 2"):
            self.save_frames_json("4" * 40)
            self.assert_unreadable()
            self.save_frames_json("4" * 39)
            self.assertEqual(self.read_frames(), [int("4" * 39)])

    def test_refuses_more_frames_than_an_import_keeps(self):
        # What the app does for each frame of a document it opens made millions
        # of them, a few kilobytes of pickle, gigabytes.
        for count, opens in ((5000, True), (5001, False)):
            with self.subTest(count=count):
                # One frame over and over, which a pickle holds once; a version
                # 2 file stores each frame's image apart.
                self.write_gzip(pickle.dumps(_version_1([_frame()] * count)))
                if opens:
                    self.assertEqual(len(self.load().frames), count)
                else:
                    self.assert_unreadable()
                image = np.zeros((1, 1, 3), dtype=np.uint8)
                frames = [_frame(_FRONTAL, image.copy()) for _ in range(count)]
                self.write(_document(frames), 2)
                if opens:
                    self.assertEqual(len(self.load().frames), count)
                else:
                    self.assert_unreadable()

    def test_refuses_a_profile_larger_than_a_crop(self):
        # add_missing_views() gives each frame saved without a view the profile,
        # and saving the document again stores it once a frame.
        frames = [_frame()] * 10
        for size, opens in ((1024, True), (9000, False)):
            with self.subTest(size=size):
                doc = _profile(
                    lateral_landmarks=np.arange(12).reshape(6, 2),
                    sagittal_x=np.zeros(size),
                    sagittal_y=np.zeros(size, dtype=np.int32),
                )
                doc[DOC_BODY][DOC_BODY_FRAMES] = frames
                self.write_gzip(pickle.dumps(doc))
                if opens:
                    self.assertTrue(self.load().lateral)
                else:
                    self.assert_unreadable()

    def test_frames_given_a_view_share_its_arrays(self):
        # load_state() copies a view's arrays, so the frames share one copy: one
        # each, for 5,000 frames of a crafted profile, was gigabytes.
        doc = _sample_document()
        doc.frames = doc.frames * 3
        doc.save(self.path)
        loaded = self.load()
        loaded.add_missing_views()
        views = [frame[6] for frame in loaded.frames]
        self.assertIs(views[0]["sagittal_x"], views[2]["sagittal_x"])
        self.assertIsNot(views[0], views[2])
        face = facial.AnalyzeFace(measures=[AnalyzeLateral()])
        face.load_state(loaded.frames[0])
        self.assertIsNot(face.sagittal_x, views[0]["sagittal_x"])

    def test_refuses_floods_of_containers_within_their_memory(self):
        # An empty set is 216 bytes, and a file of EMPTY_SET opcodes gzips 1000
        # to 1: eight million of them took 2 GB before they were refused. Each
        # container takes steps for its memory, about 32 bytes a step; checked
        # here on a budget of a tenth.
        floods = {
            "sets": pickle.EMPTY_SET,
            "dicts": pickle.EMPTY_DICT,
            "lists": pickle.EMPTY_LIST,
            "marks": pickle.MARK,
            "tuples": pickle.NONE + pickle.TUPLE1,
        }
        steps = 800_000
        for name, op in floods.items():
            with self.subTest(name):
                self.write_gzip(b"\x80\x04" + op * 2_000_000 + pickle.STOP)
                tracemalloc.start()
                try:
                    with mock.patch.object(dynaface_document, "_BASE_STEPS", steps):
                        self.assert_unreadable()
                    peak = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 48 * steps)
        with self.subTest("version 2 strings"):
            self.save_frames_json("[" + '"ab",' * 2_000_000 + '"ab"]')
            tracemalloc.start()
            try:
                with mock.patch.object(dynaface_document, "_BASE_STEPS", steps):
                    self.assert_unreadable()
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            self.assertLess(peak, 48 * steps + (16 << 20))

    def test_refuses_costly_opcodes_quickly(self):
        # Each call of numpy's _frombuffer() with 64 dimensions and an axis
        # order takes about 5 microseconds, and took a step: 233 KB of them took
        # five seconds to refuse.
        args = (b"\x00", np.dtype("u1"), (1,) * 64, "K", tuple(range(64)))
        function = struct.pack("<I", (1 << 30) + 1)
        # The arguments and the function memoized, then called again and again.
        stream = (
            pickle.dumps(args, protocol=3)[:-1]
            + _PUT
            + pickle.POP
            + pickle.GLOBAL
            + b"numpy._core.numeric\n_frombuffer\n"
            + pickle.LONG_BINPUT
            + function
            + pickle.POP
        )
        again = pickle.LONG_BINGET + function + _GET + pickle.REDUCE + pickle.POP
        self.write_gzip(stream + again * 1_000_000 + pickle.NONE + pickle.STOP)
        start = time.perf_counter()
        self.assert_unreadable()
        self.assertLess(time.perf_counter() - start, 3)

    def test_refuses_empty_arrays_of_shapes_numpy_refuses(self):
        # numpy's setstate takes them; copying one fails.
        placeholder = (np.ndarray, (0,), b"b")
        state = (1, (0, 2**62, 4), np.dtype("u1"), False, b"")
        self.assert_refused(_Reduce(_np_reconstruct, placeholder, state))

    def test_a_failed_load_keeps_nothing_it_read(self):
        # 64 MB of data, then an opcode pickle doesn't have: the error refers to
        # the unpickler, which kept the data alive through its stack.
        size = 64 << 20
        with gzip.open(self.path, "wb") as f:
            f.write(b"\x80\x04" + pickle.BINBYTES8 + struct.pack("<Q", size))
            for _ in range(size >> 20):
                f.write(bytes(1 << 20))
            f.write(b"\xff")
        tracemalloc.start()
        try:
            with self.assertRaises(DocumentError) as raised:
                self.load()
            held = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
        self.assertIsNotNone(raised.exception.__cause__)
        self.assertLess(held, 16 << 20)

    def test_a_version_1_document_saved_again_opens_again(self):
        # Its values nest as deeply in the version 2 file, counted the same way
        # though they take more levels of JSON there: here, in a key of a frame's
        # view that Dynaface doesn't read, as a later version might add one,
        # which is kept.
        value = None
        for level in range(40):
            value = {level: value}
        frame = _frame(dict(_FRONTAL, later=value))
        self.write_gzip(pickle.dumps(_version_1([frame])))
        loaded = self.load()
        loaded.save(self.path)
        self.assert_same(value, self.load().frames[0][6]["later"])

    # Documents that open, holding values of the wrong type for what the app then
    # does with them. Each is refused as it opens, as a file that isn't a
    # document, in either kind of file.

    def assert_refused_as_no_document(self):
        with self.assertRaisesRegex(DocumentError, "not appear to be a valid Dynaface"):
            self.load()

    def test_refuses_values_of_types_no_document_holds(self):
        for version in (1, 2):
            # unchanged, the document opens
            self.write(_document([_frame(dict(_FRONTAL)) for _ in range(3)]), version)
            self.assertEqual(len(self.load().frames), 3)
        cases = [(name, change, (1, 2)) for name, change in _WRONG.items()]
        cases += [(name, change, (1,)) for name, change in _WRONG_IN_VERSION_1.items()]
        for name, change, versions in cases:
            for version in versions:
                with self.subTest(name, version=version):
                    doc = _document([_frame(dict(_FRONTAL)) for _ in range(3)])
                    change(doc)
                    self.write(doc, version)
                    self.assert_refused_as_no_document()

    def test_opens_every_kind_of_frame_documents_hold(self):
        # What AnalyzeFace.dump_state() writes now and what each earlier version
        # wrote, which restore as they did.
        photo = np.zeros((30, 40, 3), dtype=np.uint8)
        profile = dict(
            _PROFILE_VIEW,
            flipped=True,
            sagittal_x=np.arange(5, dtype=np.int32) + 300,
            sagittal_y=np.arange(5, dtype=np.int32),
        )
        outline_with_a_gap = dict(
            _QUARTER_VIEW, cheek_outline_x=np.array([12.5, math.nan, 13.5])
        )
        unknown = _frame(dict(_FRONTAL))
        # Saved again from a frame saved without a view, and with the pupils in
        # one place, which left no scale.
        unknown[1], unknown[4] = [math.nan] * 3, math.inf
        six = _frame(dict(_FRONTAL))
        # The model's whole head pose, a face the crop straightened, landmarks
        # past the crop's edge, and a profile's fixed scale.
        six[1], six[5] = [1.5, -2.0, 0.25, 10.0, -4.0, 900.0], 0.12
        six[2] = [(x - 300, y + 1100) for x, y in _LANDMARKS]
        cases = {
            "Dynaface 2.1": [_frame(dict(_FRONTAL)) for _ in range(2)],
            "Dynaface 1.2 to 2.0.4": [_frame() for _ in range(2)],
            "December 2023, before the face rotation": [_frame()[:5]],
            "a photo without a face, at its own size": [
                [photo, [0, 0, 0], [], 0, 1, None]
            ],
            "a profile": [_frame(profile)],
            "a three-quarter face": [_frame(outline_with_a_gap)],
            "a three-quarter face saved before its outline was kept": [
                _frame({"pose": "quarter", "flipped": False, "lateral": False})
            ],
            "a head pose and a scale not known": [unknown],
            "more of the head pose": [six],
            "a view with keys Dynaface doesn't read": [
                _frame(dict(_FRONTAL, later=[1, "two"]))
            ],
        }
        numpy_scalars = [
            np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
            np.array([1.5, -2.0, 0.25]),
            [(np.int64(x), np.int64(y)) for x, y in _LANDMARKS],
            np.float64(20.0),
            np.float64(3.15),
            np.float64(0.1),
        ]
        in_version_1 = {"numpy scalars": [numpy_scalars]}
        for versions, kinds in (((1, 2), cases), ((1,), in_version_1)):
            for name, frames in kinds.items():
                for version in versions:
                    with self.subTest(name, version=version):
                        self.write(_document(frames, fps=29.97), version)
                        loaded = self.load()
                        self.assert_same(frames, loaded.frames)
                        self.assertEqual(loaded.fps, 29.97)
                        for state in loaded.frames:
                            facial.AnalyzeFace([]).load_state(state)

    def test_frames_saved_before_the_rotation_get_a_view_where_it_is_read(self):
        # add_missing_views() gave a profile's frame of five entries its view in
        # the face rotation's place, where load_state() took it for one. A
        # profile saved without its lateral landmarks gets none, as one whose
        # outline couldn't be traced, and the lateral measure skips it.
        doc = _document([_frame()[:5] for _ in range(2)], lateral=True, pose="lateral")
        self.write(doc, 1)
        loaded = self.load()
        loaded.add_missing_views()
        for state in loaded.frames:
            self.assertEqual(len(state), 7)
            self.assertIsNone(state[5])
            self.assertEqual(state[6]["lateral_landmarks"].shape, (0, 2))
        face = facial.AnalyzeFace(measures=[AnalyzeLateral()])
        face.load_state(loaded.frames[0])
        self.assertTrue(face.lateral)
        self.assertIsNone(face.face_rotation)
        AnalyzeLateral().calc(face)

    def test_the_tab_says_a_crafted_document_is_not_one(self):
        # The tab is never shown: the window says why, and Dynaface goes on.
        doc = _document([_frame(dict(_FRONTAL)) for _ in range(2)])
        doc[DOC_BODY][DOC_BODY_FRAMES][1][0] = "image"
        self.write(doc, 2)
        tab = SimpleNamespace()
        app = SimpleNamespace(tilt_threshold=-1)
        with mock.patch(
            "tab_analyze_video.QApplication", **{"instance.return_value": app}
        ), mock.patch.object(
            dlg_modal, "display_please_wait", side_effect=_run_and_wait
        ):
            self.assertFalse(AnalyzeVideoTab.load_document(tab, self.path))
        self.assertIn(
            "does not appear to be a valid Dynaface document", tab.load_error_message
        )


if __name__ == "__main__":
    unittest.main()
