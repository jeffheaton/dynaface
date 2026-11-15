import base64
import copy
import gzip
import io
import itertools
import json
import math
import pickle
import re
import struct
import weakref
import zipfile
from typing import Any, BinaryIO, Callable, Dict, Tuple

import numpy as np
import utl_save
from dynaface.const import Pose
from jth_ui import utl_classes
from utl_general import assert_standard_python

import dynaface.measures

DOC_HEADER = "header"
DOC_HEADER_VERSION = "version"
DOC_HEADER_FPS = "fps"
DOC_HEADER_VIEW = "view"
DOC_HEADER_LATERAL = "lateral"
DOC_HEADER_POSE = "pose"
DOC_BODY = "body"
DOC_BODY_FACE = "face"
DOC_BODY_MEASURES = "measures"
DOC_BODY_MEASURE = "measure"
DOC_BODY_MEASURE_ITEMS = "items"
DOC_BODY_FRAMES = "frames"
DOC_BODY_FRAME_NUMBERS = "frame_numbers"
DOC_BODY_LATERAL_LANDMARKS = "lateral_landmarks"
DOC_BODY_SAGITTAL_X = "sagittal_x"
DOC_BODY_SAGITTAL_Y = "sagittal_y"

DOC_NAME = "name"
DOC_ENABLED = "enabled"

# Version 1 documents were gzip-compressed pickles, and opening a pickle can run
# code stored in it. Version 2 documents are zip archives that hold the document as
# JSON, plus one .npy entry per numpy array, so opening one never runs code.
DOC_FORMAT_VERSION = 2
_DOC_JSON = "document.json"
_ARRAY_PREFIX = "arrays/"
# Marks a JSON object that stands for a value JSON can't express directly.
_TYPE_KEY = "$type"


class _RefusedGlobalError(pickle.UnpicklingError):
    pass


class _UncheckedDataError(pickle.UnpicklingError):
    pass


class _ArrayClass:
    """Stands for numpy.ndarray in a version 1 file, where it is only ever the
    class _reconstruct() is told to build: the real class could be called with a
    buffer and strides from the file."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise _UncheckedDataError("an array made by calling numpy.ndarray")


_ARRAY_CLASS = _ArrayClass()

# A dtype as numpy pickles one: a type code such as "f8", "u1", "U5" or "O8". A
# structured or subarray dtype ("i4,O", "(2,3)f8") is refused: numpy pickles
# those as void with fields in the state, which _LegacyUnpickler never trusts.
_DTYPE_CODE = re.compile(r"[A-Za-z][0-9]*")


def _brief(value: Any, depth: int = 2) -> str:
    """
    A value from a file in a few words, for a refusal to quote: containers are cut
    to a few items and levels before anything is formatted, since one that holds
    itself many times over (a list of the same list twice, 64 levels deep, is a few
    hundred bytes of pickle) would format to more text than memory holds.
    """
    kind = type(value)
    if value is None or kind in (bool, float, complex):
        return repr(value)
    if kind is int:
        bits = value.bit_length()
        return repr(value) if bits <= 64 else f"an int of {bits} bits"
    if kind in (str, bytes, bytearray):
        text = repr(value[:40])
        return text if len(value) <= 40 else f"{text}…"
    if isinstance(value, np.dtype):
        return f"dtype({value.str!r})"
    if kind is np.ndarray:
        return f"an array of {_brief(value.shape)} {value.dtype.str}"
    if kind not in (tuple, list, dict, set, frozenset):
        return f"a {kind.__name__}"
    if not value:
        return repr(value)
    if depth == 0:
        return f"a {kind.__name__} of {len(value)}"
    if kind is dict:
        items = itertools.islice(value.items(), 4)
        parts = [f"{_brief(k, depth - 1)}: {_brief(v, depth - 1)}" for k, v in items]
    else:
        parts = [_brief(item, depth - 1) for item in itertools.islice(value, 4)]
    if len(value) > 4:
        parts.append("…")
    inner = ", ".join(parts)
    if kind is tuple:
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if kind is list:
        return f"[{inner}]"
    if kind is frozenset:
        return f"frozenset({{{inner}}})"
    return f"{{{inner}}}"


def _check_shape(shape: Any) -> int:
    """Return the number of elements in shape, or raise unless it is a tuple of
    sizes within numpy's limits (64 dimensions, each smaller than 2**63)."""
    if (
        type(shape) is not tuple
        or len(shape) > 64
        or not all(type(size) is int and 0 <= size < 2**63 for size in shape)
    ):
        raise _UncheckedDataError(f"an array shape of {_brief(shape)}")
    return math.prod(shape)


def _check_raw(data: Any, dtype: Any, count: int) -> None:
    """Raise unless data holds exactly count elements of a dtype with no Python
    objects in it."""
    if not isinstance(dtype, np.dtype) or dtype.hasobject:
        raise _UncheckedDataError(f"raw bytes for an array of {_brief(dtype)}")
    if type(data) not in (bytes, bytearray):
        raise _UncheckedDataError(f"array data of type {type(data).__name__}")
    if dtype.itemsize == 0 and count:
        # No bytes stand for any number of elements of no size, and copying an
        # array of 2**63 of them never ends.
        raise _UncheckedDataError(f"{count} {dtype} elements of no size")
    if len(data) != count * dtype.itemsize:
        raise _UncheckedDataError(f"{count} {dtype} elements in {len(data)} bytes")


def _dtype(code: Any, align: Any = False, copy: Any = False) -> np.dtype:
    """numpy.dtype() for the type codes numpy pickles, always a new dtype, so the
    BUILD that follows can never reach a dtype numpy shares (dtype("f8") is one
    object in the whole process)."""
    if (
        type(code) is not str
        or not _DTYPE_CODE.fullmatch(code)
        or type(align) is not bool
        or type(copy) is not bool
    ):
        raise _UncheckedDataError(f"a dtype of {_brief(code)}")
    return np.dtype(code, align, True)


def _reconstruct(cls: Any, shape: Any, typecode: Any) -> np.ndarray:
    """numpy's placeholder for an array that a BUILD then fills in."""
    if cls is not _ARRAY_CLASS:
        raise _UncheckedDataError(f"an array of class {_brief(cls)}")
    return np.ndarray((0,), dtype=np.int8)


def _frombuffer(
    data: Any, dtype: Any, shape: Any, order: Any, axis_order: Any = None
) -> np.ndarray:
    """An array pickled with protocol 5, as numpy's own _frombuffer builds it."""
    _check_raw(data, dtype, _check_shape(shape))
    array = np.frombuffer(data, dtype=dtype)
    if order == "K" and axis_order is not None:
        # The axes' types first: comparing what the file put there could take
        # any amount of time.
        if (
            type(axis_order) is not tuple
            or len(axis_order) != len(shape)
            or not all(type(axis) is int for axis in axis_order)
            or sorted(axis_order) != list(range(len(shape)))
        ):
            raise _UncheckedDataError(f"an axis order of {_brief(axis_order)}")
        return array.reshape(shape, order="C").transpose(axis_order)
    if order not in ("C", "F"):
        raise _UncheckedDataError(f"an array order of {_brief(order)}")
    return array.reshape(shape, order=order)


def _scalar(dtype: Any, data: Any) -> np.generic:
    """A numpy scalar, such as numpy.float64(0.25)."""
    _check_raw(data, dtype, 1)
    return np.frombuffer(data, dtype=dtype)[0]


def _check_array_state(state: Any) -> None:
    """Raise unless state is the state numpy pickles an array with: (version,
    shape, dtype, Fortran order, data), the data being raw bytes for exactly that
    many elements, or for an object array, a list of them."""
    if type(state) is not tuple or len(state) != 5:
        raise _UncheckedDataError("an array state that numpy doesn't write")
    version, shape, dtype, fortran, data = state
    if version != 1 or type(fortran) is not bool:
        raise _UncheckedDataError("an array state that numpy doesn't write")
    count = _check_shape(shape)
    if isinstance(dtype, np.dtype) and dtype.hasobject:
        # Only a plain object array: its elements are the file's own values,
        # which this unpickler made.
        if dtype.kind != "O" or type(data) is not list or len(data) != count:
            raise _UncheckedDataError(f"an object array of {_brief(dtype)}")
    else:
        _check_raw(data, dtype, count)


# The only globals a version 1 pickle needs: numpy array and scalar reconstruction,
# under both the numpy 1 and numpy 2 module names, and the builtins that pickle
# can't write with its own opcodes. Each is resolved to the function or class
# here, never to whatever the file names, and anything else is refused before it
# can run.
_LEGACY_GLOBALS: Dict[Tuple[str, str], Any] = {
    ("builtins", "bytearray"): bytearray,
    ("builtins", "complex"): complex,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "set"): set,
    ("numpy", "dtype"): _dtype,
    ("numpy", "ndarray"): _ARRAY_CLASS,
}
for _numpy_core in ("numpy.core", "numpy._core"):
    _LEGACY_GLOBALS[(f"{_numpy_core}.multiarray", "_reconstruct")] = _reconstruct
    _LEGACY_GLOBALS[(f"{_numpy_core}.multiarray", "scalar")] = _scalar
    _LEGACY_GLOBALS[(f"{_numpy_core}.numeric", "_frombuffer")] = _frombuffer


class _LegacyUnpickler(pickle._Unpickler):
    """Loads version 1 documents without running anything they reference.

    Checking the names a file refers to isn't enough on its own: pickle's BUILD
    hands a dtype or array the state stored in the file, and numpy trusts the flags
    in a dtype's state. A dtype that wrongly says it holds no Python objects turns
    raw bytes from the file into object pointers, which a crafted file can use to
    crash Dynaface or run code. So this unpickler, written in Python rather than
    pickle's C one so that BUILD can be replaced, accepts a dtype's state only when
    it is exactly the state numpy gave the dtype it built, and an array's only when
    its data matches its shape and dtype; it sets the state of nothing else.

    An array takes a state only once, and only the empty placeholder that
    _reconstruct() made: a state frees the data an array had, and a view of that
    data would go on reading the freed memory. Pickle's out-of-band buffer
    opcodes, which can make such a view, are refused; pickle writes them only for
    a caller that asks to keep buffers out of the pickle, which Dynaface never
    did.
    """

    dispatch = dict(pickle._Unpickler.dispatch)

    def __init__(self, file: BinaryIO) -> None:
        super().__init__(file)
        # The placeholders waiting for their state, by id. An entry goes with
        # its placeholder, before another object can take the id, so a file
        # that drops placeholders doesn't keep them all alive.
        self._unbuilt: "weakref.WeakValueDictionary[int, np.ndarray]" = (
            weakref.WeakValueDictionary()
        )

    def load(self) -> Any:
        try:
            return super().load()
        finally:
            # The memo holds what find_class() handed out, this unpickler's own
            # _reconstruct among it, and so this unpickler: cleared, the memo
            # and every array in it go when the document does, not when the
            # garbage collector next runs. A failed load's partial values go too.
            self.memo.clear()
            self.stack = []
            self.metastack = []

    def find_class(self, module: str, name: str) -> Any:
        try:
            found = _LEGACY_GLOBALS[(module, name)]
        except KeyError:
            raise _RefusedGlobalError(f"{module}.{name}") from None
        return self._reconstruct if found is _reconstruct else found

    def _reconstruct(self, cls: Any, shape: Any, typecode: Any) -> np.ndarray:
        array = _reconstruct(cls, shape, typecode)
        self._unbuilt[id(array)] = array
        return array

    def load_build(self) -> None:
        state = self.stack.pop()
        target = self.stack[-1]
        if type(target) is np.ndarray:
            if self._unbuilt.pop(id(target), None) is not target:
                raise _UncheckedDataError(
                    "a state for an array that already has its data"
                )
            _check_array_state(state)
            target.__setstate__(state)
        elif isinstance(target, np.dtype):
            built = target.__reduce__()[2]
            # The state numpy built a type code's dtype with holds only None,
            # numbers and strings; compare their types too, so nothing in the
            # file decides what the comparison runs.
            same = (
                type(state) is tuple
                and len(state) == len(built)
                and all(type(a) is type(b) and a == b for a, b in zip(state, built))
            )
            if not same:
                raise _UncheckedDataError(
                    f"{_brief(target)} with a state numpy didn't give it"
                )
            # Equal to the dtype's own state, so there is nothing to set.
        else:
            raise _UncheckedDataError(f"a state for a {type(target).__name__}")

    def load_buffer(self) -> None:
        raise _UncheckedDataError("an out-of-band buffer")

    def load_bytearray8(self) -> None:
        # Pickle's own makes a bytearray of the length the file gives and zeroes
        # it before reading any data, so a damaged length (one flipped bit can
        # make terabytes) filled memory; read the data first, as pickle's C
        # unpickler does, and refuse it when the file ends sooner.
        (size,) = struct.unpack("<Q", self.read(8))
        data = bytearray()
        while len(data) < size:
            wanted = min(size - len(data), 1 << 20)
            chunk = self.read(wanted)
            if len(chunk) < wanted:
                raise pickle.UnpicklingError("pickle data was truncated")
            data += chunk
        self.append(data)

    dispatch[pickle.BUILD[0]] = load_build
    dispatch[pickle.NEXT_BUFFER[0]] = load_buffer
    dispatch[pickle.READONLY_BUFFER[0]] = load_buffer
    dispatch[pickle.BYTEARRAY8[0]] = load_bytearray8


def _encode(obj: Any, store_array: Callable[[np.ndarray], str]) -> Any:
    """Converts a document value to JSON, storing numpy arrays through store_array."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [_encode(item, store_array) for item in obj]
    if isinstance(obj, dict):
        if all(isinstance(key, str) for key in obj) and _TYPE_KEY not in obj:
            return {key: _encode(value, store_array) for key, value in obj.items()}
        pairs = [
            [_encode(key, store_array), _encode(value, store_array)]
            for key, value in obj.items()
        ]
        return {_TYPE_KEY: "dict", "items": pairs}
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            # np.save can't write these without pickle, so encode each element.
            items = [_encode(item, store_array) for item in obj.flat]
            return {_TYPE_KEY: "ndarray", "shape": list(obj.shape), "items": items}
        return {_TYPE_KEY: "ndarray", "file": store_array(obj)}
    if isinstance(obj, tuple):
        kind = "tuple"
    elif isinstance(obj, frozenset):
        kind = "frozenset"
    elif isinstance(obj, set):
        kind = "set"
    elif isinstance(obj, complex):
        return {_TYPE_KEY: "complex", "real": obj.real, "imag": obj.imag}
    elif isinstance(obj, (bytes, bytearray)):
        kind = "bytearray" if isinstance(obj, bytearray) else "bytes"
        return {_TYPE_KEY: kind, "base64": base64.b64encode(obj).decode("ascii")}
    else:
        raise TypeError(f"Can't save a value of type {type(obj).__name__}.")
    return {_TYPE_KEY: kind, "items": [_encode(item, store_array) for item in obj]}


def _decode(obj: Any, read_array: Callable[[str], np.ndarray]) -> Any:
    """Rebuilds a document value from the JSON that _encode wrote."""
    if isinstance(obj, list):
        return [_decode(item, read_array) for item in obj]
    if not isinstance(obj, dict):
        return obj
    kind = obj.get(_TYPE_KEY)
    if kind is None:
        return {key: _decode(value, read_array) for key, value in obj.items()}
    if kind == "complex":
        return complex(obj["real"], obj["imag"])
    if kind == "bytes":
        return base64.b64decode(obj["base64"])
    if kind == "bytearray":
        return bytearray(base64.b64decode(obj["base64"]))
    if kind == "ndarray" and "file" in obj:
        return read_array(obj["file"])

    items = [_decode(item, read_array) for item in obj["items"]]
    if kind == "ndarray":
        array = np.empty(len(items), dtype=object)
        for i, item in enumerate(items):
            array[i] = item
        return array.reshape(obj["shape"])
    if kind == "tuple":
        return tuple(items)
    if kind == "set":
        return set(items)
    if kind == "frozenset":
        return frozenset(items)
    if kind == "dict":
        return dict(items)
    raise ValueError(f"Unknown value type {kind!r}.")


def _read_zip(filename: str) -> Any:
    with zipfile.ZipFile(filename) as zf:

        def read_array(name: str) -> np.ndarray:
            with zf.open(name) as f:
                array = np.load(io.BytesIO(f.read()), allow_pickle=False)
            if not isinstance(array, np.ndarray):
                raise ValueError(f"{name} is not a numpy array.")
            return array

        return _decode(json.loads(zf.read(_DOC_JSON)), read_array)


class DocumentError(TypeError):
    """A file that can't be opened as a Dynaface document. The message says why,
    in words for the user."""


def _not_a_document(filename: str) -> DocumentError:
    return DocumentError(
        f"The file '{filename}' does not appear to be a valid Dynaface document."
    )


def _shorten(text: str, limit: int = 80) -> str:
    """Text a refusal quotes from the file, fit for a message: a crafted file can
    put any text there, at any length."""
    short = "".join(c if c.isprintable() else "?" for c in text[:limit])
    return short if len(text) <= limit else short[:-1] + "…"


def _read_document(filename: str) -> Dict[str, Any]:
    with open(filename, "rb") as f:
        magic = f.read(2)

    try:
        if magic == b"PK":
            doc = _read_zip(filename)
        elif magic == b"\x1f\x8b":
            with gzip.open(filename, "rb") as f:
                doc = _LegacyUnpickler(f).load()
        else:
            doc = None
    except _RefusedGlobalError as e:
        raise DocumentError(
            f"The file '{filename}' was not opened because it contains code "
            f"({_shorten(str(e))}) that would run when it's loaded."
        ) from e
    except _UncheckedDataError as e:
        raise DocumentError(
            f"The file '{filename}' was not opened because it contains data "
            f"({_shorten(str(e))}) that Dynaface documents never hold, which could "
            "run code when it's loaded."
        ) from e
    except Exception as e:
        raise _not_a_document(filename) from e

    if not isinstance(doc, dict) or not isinstance(doc.get(DOC_HEADER), dict):
        raise _not_a_document(filename)
    version = doc[DOC_HEADER].get(DOC_HEADER_VERSION, 1)
    if not isinstance(version, int):
        raise _not_a_document(filename)
    if version > DOC_FORMAT_VERSION:
        raise DocumentError(
            f"The file '{filename}' was saved by a newer version of Dynaface."
        )
    return doc


def _frame_numbers(value: Any, count: int) -> Any:
    """
    The frame numbers a document holds: each frame's number in the video, a whole
    number from 0, rising, one per frame (count). None when it holds none, as a
    document saved before they were kept, or holds anything else, so its frames
    are numbered by position.
    """
    if (
        isinstance(value, list)
        and len(value) == count
        and all(type(n) is int and 0 <= n < 2**31 for n in value)
        and all(a < b for a, b in zip(value, value[1:]))
    ):
        return value
    return None


def _with_view(frame: Any, view: Dict[str, Any]) -> list:
    """
    A frame state saved without a view given this one, with an unknown head pose
    (NaN). Such a state holds a [0, 0, 0] placeholder where states with a view
    hold the pose the face was analyzed with, and load_state() takes a state's
    pose as real once it has a view, so the placeholder would come back as a
    level head.
    """
    state = list(frame)
    if len(state) > 1:
        state[1] = [math.nan] * 3
    return state + [copy.deepcopy(view)]


class DynafaceDocument:
    def __init__(self):
        self._version = DOC_FORMAT_VERSION
        self.face = None
        self.frames = []
        # Each frame's number in the video, counting from 0, as the import kept
        # it (it skips frames without a face), or None to number the frames by
        # position, as documents saved before the numbers were kept do.
        self.frame_numbers = None
        self.measures = []
        self.fps = 0
        self.lateral = False
        self.lateral_landmarks = None
        self.sagittal_x = None
        self.sagittal_y = None
        self.pose = None

    def save(self, filename: str):
        measures = self._save_measures(self.measures)
        # A document whose pose was never set (one loaded from a version 1 file,
        # or a new one) is frontal, so a re-save never fails on a missing pose.
        pose = Pose.FRONTAL if self.pose is None else self.pose

        doc = {
            DOC_HEADER: {
                DOC_HEADER_VERSION: self._version,
                DOC_HEADER_FPS: self.fps,
                DOC_HEADER_LATERAL: self.lateral,
                DOC_HEADER_POSE: pose.value,
            },
            DOC_BODY: {
                DOC_BODY_MEASURES: measures,
                DOC_BODY_FRAMES: self.frames,
                DOC_BODY_LATERAL_LANDMARKS: self.lateral_landmarks,
                DOC_BODY_SAGITTAL_X: self.sagittal_x,
                DOC_BODY_SAGITTAL_Y: self.sagittal_y,
            },
        }
        if self.frame_numbers is not None:
            doc[DOC_BODY][DOC_BODY_FRAME_NUMBERS] = [int(n) for n in self.frame_numbers]
        assert_standard_python(doc)
        # A save that fails part way (a full disk, say) leaves the document as it
        # was (see save_file()).
        utl_save.save_file(filename, lambda file: self._write_zip(file, doc))

    @staticmethod
    def _write_zip(file: BinaryIO, doc: Dict[str, Any]) -> None:
        with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            counter = itertools.count()

            def store_array(array: np.ndarray) -> str:
                name = f"{_ARRAY_PREFIX}{next(counter)}.npy"
                with zf.open(name, "w") as f:
                    np.save(f, array, allow_pickle=False)
                return name

            zf.writestr(_DOC_JSON, json.dumps(_encode(doc, store_array)))

    def load(self, filename: str):
        """
        Open a document. Raises DocumentError, whose message says why, when the
        file isn't a Dynaface document that can be opened, and OSError when it
        can't be read.
        """
        doc = _read_document(filename)
        try:
            self._load_contents(doc)
        except Exception as e:
            raise _not_a_document(filename) from e

    def _load_contents(self, doc: Dict[str, Any]) -> None:
        measures = self._load_measures(doc[DOC_BODY][DOC_BODY_MEASURES])
        self._add_missing_measures(measures, dynaface.measures.all_measures())

        self.fps = doc[DOC_HEADER][DOC_HEADER_FPS]
        self.frames = doc[DOC_BODY][DOC_BODY_FRAMES]
        self.frame_numbers = _frame_numbers(
            doc[DOC_BODY].get(DOC_BODY_FRAME_NUMBERS), len(self.frames)
        )
        self.measures = measures
        if DOC_HEADER_LATERAL not in doc[DOC_HEADER]:
            # A document without a lateral key was saved before 1.2.0, when every
            # document was frontal; setting the pose keeps it ready to save again.
            self.lateral = False
            self.lateral_landmarks = None
            self.sagittal_x = None
            self.sagittal_y = None
            self.pose = Pose.FRONTAL
        else:
            # If the key is present, set it to the value in the document
            # This allows for backward compatibility with older documents
            # that do not have this key.
            self.lateral = doc[DOC_HEADER][DOC_HEADER_LATERAL]
            self.lateral_landmarks = doc[DOC_BODY][DOC_BODY_LATERAL_LANDMARKS]
            self.sagittal_x = doc[DOC_BODY][DOC_BODY_SAGITTAL_X]
            self.sagittal_y = doc[DOC_BODY][DOC_BODY_SAGITTAL_Y]
            if DOC_HEADER_POSE in doc[DOC_HEADER]:
                self.pose = Pose(doc[DOC_HEADER][DOC_HEADER_POSE])
            else:
                # Saved by 1.2.0, before the pose was stored: the lateral flag
                # says which of the two views it had.
                self.pose = Pose.LATERAL if self.lateral else Pose.FRONTAL

    def add_missing_views(self) -> None:
        """
        Give every frame state that holds no view (saved by Dynaface 2.0.4 or
        earlier, six entries) this document's view, as AnalyzeFace.dump_state()
        writes it, so whatever restores the frame with load_state() measures it in
        the document's view. A profile document kept its lateral landmarks and
        sagittal outline in the header alone, which only the face that is told
        about them restored; a frontal document's frames restore as frontal either
        way, so they are left as they are. Those versions kept no head pose, only
        a placeholder, so a state given a view gets an unknown one (NaN), as
        load_state() gives a state without a view.
        """
        if not self.lateral:
            return
        view: Dict[str, Any] = {
            "pose": (self.pose or Pose.LATERAL).value,
            "flipped": False,
            "lateral": True,
            "lateral_landmarks": np.array(self.lateral_landmarks),
        }
        for name in ("sagittal_x", "sagittal_y"):
            value = getattr(self, name)
            view[name] = None if value is None else np.array(value)
        self.frames = [
            _with_view(frame, view) if len(frame) <= 6 else frame
            for frame in self.frames
        ]

    def _load_measures(self, measures):
        # Only Dynaface's own measure classes can be created. The class names come
        # from the file, and importing whatever module a file names could run code.
        known = {
            utl_classes.get_class_full_name(measure): type(measure)
            for measure in dynaface.measures.all_measures()
        }
        result = []
        for measure in measures:
            name = measure[DOC_NAME]
            enabled = measure[DOC_ENABLED]
            source_items = measure[DOC_BODY_MEASURE_ITEMS]
            cls = known.get(name)
            # Make sure we have a class to handle this measure
            if cls:
                obj = cls()
                obj.enabled = enabled
                self._sync_items(source_items, obj)
                result.append(obj)
        return result

    def _sync_items(self, source_items, obj):
        """Update the disabled flag on the target items, based on the source"""

        for item in source_items:
            obj.set_item_enabled(item[DOC_NAME], item[DOC_ENABLED])

    def _save_measures(self, measures):
        result = []
        for measure in measures:
            items = []
            for item in measure.items:
                items.append({DOC_NAME: item.name, DOC_ENABLED: item.enabled})

            name = utl_classes.get_class_full_name(measure)
            measure_encoded = {
                DOC_NAME: name,
                DOC_ENABLED: measure.enabled,
                DOC_BODY_MEASURE_ITEMS: items,
            }
            result.append(measure_encoded)

        return result

    def _has_measure(self, doc_measures, measure):
        for m in doc_measures:
            if m.abbrev() == measure.abbrev():
                return True
        return False

    def _add_missing_measures(self, doc_measures, all_measures):

        for measure in all_measures:
            if not self._has_measure(doc_measures, measure):
                doc_measures.append(measure)
                measure.set_enabled(False)
