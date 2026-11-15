import base64
import gzip
import io
import itertools
import json
import math
import os
import pickle
import re
import struct
import weakref
import zipfile
from typing import Any, BinaryIO, Callable, Dict, List, Tuple

import numpy as np
import utl_save
from dynaface.const import Pose
from dynaface.video import MAX_FRAMES
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

# What reading a document may cost (see _Limits): a crafted file is refused within a
# second or so, instead of crashing Dynaface or keeping it busy for minutes. A
# genuine document stays far inside these: 5,000 frames (MAX_FRAMES, the most an
# import keeps) take 3.4 million steps in a version 1 file of 2.7 MB or more, or
# 5.6 million values in a version 2 file of 4 MB or more, and nest 8 levels deep.
_BASE_STEPS = 8_000_000
_KEY_STEPS = 10_000
_MAX_DEPTH = 64
# Object arrays held directly by object arrays. Freeing an object array frees what
# it holds in C, about 1.6 KB of stack an array, and the thread that loads a
# document has 512 KB; Python puts off freeing a container 50 deep, so 50 links of
# this many arrays free at once, well inside it.
_MAX_OBJECT_NESTING = 2
# The values in a profile's landmarks and outline (6 points, and a point per row
# of the 1024-pixel crop), which add_missing_views() gives each frame saved
# without a view, so a document saved again holds them once a frame.
_MAX_PROFILE_VALUES = 8192
# The longest line of a version 1 file's GLOBAL opcode, a module and a name.
_MAX_LINE = 256
# The most document.json may hold, in bytes a step.
_JSON_BYTES_A_STEP = 16
# The steps reading one array of a version 2 file takes (np.load() takes about
# 30 microseconds).
_ARRAY_STEPS = 256
# How much of a version 1 file is decompressed at once.
_READ_BUFFER = 1 << 20
# How many opcodes run between charges for them.
_CHUNK = 1 << 16

# Keys whose hashing and comparing a file can't make slow: str and bytes hashes are
# randomized, so keys can't be chosen to collide.
_QUICK_KEYS = frozenset({str, bytes, type(None), bool})
# The values that hold other values in a version 1 document.
_NESTED = frozenset({list, tuple, dict, set, frozenset, np.ndarray})
_POSES = frozenset(pose.value for pose in Pose)
# Maps digits to "0" and anything else to "-", to find long runs of digits.
_DIGITS = bytes(48 if 48 <= i <= 57 else 45 for i in range(256))


class _RefusedGlobalError(pickle.UnpicklingError):
    pass


class _UncheckedDataError(pickle.UnpicklingError):
    pass


class _TooComplexError(ValueError):
    """A file that would take more work to read than any document does, or nests its
    values deeper: one made to crash Dynaface or keep it busy. It is refused as a
    file that isn't a document."""


class _Limits:
    """
    The work reading one document may still take, so that no file can crash Dynaface
    or keep it busy for long.

    Each pickle opcode is a step, and so is each value an operation goes through (a
    call's arguments, an object array's elements, a set's members). Opcodes that
    make a container take two (a set seven), and the costliest calls more, so that
    the steps bound the memory (about 32 bytes a step) and the time (about 0.1
    microseconds) a file can take as well. A document may
    take _BASE_STEPS steps, plus one per byte of its file: a crafted file repeats
    cheap opcodes, which compress to almost nothing, and a 1 MB file of them unpacks
    to a billion, which took Dynaface minutes to read.

    Keys and set members are hashed and compared in C, where a crafted key can crash
    Dynaface (hashing a tuple nested 9,000 deep, a few hundred bytes of gzip,
    overflows the stack) or keep it busy for good (a tuple of a tuple twice, itself of
    a tuple twice, 100 levels deep, is 2**100 tuples to hash), and ints can be chosen
    to hash alike. So a key nests at most _MAX_DEPTH deep, and the keys a document
    holds, other than strings, bytes, None and bools, may have _KEY_STEPS values in
    them in all, counting a value each time it appears, and a long int by its size.
    """

    def __init__(self, file_size: int) -> None:
        self.allowed = _BASE_STEPS + file_size
        self.steps = self.allowed
        self._key_steps = _KEY_STEPS

    def charge(self, steps: int) -> None:
        self.steps -= steps
        if self.steps < 0:
            raise _TooComplexError("more steps than a document of its size takes")

    def check_key(self, key: Any) -> None:
        """Raise unless hashing key, a dict key or set member, is quick."""
        if type(key) in _QUICK_KEYS:
            return
        left = self._key_steps
        pending = [(key, 0)]
        while pending:
            value, depth = pending.pop()
            kind = type(value)
            if kind is int:
                left -= 1 + value.bit_length() // 64
            elif isinstance(value, np.generic):
                left -= 1 + value.nbytes // 64
            else:
                left -= 1
                if kind is tuple or kind is frozenset:
                    if len(value) > left or depth == _MAX_DEPTH:
                        left = -1
                    else:
                        pending.extend((member, depth + 1) for member in value)
            if left < 0:
                raise _TooComplexError("keys that take long to hash")
        self._key_steps = left


class _RawStream(io.RawIOBase):
    """A decompressed file as io.BufferedReader reads it, through readinto(). The
    unpickler reads every opcode from the BufferedReader, which checks that the
    stream it wraps isn't closed on each read: a gzip file's check is in Python,
    this one's in C."""

    def __init__(self, file: BinaryIO) -> None:
        self._file = file

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        return self._file.readinto(buffer)


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
        or len(code) > 32
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
    if isinstance(dtype, np.dtype):
        # numpy's own limit, which its setstate doesn't check for an array with
        # no elements: one of shape (0, 2**62, 4) loads, and copying it fails.
        sizes = math.prod(size for size in shape if size)
        if sizes * max(dtype.itemsize, 1) >= 2**63:
            raise _UncheckedDataError(f"an array shape of {_brief(shape)}")
    if isinstance(dtype, np.dtype) and dtype.hasobject:
        # Only a plain object array: its elements are the file's own values,
        # which this unpickler made.
        if dtype.kind != "O" or type(data) is not list or len(data) != count:
            raise _UncheckedDataError(f"an object array of {_brief(dtype)}")
    else:
        _check_raw(data, dtype, count)


def _check_tree(value: Any, limits: _Limits) -> None:
    """
    Raise unless value, walked as a tree, nests no deeper than _MAX_DEPTH and holds
    no more values than the steps left allow, at two values a step (counting a
    value each time it appears). Code that goes through a document, such as
    np.array() of a profile's landmarks or saving it again, makes two of a value
    that a pickle holds twice: a list of the same list twice, 64 levels deep, is a
    few hundred bytes of pickle, and 2**64 lists to that code.
    """
    left = 2 * limits.steps
    level = [value]
    for _ in range(_MAX_DEPTH):
        below: List[Any] = []
        for item in level:
            kind = type(item)
            if kind is dict:
                members: Any = [*item, *item.values()]
            elif kind is np.ndarray:
                if not item.dtype.hasobject:
                    continue
                members = list(item.flat)
            elif kind in _NESTED:
                members = item
            else:
                continue
            left -= len(members)
            if left < 0:
                raise _TooComplexError("more values than a document of its size")
            below.extend([member for member in members if type(member) in _NESTED])
        if not below:
            limits.charge((2 * limits.steps - left) // 2)
            return
        level = below
    raise _TooComplexError(f"values nested more than {_MAX_DEPTH} deep")


def _line_reader(readline: Callable[[int], bytes]) -> Callable[[], bytes]:
    """readline() for GLOBAL, whose module and name each end at a newline, which a
    file can leave out for any length."""

    def read_line() -> bytes:
        line = readline(_MAX_LINE)
        if len(line) == _MAX_LINE and not line.endswith(b"\n"):
            raise pickle.UnpicklingError("a pickle line too long to be a document's")
        return line

    return read_line


# The only globals a version 1 pickle needs: numpy array and scalar reconstruction,
# under both the numpy 1 and numpy 2 module names, and the builtins that pickle
# can't write with its own opcodes. Each is resolved to an attribute of
# _LegacyUnpickler, never to whatever the file names, and anything else is refused
# before it can run.
_LEGACY_GLOBALS: Dict[Tuple[str, str], str] = {
    ("builtins", "bytearray"): "_bytearray",
    ("builtins", "complex"): "_complex",
    ("builtins", "frozenset"): "_frozenset",
    ("builtins", "set"): "_set",
    ("numpy", "dtype"): "_dtype",
    ("numpy", "ndarray"): "_array_class",
}
for _numpy_core in ("numpy.core", "numpy._core"):
    _LEGACY_GLOBALS[(f"{_numpy_core}.multiarray", "_reconstruct")] = "_reconstruct"
    _LEGACY_GLOBALS[(f"{_numpy_core}.multiarray", "scalar")] = "_scalar"
    _LEGACY_GLOBALS[(f"{_numpy_core}.numeric", "_frombuffer")] = "_frombuffer"


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

    Nor can a file make it do more work than a document needs (_Limits): the
    builtins a file calls take only the arguments pickle gives them, keys must hash
    quickly, opcodes add to dicts, lists and sets only, object arrays nest at most
    _MAX_OBJECT_NESTING deep, strings decode strictly, and protocol 0's text
    opcodes, which no document holds, are refused.
    """

    dispatch = dict(pickle._Unpickler.dispatch)

    def __init__(self, file: BinaryIO, limits: _Limits) -> None:
        super().__init__(file)
        self._limits = limits
        # The placeholders waiting for their state, by id. An entry goes with
        # its placeholder, before another object can take the id, so a file
        # that drops placeholders doesn't keep them all alive.
        self._unbuilt: "weakref.WeakValueDictionary[int, np.ndarray]" = (
            weakref.WeakValueDictionary()
        )
        # The object arrays this unpickler built, by id, and how deeply each
        # nests object arrays (1 for one that holds none). An entry is checked
        # against its array, since the array may be gone and the id reused.
        self._nesting: Dict[int, Tuple["weakref.ref[np.ndarray]", int]] = {}

    def load(self) -> Any:
        # Opcodes are read straight from the file, a buffered stream, and frames
        # are read as the opcodes they hold (load_frame): pickle's own framing
        # reads each opcode through another Python call, twice the time.
        self.read = self._file_read
        self.readline = _line_reader(self._file_readline)
        self.metastack = []
        self.stack = []
        self.append = self.stack.append
        self.proto = 0
        read = self.read
        dispatch = self.dispatch
        charge = self._limits.charge
        try:
            while True:
                for _ in range(_CHUNK):
                    key = read(1)
                    if not key:
                        raise EOFError
                    dispatch[key[0]](self)
                charge(_CHUNK)
        except pickle._Stop as stop:
            return stop.value
        finally:
            # The memo holds what find_class() handed out, this unpickler's own
            # methods among it, and so this unpickler: cleared, the memo and
            # every array in it go when the document does, not when the garbage
            # collector next runs. A failed load's partial values go too.
            self.memo.clear()
            self.stack = []
            self.append = self.stack.append
            self.metastack = []

    def find_class(self, module: str, name: str) -> Any:
        self._limits.steps -= 2
        try:
            attribute = _LEGACY_GLOBALS[(module, name)]
        except KeyError:
            raise _RefusedGlobalError(f"{module}.{name}") from None
        return getattr(self, attribute)

    # What a version 1 file can call (see _LEGACY_GLOBALS), with the arguments
    # pickle gives each.

    _array_class = _ARRAY_CLASS

    def _dtype(self, code: Any, align: Any = False, copy: Any = False) -> np.dtype:
        self._limits.charge(4)
        return _dtype(code, align, copy)

    def _frombuffer(
        self, data: Any, dtype: Any, shape: Any, order: Any, axis_order: Any = None
    ) -> np.ndarray:
        # Checking a shape and an axis order goes through each of their entries.
        self._limits.charge(4 + (len(shape) if type(shape) is tuple else 0))
        if type(axis_order) is tuple:
            self._limits.charge(len(axis_order))
        return _frombuffer(data, dtype, shape, order, axis_order)

    def _reconstruct(self, cls: Any, shape: Any, typecode: Any) -> np.ndarray:
        self._limits.charge(4)
        array = _reconstruct(cls, shape, typecode)
        self._unbuilt[id(array)] = array
        return array

    def _scalar(self, dtype: Any, data: Any) -> np.generic:
        """A numpy scalar, such as numpy.float64(0.25)."""
        _check_raw(data, dtype, 1)
        self._limits.charge(4 + dtype.itemsize // 64)
        return np.frombuffer(data, dtype=dtype)[0]

    def _bytearray(self, data: Any = b"", encoding: Any = None) -> bytearray:
        # Pickle makes a bytearray from bytes, or (protocol 2) from a str and
        # "latin-1". Of an int, bytearray() makes that many zeros: terabytes.
        if type(data) is bytes and encoding is None:
            self._limits.charge(len(data) // 64)
            return bytearray(data)
        if type(data) is str and type(encoding) is str and encoding == "latin-1":
            self._limits.charge(len(data) // 64)
            return bytearray(data, encoding)
        raise pickle.UnpicklingError(f"a bytearray of {_brief(data)}")

    def _complex(self, real: Any = 0.0, imag: Any = 0.0) -> complex:
        # Of a str, complex() parses it, however long.
        if type(real) not in (int, float) or type(imag) not in (int, float):
            raise pickle.UnpicklingError(f"a complex number of {_brief(real)}")
        return complex(real, imag)

    def _set(self, items: Any = ()) -> set:
        return set(self._members(items))

    def _frozenset(self, items: Any = ()) -> frozenset:
        return frozenset(self._members(items))

    def _members(self, items: Any) -> Any:
        """Items pickle passes to set() or frozenset(): a list (protocol 2 and
        3), each checked to hash quickly."""
        if type(items) not in (list, tuple):
            raise pickle.UnpicklingError(f"set members in {_brief(items)}")
        self._limits.charge(len(items))
        for item in items:
            self._limits.check_key(item)
        return items

    # The opcodes that call, hash, nest or add to what is on the stack.

    def load_reduce(self) -> None:
        stack = self.stack
        args = stack.pop()
        # Unpacked for the call, arguments a file memoized once could be used
        # any number of times.
        if type(args) is not tuple:
            raise pickle.UnpicklingError(f"call arguments in {_brief(args)}")
        self._limits.charge(len(args))
        stack[-1] = stack[-1](*args)

    def load_newobj(self) -> None:
        # NEWOBJ makes an instance of a class, and find_class() returns none.
        raise pickle.UnpicklingError("an object made by NEWOBJ")

    def load_build(self) -> None:
        state = self.stack.pop()
        target = self.stack[-1]
        if type(target) is np.ndarray:
            if self._unbuilt.pop(id(target), None) is not target:
                raise _UncheckedDataError(
                    "a state for an array that already has its data"
                )
            _check_array_state(state)
            self._limits.charge(4 + len(state[1]))
            nesting = self._object_nesting(target, state)
            target.__setstate__(state)
            if nesting:
                self._nesting[id(target)] = (weakref.ref(target), nesting)
        elif isinstance(target, np.dtype):
            self._limits.charge(8)
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

    def _object_nesting(self, target: np.ndarray, state: Tuple) -> int:
        """
        How deeply the array a checked state makes nests object arrays: 0 for one
        of numbers, 1 for an object array that holds none, and so on. Raises past
        _MAX_OBJECT_NESTING, or for an array that would hold itself or an array
        still waiting for its state (and so might come to hold this one), which
        no pickle of a document does.
        """
        if not state[2].hasobject:
            return 0
        self._limits.charge(len(state[4]))
        inner = 0
        for item in state[4]:
            if type(item) is not np.ndarray:
                continue
            if item is target or self._unbuilt.get(id(item)) is item:
                raise _TooComplexError("an array that may come to hold itself")
            if item.dtype.hasobject:
                ref, nesting = self._nesting.get(id(item), (None, 0))
                if ref is None or ref() is not item:
                    raise _TooComplexError("an object array from elsewhere")
                inner = max(inner, nesting)
        if inner >= _MAX_OBJECT_NESTING:
            raise _TooComplexError(
                f"object arrays nested more than {_MAX_OBJECT_NESTING} deep"
            )
        return inner + 1

    def load_setitem(self) -> None:
        stack = self.stack
        value = stack.pop()
        key = stack.pop()
        self._dict(stack[-1])
        self._limits.check_key(key)
        stack[-1][key] = value

    def load_setitems(self) -> None:
        items = self.pop_mark()
        target = self._dict(self.stack[-1])
        check_key = self._limits.check_key
        for i in range(0, len(items), 2):
            check_key(items[i])
            target[items[i]] = items[i + 1]

    def load_dict(self) -> None:
        self._limits.steps -= 1
        items = self.pop_mark()
        target: Dict[Any, Any] = {}
        check_key = self._limits.check_key
        for i in range(0, len(items), 2):
            check_key(items[i])
            target[items[i]] = items[i + 1]
        self.append(target)

    def load_additems(self) -> None:
        items = self.pop_mark()
        target = self.stack[-1]
        if type(target) is not set:
            raise pickle.UnpicklingError(f"set members for {_brief(target)}")
        for item in items:
            self._limits.check_key(item)
        target.update(items)

    def load_frozenset(self) -> None:
        self._limits.steps -= 6
        items = self.pop_mark()
        for item in items:
            self._limits.check_key(item)
        self.append(frozenset(items))

    def load_append(self) -> None:
        stack = self.stack
        value = stack.pop()
        self._list(stack[-1]).append(value)

    def load_appends(self) -> None:
        items = self.pop_mark()
        self._list(self.stack[-1]).extend(items)

    @staticmethod
    def _dict(target: Any) -> Dict[Any, Any]:
        # Pickle adds items to a dict only; assigning them to a numpy array could
        # make it do any amount of work.
        if type(target) is not dict:
            raise pickle.UnpicklingError(f"dict items for {_brief(target)}")
        return target

    @staticmethod
    def _list(target: Any) -> List[Any]:
        if type(target) is not list:
            raise pickle.UnpicklingError(f"list items for {_brief(target)}")
        return target

    # Opcodes that make a container take a second step, and a set six more, for
    # the memory each takes (an empty list is 56 bytes, an empty set 216). The
    # loop checks these steps with the opcodes' own (load()).

    def load_empty_list(self) -> None:
        self._limits.steps -= 1
        self.append([])

    def load_empty_dictionary(self) -> None:
        self._limits.steps -= 1
        self.append({})

    def load_empty_set(self) -> None:
        self._limits.steps -= 6
        self.append(set())

    def load_list(self) -> None:
        self._limits.steps -= 1
        items = self.pop_mark()
        self.append(items)

    def load_tuple(self) -> None:
        self._limits.steps -= 1
        items = self.pop_mark()
        self.append(tuple(items))

    def load_tuple1(self) -> None:
        self._limits.steps -= 1
        self.stack[-1] = (self.stack[-1],)

    def load_tuple2(self) -> None:
        self._limits.steps -= 1
        self.stack[-2:] = [(self.stack[-2], self.stack[-1])]

    def load_tuple3(self) -> None:
        self._limits.steps -= 1
        self.stack[-3:] = [(self.stack[-3], self.stack[-2], self.stack[-1])]

    def load_mark(self) -> None:
        self._limits.steps -= 1
        self.metastack.append(self.stack)
        self.stack = []
        self.append = self.stack.append

    # Strings decode strictly: pickle's own decoding lets lone surrogates
    # through, calling a handler for each, so a megabyte of gzip took 20 seconds,
    # and no document holds one.

    def load_binunicode(self) -> None:
        (size,) = struct.unpack("<I", self.read(4))
        self.append(str(self.read(size), "utf-8"))

    def load_binunicode8(self) -> None:
        (size,) = struct.unpack("<Q", self.read(8))
        self.append(str(self.read(size), "utf-8"))

    def load_short_binunicode(self) -> None:
        self.append(str(self.read(self.read(1)[0]), "utf-8"))

    def load_text(self) -> None:
        # Protocol 0's text opcodes, which pickle writes from protocol 2 on only
        # for GLOBAL, and documents were protocol 3 or 4: an int of 4,300 digits
        # is a line that gzips to a few bytes and takes 60 microseconds to read.
        raise pickle.UnpicklingError("a protocol 0 opcode")

    def load_frame(self) -> None:
        # A frame's length only says how much the pickler wrote at once.
        self.read(8)

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

    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.NEWOBJ_EX[0]] = load_newobj
    dispatch[pickle.BUILD[0]] = load_build
    dispatch[pickle.SETITEM[0]] = load_setitem
    dispatch[pickle.SETITEMS[0]] = load_setitems
    dispatch[pickle.DICT[0]] = load_dict
    dispatch[pickle.ADDITEMS[0]] = load_additems
    dispatch[pickle.FROZENSET[0]] = load_frozenset
    dispatch[pickle.APPEND[0]] = load_append
    dispatch[pickle.APPENDS[0]] = load_appends
    dispatch[pickle.EMPTY_LIST[0]] = load_empty_list
    dispatch[pickle.EMPTY_DICT[0]] = load_empty_dictionary
    dispatch[pickle.EMPTY_SET[0]] = load_empty_set
    dispatch[pickle.LIST[0]] = load_list
    dispatch[pickle.TUPLE[0]] = load_tuple
    dispatch[pickle.TUPLE1[0]] = load_tuple1
    dispatch[pickle.TUPLE2[0]] = load_tuple2
    dispatch[pickle.TUPLE3[0]] = load_tuple3
    dispatch[pickle.MARK[0]] = load_mark
    dispatch[pickle.BINUNICODE[0]] = load_binunicode
    dispatch[pickle.BINUNICODE8[0]] = load_binunicode8
    dispatch[pickle.SHORT_BINUNICODE[0]] = load_short_binunicode
    for _opcode in (
        pickle.INT,
        pickle.LONG,
        pickle.FLOAT,
        pickle.STRING,
        pickle.UNICODE,
        pickle.PUT,
        pickle.GET,
        pickle.INST,
        pickle.OBJ,
        pickle.PERSID,
    ):
        dispatch[_opcode[0]] = load_text
    del _opcode
    dispatch[pickle.FRAME[0]] = load_frame
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


def _decoder(
    read_array: Callable[[Any], np.ndarray], limits: _Limits
) -> Callable[[Any], Any]:
    """
    A function that rebuilds a document value, in place, from the JSON that _encode
    wrote. Its values nest at most _MAX_DEPTH deep, counted as a version 1 file's
    are (a level a value, however many levels of JSON it takes), and object arrays
    at most _MAX_OBJECT_NESTING deep: hashing nested tuples and freeing nested
    object arrays recurse in C.
    """
    # How deeply each object array made so far nests object arrays, by id: each
    # stays alive in the value being rebuilt.
    nesting: Dict[int, int] = {}

    def decode(obj: Any, depth: int) -> Any:
        if not isinstance(obj, (list, dict)):
            return obj
        if depth >= _MAX_DEPTH:
            raise _TooComplexError(f"values nested more than {_MAX_DEPTH} deep")
        depth += 1
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = decode(item, depth)
            return obj
        kind = obj.get(_TYPE_KEY)
        if kind is None:
            for key, value in obj.items():
                obj[key] = decode(value, depth)
            return obj
        if kind == "complex":
            real, imag = obj["real"], obj["imag"]
            if type(real) not in (int, float) or type(imag) not in (int, float):
                raise ValueError(f"A complex number of {_brief(real)}.")
            return complex(real, imag)
        if kind in ("bytes", "bytearray"):
            text = obj["base64"]
            if type(text) is not str:
                raise ValueError(f"Bytes of {_brief(text)}.")
            data = base64.b64decode(text)
            return bytearray(data) if kind == "bytearray" else data
        if kind == "ndarray" and "file" in obj:
            return read_array(obj["file"])

        # Iterating a str here would make a value of each of its characters.
        items = obj["items"]
        if type(items) is not list:
            raise ValueError(f"Items of {_brief(items)}.")
        if kind == "dict":
            # Each item is a key and its value, in a list that isn't a value.
            for pair in items:
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError(f"A dict item of {_brief(pair)}.")
                pair[0] = decode(pair[0], depth)
                pair[1] = decode(pair[1], depth)
                limits.check_key(pair[0])
            return dict(items)
        items = [decode(item, depth) for item in items]
        if kind == "ndarray":
            shape = obj["shape"]
            if (
                type(shape) is not list
                or len(shape) > 64
                or not all(type(size) is int and size >= 0 for size in shape)
                or math.prod(shape) != len(items)
            ):
                raise ValueError(f"An object array of shape {_brief(shape)}.")
            inner = max(
                [
                    nesting.get(id(item), 0)
                    for item in items
                    if type(item) is np.ndarray
                ],
                default=0,
            )
            if inner >= _MAX_OBJECT_NESTING:
                raise _TooComplexError(
                    f"object arrays nested more than {_MAX_OBJECT_NESTING} deep"
                )
            # Filled through a flat view, so the array it returns owns its
            # elements: freeing a view frees its base too, a level more.
            array = np.empty(tuple(shape), dtype=object)
            flat = array.reshape(-1)
            for i, item in enumerate(items):
                flat[i] = item
            nesting[id(array)] = inner + 1
            return array
        if kind == "tuple":
            return tuple(items)
        if kind in ("set", "frozenset"):
            for item in items:
                limits.check_key(item)
            return set(items) if kind == "set" else frozenset(items)
        raise ValueError(f"Unknown value type {_brief(kind)}.")

    return lambda obj: decode(obj, 0)


def _read_zip(filename: str, limits: _Limits) -> Any:
    with zipfile.ZipFile(filename) as zf:
        # Each array is stored once, and a file that named an entry over and over
        # would have it read, and kept, as often.
        read = set()

        def read_array(name: Any) -> np.ndarray:
            if type(name) is not str or name in read:
                raise ValueError(f"An array stored as {_brief(name)}.")
            read.add(name)
            limits.charge(_ARRAY_STEPS)
            with zf.open(name) as f:
                array = np.load(io.BytesIO(f.read()), allow_pickle=False)
            if not isinstance(array, np.ndarray):
                raise ValueError(f"{name} is not a numpy array.")
            # Its header gives its shape, and 2**62 elements of no size take no
            # data at all, while copying them never ends.
            if array.dtype.itemsize == 0 and array.size:
                raise ValueError(f"{name} holds {array.size} elements of no size.")
            return array

        info = zf.getinfo(_DOC_JSON)
        if info.file_size > _JSON_BYTES_A_STEP * limits.allowed:
            raise _TooComplexError("more JSON than a document of its size")
        text = zf.read(info)
        # json.loads() makes every value before decode() sees any, so they are
        # counted first: a value is a "[" or "{" opening it, or after the first
        # in its list, a ",", and in a dict, its key's ":" as well. For the
        # memory they take, a dict counts twice, and a string once more (its
        # quotes). Nor may a number run to 40 digits: no document holds one, and
        # reading one of 4,300 takes 60 microseconds.
        marks = (b"[", b",", b":")
        values = 2 * text.count(b"{") + text.count(b'"') // 2
        values += sum(map(text.count, marks))
        if values > limits.steps or b"0" * 40 in text.translate(_DIGITS):
            raise _TooComplexError("more values than a document of its size")
        # Decoded strictly, as Dynaface writes it: json.loads() of bytes lets
        # lone surrogates through, slowly.
        return _decoder(read_array, limits)(json.loads(text.decode("utf-8")))


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
        limits = _Limits(os.fstat(f.fileno()).st_size)

    try:
        if magic == b"PK":
            doc = _read_zip(filename, limits)
        elif magic == b"\x1f\x8b":
            with gzip.open(filename, "rb") as gz:
                with io.BufferedReader(_RawStream(gz), _READ_BUFFER) as stream:
                    doc = _LegacyUnpickler(stream, limits).load()
            _check_tree(doc, limits)
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
        # A file built to crash Dynaface or keep it busy (_TooComplexError) is
        # refused as a file that isn't a document.
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


def _flag(value: Any) -> bool:
    """A measure's or an item's enabled flag, which a document stores as a bool."""
    if type(value) not in (bool, int) and not isinstance(value, np.bool_):
        raise ValueError(f"An enabled flag of {_brief(value)}.")
    return bool(value)


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
    level head. The frames share the view's arrays, which load_state() copies:
    a copy for each of 5,000 frames was gigabytes for a crafted profile.
    """
    state = list(frame)
    if len(state) > 1:
        state[1] = [math.nan] * 3
    return state + [dict(view)]


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
        # No import keeps more, and what the app does for each frame of a
        # document it opens (add_missing_views(), each frame's view saved again)
        # made millions of frames, a few kilobytes of pickle, gigabytes.
        if len(self.frames) > MAX_FRAMES:
            raise ValueError(f"A document of {len(self.frames)} frames.")
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
            profile = (self.lateral_landmarks, self.sagittal_x, self.sagittal_y)
            values = sum(np.size(part) for part in profile if part is not None)
            if values > _MAX_PROFILE_VALUES:
                raise ValueError(f"A profile of {values} values.")
            if DOC_HEADER_POSE in doc[DOC_HEADER]:
                pose = doc[DOC_HEADER][DOC_HEADER_POSE]
                # Pose() hashes what it's given, and quotes it when it isn't a
                # pose, so a crafted value goes no further than its type.
                if type(pose) is not str or pose not in _POSES:
                    raise ValueError(f"A pose of {_brief(pose)}.")
                self.pose = Pose(pose)
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
            # Only a str can name one of them; looking up anything else would
            # hash it, however long that takes. A measure named again is made
            # once: each has many items, and a file can name one 100,000 times
            # in a few hundred bytes.
            cls = known.pop(name, None) if type(name) is str else None
            # Make sure we have a class to handle this measure
            if cls:
                obj = cls()
                obj.enabled = _flag(enabled)
                self._sync_items(source_items, obj)
                result.append(obj)
        return result

    def _sync_items(self, source_items, obj):
        """Update the disabled flag on the target items, based on the source"""
        # Looked up by name, as set_item_enabled() would, but at once: a file can
        # list any number of items.
        items: Dict[str, List[Any]] = {}
        for item in obj.items:
            items.setdefault(item.name, []).append(item)
        for source in source_items:
            name = source[DOC_NAME]
            enabled = _flag(source[DOC_ENABLED])
            for item in items.get(name, []) if type(name) is str else []:
                item.enabled = enabled

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
