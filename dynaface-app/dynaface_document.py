import base64
import gzip
import io
import itertools
import json
import pickle
import zipfile
from typing import Any, Callable, Dict

import numpy as np
from dynaface.const import Pose
from jth_ui import utl_classes
from utl_general import assert_standard_python

import dynaface

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

# The only globals a version 1 pickle needs: numpy array and scalar reconstruction,
# under both the numpy 1 and numpy 2 module names, and the builtins that pickle
# can't write with its own opcodes. Anything else is refused before it can run.
_LEGACY_GLOBALS = {
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    ("builtins", "frozenset"),
    ("builtins", "set"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
}


class _RefusedGlobalError(pickle.UnpicklingError):
    pass


class _LegacyUnpickler(pickle.Unpickler):
    """Loads version 1 documents without running anything they reference."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _LEGACY_GLOBALS:
            raise _RefusedGlobalError(f"{module}.{name}")
        return super().find_class(module, name)


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
        raise TypeError(
            f"The file '{filename}' was not opened because it contains code ({e}) "
            "that would run when it's loaded."
        ) from e
    except Exception as e:
        raise TypeError(
            f"The file '{filename}' does not appear to be a valid Dynaface document."
        ) from e

    if not isinstance(doc, dict) or not isinstance(doc.get(DOC_HEADER), dict):
        raise TypeError(
            f"The file '{filename}' does not appear to be a valid Dynaface document."
        )
    if doc[DOC_HEADER].get(DOC_HEADER_VERSION, 1) > DOC_FORMAT_VERSION:
        raise TypeError(
            f"The file '{filename}' was saved by a newer version of Dynaface."
        )
    return doc


class DynafaceDocument:
    def __init__(self):
        self._version = DOC_FORMAT_VERSION
        self.face = None
        self.frames = []
        self.measures = []
        self.fps = 0
        self.lateral = False
        self.lateral_landmarks = None
        self.sagittal_x = None
        self.sagittal_y = None
        self.pose = None

    def save(self, filename: str):
        measures = self._save_measures(self.measures)

        doc = {
            DOC_HEADER: {
                DOC_HEADER_VERSION: self._version,
                DOC_HEADER_FPS: self.fps,
                DOC_HEADER_LATERAL: self.lateral,
                DOC_HEADER_POSE: self.pose.value,
            },
            DOC_BODY: {
                DOC_BODY_MEASURES: measures,
                DOC_BODY_FRAMES: self.frames,
                DOC_BODY_LATERAL_LANDMARKS: self.lateral_landmarks,
                DOC_BODY_SAGITTAL_X: self.sagittal_x,
                DOC_BODY_SAGITTAL_Y: self.sagittal_y,
            },
        }
        assert_standard_python(doc)

        with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            counter = itertools.count()

            def store_array(array: np.ndarray) -> str:
                name = f"{_ARRAY_PREFIX}{next(counter)}.npy"
                with zf.open(name, "w") as f:
                    np.save(f, array, allow_pickle=False)
                return name

            zf.writestr(_DOC_JSON, json.dumps(_encode(doc, store_array)))

    def load(self, filename: str):
        doc = _read_document(filename)

        # load
        measures = self._load_measures(doc[DOC_BODY][DOC_BODY_MEASURES])
        self._add_missing_measures(measures, dynaface.measures.all_measures())

        self.fps = doc[DOC_HEADER][DOC_HEADER_FPS]
        self.frames = doc[DOC_BODY][DOC_BODY_FRAMES]
        self.measures = measures
        if DOC_HEADER_LATERAL not in doc[DOC_HEADER]:
            # If the lateral key is not present, set it to False
            self.lateral = False
            self.lateral_landmarks = None
            self.sagittal_x = None
            self.sagittal_y = None
        else:
            # If the key is present, set it to the value in the document
            # This allows for backward compatibility with older documents
            # that do not have this key.
            self.lateral = doc[DOC_HEADER][DOC_HEADER_LATERAL]
            self.lateral_landmarks = doc[DOC_BODY][DOC_BODY_LATERAL_LANDMARKS]
            self.sagittal_x = doc[DOC_BODY][DOC_BODY_SAGITTAL_X]
            self.sagittal_y = doc[DOC_BODY][DOC_BODY_SAGITTAL_Y]
            self.pose = (
                Pose(doc[DOC_HEADER][DOC_HEADER_POSE])
                if DOC_HEADER_POSE in doc[DOC_HEADER]
                else Pose.FRONTAL
            )

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
