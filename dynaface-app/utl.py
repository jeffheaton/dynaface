from collections.abc import Mapping, Sequence, Set
from typing import Any


def assert_standard_python(
    obj: Any, *, _path: str = "root", _seen: set[int] = None
) -> None:
    """
    Recursively scans obj and raises TypeError if any sub-object
    is not a standard built-in Python primitive or container.

    Standard types allowed:
      - None, bool, int, float, complex, str, bytes, bytearray
      - list, tuple, set, frozenset, dict

    Any other object type will trigger:
        TypeError(f"Non-standard object <obj> of type <type> at <path>")

    :param obj:      The object to scan.
    :param _path:    Internal: a string showing the location in the object graph.
    :param _seen:    Internal: ids of objects already visited (to avoid infinite recursion).
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    # Allowed primitives
    if obj is None or isinstance(
        obj, (bool, int, float, complex, str, bytes, bytearray)
    ):
        return

    # dict: check keys and values
    if isinstance(obj, Mapping):
        for key, val in obj.items():
            assert_standard_python(key, _path=f"{_path}[key={key!r}]", _seen=_seen)
            assert_standard_python(val, _path=f"{_path}[{key!r}]", _seen=_seen)
        return

    # list/tuple/set/frozenset: check elements
    if isinstance(obj, (list, tuple, set, frozenset)):
        for idx, item in enumerate(obj):
            assert_standard_python(item, _path=f"{_path}[{idx}]", _seen=_seen)
        return

    # Anything else is non-standard
    raise TypeError(
        f"Non-standard object {obj!r} of type {type(obj).__name__!r} at {_path}"
    )
