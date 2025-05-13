import numpy as np
from collections.abc import Mapping
from typing import Any

# ————————————————————————————————————————————————————————————————
# Allowed built-in types
_ALLOWED_PRIMITIVES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
)
_ALLOWED_CONTAINERS = (
    list,
    tuple,
    set,
    frozenset,
    dict,
    np.ndarray,  # ← now allowed
)
_ALLOWED_TYPES = _ALLOWED_PRIMITIVES + _ALLOWED_CONTAINERS


def _full_typename(obj: Any) -> str:
    """Return the full module + class name for an object or type."""
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def assert_standard_python(
    obj: Any, *, _path: str = "root", _seen: set[int] = None
) -> None:
    """
    Recursively scans `obj`, raising TypeError if any part of it is
    not one of the allowed built-in primitives or containers.

    Now supports:
      - Primitives: None, bool, int, float, complex, str, bytes, bytearray
      - Containers: list, tuple, set, frozenset, dict, numpy.ndarray
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    # 1) Primitives are fine
    if isinstance(obj, _ALLOWED_PRIMITIVES):
        return

    # 2) numpy.ndarray → convert to nested lists and re-check
    if isinstance(obj, np.ndarray):
        try:
            nested = obj.tolist()
        except Exception:
            # fallback if .tolist() fails for some dtype
            nested = list(obj.flat)
        assert_standard_python(nested, _path=f"{_path} (numpy.ndarray)", _seen=_seen)
        return

    # 3) dict: check both keys and values
    if isinstance(obj, Mapping):
        for key, val in obj.items():
            assert_standard_python(key, _path=f"{_path} (dict key)", _seen=_seen)
            assert_standard_python(val, _path=f"{_path}[{key!r}]", _seen=_seen)
        return

    # 4) Other containers: list, tuple, set, frozenset
    if isinstance(obj, (list, tuple, set, frozenset)):
        for idx, item in enumerate(obj):
            assert_standard_python(item, _path=f"{_path}[{idx}]", _seen=_seen)
        return

    # 5) Anything else is non-standard → error out
    found_type = _full_typename(obj)
    raise TypeError(
        f"Unsupported object at {_path}!\n"
        f"  • Found: {obj!r}\n"
        f"  • Type:  {found_type}\n"
        f"  • Allowed types are:\n"
        f"    – Primitives: "
        + ", ".join(_full_typename(t) for t in _ALLOWED_PRIMITIVES)
        + "\n"
        f"    – Containers: "
        + ", ".join(_full_typename(t) for t in _ALLOWED_CONTAINERS)
    )
