import math
import numpy as np
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from dynaface.const import Pose

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

    # 2) numpy.ndarray: only object arrays can hold non-standard values, so only
    #    their elements are checked. Walking every element of numeric arrays, such
    #    as each frame's image, used to take most of the time spent saving.
    if isinstance(obj, np.ndarray):
        if not obj.dtype.hasobject:
            return
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
        "    – Containers: " + ", ".join(_full_typename(t) for t in _ALLOWED_CONTAINERS)
    )


def describe_analysis_error(error: Exception) -> str:
    """
    A message for the user about an error from analyzing an image or video.

    The library's ModelOutputError means the models computed garbage on the
    accelerator; its own advice is for programmers, so give the app's instead.
    """
    from dynaface.models import ModelOutputError

    if isinstance(error, ModelOutputError):
        return (
            "Dynaface's AI models aren't working on this computer's accelerator "
            "(GPU): they gave results that working models never give. Turn off "
            '"Use Accelerator (GPU)" in Settings, restart Dynaface, and try again.'
            f"\n\nDetails: {error}"
        )
    return str(error)


# How the app names each pose wherever a person reads it: dialogs, messages,
# tooltips and the status line. The enum's own values ("frontal", "lateral",
# "quarter") are for saved documents and logs.
POSE_LABELS: Dict[Pose, str] = {
    Pose.FRONTAL: "frontal",
    Pose.LATERAL: "lateral",
    Pose.QUARTER: "three-quarter",
}


def view_label(pose: Optional[Pose], far_left: bool = False) -> str:
    """
    The name of a face's view for the user: "frontal", "lateral" or
    "three-quarter, far cheek left" / "three-quarter, far cheek right".

    A pose that isn't lateral or quarter (frontal, detect, or None from a
    document saved before poses were stored) is frontal.
    """
    if pose == Pose.QUARTER:
        side = "left" if far_left else "right"
        return f"{POSE_LABELS[pose]}, far cheek {side}"
    return POSE_LABELS.get(pose, POSE_LABELS[Pose.FRONTAL])


def named_series(series: Any) -> List[str]:
    """
    The names of a FrameSeries' outputs that hold at least one finite number, in
    the series' order, so a chart plots only the outputs the kept frames have
    (a measure's outputs are NaN on every frame whose view it doesn't suit).
    """
    return [
        name
        for name in series.names
        if any(_is_finite(value) for value in series.values[name])
    ]


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
