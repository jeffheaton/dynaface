"""Regenerate the far-cheek outline fixtures that test the tracer and the search.

Both libraries trace the far cheek of a three-quarter face from a foreground mask,
row by row from the landmark contour, and find the malar apex and the submalar
point on the traced outline. Each case here is either a synthetic mask built to
exercise one part of the tracer (a soft-edged disc for the sub-pixel crossing, a
nose bump with contour points in the background, hair merging with the cheek with
an intensity edge to fall back on; the mask is stored in the file, run-length
encoded, with its landmarks) or one of the committed renders in
dynaface-lib-python/tests_data/cheek (whose mask is every pixel that is not the
background color) with the raw SPIGA landmarks of tests_data/cheek_landmarks.json,
and its expected results come from the Python library, the reference
implementation: the traced outline (x, rows and sources) and every field of
observed_points(), plus frontal_constructions() of two frontal faces.
dynaface-lib-python's tests/test_cheek_outline.py (TestSharedOutlines) and
dynaface-lib-dotnet's CheekOutlineTests, once the cheek module is ported, check
their library against the file this writes,
dynaface-lib-python/tests_data/cheek_outlines.json, so the .NET tests also show
that the two ports agree, the tracer included; the .NET project links the file
into its TestData as it does lateral_profiles.json.

Rerun it from the repository root, in dynaface-lib-python's venv (no models are
needed), only when the tracer or the search changes on purpose, then make the
tests pass again:

    python scripts/make_cheek_outlines.py
"""

import json
import math
import pathlib
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from dynaface import cheek, cheek_outline

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_DATA = ROOT / "dynaface-lib-python" / "tests_data"
OUT = TESTS_DATA / "cheek_outlines.json"

# The synthetic scenes: a disc of radius 100 at (160, 160) in a 320 px image, the
# far cheek the subject's left (the image's right), the far contour 1 px inside
# the disc's edge, the chin below the disc, so the chin-to-canthus distance (the
# tracer's span, and H: the levels are horizontal here) is 165 px: the walk's
# limit is 8 px and NOSE_STEP is 1.65 px. The mask rows' edge
# (0.87 px past the last disc pixel, from the binary alpha) sits about 2 px
# outside the seed: the band's reference.
SIZE = 320
CX, CY, RADIUS = 160.0, 160.0, 100.0
INSET = 1
# The renders traced from their PNGs, and the two frontal faces.
RENDERS = ("yaw45", "yaw-45", "yaw45_hollow3", "yaw50")
FRONTAL = (("img4", "pipeline"), ("yaw0", "raw"))


def disc_alpha(soft: bool) -> NDArray[np.uint8]:
    """The disc's alpha: 255 inside, or a ramp of 40 per px through MASK_ALPHA
    exactly at the radius, as a background model's alpha fades."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rr = np.hypot(xx - CX, yy - CY)
    if soft:
        ramp = cheek_outline.MASK_ALPHA + 40.0 * (RADIUS - rr)
        return np.clip(ramp, 0, 255).astype(np.uint8)
    return np.where(rr < RADIUS, 255, 0).astype(np.uint8)


def disc_edge(alpha: NDArray[np.uint8], y: int) -> int:
    """The disc's rightmost foreground column in row y."""
    return int(np.flatnonzero(alpha[y] > cheek_outline.MASK_ALPHA).max())


def disc_landmarks(alpha: NDArray[np.uint8]) -> NDArray[np.float64]:
    """The 98 landmarks of the disc head, the far cheek the subject's left."""
    lm = np.zeros((98, 2), dtype=np.float64)
    lm[60], lm[72] = (120, 110), (200, 110)  # lateral canthi
    lm[96], lm[97] = (135, 110), (165, 110)  # pupils
    lm[55], lm[59] = (145, 170), (175, 170)  # alar bases, parallel to the canthi
    lm[90], lm[94] = (160, 230), (160, 232)  # stomion at 231: h 0.267
    lm[54], lm[57] = (205, 175), (190, 190)  # nose tip h 0.606, subnasale 0.515
    last_row = int(
        np.flatnonzero(np.any(alpha > cheek_outline.MASK_ALPHA, axis=1)).max()
    )
    for k, y in enumerate(np.round(np.linspace(105, 275, 17)).astype(int)):
        if y <= last_row:
            lm[32 - k] = (disc_edge(alpha, y) - INSET, y)
        else:
            lm[32 - k] = (170, y)  # the jaw, below the disc
    lm[16] = (160, 275)  # the chin: H = 275 - 110 = 165
    for k in range(17):
        lm[k] = (SIZE - 1 - lm[32 - k][0], lm[32 - k][1])
    return lm


def extend(
    alpha: NDArray[np.uint8], rows: Sequence[int], start: int, stop: int
) -> None:
    """Foreground from edge + start to edge + stop (exclusive) on these rows."""
    for y in rows:
        e = disc_edge(alpha, y)
        alpha[y, e + start : e + stop] = 255


def rle_rows(image: NDArray[np.uint8]) -> List[List[int]]:
    """Each row as [value, count, value, count, ...]."""
    out = []
    for row in image:
        change = np.flatnonzero(np.diff(row.astype(np.int32))) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(row)]])
        flat: List[int] = []
        for s, e in zip(starts, ends):
            flat += [int(row[s]), int(e - s)]
        out.append(flat)
    return out


def number(value: Any) -> Any:
    """A JSON number: NaN becomes null, numpy scalars plain Python ones."""
    v = float(value)
    return None if math.isnan(v) else v


def points(array: Any) -> List[Any]:
    return [[number(x), number(y)] for x, y in np.asarray(array, dtype=np.float64)]


def observed_dict(found: cheek_outline.ObliqueCheek) -> Dict[str, Any]:
    """Every field of ObliqueCheek, in its declared order, as JSON values."""
    out: Dict[str, Any] = {}
    for field in fields(found):
        value = getattr(found, field.name)
        if field.name in ("far_left", "apex_prior", "apex_near_nose", "nose_on"):
            out[field.name] = bool(value)
        elif field.name == "outline":
            out[field.name] = points(value)
        elif field.name == "outline_src":
            out[field.name] = [int(v) for v in value]
        elif field.name == "chord":
            out[field.name] = points(value)
        elif field.name == "levels":
            out[field.name] = {k: points(v) for k, v in value.items()}
        elif field.name in ("apex", "submalar"):
            out[field.name] = points([value])[0]
        else:
            out[field.name] = number(value)
    return out


def frontal_dict(fc: cheek.FrontalCheek) -> Dict[str, Any]:
    """Every field of FrontalCheek, the frame first, as JSON values."""
    out: Dict[str, Any] = {
        "frame": {
            "origin": points([fc.frame.origin])[0],
            "across": points([fc.frame.across])[0],
            "down": points([fc.frame.down])[0],
            "ipd": number(fc.frame.ipd),
        }
    }
    for field in fields(fc):
        if field.name == "frame":
            continue
        value = getattr(fc, field.name)
        if isinstance(value, np.ndarray):
            out[field.name] = points([value])[0]
        elif isinstance(value, (bool, int, np.integer)):
            out[field.name] = int(value)
        else:
            out[field.name] = number(value)
    return out


def trace_case(
    name: str,
    kind: str,
    landmarks: NDArray[np.float64],
    far_left: bool,
    alpha: NDArray[np.uint8],
    gray: Optional[NDArray[np.uint8]],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Trace and search one case, and return its entry with the expected values."""
    foreground = alpha > cheek_outline.MASK_ALPHA
    xs, rows, src = cheek_outline.trace_far_cheek(
        foreground, alpha, landmarks, far_left, gray=gray
    )
    found = cheek_outline.observed_points(landmarks, xs, rows, src, far_left)
    assert found is not None, name
    entry: Dict[str, Any] = {"name": name, "kind": kind, "far_left": far_left}
    entry.update(extra)
    entry["landmarks"] = [[float(x), float(y)] for x, y in landmarks]
    entry["outline"] = {
        "x": [float(x) for x in xs],  # float32 values, exact as doubles
        "rows": [int(y) for y in rows],
        "src": [int(s) for s in src],
    }
    entry["observed"] = observed_dict(found)
    return entry


def synthetic_cases() -> List[Dict[str, Any]]:
    cases = []
    # 1. The soft-edged disc: every disc row from the mask, the x the sub-pixel
    # alpha crossing on the circle.
    alpha = disc_alpha(soft=True)
    lm = disc_landmarks(alpha)
    case = trace_case(
        "soft_disc",
        "synthetic",
        lm,
        True,
        alpha,
        None,
        {"size": [SIZE, SIZE], "alpha": rle_rows(alpha), "gray": None},
    )
    src = np.array(case["outline"]["src"])
    rows = np.array(case["outline"]["rows"])
    xs = np.array(case["outline"]["x"])
    mask = src == cheek_outline.OUTLINE_SRC_MASK
    assert mask.sum() > 120 and not np.any(src == cheek_outline.OUTLINE_SRC_NOSE)
    radius = np.hypot(xs[mask] - CX, rows[mask] - CY)
    assert np.abs(radius - RADIUS).max() < 0.25, np.abs(radius - RADIUS).max()
    cases.append(case)
    # 2. The nose: a bump 4 px past the edge (3 px beyond the band's reference,
    # against a NOSE_STEP of 1.65) over the nose band and above it, marked
    # OUTLINE_SRC_NOSE between the tip's and the subnasale's levels and upward
    # while it departs; two contour points 4 px in the background, whose walk
    # goes inward to the edge.
    alpha = disc_alpha(soft=False)
    lm = disc_landmarks(alpha)
    extend(alpha, range(168, 186), 1, 5)
    for y in (211, 222):
        k = int(np.flatnonzero(lm[16:33, 1] == y)[0]) + 16
        lm[k] = (disc_edge(alpha, y) + 4, y)
    case = trace_case(
        "nose_disc",
        "synthetic",
        lm,
        True,
        alpha,
        None,
        {"size": [SIZE, SIZE], "alpha": rle_rows(alpha), "gray": None},
    )
    src = np.array(case["outline"]["src"])
    rows = np.array(case["outline"]["rows"])
    nose = set(rows[src == cheek_outline.OUTLINE_SRC_NOSE].tolist())
    assert nose == set(range(168, 186)), sorted(nose)
    assert case["observed"]["nose_on"] is True
    cases.append(case)
    # 3. Hair: the mask runs 30 px past the edge on rows 108 to 130 (the walk,
    # limited to 8 px, finds no edge there), with a gray image whose step 4 px
    # outside the disc's edge gives the intensity edge on the rows the edge
    # source is on (below a plain row ratio of 0.95: rows 119 and down) and the
    # contour on those above; a hair blob 30 px outside the edge on rows 200 to
    # 210, beyond the walk, changes nothing.
    clean = disc_alpha(soft=False)
    alpha = clean.copy()
    lm = disc_landmarks(alpha)
    extend(alpha, range(108, 131), 1, 31)
    extend(alpha, range(200, 211), 30, 41)
    gray = np.full((SIZE, SIZE), 60, dtype=np.uint8)
    for y in range(SIZE):
        # The step stops two rows above the disc's last row (259), the blur's
        # reach, so the rows below the disc, where the walk fails too, see no
        # gradient and keep the contour.
        if y < 258 and np.any(clean[y] > cheek_outline.MASK_ALPHA):
            gray[y, : disc_edge(clean, y) + 5] = 180
    case = trace_case(
        "hair_disc",
        "synthetic",
        lm,
        True,
        alpha,
        gray,
        {"size": [SIZE, SIZE], "alpha": rle_rows(alpha), "gray": rle_rows(gray)},
    )
    src = np.array(case["outline"]["src"])
    rows = np.array(case["outline"]["rows"])
    edge_rows = set(rows[src == cheek_outline.OUTLINE_SRC_EDGE].tolist())
    contour_rows = set(rows[src == cheek_outline.OUTLINE_SRC_CONTOUR].tolist())
    assert edge_rows == set(range(119, 131)), sorted(edge_rows)
    assert set(range(108, 119)) <= contour_rows, sorted(contour_rows)
    assert not np.any(
        src[(rows >= 200) & (rows <= 210)] != cheek_outline.OUTLINE_SRC_MASK
    )
    cases.append(case)
    return cases


def render_cases(
    images: Dict[str, Any], renders: Dict[str, Any]
) -> List[Dict[str, Any]]:
    background = np.array(renders["background"], dtype=np.uint8)
    cases = []
    for name in RENDERS:
        render = renders["renders"][name]
        path = TESTS_DATA / "cheek" / render["file"]
        image = np.asarray(Image.open(path).convert("RGB"))
        foreground = np.any(image != background, axis=2)
        alpha = np.where(foreground, 255, 0).astype(np.uint8)
        lm = np.array(images[name]["raw"]["landmarks"], dtype=np.float64)
        far_left = render["far_side"] == "left"
        cases.append(
            trace_case(
                name,
                "render",
                lm,
                far_left,
                alpha,
                None,
                {"file": f"cheek/{render['file']}"},
            )
        )
    return cases


def frontal_cases(images: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []
    for name, source in FRONTAL:
        lm = np.array(images[name][source]["landmarks"], dtype=np.float64)
        fc = cheek.frontal_constructions(lm)
        assert fc is not None, name
        cases.append(
            {
                "name": name,
                "landmarks_from": source,
                "landmarks": [[float(x), float(y)] for x, y in lm],
                "constructions": frontal_dict(fc),
            }
        )
    return cases


def write_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    """Compact JSON, one case per line, so the file stays small and its diffs
    readable."""

    def dumps(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("{\n")
        keys = list(data)
        for i, key in enumerate(keys):
            value = data[key]
            comma = "," if i < len(keys) - 1 else ""
            if isinstance(value, list) and value and isinstance(value[0], dict):
                f.write(f" {dumps(key)}: [\n")
                for j, item in enumerate(value):
                    item_comma = "," if j < len(value) - 1 else ""
                    f.write(f"  {dumps(item)}{item_comma}\n")
                f.write(f" ]{comma}\n")
            else:
                f.write(f" {dumps(key)}: {dumps(value)}{comma}\n")
        f.write("}\n")


def main() -> None:
    with open(TESTS_DATA / "cheek_landmarks.json", encoding="utf-8") as f:
        images = json.load(f)["images"]
    with open(TESTS_DATA / "cheek" / "cheek_renders.json", encoding="utf-8") as f:
        renders = json.load(f)
    data: Dict[str, Any] = {
        "about": (
            "Far-cheek outlines for testing the tracer and the apex and submalar "
            "search: synthetic masks and the committed renders, with the Python "
            "library's outline and observed points for each, and the frontal "
            "constructions of two faces. Written by scripts/make_cheek_outlines.py; "
            "don't edit by hand."
        ),
        "mask": {
            "synthetic": (
                "alpha and gray are rows of [value, count, ...] runs over the "
                "image's width; the foreground is alpha > mask_alpha"
            ),
            "render": (
                "every pixel of the file whose color is not background is "
                "foreground, with an alpha of 255 there and 0 elsewhere"
            ),
            "mask_alpha": cheek_outline.MASK_ALPHA,
            "background": renders["background"],
        },
        "outline": (
            "x is the traced outline's float32 x in each of rows, src each row's "
            "source: 0 contour, 1 mask, 2 nose, 3 intensity edge"
        ),
        "observed": "every field of cheek_outline.ObliqueCheek; null is NaN",
        "constructions": "every field of cheek.FrontalCheek",
        "cases": synthetic_cases() + render_cases(images, renders),
        "frontal": frontal_cases(images),
    }
    write_json(OUT, data)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
