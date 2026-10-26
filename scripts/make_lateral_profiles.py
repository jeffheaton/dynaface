"""Regenerate the synthetic side profiles that test the profile-landmark search.

Both libraries find six landmarks on the traced outline of a side profile. Each
case here is a synthetic outline, built to exercise one part of that search, and
its expected results come from the Python library, the reference implementation.
dynaface-lib-python's tests/test_lateral.py and dynaface-lib-dotnet's
LateralProfileTests check their library against the file this writes,
dynaface-lib-python/tests_data/lateral_profiles.json, so the .NET tests also show
that the two ports agree.

Rerun it from the repository root, in dynaface-lib-python's venv, only when the
search changes on purpose, then make the .NET tests pass again:

    python scripts/make_lateral_profiles.py
"""

import json
import pathlib
from unittest import mock

import numpy as np
from scipy.signal import savgol_filter

from dynaface import lateral

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "dynaface-lib-python" / "tests_data" / "lateral_profiles.json"

SHIFT_X = 180  # how far left of the image's edge the face's most forward point is

# The outline of a face, as (row, x) control points. The profile faces left, so a
# smaller x is further forward.
FACE = [
    (100, 70),  # top of the forehead
    (150, 45),  # glabella
    (195, 62),  # nasion
    (215, 55),  # a bend in the bridge of the nose
    (250, 0),  # nasal tip
    (272, 50),  # subnasal point
    (300, 34),  # upper lip
    (318, 44),  # between the lips
    (334, 36),  # lower lip
    (362, 58),  # mentolabial fold
    (395, 30),  # pogonion
    (425, 80),  # under the chin
    (499, 110),  # neck
]
# Corners for the landmark search on its own: on the nose, just below the subnasal
# point and under the chin.
FACE_CORNERS = [230, 285, 420]

# The frontal landmarks whose heights anchor the search, at the heights matching
# FACE. Index 35, on a brow, is the highest, which anchors the glabella.
ANCHORS = {35: 150, 51: 195, 54: 245, 57: 268, 79: 300, 8: 345, 14: 380, 16: 410}


def pixel_outline(points):
    """An outline traced from pixels, as the library sees it: whole columns.

    Returns its x, shifted so the smallest is 0, and its rows.
    """
    rows = np.arange(points[0][0], points[-1][0] + 1)
    xs = np.rint(np.interp(rows, [p[0] for p in points], [p[1] for p in points]))
    xs = xs.astype(int)
    return xs - xs.min(), rows


def smooth_outline(points, sigma):
    """An outline with rounded bends, for running the whole search.

    On a pixel outline, a sloping edge is a staircase whose identical steps give
    curvature peaks of exactly the same height, and which of them the corner
    search keeps then comes down to rounding noise, which differs between the
    ports. A smooth outline has no such ties. Its x is a multiple of 1/1024, which
    the .NET port's 32-bit floats hold exactly, so both ports read the same values.
    """
    rows = np.arange(points[0][0], points[-1][0] + 1)
    xs = np.interp(rows, [p[0] for p in points], [p[1] for p in points])
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    xs = np.convolve(np.pad(xs, radius, mode="edge"), kernel / kernel.sum(), "valid")
    xs = np.round(xs * 1024) / 1024
    return xs - xs.min(), rows


def frontal(count=98, **changes):
    """Frontal landmarks at the ANCHORS heights, with changes by index (i35=...)."""
    heights = dict(ANCHORS)
    heights.update({int(k[1:]): v for k, v in changes.items()})
    return [[0, heights.get(i, 300)] for i in range(count)]


def row_index(rows, row):
    return int(np.flatnonzero(rows == row)[0])


def search(xs, rows, landmarks_frontal):
    found = lateral._find_profile_landmarks(xs, rows, SHIFT_X, landmarks_frontal)
    return [np.asarray(part).tolist() for part in found]


def jittered_savgol(seed):
    """savgol_filter with relative noise of 1e-9, far above any rounding noise."""
    rng = np.random.default_rng(seed)

    def jittered(*args, **kwargs):
        out = savgol_filter(*args, **kwargs)
        return out * (1 + rng.uniform(-1e-9, 1e-9, out.shape))

    return jittered


def robust(xs, rows, landmarks_frontal):
    """Whether the search's results survive noise in its smoothing arithmetic."""
    expected = search(xs, rows, landmarks_frontal)
    for seed in range(8):
        with mock.patch.object(lateral, "savgol_filter", jittered_savgol(seed)):
            if search(xs, rows, landmarks_frontal) != expected:
                return False
    return True


outlines = {}
cases = []


def add_outline(name, xs, rows):
    xs, rows = np.asarray(xs), np.asarray(rows)
    outlines[name] = {"sagittal_x": xs.tolist(), "sagittal_y": rows.tolist()}
    return xs, rows


def add(name, about, outline_name, landmarks_frontal, check, search_only, **given):
    """Add a case, with the Python library's results.

    A whole-search case finds the outline's extrema, then its corners, then the
    landmarks. A search-only case tests the landmark search on its own, with the
    outline's extrema and the given corners, or other given indices.
    """
    xs = np.array(outlines[outline_name]["sagittal_x"])
    rows = np.array(outlines[outline_name]["sagittal_y"])
    if search_only:
        max_found, min_found = lateral._find_local_max_min(xs)
        indices = {
            "max_indices": given.get("max_indices", max_found),
            "min_indices": given.get("min_indices", min_found),
            "corner_idxs": given["corner_idxs"],
        }
        indices = {k: [int(i) for i in v] for k, v in indices.items()}
        landmarks = lateral._find_lateral_landmarks(
            xs,
            rows,
            *(np.array(v, dtype=int) for v in indices.values()),
            SHIFT_X,
            landmarks_frontal,
        ).tolist()
    else:
        assert robust(xs, rows, landmarks_frontal), f"{name} depends on rounding noise"
        found = search(xs, rows, landmarks_frontal)
        indices = dict(zip(("max_indices", "min_indices", "corner_idxs"), found))
        landmarks = found[3]
    # Fail here, rather than in the tests, if a case stops doing its job.
    assert check(landmarks, indices), f"{name} no longer does what it's for"
    cases.append(
        {
            "name": name,
            "about": about,
            "outline": outline_name,
            "search_only": search_only,
            "shift_x": SHIFT_X,
            "frontal": landmarks_frontal,
            **indices,
            "landmarks": landmarks,
        }
    )


MISSING = [-1 + SHIFT_X, -1]  # a landmark that wasn't found: (-1, -1), shifted
G, N, TIP, SUB, ML, POG = range(6)

face_x, face_rows = add_outline("face", *pixel_outline(FACE))
# A flat chin, so the mentolabial search, which ends at landmark 14, and the
# pogonion search, which starts there, can find points a pixel or two apart.
FLAT_CHIN = [p for p in FACE if p[0] <= 362] + [(392, 30), (412, 30)]
chin_x, chin_rows = add_outline(
    "flat_chin", *pixel_outline(FLAT_CHIN + [p for p in FACE if p[0] > 412])
)
add_outline("smooth_face", *smooth_outline(FACE, sigma=3))
add_outline("smoother_face", *smooth_outline(FACE, sigma=6))
add_outline("flat", [0] * 120, np.arange(240, 360))
add_outline("six_rows", [3, 1, 0, 1, 3, 6], np.arange(247, 253))
add_outline("eight_rows", [6, 3, 1, 0, 1, 3, 6, 10], np.arange(246, 254))


def on(row, xs=face_x, rows=face_rows):
    """The landmark expected on an outline at a row."""
    return [int(xs[row_index(rows, row)]) + SHIFT_X, row]


def corners(*rows, outline_rows=face_rows):
    return [row_index(outline_rows, r) for r in rows]


def near(point, row, tolerance=3):
    return point != MISSING and abs(point[1] - row) <= tolerance


# --- The whole search: extrema, corners and landmarks ---

add(
    "smooth_face",
    "A face with rounded bends: every landmark found near its bend.",
    "smooth_face",
    frontal(),
    lambda lm, ix: near(lm[TIP], 250)
    and near(lm[POG], 395)
    and near(lm[ML], 362)
    and len(ix["corner_idxs"]) > 0,
    search_only=False,
)
add(
    "smoother_face",
    "The same face with rounder bends, so it has fewer corners.",
    "smoother_face",
    frontal(),
    lambda lm, ix: MISSING not in lm and len(ix["corner_idxs"]) > 0,
    search_only=False,
)
add(
    "flat",
    "A straight vertical outline: no extrema and no corners.",
    "flat",
    frontal(),
    lambda lm, ix: ix["max_indices"] == ix["min_indices"] == ix["corner_idxs"] == []
    and lm[TIP] == MISSING,
    search_only=False,
)
add(
    "six_rows",
    "An outline too short (under 7 rows) to look for corners in.",
    "six_rows",
    frontal(),
    lambda lm, ix: ix["corner_idxs"] == [],
    search_only=False,
)
add(
    "eight_rows",
    "An outline shorter than every smoothing window, so no scale runs.",
    "eight_rows",
    frontal(),
    lambda lm, ix: ix["corner_idxs"] == [],
    search_only=False,
)

# --- The landmark search on its own, on a pixel outline ---

add(
    "face",
    "A face traced from pixels: every landmark found, each extreme unique.",
    "face",
    frontal(),
    lambda lm, ix: lm == [on(150), on(195), on(250), on(272), on(362), on(395)],
    search_only=True,
    corner_idxs=corners(*FACE_CORNERS),
)
add(
    "tied_extremes",
    "A flat chin, so the pogonion is the middle of the tied, most forward rows in"
    " its window (392 to 410).",
    "flat_chin",
    frontal(),
    lambda lm, ix: lm[POG] == on(401, chin_x, chin_rows),
    search_only=True,
    corner_idxs=corners(*FACE_CORNERS),
)
add(
    "subnasal_corner_first",
    "A corner before the deepest point of the subnasal window is chosen.",
    "face",
    frontal(),
    lambda lm, ix: lm[SUB] == on(269),
    search_only=True,
    corner_idxs=corners(269, 290),
)
add(
    "subnasal_corner_outside",
    "Corners outside the subnasal window: its deepest point is chosen.",
    "face",
    frontal(),
    lambda lm, ix: lm[SUB] == on(272),
    search_only=True,
    corner_idxs=corners(200, 420),
)
add(
    "no_corners",
    "No corners at all: the subnasal point is the window's deepest point.",
    "face",
    frontal(),
    lambda lm, ix: lm[SUB] == on(272),
    search_only=True,
    corner_idxs=[],
)
add(
    "no_minima",
    "No local minima, so the nasal tip isn't found.",
    "face",
    frontal(),
    lambda lm, ix: lm[TIP] == MISSING,
    search_only=True,
    corner_idxs=corners(*FACE_CORNERS),
    min_indices=[],
)
add(
    "overlap_corner",
    "The mentolabial point lands on the pogonion, so the nearest corner above"
    " landmark 14 replaces it.",
    "flat_chin",
    frontal(i8=398, i14=400, i16=402),
    lambda lm, ix: lm[ML] == on(362, chin_x, chin_rows),
    search_only=True,
    corner_idxs=corners(200, 362, 405, outline_rows=chin_rows),
)
add(
    "overlap_no_corner",
    "The mentolabial point lands on the pogonion and no corner is above landmark"
    " 14, so it stays.",
    "flat_chin",
    frontal(i8=398, i14=400, i16=402),
    lambda lm, ix: abs(lm[ML][1] - lm[POG][1]) <= 5,
    search_only=True,
    corner_idxs=corners(405, outline_rows=chin_rows),
)
add(
    "same_rows",
    "Window ends at the same row are widened by two rows each way.",
    "face",
    frontal(i57=280, i79=280, i8=395, i14=395, i16=395),
    lambda lm, ix: MISSING not in lm,
    search_only=True,
    corner_idxs=corners(*FACE_CORNERS),
)
add(
    "reversed_windows",
    "Window ends given bottom first are put in order.",
    "face",
    frontal(i57=300, i79=268, i14=410, i16=380),
    lambda lm, ix: lm[POG] == on(395) and lm[SUB] == on(272),
    search_only=True,
    corner_idxs=corners(*FACE_CORNERS),
)
add(
    "nasion_above_outline",
    "Landmark 51 is above the whole outline, so no row is at or above it and the"
    " nasion isn't found.",
    "face",
    frontal(i35=40, i51=50),
    lambda lm, ix: lm[N] == MISSING,
    search_only=True,
    corner_idxs=corners(*FACE_CORNERS),
)
add(
    "sixty_frontal_landmarks",
    "Only 60 frontal landmarks, too few to find the subnasal point.",
    "face",
    frontal(count=60),
    lambda lm, ix: lm[SUB] == MISSING and lm[TIP] != MISSING,
    search_only=True,
    corner_idxs=corners(*FACE_CORNERS),
)
add(
    "no_frontal_landmarks",
    "No frontal landmarks: nothing is found, and nothing is shifted.",
    "face",
    [],
    lambda lm, ix: lm == [[-1, -1]] * 6,
    search_only=True,
    corner_idxs=corners(*FACE_CORNERS),
)

ABOUT = (
    "Synthetic side-profile outlines for testing the profile-landmark search."
    " Written by scripts/make_lateral_profiles.py, with expected results from the"
    " Python library; don't edit by hand."
)


def fields(item, indent):
    pad = " " * indent
    return ",\n".join(f'{pad}"{k}": {json.dumps(v)}' for k, v in item.items())


text = "{\n" + f'  "about": {json.dumps(ABOUT)},\n' + '  "outlines": {\n'
text += ",\n".join(
    f'    "{name}": {{\n{fields(o, 6)}\n    }}' for name, o in outlines.items()
)
text += "\n  },\n" + '  "cases": [\n'
text += ",\n".join("    {\n" + fields(c, 6) + "\n    }" for c in cases)
text += "\n  ]\n}\n"
OUT.write_text(text)
print(
    f"wrote {len(cases)} cases on {len(outlines)} outlines to {OUT.relative_to(ROOT)}"
)
