"""Regenerate the cheek-landmark fixtures in dynaface-lib-python/tests_data.

The fixtures let the cheek tests run on real landmarks and on renders of a head
whose geometry is known, mostly without the AI models. This writes:

- pose-quarter-512.png: the desktop app's three-quarter sketch
  (dynaface-app/data/pose-quarter.png) at 512 px wide, so the library's tests never
  reach into the app.
- cheek/yaw0.png, yaw10.png, yaw45.png, yaw-45.png, yaw50.png, yaw45_hollow1.5.png,
  yaw45_hollow3.png and yaw45_hollow4.5.png: 800 px renders of a synthetic head
  (MediaPipe's canonical face model, Apache 2.0), flat-shaded on a flat background,
  turned by a known yaw; the last three have a submalar hollow of that depth in mm
  pressed into each cheek. Positive yaw turns the nose toward the image's right
  (the subject's left), which is SPIGA's sign; positive pitch is chin up.
  (cheek/yaw45_hair.png, yaw45.png with hair painted beside the far cheek, is
  written from the committed yaw45.png by scripts/make_cheek_hair_render.py, not
  here.)
- cheek/cheek_renders.json: for every render of the sweep, the eight above and the
  ones only analyzed in memory (the yaws 0 to 65 and the mirrored turns, the level
  head and the 45 degree turn pitched by -5, +5 and +10 degrees, the level head
  pitched 25 degrees, the 3 mm hollow at 50 degrees and a 3 mm malar bump at 45),
  its true yaw, pitch, hollow and bump, the mesh feature points projected into the
  render, and silhouette oracles read from the render's own mask (every pixel that
  is not the background color): the silhouette crossing of the level 0.3 IPD below
  the pupil line on each side, and on a turned render the far silhouette's
  chord-prominence maximum between the canthus and stomion levels, the deepest
  point below it, the nose's rows and the mesh's own maximum-projection apex.
- cheek_landmarks.json: what dynaface's current pipeline returns for the repository's
  test photos, the sketch (detected, and forced to Pose.QUARTER) and every render:
  the 98 raw landmarks and head pose found on the full image before any crop, the
  nose-offset and eye-width cues read from them, the pose classification under
  Pose.DETECT, and the crop's landmarks and scale: the frontal crop's on a frontal
  face, and on a quarter face the centered band crop's, with its far side, its
  chin-to-canthus distance H (cheek.scale) and its far-cheek outline's rows by
  source (the outline the background model, U^2-Net, traced; that entry alone runs
  it). Only the sketch forced to Pose.QUARTER is a quarter face here: quarter
  detection ships off (config.AUTO_QUARTER), so under Pose.DETECT the renders turned
  35 to 50 degrees are frontal faces and the ones turned further are profiles.

Regenerating needs the review folder's tooling, which is not in the repository:
the renderer synthetic_head/render_mesh.py and its data/canonical_face_model.obj
(--synthetic-head, default ~/projects/dynaface-review/synthetic_head). The
committed fixtures are sufficient for the tests; rerun this only when the renders,
the oracles or the pipeline's landmarks change on purpose. Every render parameter
is fixed and nothing is random, so two runs write identical files. Run it from the
repository root in dynaface-lib-python's venv (it needs the AI models):

    python scripts/make_cheek_fixtures.py [--device cpu]
"""

import argparse
import importlib
import json
import math
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from dynaface import cheek_outline, const, facial, image, models

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_DATA = ROOT / "dynaface-lib-python" / "tests_data"
SKETCH = ROOT / "dynaface-app" / "data" / "pose-quarter.png"
SKETCH_512 = "pose-quarter-512.png"
SYNTHETIC_HEAD = "~/projects/dynaface-review/synthetic_head"
MODEL_OBJ = os.path.join("data", "canonical_face_model.obj")

# The repository's test photos: a frontal face, a smile and two profiles.
PHOTOS = [
    ("img4", "img4-1024-frontal.jpg"),
    ("img1", "img1-512.jpg"),
    ("img2", "img2-1024-right-lateral.jpg"),
    ("img3", "img3-1024-left-lateral.jpg"),
]

# The render sweep: (true yaw in degrees, true pitch in degrees, hollow in mm, malar
# bump in mm). Positive yaw: the nose toward the image's right; positive pitch:
# chin up. The pitched heads are the definitions' pitch plan for the frontal core
# (0 to 10 degrees) and the oblique module's pitch -5/+5 readings, plus one gross
# pitch past cheek.PITCH_GROSS_RAW; the hollows at 1.5, 3 and 4.5 mm are the
# submalar depth series.
SWEEP: List[Tuple[float, float, float, float]] = [
    (float(yaw), 0.0, 0.0, 0.0)
    for yaw in (0, 5, 10, 15, 20, 25, 30, 35, 40, 42, 45, 48, 50, 55, 60, 65)
] + [
    (-15.0, 0.0, 0.0, 0.0),
    (-30.0, 0.0, 0.0, 0.0),
    (-45.0, 0.0, 0.0, 0.0),
    (-50.0, 0.0, 0.0, 0.0),
    (45.0, 0.0, 3.0, 0.0),
    (50.0, 0.0, 3.0, 0.0),
    (45.0, 0.0, 0.0, 3.0),
    (0.0, 5.0, 0.0, 0.0),
    (0.0, 10.0, 0.0, 0.0),
    (0.0, -5.0, 0.0, 0.0),
    (0.0, 25.0, 0.0, 0.0),
    (45.0, 5.0, 0.0, 0.0),
    (45.0, 10.0, 0.0, 0.0),
    (45.0, -5.0, 0.0, 0.0),
    (45.0, 0.0, 1.5, 0.0),
    (45.0, 0.0, 4.5, 0.0),
]
# The renders written as PNG files; the others are analyzed in memory only.
WRITTEN = (
    "yaw0",
    "yaw10",
    "yaw45",
    "yaw-45",
    "yaw50",
    "yaw45_hollow1.5",
    "yaw45_hollow3",
    "yaw45_hollow4.5",
)

# The renderer's view: 320 mm square at 0.4 mm per pixel is an 800 px render, the
# head centered 20 mm below the middle, flat-shaded on its flat background.
MM_PER_PX = 0.4
VIEW_MM = 320.0
VIEW_CENTER = (0.0, -20.0)
SUBDIV = 2  # Loop subdivisions of the canonical mesh
HEAD_CENTER = np.array([0.0, 0.0, -40.0])  # orients the mesh normals outward

ZYGION_LEVEL_IPD = 0.3  # the zygion level below the pupil line, in IPD units
SMOOTH_ROWS = 9  # box filter on the silhouette deviations before an extremum
PLATEAU_PX = 0.5  # rows whose prominence is within this of the maximum: its plateau
NOSE_HALF_WIDTH_MM = 30.0  # a visible surface nearer the midline than this is nose
NOSE_EDGE_PX = 1.0  # the nose within this of the silhouette makes a nose row
SIGN_CHECK_MIN_YAW = 10.0  # renders turned at least this far check SPIGA's sign

LM_NOSE_TIP, LM_CANTHUS_R, LM_CANTHUS_L = 54, 60, 72
LM_EYE_R, LM_EYE_L = (60, 64), (68, 72)  # outer and inner corners of each eye
LM_PUPIL_R, LM_PUPIL_L = 96, 97

# Mesh feature points projected into every render, by the renderer's names.
FEATURES = (
    "pupil_r",
    "pupil_l",
    "lat_canthus_r",
    "lat_canthus_l",
    "alar_base_r",
    "alar_base_l",
    "nose_tip",
    "subnasale",
    "chin",
    "commissure_r",
    "commissure_l",
)

PIXEL_FRAME_RENDERS = (
    "Every pixel coordinate is [column, row] in the raw render as written to the "
    "PNG: x to the right from the left edge, y down from the top edge, no crop, no "
    "flip, no straightening. Mask columns and rows are integer pixel indices; the "
    "mesh projections are the renderer's continuous coordinates. Rows increase "
    "downward, so h = (chin row - point row) / (chin row - canthus row)."
)
PIXEL_FRAME_LANDMARKS = (
    "Raw landmarks are [x, y] pixels of the full image as loaded, before any crop "
    "or flip (for a render, the raw render's own frame, the same as "
    "cheek/cheek_renders.json). Pipeline landmarks are pixels of the 1024 by 1024 "
    "crop that load_image(img, crop=True) made, with its scale: on a frontal face "
    "(pipeline.pose 'frontal') the pupil-anchored frontal crop, and on a quarter "
    "face (pipeline.pose 'quarter') the centered band crop, never flipped, with "
    "far_left (the far cheek is the subject's left), cheek_scale (H, the "
    "chin-to-canthus distance in crop pixels) and the far-cheek outline's rows by "
    "source (mask, contour, nose, edge) that the trace stored; a profile records "
    "its pose only. The frontal crop corrects the pupil distance for yaw only "
    "above +5 degrees, so a negative turn's crop is uncorrected and can leave "
    "landmarks outside it (yaw-30 and yaw-45): compare mirrored turns through the "
    "raw landmarks, never through the pipeline ones."
)


def round_to(value: Any, decimals: int) -> float:
    """A plain float rounded for the JSON."""
    return round(float(value), decimals)


def point2(p: Any) -> List[float]:
    """A pixel point as a two-decimal [x, y] list."""
    return [round_to(p[0], 2), round_to(p[1], 2)]


# ----------------------------------------------------------------------------- Scene


class Scene:
    """The synthetic head, as the design probes built it.

    MediaPipe's canonical face with its eye and mouth openings filled and the mesh
    Loop-subdivided, in millimeters with x toward the subject's left, y up and z
    anterior; optionally a Gaussian submalar hollow pressed into each cheek and a
    Gaussian malar bump pushed out of each cheek at Kaptein's height (0.793 of the
    chin-to-canthus height).

    Args:
        rm: The imported render_mesh module.
        model (str): Path of the canonical face model (.obj).
        hollow (float): Depth of the submalar hollow in mm; 0 for none.
        bump (float): Height of the malar bump in mm; 0 for none.
    """

    def __init__(
        self, rm: Any, model: str, hollow: float = 0.0, bump: float = 0.0
    ) -> None:
        self.rm = rm
        vertices, faces = rm.load_obj(model)
        vertices = vertices * 10.0  # cm -> mm
        if vertices[rm.MP["lat_canthus_l"], 0] < 0:
            vertices[:, 0] *= -1
        feats: Dict[str, NDArray[np.float64]] = {
            k: vertices[i].copy() for k, i in rm.MP.items()
        }
        feats["pupil_r"] = vertices[rm.EYE_RING_R].mean(axis=0)
        feats["pupil_l"] = vertices[rm.EYE_RING_L].mean(axis=0)
        mat = np.zeros(len(faces), dtype=int)
        vertices, faces, mat = rm.fill_holes(vertices, faces, mat)
        for _ in range(SUBDIV):
            vertices, faces, mat = rm.loop_subdivide(vertices, faces, mat)
        normals, _ = rm.vertex_normals(vertices, faces, HEAD_CENTER)
        if hollow > 0:
            # The generic face has no submalar hollow: press one into the submalar
            # triangle of each cheek, below the malar prominence.
            for sx in (1.0, -1.0):
                cx, cy, sxx, syy = sx * 42.0, -24.0, 11.0, 9.0
                w = np.exp(
                    -(
                        ((vertices[:, 0] - cx) ** 2) / (2 * sxx**2)
                        + ((vertices[:, 1] - cy) ** 2) / (2 * syy**2)
                    )
                )
                w[vertices[:, 2] < 10] = 0.0
                vertices = vertices - hollow * w[:, None] * normals
            normals, _ = rm.vertex_normals(vertices, faces, HEAD_CENTER)
        chin_y = feats["chin"][1]
        canthus_y = feats["lat_canthus_l"][1]
        self.bump_h = 0.793
        bump_y = chin_y + self.bump_h * (canthus_y - chin_y)
        if bump > 0:
            # A malar bump at Kaptein's height, so a detector has a prominence with
            # a known h.
            for sx in (1.0, -1.0):
                cx, cy, sxx, syy = sx * 50.0, bump_y, 10.0, 8.0
                w = np.exp(
                    -(
                        ((vertices[:, 0] - cx) ** 2) / (2 * sxx**2)
                        + ((vertices[:, 1] - cy) ** 2) / (2 * syy**2)
                    )
                )
                w[vertices[:, 2] < 5] = 0.0
                vertices = vertices + bump * w[:, None] * normals
            normals, _ = rm.vertex_normals(vertices, faces, HEAD_CENTER)
        self.vertices = vertices
        self.faces = faces
        self.mat = mat
        self.normals = normals
        self.feats = feats

    def pitched(
        self, pitch: float
    ) -> Tuple[
        NDArray[np.float64], NDArray[np.float64], Dict[str, NDArray[np.float64]]
    ]:
        """The vertices, normals and features rotated about x.

        Args:
            pitch (float): Degrees, positive chin up.

        Returns:
            The rotated vertices, vertex normals and feature points.
        """
        # The mesh has y up and z anterior; a positive rotation about x takes +y
        # toward +z, which brings the forehead forward and the chin back (chin
        # down), so chin up is the negative angle.
        a = -math.radians(pitch)
        rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(a), -math.sin(a)],
                [0.0, math.sin(a), math.cos(a)],
            ]
        )
        vertices = self.vertices @ rot.T
        normals = self.normals @ rot.T
        feats = {k: v @ rot.T for k, v in self.feats.items()}
        return vertices, normals, feats

    def render(
        self, yaw: float, pitch: float = 0.0
    ) -> Tuple[Any, NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        """Render the head turned by a yaw.

        Args:
            yaw (float): Degrees; positive turns the nose toward the image's right,
                the subject's left (the camera stands at the subject's right), which
                is the sign SPIGA reports.
            pitch (float): Degrees, positive chin up.

        Returns:
            The renderer's Render (image, mask, depth and project()), the vertices
            and the feature points as rendered.
        """
        vertices, normals, feats = self.pitched(pitch)
        view = self.rm.View(
            f"yaw{yaw:g}",
            "",
            yaw=-yaw,
            width_mm=VIEW_MM,
            height_mm=VIEW_MM,
            center=VIEW_CENTER,
        )
        render = self.rm.render_view(
            view, vertices, self.faces, self.mat, normals, MM_PER_PX
        )
        return render, vertices, feats


def render_name(yaw: float, pitch: float, hollow: float, bump: float) -> str:
    """The sweep entry's name, which is also its PNG's stem."""
    name = f"yaw{yaw:g}"
    if pitch:
        name += f"_pitch{pitch:g}"
    if hollow:
        name += f"_hollow{hollow:g}"
    if bump:
        name += f"_bump{bump:g}"
    return name


# ------------------------------------------------------------------- Silhouette oracles


def box_smooth(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """A SMOOTH_ROWS box filter on an edge-padded array, the length kept."""
    pad = SMOOTH_ROWS // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(SMOOTH_ROWS) / SMOOTH_ROWS, mode="valid")


def chord_deviation(
    outward: NDArray[np.float64], rows: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Each point's perpendicular deviation from the chord through the first and
    last, positive outward (toward larger outward values)."""
    a = np.array([outward[0], rows[0]], dtype=float)
    b = np.array([outward[-1], rows[-1]], dtype=float)
    chord = (b - a) / np.linalg.norm(b - a)
    normal = np.array([chord[1], -chord[0]])
    points = np.stack([outward, rows], axis=1).astype(float)
    result: NDArray[np.float64] = (points - a) @ normal
    return result


def far_silhouette(
    mask: NDArray[np.bool_], rows: NDArray[np.int64], image_right: bool
) -> NDArray[np.float64]:
    """The silhouette's column in each row: the outermost face pixel on the far side."""
    cols = np.empty(len(rows), dtype=float)
    for i, y in enumerate(rows):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            raise ValueError(f"no face pixel in row {y}")
        cols[i] = xs.max() if image_right else xs.min()
    return cols


def midline_mask(render: Any) -> NDArray[np.bool_]:
    """The pixels whose visible surface lies within NOSE_HALF_WIDTH_MM of the
    midsagittal plane, from the renderer's depth buffer: in the cheek band, the nose.
    Edge pixels covered less than half have no depth and are never midline."""
    right, up, forward = render.view.axes()
    cx, cy = render.view.center
    cols = np.arange(render.w)[None, :]
    rows = np.arange(render.h)[:, None]
    u = (cols - render.w / 2) * render.mm_per_px + cx
    v = -(rows - render.h / 2) * render.mm_per_px + cy
    finite = np.isfinite(render.depth)
    depth = np.where(finite, render.depth, 0.0)
    x_mesh = u * right[0] + v * up[0] + depth * forward[0]
    result: NDArray[np.bool_] = finite & (np.abs(x_mesh) < NOSE_HALF_WIDTH_MM)
    return result


def nose_edge_gaps(
    midline: NDArray[np.bool_],
    rows: NDArray[np.int64],
    cols: NDArray[np.float64],
    image_right: bool,
) -> NDArray[np.float64]:
    """How far inside the silhouette the visible nose reaches in each band row: the
    silhouette's column minus the outermost midline pixel's, outward positive; NaN
    where the row has no midline pixel."""
    sign = 1.0 if image_right else -1.0
    gaps = np.full(len(rows), np.nan)
    for i, y in enumerate(rows):
        xs = np.flatnonzero(midline[y])
        if xs.size:
            edge = xs.max() if image_right else xs.min()
            gaps[i] = sign * (cols[i] - edge)
    return gaps


def extremum(values: NDArray[np.float64]) -> int:
    """The index of the largest value outside the filter's halo at both ends."""
    halo = SMOOTH_ROWS // 2
    if len(values) <= 2 * halo + 1:
        raise ValueError("too few rows for an extremum")
    return halo + int(np.argmax(values[halo : len(values) - halo]))


def silhouette_oracle(
    rm: Any,
    yaw: float,
    render: Any,
    vertices: NDArray[np.float64],
    feats: Dict[str, NDArray[np.float64]],
    img: NDArray[np.uint8],
) -> Dict[str, Any]:
    """The oracles of one render, read from its own mask and the mesh.

    Args:
        rm: The render_mesh module (for find_apex and find_zygion).
        yaw (float): The render's true yaw.
        render: The renderer's Render.
        vertices: The mesh as rendered.
        feats: The feature points as rendered.
        img: The render's pixels.

    Returns:
        Dict[str, Any]: The projected features, the zygion-level crossings and, on
            a turned render, the far side's apex, submalar and nose oracles.
    """
    background = img[0, 0]
    mask = np.any(img != background, axis=2)
    height, width = mask.shape
    stomion = 0.5 * (feats["stomion_upper"] + feats["stomion_lower"])
    px = {k: np.array(render.project(feats[k])[:2]) for k in FEATURES}
    px["stomion"] = np.array(render.project(stomion)[:2])
    ipd = float(np.hypot(*(px["pupil_l"] - px["pupil_r"])))
    level_row = 0.5 * (px["pupil_r"][1] + px["pupil_l"][1]) + ZYGION_LEVEL_IPD * ipd
    zy_row = int(round(level_row))
    xs = np.flatnonzero(mask[zy_row])
    entry: Dict[str, Any] = {
        "size": [int(width), int(height)],
        "features": {k: point2(v) for k, v in px.items()},
        "ipd_px": round_to(ipd, 2),
        "zygion_level_row": round_to(level_row, 2),
        "zygion": {"right": [int(xs.min()), zy_row], "left": [int(xs.max()), zy_row]},
        "mesh_zygion": {
            side: point2(render.project(rm.find_zygion(vertices, feats, side[0])))
            for side in ("right", "left")
        },
        "far_side": None,
    }
    if yaw == 0:
        return entry
    far = "l" if yaw > 0 else "r"
    image_right = yaw > 0
    sign = 1.0 if image_right else -1.0
    canthus = px[f"lat_canthus_{far}"]
    chin = px["chin"]
    rows = np.arange(int(round(canthus[1])), int(round(px["stomion"][1])) + 1)
    cols = far_silhouette(mask, rows, image_right)
    outward = sign * cols
    scale = float(chin[1] - canthus[1])  # chin-to-canthus height, px

    def h_of(row: float) -> float:
        return (float(chin[1]) - float(row)) / scale

    prominence = box_smooth(chord_deviation(outward, rows.astype(float)))
    i_apex = extremum(prominence)
    # The generic mesh has no distinct malar prominence, so on a plain render that
    # maximum is the argmax of a flat plateau: the rows within PLATEAU_PX of it,
    # outside the filter's halo, say how flat.
    halo = SMOOTH_ROWS // 2
    inner = slice(halo, len(prominence) - halo)
    plateau = rows[inner][prominence[inner] >= prominence[i_apex] - PLATEAU_PX]
    # The nose rows: band rows down to the subnasale level whose silhouette pixel
    # is the visible nose, or within a pixel of it (the edge pixel has no depth).
    gaps = nose_edge_gaps(midline_mask(render), rows, cols, image_right)
    eligible = rows <= int(round(px["subnasale"][1]))
    with np.errstate(invalid="ignore"):
        nose_rows = [
            int(y) for y, g, ok in zip(rows, gaps, eligible) if ok and g <= NOSE_EDGE_PX
        ]
    edge_gap = (
        np.nanmin(gaps[eligible]) if np.any(np.isfinite(gaps[eligible])) else None
    )
    inward = box_smooth(-chord_deviation(outward[i_apex:], rows[i_apex:].astype(float)))
    i_sub = i_apex + extremum(inward)
    mesh_apex = render.project(rm.find_apex(vertices, feats, far, abs(yaw)))
    tip = px["nose_tip"]
    tip_row = int(round(tip[1]))
    tip_xs = np.flatnonzero(mask[tip_row])
    tip_edge = float(tip_xs.max() if image_right else tip_xs.min())
    entry.update(
        {
            "far_side": "left" if far == "l" else "right",
            "far_image_side": "right" if image_right else "left",
            "band": {
                "canthus_row": round_to(canthus[1], 2),
                "stomion_row": round_to(px["stomion"][1], 2),
                "chin_row": round_to(chin[1], 2),
                "rows": [int(rows[0]), int(rows[-1])],
                "scale_px": round_to(scale, 2),
            },
            "apex": {
                "point": [int(cols[i_apex]), int(rows[i_apex])],
                "h": round_to(h_of(rows[i_apex]), 4),
                "prominence_px": round_to(prominence[i_apex], 2),
                "prominence_mm": round_to(prominence[i_apex] * MM_PER_PX, 3),
                "in_nose_rows": bool(int(rows[i_apex]) in nose_rows),
                "plateau_rows": [int(plateau[0]), int(plateau[-1])],
                "plateau_row_count": int(len(plateau)),
                "plateau_h": [
                    round_to(h_of(plateau[-1]), 4),
                    round_to(h_of(plateau[0]), 4),
                ],
            },
            "submalar": {
                "point": [int(cols[i_sub]), int(rows[i_sub])],
                "h": round_to(h_of(rows[i_sub]), 4),
                "depth_px": round_to(inward[i_sub - i_apex], 2),
                "depth_mm": round_to(inward[i_sub - i_apex] * MM_PER_PX, 3),
                "in_nose_rows": bool(int(rows[i_sub]) in nose_rows),
            },
            "mesh_apex": {
                "point": point2(mesh_apex),
                "h": round_to(h_of(mesh_apex[1]), 4),
            },
            "nose": {
                "tip_h": round_to(h_of(tip[1]), 4),
                "subnasale_h": round_to(h_of(px["subnasale"][1]), 4),
                "tip_gap_px": round_to(sign * (tip_edge - tip[0]), 2),
                "edge_gap_px": None if edge_gap is None else round_to(edge_gap, 2),
                "rows": [nose_rows[0], nose_rows[-1]] if nose_rows else None,
                "row_count": len(nose_rows),
            },
        }
    )
    return entry


# --------------------------------------------------------------------- The pipeline


def pose_cues(landmarks: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """The two yaw cues the cheek module reads from raw landmarks.

    The nose offset is landmark 54's offset from the midpoint of the lateral canthi
    (60, 72) along the pupil axis (96 toward 97), in canthus spans: roll-corrected,
    positive toward the image's right. nose_offset_raw is the same cue on the image
    x axis alone, as the design probes measured it. The eye-width ratio is the
    narrower eye's corner-to-corner width over the wider one's.
    """
    lm = np.asarray(landmarks, dtype=float)
    across = lm[LM_PUPIL_L] - lm[LM_PUPIL_R]
    across = across / np.linalg.norm(across)
    mid = 0.5 * (lm[LM_CANTHUS_R] + lm[LM_CANTHUS_L])
    span = float((lm[LM_CANTHUS_L] - lm[LM_CANTHUS_R]) @ across)
    offset = float((lm[LM_NOSE_TIP] - mid) @ across) / span
    raw = (lm[LM_NOSE_TIP][0] - mid[0]) / (lm[LM_CANTHUS_L][0] - lm[LM_CANTHUS_R][0])
    width_r = float(np.linalg.norm(lm[LM_EYE_R[1]] - lm[LM_EYE_R[0]]))
    width_l = float(np.linalg.norm(lm[LM_EYE_L[1]] - lm[LM_EYE_L[0]]))
    return {
        "nose_offset": round_to(offset, 4),
        "nose_offset_raw": round_to(raw, 4),
        "eye_width_r": round_to(width_r, 2),
        "eye_width_l": round_to(width_l, 2),
        "eye_width_ratio": round_to(min(width_r, width_l) / max(width_r, width_l), 4),
    }


def head_pose(face: facial.AnalyzeFace) -> Dict[str, float]:
    """The face's yaw, pitch and roll, rounded."""
    return {
        "yaw": round_to(face.yaw, 2),
        "pitch": round_to(face.pitch, 2),
        "roll": round_to(face.roll, 2),
    }


def run_pipeline(img: NDArray[np.uint8], force_pose: const.Pose) -> Dict[str, Any]:
    """What dynaface's pipeline returns for one image.

    Args:
        img: The RGB image.
        force_pose (const.Pose): The pose passed to the cropping load_image():
            DETECT to classify, or a pose to force.

    Returns:
        Dict[str, Any]: The raw detection (load_image(img, crop=False) with the
            frontal pose forced, so nothing is flipped or cropped), the cues, and
            the pipeline's result: the pose, and on a frontal or a quarter face
            the crop's landmarks and scale, a quarter face adding its far side,
            H (cheek.scale) and its outline's rows by source (a profile records
            its pose only).
    """
    face = facial.AnalyzeFace(measures=[])
    face.load_image(img, crop=False, force_pose=const.Pose.FRONTAL)
    if len(face.landmarks) != 98:
        raise RuntimeError(f"{len(face.landmarks)} landmarks found; expected 98")
    entry: Dict[str, Any] = {
        "size": [int(img.shape[1]), int(img.shape[0])],
        "raw": {
            **head_pose(face),
            "landmarks": [[int(x), int(y)] for x, y in face.landmarks],
        },
        "cues": pose_cues(face.landmarks),
    }
    face = facial.AnalyzeFace(measures=[])
    face.load_image(img, crop=True, force_pose=force_pose)
    pipeline: Dict[str, Any] = {
        "force_pose": force_pose.value,
        "pose": face.pose.value,
        "lateral": bool(face.lateral),
        "flipped": bool(face.flipped),
    }
    if face.pose in (const.Pose.FRONTAL, const.Pose.QUARTER):
        pipeline.update(
            {
                "size": [int(face.width), int(face.height)],
                **head_pose(face),
                "pix2mm": round_to(face.pix2mm, 6),
                "pupillary_distance": round_to(face.pupillary_distance, 2),
                "face_rotation": (
                    None
                    if face.face_rotation is None
                    else round_to(face.face_rotation, 6)
                ),
                "landmarks": [[int(x), int(y)] for x, y in face.landmarks],
            }
        )
    if face.pose == const.Pose.QUARTER:
        pipeline.update(quarter_view(face))
    entry["pipeline"] = pipeline
    return entry


def quarter_view(face: facial.AnalyzeFace) -> Dict[str, Any]:
    """A quarter face's far side, its H and its stored outline's rows by source."""
    observed = face.oblique_cheek()
    rows = face.cheek_outline_y
    src = face.cheek_outline_src
    if observed is None or rows is None or src is None:
        raise RuntimeError("the quarter face has no far-cheek outline")
    sources = {
        "mask": cheek_outline.OUTLINE_SRC_MASK,
        "contour": cheek_outline.OUTLINE_SRC_CONTOUR,
        "nose": cheek_outline.OUTLINE_SRC_NOSE,
        "edge": cheek_outline.OUTLINE_SRC_EDGE,
    }
    return {
        "far_left": bool(face.far_left),
        "cheek_scale": round_to(observed.scale, 2),
        "outline": {
            "rows": [int(rows[0]), int(rows[-1])],
            **{
                f"{name}_rows": int(np.sum(src == code))
                for name, code in sources.items()
            },
        },
    }


def check_yaw_sign(entries: Dict[str, Dict[str, Any]]) -> str:
    """Check that SPIGA's yaw and the nose offset carry the render's yaw sign on
    every render turned at least SIGN_CHECK_MIN_YAW degrees, and describe it."""
    checked = []
    for name, entry in entries.items():
        render = entry.get("render")
        if render is None or abs(render["yaw"]) < SIGN_CHECK_MIN_YAW:
            continue
        true_sign = math.copysign(1.0, render["yaw"])
        spiga = entry["raw"]["yaw"]
        offset = entry["cues"]["nose_offset"]
        if (
            math.copysign(1.0, spiga) != true_sign
            or math.copysign(1.0, offset) != true_sign
        ):
            raise RuntimeError(
                f"{name}: true yaw {render['yaw']:+g} but SPIGA yaw {spiga:+.2f} and "
                f"nose offset {offset:+.4f}: the yaw sign does not match"
            )
        checked.append(f"{name} {spiga:+.1f}")
    return (
        "Positive yaw, true or as SPIGA reports it, means the nose points toward the "
        "image's right (the subject's left). Checked: SPIGA's yaw and the nose offset "
        "carry the true yaw's sign on " + ", ".join(checked) + "."
    )


# --------------------------------------------------------------------------- Output


def write_json(
    path: pathlib.Path, header: Dict[str, Any], key: str, entries: Dict[str, Any]
) -> None:
    """Write compact JSON with one entry per line, so the file stays small and
    its diffs readable."""

    def dumps(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("{\n")
        for k, v in header.items():
            f.write(f" {dumps(k)}: {dumps(v)},\n")
        f.write(f" {dumps(key)}: {{\n")
        names = list(entries)
        for i, name in enumerate(names):
            comma = "," if i < len(names) - 1 else ""
            f.write(f"  {dumps(name)}: {dumps(entries[name])}{comma}\n")
        f.write(" }\n}\n")


def make_sketch(out: pathlib.Path) -> None:
    """The app's three-quarter sketch at 512 px wide, the aspect kept."""
    sketch = Image.open(SKETCH).convert("RGB")
    if sketch.width != 512:
        size = (512, round(sketch.height * 512 / sketch.width))
        sketch = sketch.resize(size, Image.Resampling.LANCZOS)
    sketch.save(out / SKETCH_512, optimize=True)


def import_renderer(folder: str) -> Any:
    """Import render_mesh from the review folder's synthetic_head directory."""
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isfile(os.path.join(folder, "render_mesh.py")):
        raise FileNotFoundError(f"render_mesh.py not found in {folder}")
    sys.path.insert(0, folder)
    return importlib.import_module("render_mesh")


def size_of(path: pathlib.Path) -> str:
    """A file's size for the summary."""
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return f"{shown}: {path.stat().st_size:,} bytes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--synthetic-head",
        default=SYNTHETIC_HEAD,
        help=f"the review folder's synthetic_head directory (default {SYNTHETIC_HEAD})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="the ONNX device: auto picks the fastest, cpu avoids Core ML",
    )
    parser.add_argument(
        "--out",
        default=str(TESTS_DATA),
        help=f"the tests_data folder to write into (default {TESTS_DATA})",
    )
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    cheek_dir = out / "cheek"
    cheek_dir.mkdir(parents=True, exist_ok=True)
    rm = import_renderer(args.synthetic_head)
    model = os.path.join(
        os.path.abspath(os.path.expanduser(args.synthetic_head)), MODEL_OBJ
    )

    if args.device == "cpu":

        def cpu_only() -> str:
            return "cpu"

        models.detect_device = cpu_only
    device = models.detect_device()
    models.init_models(models.download_models(), device)
    print(f"models on {device}")

    make_sketch(out)
    sketch = image.load_image(str(out / SKETCH_512))

    # The renders, with their oracles, and dynaface on each of them.
    scenes: Dict[Tuple[float, float], Scene] = {}
    renders: Dict[str, Dict[str, Any]] = {}
    landmarks: Dict[str, Dict[str, Any]] = {}
    background = [int(c) for c in (rm.BG * 255).astype(np.uint8)]
    for yaw, pitch, hollow, bump in SWEEP:
        if (hollow, bump) not in scenes:
            scenes[(hollow, bump)] = Scene(rm, model, hollow=hollow, bump=bump)
        scene = scenes[(hollow, bump)]
        name = render_name(yaw, pitch, hollow, bump)
        render, vertices, feats = scene.render(yaw, pitch)
        img = np.asarray(render.image, dtype=np.uint8)
        if [int(c) for c in img[0, 0]] != background:
            raise RuntimeError(f"{name}: the corner pixel is not the background")
        params = {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "hollow_mm": hollow,
            "bump_mm": bump,
        }
        file: Optional[str] = None
        if name in WRITTEN:
            file = f"{name}.png"
            render.image.save(cheek_dir / file, optimize=True)
            if not np.array_equal(image.load_image(str(cheek_dir / file)), img):
                raise RuntimeError(f"{file} does not load back as rendered")
        oracle = silhouette_oracle(rm, yaw, render, vertices, feats, img)
        renders[name] = {"file": file, **params, **oracle}
        landmarks[name] = {"file": f"cheek/{file}" if file else None, "render": params}
        landmarks[name].update(run_pipeline(img, const.Pose.DETECT))
        print(
            f"{name:>14s}: SPIGA yaw {landmarks[name]['raw']['yaw']:+6.2f} "
            f"nose offset {landmarks[name]['cues']['nose_offset']:+.4f} "
            f"eye ratio {landmarks[name]['cues']['eye_width_ratio']:.4f} "
            f"-> {landmarks[name]['pipeline']['pose']}"
        )
    write_json(
        cheek_dir / "cheek_renders.json",
        {
            "about": (
                "Silhouette oracles of the synthetic-head renders in this folder and "
                "of the sweep renders only analyzed in memory. Written by "
                "scripts/make_cheek_fixtures.py; don't edit by hand."
            ),
            "pixel_frame": PIXEL_FRAME_RENDERS,
            "mm_per_px": MM_PER_PX,
            "background": background,
            "mask": "every pixel whose color is not the background color",
            "zygion_level_ipd": ZYGION_LEVEL_IPD,
            "smooth_rows": SMOOTH_ROWS,
            "plateau_px": PLATEAU_PX,
            "nose_half_width_mm": NOSE_HALF_WIDTH_MM,
            "nose_edge_px": NOSE_EDGE_PX,
            "apex": (
                "On the far side, the silhouette's greatest outward perpendicular "
                "deviation from the chord joining it at the far canthus row and at "
                "the stomion row, after the box filter; the filter's halo at both "
                "ends is excluded. The generic mesh has no distinct malar "
                "prominence, so on a plain render that maximum is the argmax of a "
                "flat plateau: plateau_rows (first, last), plateau_row_count and "
                "plateau_h (low, high) are the rows whose prominence lies within "
                "plateau_px of it, and the two mirrored turns' apex rows need not "
                "match. Only the hollow and bump renders carry a distinct apex, so "
                "assert apex.h against this oracle on those alone, and compare the "
                "two plain turns through the measure rather than through these "
                "rows. mesh_apex is the mesh's own maximum-projection apex "
                "(render_mesh.find_apex) projected, for information only."
            ),
            "submalar": (
                "Below the apex to the stomion row, the greatest inward deviation "
                "from the apex-to-stomion chord, after the same filter."
            ),
            "nose": (
                "tip_gap_px is the silhouette's column at the mesh nose tip's row "
                "minus the tip's column, outward positive (the tip inside the face). "
                "The visible nose is every pixel whose surface, from the depth "
                "buffer, lies within nose_half_width_mm of the midsagittal plane; "
                "edge_gap_px is the least distance it keeps inside the silhouette "
                "over the band rows down to the subnasale level, and rows (first, "
                "last, count) are the rows there where it comes within nose_edge_px "
                "of the silhouette: the nose on the outline."
            ),
        },
        "renders",
        renders,
    )

    # The photos and the sketch.
    photos: Dict[str, Dict[str, Any]] = {}
    for name, file in PHOTOS:
        img = image.load_image(str(TESTS_DATA / file))
        photos[name] = {
            "file": file,
            "render": None,
            **run_pipeline(img, const.Pose.DETECT),
        }
        print(
            f"{name:>14s}: SPIGA yaw {photos[name]['raw']['yaw']:+6.2f} -> {photos[name]['pipeline']['pose']}"
        )
    for name, force_pose in (
        ("quarter_sketch", const.Pose.DETECT),
        ("quarter_sketch_forced", const.Pose.QUARTER),
    ):
        photos[name] = {
            "file": SKETCH_512,
            "render": None,
            **run_pipeline(sketch, force_pose),
        }
        print(
            f"{name:>14s}: SPIGA yaw {photos[name]['raw']['yaw']:+6.2f} -> {photos[name]['pipeline']['pose']}"
        )
    entries = {**photos, **landmarks}
    sign_check = check_yaw_sign(entries)
    print(sign_check)
    write_json(
        out / "cheek_landmarks.json",
        {
            "about": (
                "What dynaface's pipeline returns for the test photos, the "
                "three-quarter sketch and the synthetic-head renders (the written "
                "ones in cheek/, the rest rendered in memory), so the cheek tests "
                "run on real landmarks without the models. Written by "
                "scripts/make_cheek_fixtures.py; don't edit by hand."
            ),
            "pixel_frame": PIXEL_FRAME_LANDMARKS,
            "pd_mm": float(facial.AnalyzeFace.pd),
            "yaw_sign": sign_check,
            "cues": (
                "nose_offset: landmark 54's offset from the lateral canthi's midpoint "
                "along the pupil axis, in canthus spans, roll-corrected, positive "
                "toward the image's right; nose_offset_raw: the same on the image x "
                "axis alone, as the design probes measured it; eye_width_ratio: the "
                "narrower eye's corner-to-corner width over the wider one's. All from "
                "the raw landmarks."
            ),
        },
        "images",
        entries,
    )

    print("\nwrote")
    for path in [
        out / SKETCH_512,
        out / "cheek_landmarks.json",
        cheek_dir / "cheek_renders.json",
    ] + [cheek_dir / f"{name}.png" for name in WRITTEN]:
        print("  " + size_of(path))


if __name__ == "__main__":
    main()
