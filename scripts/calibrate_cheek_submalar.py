"""Calibrate the submalar depth threshold of the cheek landmarks.

The Cheek Measures measure (dynaface.measures_cheek) reads the far cheek of a
three-quarter face from the outline the background model traced
(dynaface.cheek_outline): the malar apex, and below it the submalar point, the
deepest inward deviation of the outline from the apex-to-stomion chord as a
fraction of H, the chin-to-canthus distance. A hollow is "present"
(sub.*.present) when that depth reaches cheek_outline.SUBMALAR_MIN_DEPTH. The
threshold has two anchors, and this script measures both:

- the synthetic renders in dynaface-lib-python/tests_data/cheek (the 45 and 50
  degree turns, the mirrored turn, the hair variant and the heads with a 1.5, 3
  and 4.5 mm hollow pressed into each cheek): what each one's depth reads, so a
  known hollow's reading and a plain head's spread stay on record;
- real photos of both turns, which are never committed: the no-hollow floor of
  the method itself. Every photo is read five times when --perturb is given
  (as is, resized to 0.97 and 1.03 of its size, and shifted 4 px left and
  right), and the spread of the depth over those readings is the noise the
  threshold must clear by a margin: on the calibration on record (below) the
  shipped 0.003 H is 2.7 to 3.6 times the no-hollow floor by statistic (the
  90th percentile of that spread on faces without a hollow, 0.0008; the worst
  reading of a convex cheek, -0.0011; the spread over all faces, 0.0015), and
  the comment on cheek_outline.SUBMALAR_MIN_DEPTH states it. The mirrored pairs
  (the same subject turned each way) show how far a face's two cheeks read
  apart, and the yaw distribution shows how many photos fall inside the apex
  gate at all.

Beside the depth it records, per image, every output of the measure and the cheek
landmarks' coordinates (the Landmarks measure's Cheek Landmarks outputs), the
outline's rows by source, and an edge-fidelity check: on every mask row of the apex
window, where the intensity edge (the tracer's stand-in source for rows the mask has
no edge on) would have fallen, against the mask's edge, so the stand-in's accuracy
on real skin is known before it is ever trusted for a gate.

Rerun it after any change to cheek_outline.SMOOTH_ROWS, the chord rule or the apex
rule, and read the floor off summary.md again. Every image goes through the
forced quarter path (Pose.QUARTER) with the background model, so the AI models are
needed; run it from the repository root in dynaface-lib-python's venv. The photo
folders are given on the command line (one folder per set; the set is named after
the folder) and stay outside the repository. The calibration on record used the
Face Research Lab London set (CC BY): the neutral and smiling three-quarter
photos of both turns, 102 subjects each, as

    python -I scripts/calibrate_cheek_submalar.py --device cpu \\
        --photos neutral_left_3quarter neutral_right_3quarter \\
                 smiling_left_3quarter smiling_right_3quarter \\
        --out calib --overlays --perturb 20

A folder named <expression>_<turn>_3quarter is parsed into the expression and the
turn; "left" means the head turned to the subject's left, so the far cheek is the
subject's right and the .r outputs carry the apex. Results go to <out>/results.jsonl
(one line per reading), <out>/summary.md, and with --overlays the measure's own
drawing of every image under <out>/overlays/<set>/ plus a contact sheet of the
first three subjects of each set.
"""

import argparse
import json
import logging
import math
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from dynaface import cheek, cheek_outline, const, facial, image, models
from dynaface.measures_cheek import AnalyzeCheekMeasures

ROOT = pathlib.Path(__file__).resolve().parent.parent
RENDERS = ROOT / "dynaface-lib-python" / "tests_data" / "cheek"
FRONTAL_RENDERS = ("yaw0.png", "yaw10.png")  # level or nearly level heads
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png")
OUTPUT_PREFIXES = ("zy.", "cr.", "sr.", "apex.", "sub.", "cheek.")
SOURCES = ("contour", "mask", "nose", "edge")  # OUTLINE_SRC_* in order
# The readings of one photo: as is, and the four perturbations of --perturb.
READINGS = ("plain", "scale0.97", "scale1.03", "shift-4", "shift+4")
SHIFT_PX = 4
SCALES = {"scale0.97": 0.97, "scale1.03": 1.03}
# A photo whose median depth over its readings is below this (H) is listed in
# the perturbation table's "flat" subset: a cut on the measured depth itself,
# so it selects on the outcome, and the by-expression split stands beside it.
FLAT_DEPTH_MEDIAN = 0.002
# The yaw window of a quarter face, |cheek.yaw| in degrees, and the apex gate,
# |cheek.yaw| within OBLIQUE_YAW_GATE of OBLIQUE_YAW; both rates are reported.
YAW_WINDOW = cheek.QUARTER_YAW_RANGE
APEX_WINDOW = cheek_outline.APEX_H_WINDOW
EDGE_NEAR_PX = 2.0  # an intensity edge within this of the mask's edge agrees
OVERLAY_LONG_SIDE = 900
CONTACT_TILE_WIDTH = 450
CONTACT_PER_SET = 3
JPEG_QUALITY = 85

logger = logging.getLogger("calibrate_cheek_submalar")


def parse_set(name: str) -> Tuple[str, str]:
    """The expression and the turn of a photo folder named like
    neutral_left_3quarter; "unknown" for a token that isn't there."""
    tokens = name.lower().split("_")
    expression = next((t for t in tokens if t in ("neutral", "smiling")), "unknown")
    turn = next((t for t in tokens if t in ("left", "right")), "unknown")
    return expression, turn


def perturbed(rgb: NDArray[np.uint8], reading: str) -> NDArray[np.uint8]:
    """The image of one reading: as is, resized (INTER_AREA), or shifted
    SHIFT_PX left or right with the edge column replicated."""
    if reading == "plain":
        return rgb
    if reading in SCALES:
        s = SCALES[reading]
        size = (round(rgb.shape[1] * s), round(rgb.shape[0] * s))
        return np.asarray(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))
    n = SHIFT_PX
    if reading == "shift-4":  # the content moves left
        fill = np.repeat(rgb[:, -1:], n, axis=1)
        return np.ascontiguousarray(np.concatenate([rgb[:, n:], fill], axis=1))
    if reading == "shift+4":  # the content moves right
        fill = np.repeat(rgb[:, :1], n, axis=1)
        return np.ascontiguousarray(np.concatenate([fill, rgb[:, :-n]], axis=1))
    raise ValueError(reading)


def jsonable(value: Any) -> Any:
    """Plain JSON: NaN and infinities as null, numpy scalars and arrays as
    Python numbers and lists."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def edge_fidelity(
    face: facial.AnalyzeFace,
    gray: NDArray[np.uint8],
    h_rows: Optional[NDArray[np.float64]],
) -> Dict[str, Any]:
    """
    Where the intensity edge would fall on every mask row of the apex window,
    against the mask's own edge, through the tracer's edge search (its own
    helper, on those rows, with the walk limit the tracer used).

    Returns the number of window mask rows, how many gave an edge, the median
    and 90th percentile of |x_edge - x_mask| in px, the fraction within
    EDGE_NEAR_PX, the fraction of rows where no edge reached EDGE_GRAD_MIN, and
    the differences themselves (rounded) for pooling.
    """
    empty = {
        "n_window_mask": 0,
        "n_compared": 0,
        "median_px": math.nan,
        "p90_px": math.nan,
        "within_2px": math.nan,
        "no_edge": math.nan,
        "diffs_px": [],
    }
    xs_out, ys_out, src_out = (
        face.cheek_outline_x,
        face.cheek_outline_y,
        face.cheek_outline_src,
    )
    if xs_out is None or ys_out is None or src_out is None or h_rows is None:
        return empty
    pts = cheek_outline.as_landmarks(face.landmarks)
    if pts is None:
        return empty
    frame = cheek_outline.oblique_frame(pts)
    if frame is None:
        return empty
    rows, seed = cheek_outline._band_rows(pts, face.far_left)
    if len(rows) != len(ys_out) or not np.array_equal(rows, ys_out):
        logger.warning("The outline's rows are not the band's rows; no edge check")
        return empty
    chin = pts[cheek_outline.LM_CHIN]
    canthus = pts[cheek_outline.FAR_CANTHUS[face.far_left]]
    span = float((chin - canthus) @ frame.down)
    limit = int(cheek_outline.OUTLINE_TOL * span) if span > 0.0 else 0
    plain_span = float(chin[1] - canthus[1])
    if plain_span > 0.0:
        h_plain = (float(chin[1]) - rows.astype(np.float64)) / plain_span
    else:
        h_plain = np.full(len(rows), math.inf, dtype=np.float64)
    window = (
        (src_out == cheek_outline.OUTLINE_SRC_MASK)
        & (h_rows >= APEX_WINDOW[0])
        & (h_rows <= APEX_WINDOW[1])
    )
    idx = [int(i) for i in np.flatnonzero(window)]
    if not idx:
        return empty
    xs = np.array(seed, dtype=np.float64)
    src = np.full(len(rows), cheek_outline.OUTLINE_SRC_CONTOUR, dtype=np.uint8)
    cheek_outline._edge_rows(gray, rows, seed, xs, src, idx, limit, h_plain)
    found = [i for i in idx if src[i] == cheek_outline.OUTLINE_SRC_EDGE]
    diffs = np.abs(xs[found] - np.asarray(xs_out, dtype=np.float64)[found])
    result = dict(empty)
    result["n_window_mask"] = len(idx)
    result["n_compared"] = len(found)
    result["no_edge"] = 1.0 - len(found) / len(idx)
    result["diffs_px"] = [round(float(d), 2) for d in diffs]
    if len(found):
        result["median_px"] = float(np.median(diffs))
        result["p90_px"] = float(np.percentile(diffs, 90))
        result["within_2px"] = float(np.mean(diffs <= EDGE_NEAR_PX))
    return result


def outline_heights(face: facial.AnalyzeFace) -> Optional[NDArray[np.float64]]:
    """h of every outline row, through the levels, as the search reads it."""
    xs, ys = face.cheek_outline_x, face.cheek_outline_y
    if xs is None or ys is None:
        return None
    pts = cheek_outline.as_landmarks(face.landmarks)
    if pts is None:
        return None
    frame = cheek_outline.oblique_frame(pts)
    if frame is None:
        return None
    points = np.stack(
        [np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)], axis=1
    )
    chin = pts[cheek_outline.LM_CHIN]
    canthus = pts[cheek_outline.FAR_CANTHUS[face.far_left]]
    h, _ = cheek_outline._height_ratios(points, chin, canthus, frame)
    return np.asarray(h, dtype=np.float64)


def measure(rgb: NDArray[np.uint8]) -> Tuple[Dict[str, Any], Optional[NDArray[Any]]]:
    """
    One reading: a fresh face with the Cheek Measures measure alone, the forced
    quarter path, analyze(), and everything the record keeps, with the cheek
    landmarks' coordinates (the Landmarks measure's Cheek Landmarks outputs)
    from face.cheek_landmarks; the drawn image goes along when the face was
    found.
    """
    face = facial.AnalyzeFace(measures=[AnalyzeCheekMeasures()])
    started = time.perf_counter()
    found = face.load_image(rgb, force_pose=const.Pose.QUARTER)
    if not found:
        return {"ok": False, "seconds": time.perf_counter() - started}, None
    gray = cheek_outline.gray_image(face.render_img)  # before the drawing
    outputs = face.analyze() or {}
    seconds = time.perf_counter() - started
    data = {
        k: float(v)
        for k, v in outputs.items()
        if k.startswith(OUTPUT_PREFIXES) and isinstance(v, (int, float))
    }
    # The coordinates, as the Landmarks measure's Cheek Landmarks item reports
    # them, without that measure's drawing of the 98 landmarks on the overlay.
    for name, value in zip(
        cheek.CHEEK_LANDMARK_OUTPUTS, face.cheek_landmarks.reshape(-1)
    ):
        data[name] = float(value)
    side = "l" if face.far_left else "r"
    src = face.cheek_outline_src
    counts = {name: 0 for name in SOURCES}
    window_counts = {name: 0 for name in SOURCES}
    h_rows = outline_heights(face)
    if src is not None:
        for code, name in enumerate(SOURCES):
            counts[name] = int(np.sum(src == code))
            if h_rows is not None:
                in_window = (h_rows >= APEX_WINDOW[0]) & (h_rows <= APEX_WINDOW[1])
                window_counts[name] = int(np.sum((src == code) & in_window))
    ob = face.oblique_cheek()
    yaw = data.get("cheek.yaw", math.nan)
    yaw_model = data.get("cheek.yaw.model", math.nan)
    record: Dict[str, Any] = {
        "ok": True,
        "seconds": seconds,
        "far_left": bool(face.far_left),
        "side": side,
        "crop_hw": [int(face.render_img.shape[0]), int(face.render_img.shape[1])],
        "pix2mm": float(face.pix2mm),
        "pd_mm": float(face.pd),
        "pupil_px": float(face.pupillary_distance),
        "outputs": data,
        "rows": int(0 if src is None else len(src)),
        "rows_by_source": counts,
        "window_rows_by_source": window_counts,
        "H": float(ob.scale) if ob is not None else math.nan,
        "window_mask": float(ob.window_mask) if ob is not None else math.nan,
        "below_mask": float(ob.below_mask) if ob is not None else math.nan,
        "in_window": bool(YAW_WINDOW[0] <= abs(yaw) <= YAW_WINDOW[1]),
        "in_gate": bool(abs(abs(yaw) - cheek.OBLIQUE_YAW) <= cheek.OBLIQUE_YAW_GATE),
        "agree": bool(abs(yaw - yaw_model) <= cheek.YAW_AGREE_DEG),
        # The far side's values, so a reader needn't pick .r or .l.
        "apex_h": data.get(f"apex.{side}.h", math.nan),
        "apex_dev": data.get(f"apex.{side}.dev", math.nan),
        "apex_prior": data.get(f"apex.{side}.prior", math.nan),
        "apex_near_nose": data.get(f"apex.{side}.near_nose", math.nan),
        "sub_h": data.get(f"sub.{side}.h", math.nan),
        "depth": data.get(f"sub.{side}.depth", math.nan),
        "depth_px": data.get(f"sub.{side}.depth.px", math.nan),
        "present": data.get(f"sub.{side}.present", math.nan),
        "edge": edge_fidelity(face, gray, h_rows),
    }
    return record, face.render_img


def save_overlay(render_rgb: NDArray[Any], path: pathlib.Path) -> None:
    """The drawn image as a JPEG whose longer side is OVERLAY_LONG_SIDE px."""
    h, w = render_rgb.shape[:2]
    s = OVERLAY_LONG_SIDE / max(h, w)
    small = cv2.resize(
        render_rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )


def list_images(folder: pathlib.Path, limit: Optional[int]) -> List[pathlib.Path]:
    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES
    )
    return files[:limit] if limit else files


def run_image(
    path: pathlib.Path,
    set_name: str,
    subject: str,
    expression: str,
    turn: str,
    readings: Sequence[str],
    out: pathlib.Path,
    overlays: bool,
    results: Any,
) -> None:
    rgb = image.load_image(str(path))
    for reading in readings:
        record, drawn = measure(perturbed(rgb, reading))
        record = {
            "file": path.name,
            "set": set_name,
            "subject": subject,
            "expression": expression,
            "turn": turn,
            "reading": reading,
            **record,
        }
        results.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
        results.flush()
        if overlays and reading == "plain" and drawn is not None:
            save_overlay(drawn, out / "overlays" / set_name / f"{path.stem}.jpg")
        if record["ok"]:
            print(
                f"{set_name}/{path.name} {reading}: apex {record['apex_h']:.3f}h "
                f"depth {record['depth']:.4f} ({record['depth_px']:.2f} px) "
                f"yaw {record['outputs'].get('cheek.yaw', math.nan):.1f} "
                f"{record['seconds']:.2f}s",
                flush=True,
            )
        else:
            print(f"{set_name}/{path.name} {reading}: no face", flush=True)


# --- the summary -------------------------------------------------------------


def _num(value: Any) -> float:
    """A JSON number as a float; null (a NaN written out) as NaN."""
    return math.nan if value is None else float(value)


def _below(value: Any, threshold: float) -> float:
    """1.0 / 0.0 whether value is below threshold, NaN when unknown."""
    v = _num(value)
    return math.nan if not math.isfinite(v) else float(v < threshold)


def _finite(values: Sequence[Any]) -> NDArray[np.float64]:
    a = np.array([math.nan if v is None else float(v) for v in values], dtype=float)
    return a[np.isfinite(a)]


def _f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    v = float(value)
    return "-" if not math.isfinite(v) else f"{v:.{digits}f}"


def _rate(flags: Sequence[Any]) -> str:
    a = _finite(flags)
    return "-" if len(a) == 0 else f"{a.mean():.3f}"


def _pct(a: NDArray[np.float64], q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else math.nan


def _median(values: Sequence[Any]) -> float:
    a = _finite(values)
    return float(np.median(a)) if len(a) else math.nan


def _stats(values: Sequence[Any], digits: int = 2) -> str:
    """mean / sd / min / max."""
    a = _finite(values)
    if len(a) == 0:
        return "- / - / - / -"
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return f"{_f(a.mean(), digits)} / {_f(sd, digits)} / {_f(a.min(), digits)} / {_f(a.max(), digits)}"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def render_table(records: List[Dict[str, Any]]) -> str:
    rows = []
    for r in records:
        if not r["ok"]:
            rows.append([r["subject"], "no face"] + ["-"] * 16)
            continue
        o = r["outputs"]
        src = r["rows_by_source"]
        rows.append(
            [
                r["subject"],
                _f(o.get("cheek.yaw"), 1),
                _f(o.get("cheek.yaw.model"), 1),
                "left" if r["far_left"] else "right",
                f"c{src['contour']} m{src['mask']} n{src['nose']} e{src['edge']}",
                _f(o.get("cheek.mask"), 3),
                _f(r["apex_h"], 3),
                _f(r["apex_dev"], 4),
                _f(r["apex_prior"], 0),
                _f(r["apex_near_nose"], 0),
                _f(o.get("apex.ok"), 0),
                _f(r["depth"], 4),
                _f(r["depth_px"], 2),
                _f(r["present"], 0),
                _f(o.get("sub.ok"), 0),
                _f(o.get("cheek.nose"), 0),
                _f(o.get("cheek.nose.gap"), 3),
            ]
        )
    header = [
        "render",
        "cheek.yaw",
        "model yaw",
        "far side",
        "rows by source",
        "cheek.mask",
        "apex h",
        "apex dev",
        "prior",
        "near_nose",
        "apex.ok",
        "depth (H)",
        "depth px",
        "present",
        "sub.ok",
        "cheek.nose",
        "nose gap",
    ]
    return _table(header, rows)


def set_tables(by_set: Dict[str, List[Dict[str, Any]]]) -> str:
    """The per-set tables of the plain readings."""
    rate_rows, yaw_rows, apex_rows, depth_rows, edge_rows = [], [], [], [], []
    for name, recs in by_set.items():
        ok = [r for r in recs if r["ok"]]
        o = [r["outputs"] for r in ok]
        rate_rows.append(
            [
                name,
                len(recs),
                len(ok),
                _rate([r["in_window"] for r in ok]),
                _rate([r["in_gate"] for r in ok]),
                _rate([r["agree"] for r in ok]),
                _rate([x.get("apex.ok") for x in o]),
                _rate([x.get("sub.ok") for x in o]),
                _rate([r["apex_prior"] for r in ok]),
                _rate([r["apex_near_nose"] for r in ok]),
                _rate([x.get("cheek.nose") for x in o]),
                _rate([_below(x.get("cheek.mask"), 0.8) for x in o]),
                _rate([_below(r["window_mask"], 0.8) for r in ok]),
                _rate([r["rows_by_source"]["edge"] > 0 for r in ok]),
                _rate([r["window_rows_by_source"]["contour"] > 0 for r in ok]),
                _rate([r["window_rows_by_source"]["edge"] > 0 for r in ok]),
            ]
        )
        yaw_rows.append(
            [
                name,
                _stats([x.get("cheek.yaw") for x in o], 1),
                _stats([x.get("cheek.yaw.model") for x in o], 1),
                _stats([x.get("cheek.pitch") for x in o], 1),
            ]
        )
        observed = _finite([r["apex_h"] for r in ok if r["apex_prior"] == 0.0])
        apex_rows.append(
            [
                name,
                len(observed),
                _f(np.median(observed) if len(observed) else None, 3),
                f"{_f(_pct(observed, 25), 3)} - {_f(_pct(observed, 75), 3)}",
                _f(observed.min() if len(observed) else None, 3),
                _f(observed.max() if len(observed) else None, 3),
                _rate([APEX_WINDOW[0] <= h <= APEX_WINDOW[1] for h in observed]),
                _f(_median([r["apex_dev"] for r in ok if r["apex_prior"] == 0.0]), 4),
            ]
        )
        depth = _finite([r["depth"] for r in ok])
        depth_px = _finite([r["depth_px"] for r in ok])
        for label, a, digits in (("H", depth, 4), ("px", depth_px, 2)):
            depth_rows.append(
                [name, label, len(a)]
                + [_f(_pct(a, q), digits) for q in (5, 25, 50, 75, 90, 95)]
                + [_f(a.max() if len(a) else None, digits)]
                + [_rate([r["present"] for r in ok]) if label == "H" else ""]
            )
        diffs = np.concatenate(
            [np.asarray(r["edge"]["diffs_px"], dtype=float) for r in ok]
            if ok
            else [np.zeros(0)]
        )
        n_window = sum(r["edge"]["n_window_mask"] for r in ok)
        n_compared = sum(r["edge"]["n_compared"] for r in ok)
        edge_rows.append(
            [
                name,
                n_window,
                n_compared,
                _f(np.median(diffs) if len(diffs) else None, 2),
                _f(_pct(diffs, 90), 2),
                _rate(list(diffs <= EDGE_NEAR_PX)),
                _f(1.0 - n_compared / n_window if n_window else None, 3),
                _f(_median([r["edge"]["median_px"] for r in ok]), 2),
            ]
        )
    text = "Gates and rows (rates over the faces found):\n\n"
    text += _table(
        [
            "set",
            "N",
            "found",
            "in_window",
            "in_gate",
            "agree",
            "apex.ok",
            "sub.ok",
            "prior",
            "near_nose",
            "cheek.nose",
            "mask<0.8",
            "window mask<0.8",
            "any EDGE row",
            "contour row in window",
            "EDGE row in window",
        ],
        rate_rows,
    )
    text += "\nYaw and pitch (mean / sd / min / max, degrees):\n\n"
    text += _table(["set", "cheek.yaw", "cheek.yaw.model", "cheek.pitch"], yaw_rows)
    text += "\nObserved apexes (prior 0): apex h:\n\n"
    text += _table(
        [
            "set",
            "n observed",
            "median",
            "IQR",
            "min",
            "max",
            "inside [0.71, 0.83]",
            "median dev",
        ],
        apex_rows,
    )
    text += "\nSubmalar depth percentiles (as a fraction of H, and in px):\n\n"
    text += _table(
        ["set", "unit", "n", "p5", "p25", "p50", "p75", "p90", "p95", "max", "present"],
        depth_rows,
    )
    text += "\nEdge fidelity on the mask rows of the apex window (px, pooled):\n\n"
    text += _table(
        [
            "set",
            "window mask rows",
            "with an edge",
            "median",
            "p90",
            "within 2 px",
            "no edge",
            "median of per-photo medians",
        ],
        edge_rows,
    )
    return text


def pair_table(records: List[Dict[str, Any]]) -> str:
    """The mirrored pairs: each subject and expression with both turns found."""
    plain = [r for r in records if r["reading"] == "plain" and r["ok"]]
    by_key: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for r in plain:
        if r["turn"] in ("left", "right"):
            by_key.setdefault((r["subject"], r["expression"]), {})[r["turn"]] = r
    rows_out = []
    for expression in sorted({k[1] for k in by_key}):
        yaw_diff, depth_diff, depth_px_diff, apex_diff = [], [], [], []
        n_pairs = 0
        for (subject, expr), turns in by_key.items():
            if expr != expression or "left" not in turns or "right" not in turns:
                continue
            n_pairs += 1
            lt, rt = turns["left"], turns["right"]
            yl = _num(lt["outputs"].get("cheek.yaw"))
            yr = _num(rt["outputs"].get("cheek.yaw"))
            yaw_diff.append(abs(yl) - abs(yr))
            depth_diff.append(_num(lt["depth"]) - _num(rt["depth"]))
            depth_px_diff.append(_num(lt["depth_px"]) - _num(rt["depth_px"]))
            apex_diff.append(_num(lt["apex_h"]) - _num(rt["apex_h"]))
        for label, values, digits in (
            ("abs yaw left - right (deg)", yaw_diff, 1),
            ("depth left - right (H)", depth_diff, 4),
            ("depth left - right (px)", depth_px_diff, 2),
            ("apex h left - right", apex_diff, 3),
        ):
            a = _finite(values)
            sd = float(a.std(ddof=1)) if len(a) > 1 else math.nan
            rows_out.append(
                [
                    expression,
                    n_pairs,
                    label,
                    len(a),
                    _f(a.mean(), digits) if len(a) else "-",
                ]
                + [_f(sd, digits)]
                + [_f(_pct(a, q), digits) for q in (5, 25, 50, 75, 95)]
                + [_f(np.median(np.abs(a)) if len(a) else None, digits)]
            )
    return _table(
        [
            "expression",
            "pairs",
            "difference",
            "n",
            "mean",
            "sd",
            "p5",
            "p25",
            "p50",
            "p75",
            "p95",
            "median abs diff",
        ],
        rows_out,
    )


def perturbation_table(records: List[Dict[str, Any]]) -> str:
    """Per photo, the spread of the depth and of apex h over its readings;
    pooled, the median and 90th percentile of the per-photo std and range.

    The depth rows count a photo only when every reading has a finite depth (a
    reading whose depth is NaN is not noise but a gate); the apex h rows count
    every photo with all its readings. A change in sub.*.present is counted as
    a 0/1 flip or as a reading whose depth became NaN, separately. The depth
    spread is also split by expression (the smiling sets, where almost no hollow
    is read, are the nearest thing to a no-hollow population) and by the photo's
    median depth over its readings (below FLAT_DEPTH_MEDIAN, a cut on the
    measured depth itself), each labeled by its selection rule."""
    by_photo: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for r in records:
        by_photo.setdefault((r["set"], r["file"]), {})[r["reading"]] = r
    per_photo = []
    for (set_name, file), readings in by_photo.items():
        if len(readings) < len(READINGS):
            continue
        if not all(r["ok"] for r in readings.values()):
            continue
        row: Dict[str, Any] = {"set": set_name, "file": file}
        row["expression"] = next(iter(readings.values()))["expression"]
        depths = _finite([r["depth"] for r in readings.values()])
        row["depth_all_finite"] = len(depths) == len(READINGS)
        row["depth_median"] = float(np.median(depths)) if len(depths) else math.nan
        for key in ("depth", "depth_px", "apex_h"):
            a = _finite([r[key] for r in readings.values()])
            complete = len(a) == len(READINGS)
            row[key + "_std"] = float(a.std(ddof=1)) if complete else math.nan
            row[key + "_range"] = float(a.max() - a.min()) if complete else math.nan
        row["prior_changes"] = len({r["apex_prior"] for r in readings.values()}) > 1
        present = [r["present"] for r in readings.values()]
        row["present_flip"] = len({p for p in present if p is not None}) > 1
        row["present_nan"] = any(p is None for p in present) and any(
            p is not None for p in present
        )
        per_photo.append(row)
    if not per_photo:
        return "No photo has all five readings.\n"
    finite = [p for p in per_photo if p["depth_all_finite"]]
    text = (
        f"{len(per_photo)} photos with all {len(READINGS)} readings found, "
        f"{len(finite)} of them with a finite depth on every reading (the depth "
        f"rows below); the apex h rows count all {len(per_photo)}.\n\n"
    )

    def pooled(
        label: str, key: str, stat: str, photos: List[Dict[str, Any]], digits: int
    ) -> List[Any]:
        a = _finite([p[f"{key}_{stat}"] for p in photos])
        return [
            label,
            stat,
            len(a),
            _f(np.median(a) if len(a) else None, digits),
            _f(_pct(a, 90), digits),
            _f(a.max() if len(a) else None, digits),
        ]

    rows = []
    for key, label, digits, photos in (
        ("depth", "depth (H)", 4, finite),
        ("depth_px", "depth (px)", 2, finite),
        ("apex_h", "apex h", 3, per_photo),
    ):
        for stat in ("std", "range"):
            rows.append(pooled(label, key, stat, photos, digits))
    text += _table(["quantity", "per-photo", "n", "median", "p90", "max"], rows)
    text += (
        f"\nThe apex changed between prior and observed on "
        f"{sum(p['prior_changes'] for p in per_photo)} photos; sub.*.present "
        f"flipped between 0 and 1 on {sum(p['present_flip'] for p in per_photo)} "
        f"and became NaN on some reading on "
        f"{sum(p['present_nan'] for p in per_photo)}.\n"
    )
    text += "\nThe depth spread (H) by photo subset:\n\n"
    subsets = [
        (f"{expression} sets", [p for p in finite if p["expression"] == expression])
        for expression in sorted({p["expression"] for p in finite})
    ] + [
        (
            f"median depth over the readings below {FLAT_DEPTH_MEDIAN} H",
            [p for p in finite if p["depth_median"] < FLAT_DEPTH_MEDIAN],
        ),
        (
            f"median depth at or above {FLAT_DEPTH_MEDIAN} H",
            [p for p in finite if p["depth_median"] >= FLAT_DEPTH_MEDIAN],
        ),
    ]
    rows = []
    for label, photos in subsets:
        for stat in ("std", "range"):
            rows.append(pooled(label, "depth", stat, photos, 4))
    text += _table(["subset", "per-photo", "n", "median", "p90", "max"], rows)
    text += "\nPer photo (depth cells are blank unless every reading has one):\n\n"
    text += _table(
        [
            "set",
            "file",
            "depth std (H)",
            "depth range (H)",
            "depth std (px)",
            "depth range (px)",
            "apex h std",
            "apex h range",
        ],
        [
            [
                p["set"],
                p["file"],
                _f(p["depth_std"], 4),
                _f(p["depth_range"], 4),
                _f(p["depth_px_std"], 2),
                _f(p["depth_px_range"], 2),
                _f(p["apex_h_std"], 3),
                _f(p["apex_h_range"], 3),
            ]
            for p in per_photo
        ],
    )
    return text


def write_summary(
    records: List[Dict[str, Any]], out: pathlib.Path, wall: float, device: str
) -> None:
    renders = [r for r in records if r["set"] == "renders" and r["reading"] == "plain"]
    photos = [r for r in records if r["set"] != "renders"]
    plain_photos = [r for r in photos if r["reading"] == "plain"]
    by_set: Dict[str, List[Dict[str, Any]]] = {}
    for r in plain_photos:
        by_set.setdefault(r["set"], []).append(r)
    parts = [
        "# Submalar depth calibration\n",
        f"Threshold in the library: SUBMALAR_MIN_DEPTH = "
        f"{cheek_outline.SUBMALAR_MIN_DEPTH} H; SMOOTH_ROWS = "
        f"{cheek_outline.SMOOTH_ROWS}; APEX_H_PRIOR = {cheek_outline.APEX_H_PRIOR}; "
        f"APEX_H_WINDOW = {APEX_WINDOW}; APEX_MIN_DEV = "
        f"{cheek_outline.APEX_MIN_DEV}; OUTLINE_MIN_MASK = "
        f"{cheek_outline.OUTLINE_MIN_MASK}; EDGE_GRAD_MIN = "
        f"{cheek_outline.EDGE_GRAD_MIN}. in_window is |cheek.yaw| in "
        f"[{YAW_WINDOW[0]}, {YAW_WINDOW[1]}] (a quarter face); in_gate is |cheek.yaw| "
        f"within {cheek.OBLIQUE_YAW_GATE} of {cheek.OBLIQUE_YAW} (the apex.ok gate); "
        f"agree is |cheek.yaw - cheek.yaw.model| <= {cheek.YAW_AGREE_DEG}.\n",
        "## 1. The renders\n",
        render_table(renders) if renders else "No renders.\n",
        "## 2. The photo sets (plain readings)\n",
        set_tables(by_set) if by_set else "No photos.\n",
        "## 3. The mirrored pairs\n",
        pair_table(plain_photos) if plain_photos else "No photos.\n",
        "## 4. The perturbation noise\n",
        perturbation_table(photos) if photos else "No photos.\n",
        "## 5. Timing\n",
    ]
    seconds = _finite([r["seconds"] for r in records])
    parts.append(
        f"Device {device}: {len(records)} readings "
        f"({len(renders)} renders, {len(plain_photos)} photos, "
        f"{len(photos) - len(plain_photos)} perturbed readings) in {wall:.0f} s wall; "
        f"per reading mean {_f(seconds.mean(), 2)} s, median "
        f"{_f(np.median(seconds), 2)} s, max {_f(seconds.max(), 2)} s "
        f"(load_image with the background model plus analyze).\n"
    )
    (out / "summary.md").write_text("\n".join(parts), encoding="utf-8")


def contact_sheet(
    records: List[Dict[str, Any]], out: pathlib.Path, set_names: Sequence[str]
) -> None:
    """The first CONTACT_PER_SET overlays of each set, one row per set."""
    tiles: List[List[NDArray[np.uint8]]] = []
    for name in set_names:
        files = [
            r["file"]
            for r in records
            if r["set"] == name and r["reading"] == "plain" and r["ok"]
        ][:CONTACT_PER_SET]
        row = []
        for file in files:
            path = out / "overlays" / name / f"{pathlib.Path(file).stem}.jpg"
            tile = cv2.imread(str(path))
            if tile is None:
                continue
            h, w = tile.shape[:2]
            tile = cv2.resize(
                tile,
                (CONTACT_TILE_WIDTH, round(h * CONTACT_TILE_WIDTH / w)),
                interpolation=cv2.INTER_AREA,
            )
            # The file name on a dark band along the bottom edge, clear of the
            # measure's text column.
            label = f"{name}/{file}"
            band = tile.shape[0] - 28
            tile[band:] = (tile[band:] * 0.25).astype(np.uint8)
            cv2.putText(
                tile,
                label,
                (8, tile.shape[0] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            row.append(tile)
        if row:
            tiles.append(row)
    if not tiles:
        return
    height = max(t.shape[0] for row in tiles for t in row)
    columns = max(len(row) for row in tiles)
    sheet = np.full(
        (height * len(tiles), CONTACT_TILE_WIDTH * columns, 3), 255, dtype=np.uint8
    )
    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            y, x = i * height, j * CONTACT_TILE_WIDTH
            sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    cv2.imwrite(
        str(out / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--renders",
        type=pathlib.Path,
        default=RENDERS,
        help="the folder of render PNGs (every PNG but the frontal yaw0 and yaw10)",
    )
    parser.add_argument(
        "--photos",
        type=pathlib.Path,
        nargs="*",
        default=[],
        help="folders of photos (.jpg, .jpeg, .png), one set each, named after it",
    )
    parser.add_argument("--device", choices=("cpu", "auto"), default="auto")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--overlays", action="store_true", help="write every image's drawn result"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="the first N photos of each folder"
    )
    parser.add_argument(
        "--perturb",
        type=int,
        default=0,
        help="the first N photos of each folder also get the four perturbed readings",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    out: pathlib.Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cpu" else models.detect_device()
    models.init_models(models.download_models(), device)
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []
    set_names: List[str] = []
    with open(out / "results.jsonl", "w", encoding="utf-8") as results:
        renders = [
            p
            for p in sorted(args.renders.glob("*.png"))
            if p.name not in FRONTAL_RENDERS
        ]
        for path in renders:
            run_image(
                path,
                "renders",
                path.stem,
                "render",
                "render",
                ("plain",),
                out,
                args.overlays,
                results,
            )
        for folder in args.photos:
            set_name = folder.name
            set_names.append(set_name)
            expression, turn = parse_set(set_name)
            for i, path in enumerate(list_images(folder, args.limit)):
                readings = READINGS if i < args.perturb else ("plain",)
                run_image(
                    path,
                    set_name,
                    path.stem.split("_")[0],
                    expression,
                    turn,
                    readings,
                    out,
                    args.overlays,
                    results,
                )
    with open(out / "results.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    wall = time.perf_counter() - started
    write_summary(records, out, wall, device)
    if args.overlays and set_names:
        contact_sheet(records, out, set_names)
    print(f"wrote {out / 'summary.md'} ({len(records)} readings, {wall:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
