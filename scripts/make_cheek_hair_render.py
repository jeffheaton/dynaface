"""Paint hair beside the far cheek of the 45 degree render: the cheek tests' hair case.

dynaface-lib-python/tests_data/cheek/yaw45_hair.png is tests_data/cheek/yaw45.png
with a dark, tapered strip painted against the far cheek (the subject's left,
the image's right) from a pixel outside the far WFLW contour, 10 px above the far
canthus row to 10 px below the stomion row and 40 px wide at its middle: hair
hanging beside the cheek, which U^2-Net keeps as foreground at the band-fit crop
(the sketch's own hair it drops there, so the sketch is no hair case; see
tests/test_cheek_photos.py). The mask then has no edge within the tracer's reach
on any band row, and the rows come from the intensity edge at the skin and from
the contour, never from the mask. The contour comes from the render's raw SPIGA
landmarks in tests_data/cheek_landmarks.json, so no model is needed.

Rerun it from the repository root in dynaface-lib-python's venv only when the
render or the strip changes on purpose; the noise is seeded, so two runs write
identical files:

    python scripts/make_cheek_hair_render.py
"""

import json
import math
import pathlib

import numpy as np
from PIL import Image

from dynaface import cheek_outline

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_DATA = ROOT / "dynaface-lib-python" / "tests_data"
RENDER = TESTS_DATA / "cheek" / "yaw45.png"
OUT = TESTS_DATA / "cheek" / "yaw45_hair.png"
LANDMARKS_JSON = TESTS_DATA / "cheek_landmarks.json"

# The strip: rows from ROW_PAD above the far canthus row to ROW_PAD below the
# stomion row, from INSET px outside the contour seed outward, WIDTH px wide at
# the middle, tapering to END_WIDTH at both ends; a flat hair color with seeded
# noise of NOISE gray levels.
ROW_PAD = 10
INSET = 1.0
WIDTH, END_WIDTH = 40.0, 4.0
COLOR = (62, 42, 30)
NOISE = 6.0
SEED = 3


def paint_hair(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """The render with the strip painted beside the far (left) cheek."""
    out = np.array(image, copy=True)
    canthus_row = int(round(landmarks[cheek_outline.LM_CANTHUS_L][1]))
    stomion_row = int(
        round(np.mean([landmarks[i][1] for i in cheek_outline.LM_STOMION]))
    )
    rows = np.arange(canthus_row - ROW_PAD, stomion_row + ROW_PAD + 1)
    contour = landmarks[list(cheek_outline.LEFT_FAR_CONTOUR)]
    seed = cheek_outline.contour_seed(contour, rows, far_left=True)
    rng = np.random.default_rng(SEED)
    n = len(rows)
    for k, (y, s) in enumerate(zip(rows, seed)):
        t = (k - n / 2.0) / (n / 2.0)
        width = (WIDTH - END_WIDTH) * math.sqrt(max(0.0, 1.0 - t * t)) + END_WIDTH
        x0 = int(round(s + INSET))
        x1 = min(int(round(s + INSET + width)), out.shape[1])
        noise = rng.normal(0.0, NOISE, size=(x1 - x0, 3))
        line = np.array(COLOR, dtype=np.float64)[None, :] + noise
        out[y, x0:x1] = np.clip(line, 0, 255).astype(np.uint8)
    return out


def main() -> None:
    with open(LANDMARKS_JSON, encoding="utf-8") as f:
        entry = json.load(f)["images"]["yaw45"]
    landmarks = np.array(entry["raw"]["landmarks"], dtype=np.float64)
    image = np.asarray(Image.open(RENDER).convert("RGB"))
    Image.fromarray(paint_hair(image, landmarks)).save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
