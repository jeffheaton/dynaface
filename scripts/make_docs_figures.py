"""Regenerate the figures on docs.dynaface.org.

Each figure shows one measure on a sample image from dynaface-lib-python/tests_data.
Run it from the repository root in dynaface-lib-python's venv, since it needs the AI
models, whenever a measure's drawing changes:

    python scripts/make_docs_figures.py
"""

import pathlib

import cv2
import matplotlib.patheffects as patheffects
import numpy as np
from matplotlib.figure import Figure

from dynaface import facial, measures, models

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "dynaface-lib-python" / "tests_data"
OUT = ROOT / "docs" / "assets"

FRONTAL = "img4-1024-frontal.jpg"
SMILE = "img1-512.jpg"
LATERAL = "img2-1024-right-lateral.jpg"

# Figure name, sample image and the measure it shows.
MEASURE_FIGURES = [
    ("fai", FRONTAL, measures.AnalyzeFAI),
    ("oral-ce", FRONTAL, measures.AnalyzeOralCommissureExcursion),
    ("brow", FRONTAL, measures.AnalyzeBrows),
    ("dental-display", SMILE, measures.AnalyzeDentalArea),
    ("eye-area", FRONTAL, measures.AnalyzeEyeArea),
    ("intercanthal-distance", FRONTAL, measures.AnalyzeIntercanthalDistance),
    ("outer-eye-corners", FRONTAL, measures.AnalyzeOuterEyeCorners),
    ("mouth-length", FRONTAL, measures.AnalyzeMouthLength),
    ("nose-frontal", FRONTAL, measures.AnalyzeNoseFrontal),
    ("position", FRONTAL, measures.AnalyzePosition),
    ("skin-tone", FRONTAL, measures.AnalyzeSkinTone),
    ("lateral", LATERAL, measures.AnalyzeLateral),
]


def save(image_rgb, path: pathlib.Path, width: int) -> None:
    height = round(image_rgb.shape[0] * width / image_rgb.shape[1])
    small = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    print(f"wrote {path.relative_to(ROOT)}")


# Landmark groups, with the direction each group's labels are pushed: away from
# the group's center ("out"), toward it ("in"), or a fixed (x, y) direction.
LANDMARK_GROUPS = [
    (range(0, 33), "out"),  # face contour
    (range(33, 42), "out"),  # subject's right brow
    (range(42, 51), "out"),  # subject's left brow
    (range(51, 55), (1, 0)),  # nose bridge
    (range(55, 60), "out"),  # base of the nose
    (range(60, 68), "out"),  # subject's right eye
    (range(68, 76), "out"),  # subject's left eye
    (range(76, 88), "out"),  # outer lip
    (range(88, 96), "in"),  # inner lip
    (range(96, 98), (0.7, -0.7)),  # pupils
]


def save_landmark_figure(path: pathlib.Path) -> None:
    """Draw every landmark with its index, on a smile so the inner lip is open."""
    face = facial.load_face_image(str(SAMPLES / SMILE), measures=[])
    points = np.array(face.landmarks, dtype=float)

    x0, y0 = (points.min(axis=0) - 40).astype(int)
    x1, y1 = (points.max(axis=0) + 40).astype(int)
    # Fade the face so the labels stand out.
    region = face.render_img[y0:y1, x0:x1].astype(float) * 0.5 + 127.5

    # A Figure made without pyplot saves without a GUI backend.
    fig = Figure(figsize=(8, 8 * (y1 - y0) / (x1 - x0)), dpi=150)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(region.astype(np.uint8), extent=(x0, x1, y1, y0))
    ax.scatter(points[:, 0], points[:, 1], s=9, c="#d62728", zorder=2)
    ax.set_axis_off()

    outline = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    for indices, direction in LANDMARK_GROUPS:
        center = points[list(indices)].mean(axis=0)
        for i in indices:
            if isinstance(direction, tuple):
                d = np.array(direction, dtype=float)
            else:
                d = points[i] - center if direction == "out" else center - points[i]
            d = d / (np.linalg.norm(d) or 1.0)
            # Offsets are in points; matplotlib's y axis points up, the image's down.
            ax.annotate(
                str(i),
                points[i],
                xytext=(9 * d[0], -9 * d[1]),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                path_effects=outline,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, pil_kwargs={"quality": 90})
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    models.init_models(models.download_models(), models.detect_device())

    for name, sample, measure in MEASURE_FIGURES:
        face = facial.load_face_image(str(SAMPLES / sample), measures=[measure()])
        face.analyze()
        save(face.render_img, OUT / "measures" / f"{name}.jpg", width=600)

    save_landmark_figure(OUT / "landmarks.jpg")


if __name__ == "__main__":
    main()
