# Python library

The `dynaface` package does all of Dynaface's analysis. It runs on Python 3.10 or newer, on Linux, macOS and Windows.

```bash
pip install dynaface
```

[Getting started](../getting-started.md) shows how to measure a face, and the [tutorial](tutorial.ipynb) walks through it step by step.

## The main pieces

| | |
|---|---|
| [`dynaface.models`][dynaface.models] | Download the AI models, load them and choose a device. |
| [`load_face_image()`][dynaface.facial.load_face_image] | Load a photo from a file or URL. |
| [`AnalyzeFace`][dynaface.facial.AnalyzeFace] | Analyze a face: find the landmarks, crop it, run the measures and draw them. |
| [`all_measures()`][dynaface.measures.all_measures] | Every measure, ready to pass to `AnalyzeFace`. |

## Analyzing an image you already have

`load_face_image()` reads files and URLs. For an image already in memory, such as a video frame, create an `AnalyzeFace` and pass the image to `load_image()` as an RGB array:

```python
import cv2
from dynaface import facial, measures, models

models.init_models(models.download_models(), models.detect_device())

face = facial.AnalyzeFace(measures=measures.all_measures())
frame = cv2.cvtColor(cv2.imread("frame.png"), cv2.COLOR_BGR2RGB)  # OpenCV loads BGR
face.load_image(frame, crop=True)
results = face.analyze()
```

## Adding a measure

Measures are plugins. Subclass [`MeasureBase`][dynaface.measures_base.MeasureBase], declare one `MeasureItem` per output in the constructor, and implement `abbrev()` and `calc()`. `calc()` must return every output it declares, whether or not its item is enabled; an item's enabled flag only controls drawing. To use the new measure, include it in the `measures` list you give `AnalyzeFace`.
