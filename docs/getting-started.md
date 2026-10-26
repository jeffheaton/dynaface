# Getting started

## Python

Dynaface needs Python 3.10 or newer. Install it from PyPI:

```bash
pip install dynaface
```

Then measure a face in a photo:

```python
from dynaface import facial, measures, models

# Load the AI models, using a GPU or Apple silicon when one is available.
models.init_models(models.download_models(), models.detect_device())

# Find the face, crop it to a standard size and run every measure.
face = facial.load_face_image(
    "https://data.heatonresearch.com/dynaface/sample/img4-1024-frontal.jpg",
    measures=measures.all_measures(),
)
results = face.analyze()

# results maps each measure to its value, such as results["oce.l"] in millimeters.
print(f"Facial Asymmetry Index: {results['fai']:.2f}")

# Save the cropped face with every measure drawn on it.
face.save("face_measures.jpg")
```

The first run downloads Dynaface's three AI models, about 400 MB, to `~/.dynaface/models`. `detect_device()` runs them on an NVIDIA GPU (`"cuda"`) or Apple silicon (`"mps"`) when one is available, and on the CPU otherwise.

Next, work through the [tutorial](python/tutorial.ipynb), try the [examples](python/examples.md), or browse the [Python API](python/api/facial.md).

## .NET

Install the packages from NuGet:

```bash
dotnet add package Dynaface.Onnx
```

`Dynaface.Onnx` runs the models with ONNX Runtime and brings in the core `Dynaface` package. The .NET library doesn't download the models itself. Use the same three `.onnx` files the Python library downloads, or get them from the [model bundle](https://data.heatonresearch.com/dynaface/model/2/dynaface_models.zip). The [.NET page](dotnet/index.md) shows how to analyze a face.
