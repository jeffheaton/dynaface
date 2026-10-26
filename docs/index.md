# Dynaface

Dynaface measures facial symmetry and movement in photos and videos, mainly to assess people with facial paralysis. It finds the face, places 98 landmarks on it with an AI model that copes well with asymmetric faces, and calculates clinical measures such as the Facial Asymmetry Index (FAI) and oral commissure excursion.

Dynaface comes in three forms:

- **[Python library](python/index.md)**: `pip install dynaface`. It does all of the analysis, for research code and notebooks.
- **[.NET library](dotnet/index.md)**: the `Dynaface` and `Dynaface.Onnx` NuGet packages, a port of the Python library for C# and Unity apps.
- **Desktop and mobile apps**: for clinicians and researchers who don't write code. Get them from the [Dynaface website](https://www.dynaface.org).

## Where to start

- [Getting started](getting-started.md): install a library and measure your first face.
- [How it works](how-it-works.md): the steps from a photo to its measurements.
- [Measures](measures.md): every measurement Dynaface makes.
- [Tutorial](python/tutorial.ipynb): a step-by-step notebook you can run in Colab.

## Citing Dynaface

If you use Dynaface in research, please cite:

> Renne, A., Heaton, J., & Boahene, K. D. O. (2026). Associations of AI-based facial metrics with patient-reported outcomes in idiopathic facial paralysis. *The Laryngoscope*, 136(7), 2991–3001. <https://doi.org/10.1002/lary.70417>

Dynaface is open source under the [Apache License 2.0](https://github.com/jeffheaton/dynaface/blob/main/LICENSE). It is a research tool, not a medical device.
