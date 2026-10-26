![Dynaface](https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/images/dynaface_logo.jpg)

# Dynaface Python Library

[![PyPI version](https://img.shields.io/pypi/v/dynaface?style=flat-square)](https://pypi.org/project/dynaface/)
[![Tests](https://img.shields.io/github/actions/workflow/status/jeffheaton/dynaface/test-lib.yml?branch=main&label=tests&style=flat-square)](https://github.com/jeffheaton/dynaface/actions/workflows/test-lib.yml)
[![Coverage](https://img.shields.io/codecov/c/github/jeffheaton/dynaface/main?flag=python&label=coverage&style=flat-square)](https://app.codecov.io/gh/jeffheaton/dynaface)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeffheaton/dynaface/blob/main/dynaface-lib-python/examples/dynaface_intro.ipynb)

Dynaface is an AI-driven facial tracking Python Library utilizing computer vision techniques that integrate Convolutional Neural Networks (CNNs) with cascaded Graph Attention Network (GAT) regressors to enhance facial landmark detection by capturing both local appearance and global structural relationships ([Prados-Torreblanca1, A & Buenaposada, J](https://bmvc2022.mpi-inf.mpg.de/0155.pdf)). This approach encodes both the appearance and spatial positioning of facial landmarks while employing an attention mechanism to prioritize reliable information. Such a method is particularly advantageous for assessing facial asymmetry in patients with facial paralysis, where conventional landmarking algorithms are often biased toward symmetric faces. By leveraging a global representation of facial structure, Dynaface enables precise detection of key landmarks despite asymmetries, facilitating the objective quantification of facial movement and asymmetry. These measurements, including the Facial Asymmetry Index (FAI) and Oral Commissure Excursion (OCE), serve as critical indicators of facial function and can be correlated with patient-reported outcomes to evaluate recovery and patient satisfaction.

# Quick Start

Install from [PyPI](https://pypi.org/project/dynaface/):

```bash
pip install dynaface
```

Measure a face in a photo. The first run downloads Dynaface's AI models (about 400 MB) to `~/.dynaface/models`.

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

For a step-by-step walkthrough, open the [tutorial notebook](https://github.com/jeffheaton/dynaface/blob/main/dynaface-lib-python/examples/dynaface_intro.ipynb) in [Colab](https://colab.research.google.com/github/jeffheaton/dynaface/blob/main/dynaface-lib-python/examples/dynaface_intro.ipynb).

# Helpful Links

- [Dynaface Application](https://github.com/jeffheaton/dynaface/tree/main/dynaface-app)

# Helpful Python Commands

**Activate Environment**

```
source venv/bin/activate
.\venv\Scripts\activate.bat
.\venv\Scripts\Activate.ps1
```

**Allow Windows to Use Environment**

```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Run Unit Tests**

For Mac/Linux:

```
cd dynaface-lib-python
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests
```

For Windows:

```
cd dynaface-lib-python
python3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m unittest discover -s tests
```

⚠️ Note for Windows users:
If you get an error like execution of scripts is disabled on this system when activating the virtual environment, you can temporarily bypass it with:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

# Running Examples

The [examples folder](https://github.com/jeffheaton/dynaface/tree/main/dynaface-lib-python/examples) has scripts that measure images and videos. From `dynaface-lib-python` in a clone of this repository, run them on the bundled sample files:

```bash
# Measure a frontal photo and save it with the measures drawn on it.
python examples/process_image.py tests_data/img4-1024-frontal.jpg frontal.jpg

# Measure a side profile.
python examples/process_media.py --crop tests_data/img2-1024-right-lateral.jpg lateral.jpg

# Measure every frame of a video. This writes the annotated video-analyze.mp4,
# a chart of each measure over time (video-graph.png) and the values for every
# frame (video-data.csv).
python examples/process_media.py tests_data/video1-512.mp4 video.mp4
```

# To Cite Dynaface

- Renne, A., Heaton, J., & Boahene, K. D. O. (2026). Associations of AI-based facial metrics with patient-reported outcomes in idiopathic facial paralysis. _Laryngoscope_. Advance online publication. https://doi.org/10.1002/lary.70417
- BibTeX: https://github.com/jeffheaton/dynaface/blob/main/CITATION.bib
