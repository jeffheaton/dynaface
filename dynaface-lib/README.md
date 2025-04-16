# Dynaface Python Library

[![PyPI version](https://badge.fury.io/py/dynaface.svg)](https://badge.fury.io/py/dynaface)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeffheaton/dynaface/blob/main/dynaface-lib/examples/dynaface_intro.ipynb)

Dynaface is an AI-driven facial tracking Python Library utilizing computer vision techniques that integrate Convolutional Neural Networks (CNNs) with cascaded Graph Attention Network (GAT) regressors to enhance facial landmark detection by capturing both local appearance and global structural relationships (Zhou et al., 2022). This approach encodes both the appearance and spatial positioning of facial landmarks while employing an attention mechanism to prioritize reliable information. Such a method is particularly advantageous for assessing facial asymmetry in patients with facial paralysis, where conventional landmarking algorithms are often biased toward symmetric faces. By leveraging a global representation of facial structure, Dynaface enables precise detection of key landmarks despite asymmetries, facilitating the objective quantification of facial movement and asymmetry. These measurements, including the Facial Asymmetry Index (FAI) and Oral Commissure Excursion (OCE), serve as critical indicators of facial function and can be correlated with patient-reported outcomes to evaluate recovery and patient satisfaction.

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
cd dynaface-lib
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests
```

For Windows:

```
cd dynaface-lib
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

- [Dynaface Examples]()

```
python ./examples/process_media.py /Users/jeff/data/facial/samples/tracy-ref-blink.mp4

python ./examples/process_media.py --crop /Users/jeff/data/facial/samples/2021-8-19.png

python ./examples/process_media.py --crop /Users/jeff/data/facial/samples/tracy_frame.png

python ./examples/process_media.py --crop /Users/jeff/data/facial/samples/tracy-blink-single.mp4
```
