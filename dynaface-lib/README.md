# Dynaface Python Application

[![PyPI version](https://badge.fury.io/py/dynaface.svg)](https://badge.fury.io/py/dynaface)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeffheaton/dynaface/blob/main/dynaface-lib/examples/dynaface_intro.ipynb)

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

```
python -m unittest discover -s tests
```

# Running Examples

- [Dynaface Examples]()

```
python ./examples/process_media.py /Users/jeff/data/facial/samples/tracy-ref-blink.mp4

python ./examples/process_media.py --crop /Users/jeff/data/facial/samples/2021-8-19.png

python ./examples/process_media.py --crop /Users/jeff/data/facial/samples/tracy_frame.png

python ./examples/process_media.py --crop /Users/jeff/data/facial/samples/tracy-blink-single.mp4
```
