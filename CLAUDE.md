# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dynaface is a facial analysis tool for measuring facial symmetry and movement, primarily for assessing facial paralysis patients. It is organized as a monorepo with three components:

- **`dynaface-lib-python/`** — Python library (PyPI package `dynaface`)
- **`dynaface-app/`** — Desktop GUI application (PyQt6)
- **`dynaface-lib-dotnet/`** — C#/.NET port of the library (NuGet packages `Dynaface`/`Dynaface.Onnx`; also ships as a Unity package); see `dynaface-lib-dotnet/CLAUDE.md`

## Versioning

The three components are typically released in lockstep at the same version. When bumping a version, update every location for that component:

- **dynaface-lib-python** — `setup.cfg` `[metadata] version` is the source of truth: Build Library (`build-lib.yml`) reads it, regenerates `dynaface/version.py` from it, and stages the wheel and sdist on S3 under it. Also bump the checked-in `dynaface/version.py` `VERSION` so local/editable installs report correctly.
- **dynaface-lib-dotnet** — `facial_dll/FacialDll.csproj` `<Version>` is the source of truth: Build Library (.NET) (`build-lib-dotnet.yml`) greps it, generates `facial_dll/Version.cs`, and packs and stages both NuGet packages and their `.snupkg` symbols with it. Keep `FacialDll.Onnx/FacialDll.Onnx.csproj` `<Version>` in lockstep for local builds.
- **dynaface-app** — `deploy/macos/build.sh` `DYNAFACE_VERSION`, and `deploy/windows/build.ps1` **both** `$script:DYNAFACE_VERSION` and `$currentVersion`. These stamp the app bundle version and regenerate the checked-in `version.py` at build time, and they also pin the PyPI `dynaface` lib version the app is built against — so the lib must be published to PyPI at that version before the app can build. Also bump the checked-in `version.py` `VERSION` for local runs.

Each library is released in two manual steps: a build workflow that stages exactly what ships on S3 with a build record (`<name>-<version>-build.json`), rerunnable at the same version, and a deploy workflow that takes the version, publishes what was staged without rebuilding, and tags `v<version>` at the commit the build recorded. Python and .NET share the version and the tag. See `RELEASING.md`.

## Important: Always Use the venv

The two Python components use Python virtual environments. **Never install packages or run tests using the system/base Python.** Always activate the venv first. If `venv/` doesn't exist in the component directory, create it before proceeding.

## Commands

### dynaface-lib-python (Python Library)

```bash
cd dynaface-lib-python
python3.11 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run all tests
python -m pytest tests --cov=dynaface --cov-branch --cov-report=html

# Run a single test file
python -m pytest tests/test_facial.py

# Lint & format
flake8 dynaface --config ../.flake8
black dynaface
isort dynaface

# Type checking
mypy dynaface --install-types --non-interactive

# Security scan
bandit -r dynaface

# Build the wheel and sdist into dist/ (pip install build first)
python -m build
```

### dynaface-app (Desktop Application)

```bash
cd dynaface-app
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install dynaface

# Run the application
python dynaface_app.py
```

(The app's only automated tests cover `.dyfc` project files: `python -m unittest discover -s tests -t .` from `dynaface-app`, in its venv. Otherwise, verify changes by running the app.)

### dynaface-lib-dotnet (C#/.NET Library)

C# port of the Python library: `facial_dll/` core (netstandard2.1, NuGet package `Dynaface`, Unity package via `Dynaface.asmdef`), `FacialDll.Onnx/` ONNX Runtime backend (NuGet `Dynaface.Onnx`), `FacialDllConsole/` CLI harness, and `DynafaceTests/` xUnit suite. Requires the .NET 10 SDK; no venv. See `dynaface-lib-dotnet/CLAUDE.md` for architecture and the full command list.

```bash
cd dynaface-lib-dotnet

# Build everything
dotnet build FacialDll.sln

# Run tests (model-gated integration tests self-skip if the ONNX models aren't found)
dotnet test DynafaceTests/DynafaceTests.csproj
```

## Code Style

- Line length: 88 characters (Black-compatible)
- Formatter: Black
- Import sorter: isort
- Linter: flake8 (configured in `.flake8` at repo root)
- Python: 3.10+ required; CI tests 3.10 through 3.14 on Linux, macOS and Windows (and the .NET library on all three)
- **Spelling: American English everywhere** — identifiers, comments, docstrings, log
  and UI strings, Markdown docs, and commit messages. Use `color`, `canceled`,
  `center`, `behavior`, `gray`, `analyze`, `normalize`, `initialize`, `modeling`,
  `labeled`, `license`, `defense`, `millimeter` — never their British forms. The only
  exception is an external API that dictates the spelling: mirror those exactly and do
  not "fix" them (e.g. Unity's `MonoBehaviour`).

## Architecture

### Library (`dynaface-lib-python/dynaface/`)

The library has three analysis paths: **frontal** (face-on), **quarter** (three-quarter: the head turned about 45 degrees, for the far cheek's landmarks) and **lateral** (side-profile).

**Core classes:**
- `facial.py` — `AnalyzeFace`: main entry point for image/video analysis; handles landmark detection, pose estimation, face cropping
- `lateral.py` — Handles lateral/profile view analysis
- `cheek.py` — The cheek landmarks' frontal constructions (zygion, cheek and submalar reference points, in a roll-corrected pupil frame), the yaw calibration tables (re-measured from `tests_data/cheek_landmarks.json`), the far-side rule, the quarter classification (`classify_quarter()`) and `locate()` / `locate_all()`, the one rule behind `AnalyzeFace.find_cheek_landmarks()`; numpy only, for the .NET port
- `cheek_outline.py` — The far-cheek tracer of a quarter face (`trace_far_cheek()`: the mask's edge row by row, with the landmark contour, the nose rows and an intensity edge standing in, each row's source kept) and the oblique search (`observed_points()`: Kaptein's levels, the malar apex and the submalar point); numpy only
- `models.py` — Model management: MTCNN2 face detector, SPIGA landmark detection (GAT-based), background removal (rembg); handles CPU/GPU/MPS device detection automatically
- `image.py` — `ImageAnalysis` base class, image I/O and transforms

**Measurement plugin system:**
- `measures_base.py` — `MeasureBase` abstract class and `MeasureItem` data class (an item can be a group, `outputs=`, a switch named for what it draws standing for several outputs, and can be limited to some views, `views=`)
- `measures_frontal.py` — Frontal measurements: `AnalyzeFAI` (Facial Asymmetry Index), `AnalyzeOralCommissureExcursion`, brows, eyes, dental area, etc.; `AnalyzeLandmarks` ("Landmarks") exports the 98 landmarks and, through its last item, the "Cheek Landmarks" group, the ten cheek landmarks' 20 coordinates (`cheek.CHEEK_LANDMARK_OUTPUTS`, from `face.cheek_landmarks`), drawn as orange dots
- `measures_lateral.py` — Lateral measurements
- `measures_skin.py` — Skin tone analysis
- `measures_cheek.py` — `AnalyzeCheekMeasures` ("Cheek Measures"), registered last in `all_measures()` (the 15th entry, as `MeasureCheekMeasures` is in the .NET registry): six group items, each drawing one thing and carrying its numbers (41 outputs in all): Zygion width and Reference points (frontal, millimeters), Far cheek outline, Malar apex and Submalar point (quarter, fractions of `cheek.scale`), and Head turn (both views, the yaw and pitch flags); the coordinates are the Landmarks measure's Cheek Landmarks item
- `measures.py` — `all_measures()` returns the full list of active measurement plugins; `MeasureSelection` keeps a user's choice of measures and items
- `video.py` — `VideoAnalyzer`: the import loop (skips frames without a face but keeps the others' numbers, smooths landmarks with `LandmarkSmoother`, caps at 5000 frames); `FrameSeries`: a video's outputs frame by frame (from `analyze()` results or saved states), its CSV, peak frames, and the numbers behind the apps' Evaluation report. The desktop app's chart, CSV, jump-to-max and Evaluation use it, and the Unity app should too once it takes the 2.1 library (the .NET port has `FrameSeries` as well); keep that logic here, not in the apps

**Other:**
- `const.py` — Landmark indices and thresholds
- `config.py` — Configuration
- `spiga/` — Embedded SPIGA framework (Graph Attention Network for 97-point landmark detection)

### Application (`dynaface-app/`)

- `dynaface_app.py` — Entry point; `AppDynaface` extends `AppJTH` (Jeff Heaton's UI framework)
- `dynaface_window.py` — `DynafaceWindow`: main window, menu, file handling, tab management
- `dynaface_document.py` — Project file management (`.dyfc` format: a zip of JSON plus `.npy` arrays, never pickle; older gzip-pickle files load through an allowlisted unpickler, and measures are created only from `all_measures()` classes)
- `tab_analyze_video.py` — Primary analysis tab: landmark visualization, data export (largest file at ~51KB)
- `tab_settings.py` — Preferences (pupillary distance, accelerator, smoothing)
- `tab_eval.py` — Evaluation/testing tab
- `jth_ui/` — Reusable PyQt6 components, settings management, logging utilities

### Key Design Patterns

- **Plugin architecture** for measurements: add a new `MeasureBase` subclass and register it in `measures.py`
- **Measure output contract** (both ports): `calc()` returns every declared output whether or not its item is enabled or rendering is on; item flags only gate drawing, and callers pick what to show or export (`get_all_items()` / `GetAllItems`). Drawing-only switches are declared `display_only` and are never returned; a group item (Python first, the .NET port follows) stands for its outputs, listed together, and an item with views is off on a view it doesn't suit (see the `measures_base.py` docstring)
- **Scale contract** (both ports): a frontal or quarter face's `pix2mm` is always measured on the landmarks about to be measured, from a pupillary distance in mm that belongs to the analysis (`face.pd` / `FacePipelineResult.PupilDistMm`). Never pair a saved or pre-crop scale with other landmarks: `analyze()` re-measures it and `FaceMeasureContext` measures it itself, and `tests/test_scale.py` / `ScaleConsistencyTests` guard it (see the units note in `dynaface-lib-dotnet/facial_dll/CLAUDE.md`). A quarter face's pupil scale is foreshortened by the turn, so no millimeter is defined on it: `AnalyzePosition` reports `px2mm` as NaN there and the Cheek Measures' oblique outputs are fractions of `cheek.scale`
- **Measure choices** (both ports): keep a user's choice of measures and items in a `MeasureSelection` and `apply(measures, lateral, quarter=face.quarter)` it to each face, so the millimeter measures stay off a turned head. `update_for_type(lateral, quarter=False)` / `UpdateForType` overwrites the item flags, so it loses the choices
- **Model output checks** (both ports): output no working model gives (NaN, a constant output, landmarks all at one point, or in .NET none or the wrong size) raises `ModelOutputError` / `ModelOutputException`, so a backend that computes garbage isn't reported as "no face"
- **Lazy model loading**: models are downloaded on first use and cached
- **Three views**: frontal, quarter and lateral are separate code paths in `load_image()` that share base infrastructure. The quarter path (opt-in through `config.AUTO_QUARTER`, which ships off, or a forced `Pose.QUARTER`) takes the centered band crop (`crop_lateral(center=True)`), never flips, traces the far cheek with the background model, and saves `far_left` and the outline in the state's view dict, so `load_state()` measures as the original did
- **MVC separation**: `dynaface-lib-python` handles all computation; `dynaface-app` handles presentation
