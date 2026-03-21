# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dynaface is a facial analysis tool for measuring facial symmetry and movement, primarily for assessing facial paralysis patients. It is organized as a monorepo with two components:

- **`dynaface-lib/`** — Python library (PyPI package `dynaface`)
- **`dynaface-app/`** — Desktop GUI application (PyQt6)

## Commands

### dynaface-lib (Python Library)

```bash
cd dynaface-lib
python3.11 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run all tests
python -m pytest tests --cov=. --cov-report=html

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

# Build wheel
python -m build --wheel
```

### dynaface-app (Desktop Application)

```bash
cd dynaface-app
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install dynaface

# Run the application
python dynaface_app.py

# Run tests
python -m unittest discover -s tests
```

## Code Style

- Line length: 88 characters (Black-compatible)
- Formatter: Black
- Import sorter: isort
- Linter: flake8 (configured in `.flake8` at repo root)
- Python: 3.10+ required, tested on 3.11

## Architecture

### Library (`dynaface-lib/dynaface/`)

The library has two main analysis paths: **frontal** (face-on) and **lateral** (side-profile).

**Core classes:**
- `facial.py` — `AnalyzeFace`: main entry point for image/video analysis; handles landmark detection, pose estimation, face cropping
- `lateral.py` — Handles lateral/profile view analysis
- `models.py` — Model management: MTCNN2 face detector, SPIGA landmark detection (GAT-based), background removal (rembg); handles CPU/GPU/MPS device detection automatically
- `image.py` — `ImageAnalysis` base class, image I/O and transforms

**Measurement plugin system:**
- `measures_base.py` — `MeasureBase` abstract class and `MeasureItem` data class
- `measures_frontal.py` — Frontal measurements: `AnalyzeFAI` (Facial Asymmetry Index), `AnalyzeOralCommissureExcursion`, brows, eyes, dental area, etc.
- `measures_lateral.py` — Lateral measurements
- `measures_skin.py` — Skin tone analysis
- `measures.py` — `all_measures()` returns the full list of active measurement plugins

**Other:**
- `const.py` — Landmark indices and thresholds
- `config.py` — Configuration
- `spiga/` — Embedded SPIGA framework (Graph Attention Network for 97-point landmark detection)

### Application (`dynaface-app/`)

- `dynaface_app.py` — Entry point; `AppDynaface` extends `AppJTH` (Jeff Heaton's UI framework)
- `dynaface_window.py` — `DynafaceWindow`: main window, menu, file handling, tab management
- `dynaface_document.py` — Project file management (`.dyfc` format)
- `tab_analyze_video.py` — Primary analysis tab: landmark visualization, data export (largest file at ~51KB)
- `tab_settings.py` — Preferences (pupillary distance, accelerator, smoothing)
- `tab_eval.py` — Evaluation/testing tab
- `jth_ui/` — Reusable PyQt6 components, settings management, logging utilities

### Key Design Patterns

- **Plugin architecture** for measurements: add a new `MeasureBase` subclass and register it in `measures.py`
- **Lazy model loading**: models are downloaded on first use and cached
- **Dual-view analysis**: frontal and lateral are separate code paths that share base infrastructure
- **MVC separation**: `dynaface-lib` handles all computation; `dynaface-app` handles presentation
