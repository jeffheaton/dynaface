# Contributing to Dynaface

Thanks for helping improve Dynaface! It measures facial symmetry and movement, mainly to assess people with facial paralysis. Bug reports, feature ideas, documentation fixes and code are all welcome.

Everyone who takes part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Protect patient privacy

Dynaface is used with patient photos and videos, and everything posted here is public, so please be careful about what you share.

- Don't post photos, videos or screenshots of identifiable patients, or measurements exported from them, in issues, pull requests or test data.
- Reproduce problems with the sample images in [`dynaface-lib-python/tests_data`](dynaface-lib-python/tests_data), or with images you have the right to share publicly.
- If a problem only happens on an image you can't share, describe the image instead: the pose, lighting and resolution, and what goes wrong. If that isn't enough, email jeff@jeffheaton.com before sending anything.
- Only add test images that you have the right to publish in this repository.

## Reporting bugs and suggesting features

Search the [existing issues](https://github.com/jeffheaton/dynaface/issues) first. Then open a new issue with the **Bug report** or **Feature request** form, which asks for the details needed to act on it.

Report security vulnerabilities privately instead, as the [security policy](SECURITY.md) describes. Accessibility problems are bugs too, and the [accessibility statement](ACCESSIBILITY.md) says what to include.

For a new measurement, explain which landmarks it uses and how it's calculated, and link a clinical reference if there is one.

If you'd like to help with code, issues labeled [`good first issue`](https://github.com/jeffheaton/dynaface/labels/good%20first%20issue) are a good place to start.

## How the repository is organized

| Directory | Contents |
|---|---|
| [`dynaface-lib-python`](dynaface-lib-python) | The Python library, published to PyPI as `dynaface`. It does all of the analysis: face detection, landmarks, pose and the measurements. |
| [`dynaface-lib-dotnet`](dynaface-lib-dotnet) | A C# port of the library, published to NuGet as `Dynaface` and `Dynaface.Onnx`. The core also ships as a Unity package. |
| [`dynaface-app`](dynaface-app) | The PyQt6 desktop app, built on the Python library. |

The Python library is the reference implementation. The C# port follows the same formulas and pipeline order, so a change to a measurement or to the pipeline in one library should be made in the other too. If you only change one, open an issue so the other can catch up.

## Setting up

### The models

Both libraries use three ONNX models (about 400 MB) stored in `~/.dynaface/models`. The Python library downloads them the first time it needs them. The .NET tests look in the same folder, or in the folder named by the `DYNAFACE_ONNX_MODELS` environment variable, so setting up the Python library first gets the models for both.

### Python library

Python 3.10 or newer is required, and CI tests with 3.11. Always work in a virtual environment:

```bash
cd dynaface-lib-python
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[test,lint,dev]"
python -m pytest tests
```

### .NET library

You need the [.NET 10 SDK](https://dotnet.microsoft.com/download).

```bash
cd dynaface-lib-dotnet
dotnet build FacialDll.sln
dotnet test DynafaceTests/DynafaceTests.csproj
```

Without the models, the integration tests are reported as skipped. Set `DYNAFACE_REQUIRE_MODELS=1` to make them fail instead.

### Desktop app

```bash
cd dynaface-app
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ../dynaface-lib-python   # your local library; or `pip install dynaface` for the released one
python dynaface_app.py
```

The app has no automated tests, so check your change by running it, and say in the pull request what you tried.

## Making changes

### Code style

- **Python:** format with Black and isort, and check with flake8 and mypy. New functions need type hints. From `dynaface-lib-python`:

  ```bash
  black dynaface
  isort dynaface
  flake8 dynaface --config ../.flake8
  mypy dynaface
  ```

- **C#:** run `dotnet format FacialDll.sln` from `dynaface-lib-dotnet`. The release build fails on unformatted code.
- **Spelling:** use American English in code, comments, docs and messages (`color`, `analyze`, `center`, `canceled`). The only exception is a name an external API dictates, such as Unity's `MonoBehaviour`.

### Adding a measurement

Measurements are plugins:

- **Python:** subclass `MeasureBase` and register the new class in `measures.all_measures()`.
- **C#:** subclass `FaceMeasureBase` in `facial_dll/Measures/` and register it in `DynafaceMeasures.AllMeasures()`, in the same position as in Python.

A measure's `calc()` must return every output it declares, whether or not that output is enabled. Enabling an output only controls whether it's drawn. The docstring at the top of [`measures_base.py`](dynaface-lib-python/dynaface/measures_base.py) has the full rules, and the tests in both libraries check every registered measure against them.

### Tests and versions

- Add or update tests for any change in behavior.
- Don't change version numbers. The maintainer bumps them when making a release.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one change, and link the issue it addresses.
3. Run the tests and formatters for each part you changed.
4. Describe what changed and why.

The **Test Library** and **Test Library (.NET)** workflows run automatically on pull requests that touch each library, and they need to pass.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), the same license as the rest of Dynaface.
