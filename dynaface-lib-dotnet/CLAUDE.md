# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Solution Layout

- `facial_dll/` — core library (`netstandard2.1`), no Unity or ONNX dependencies; ships as a Unity package and as the NuGet package **`Dynaface`**
- `FacialDll.Onnx/` — ONNX Runtime inference backend (`netstandard2.1`): `OnnxDynafaceInference`, depends on the core + `Microsoft.ML.OnnxRuntime`; ships as the NuGet package **`Dynaface.Onnx`**
- `FacialDllConsole/` — standalone CLI harness / example (`net10.0`, not packaged) using both projects + `SkiaSharp`
- `DynafaceTests/` — xUnit tests (`net10.0`); pure unit tests always run, model-gated integration tests self-skip when the 3 real `.onnx` files aren't found locally (see `DynafaceTests/ModelPathResolver.cs`)
- `FacialDll.sln` — solution covering all four projects

`Dynaface` and `Dynaface.Onnx` are published to NuGet in lockstep at the same `<Version>`. As of 2.0.2 the assembly names match the package ids (`Dynaface.dll` / `Dynaface.Onnx.dll`); before that they were `FacialDll` / `FacialDll.Onnx`, so prebuilt-DLL and Unity consumers (the external `DynafaceRuntime` asmdef) must update their references when crossing 2.0.2.

See [`facial_dll/CLAUDE.md`](facial_dll/CLAUDE.md) for the full pipeline architecture, coordinate system contract, and WFLW-98 landmark index reference.

This is a contract-faithful C# port of the Python `dynaface-lib` package (`../dynaface-lib-python/`) — same measurement formulas and pipeline order, adapted to .NET idioms (nullable types, PascalCase, typed DTOs) where Python's idioms didn't fit. When porting new functionality from there, treat `dynaface-lib`'s Python source as the reference implementation.

## Build & Run

```bash
# Build everything from the repo root
dotnet build FacialDll.sln

# Run the test suite
dotnet test DynafaceTests/DynafaceTests.csproj

# Run the console harness (all 3 models required)
dotnet run --project FacialDllConsole -- photo.jpg blaze_face_short_range.onnx spiga_wflw.onnx u2net.onnx

# Print auto-detected tensor names for a new ONNX export
dotnet run --project FacialDllConsole -- photo.jpg blaze_face_short_range.onnx spiga_wflw.onnx u2net.onnx --list-tensors
```

## Project Relationship

`facial_dll` defines the single inference interface; `FacialDll.Onnx` provides the concrete ONNX implementation; `FacialDllConsole` is an example that wires them together:

| Interface (facial_dll / `Dynaface`) | Implementation (FacialDll.Onnx / `Dynaface.Onnx`) |
|---|---|
| `IDynafaceInference` (`RunBlazeFace`/`RunSpiga`/`RunU2Net`) | `OnnxDynafaceInference` |

The interface lives in `facial_dll/IDynafaceInference.cs`; `OnnxDynafaceInference` lives in `FacialDll.Onnx/`. It auto-detects tensor names from model metadata; pass explicit overrides to the constructor or use `--list-tensors` to inspect a model before running inference.

The Unity runtime adapter layer (`DynafaceRuntime` assembly) implements the same interface for Unity Inference Engine — it lives outside this repo, and (unlike `Dynaface.Onnx`) is not a NuGet package.

## Adding a New Measurement

Subclass `FaceMeasureBase` in `facial_dll/Measures/`, declare its `MeasureItemInfo` items in the constructor (mirrors dynaface-lib's per-field enable granularity), and implement `Label` and `Calc(FaceMeasureContext ctx, bool render = true)`. Register the new class in `facial_dll/DynafaceMeasures.cs` `AllMeasures()` (the library-owned registry mirroring dynaface-lib's `measures.all_measures()` — consumers run it via `FaceMeasureContext.Analyze()`, mirroring `AnalyzeFace.analyze()`). Use `ctx.Measure`, `ctx.MeasurePolygon`, and `ctx.AddHeader`/`ctx.AddValue`/`ctx.Values` for output. See `facial_dll/CLAUDE.md` for the full contract.
