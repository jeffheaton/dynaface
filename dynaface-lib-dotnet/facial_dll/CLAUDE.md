# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a pure C# face-processing pipeline with no Unity dependencies (`noEngineReferences: true` in `FacialDll.asmdef`). It targets `netstandard2.1` so it works both as a Unity package and as a regular .NET library. It's a contract-faithful port of the Python `dynaface-lib` package (see `../../dynaface-lib-python/`), adapted to .NET idioms (nullable types instead of sentinel tuples, PascalCase, typed DTOs instead of loose dicts) where Python's own idioms didn't fit.

**Solution layout** (rooted at `dynaface-lib-dotnet/`):
- `facial_dll/` — the core library (`FacialDll.csproj`, `netstandard2.1`), NuGet package `Dynaface`
- `FacialDll.Onnx/` — ONNX Runtime inference backend (`netstandard2.1`), NuGet package `Dynaface.Onnx`; holds `OnnxDynafaceInference`
- `FacialDllConsole/` — standalone console app / example (`net10.0`), uses OnnxRuntime (via `FacialDll.Onnx`) + SkiaSharp
- `DynafaceTests/` — xUnit tests (`net10.0`); pure unit tests always run, model-gated integration tests self-skip without local `.onnx` files (see its own files for the env var)
- `FacialDll.sln` — solution file covering all four projects

## Build & Run

```bash
# From dynaface-lib-dotnet/
dotnet build FacialDll.sln
dotnet test DynafaceTests/DynafaceTests.csproj
dotnet run --project FacialDllConsole -- photo.jpg blaze_face_short_range.onnx spiga_wflw.onnx u2net.onnx

# Print auto-detected tensor names (useful when first using a new ONNX export)
dotnet run --project FacialDllConsole -- photo.jpg blaze_face_short_range.onnx spiga_wflw.onnx u2net.onnx --list-tensors
```

The Unity adapter layer (model runners, camera source, app controller) lives in a separate `DynafaceRuntime` assembly.

## Pipeline Architecture

Mirrors dynaface-lib's `AnalyzeFace.load_image` order exactly (not the pipeline's own earlier design, which aligned directly off BlazeFace's eye keypoints before landmarks even ran):

1. **`BlazeFaceDetector.TryDetectBbox`** — runs the BlazeFace short-range model (128×128 NHWC, values in [-1,1]) via `IDynafaceInference.RunBlazeFace`, decodes + NMS's all above-threshold anchors, and returns a face bbox in the source image's own raw pixel-index space. Rotation-retry (0/90/180/270) and the eye-keypoint sanity heuristic (`LastDetectionEyesOk`) are .NET-only additions with no Python equivalent — Python assumes a single, already-upright orientation.

2. **`SpigaLandmarkDetector.Detect`** — given the bbox, builds SPIGA's own crop (pad to a square of `max(w,h)*1.6`, centered on the bbox, warped to 256×256 via `ImageUtils.WarpAffine`, nearest-neighbor + zero-fill), runs SPIGA via `IDynafaceInference.RunSpiga`, and inverse-maps the 98 landmarks back into the source image's own pixel space, converting to **top-left semantic coordinates** as the final step. Also returns SPIGA's raw 6-value headpose `[yaw,pitch,roll,tx,ty,tz]`.

3. **`PoseClassifier`** — classifies frontal vs. lateral from headpose yaw + a nose-asymmetry ratio, gated by `DynafaceConfig.AutoLateral`.

4. **Frontal: `StyleGanCropper.Crop`** — optional tilt-threshold rotation correction (off by default), yaw-based foreshortening correction, scales so pupil distance becomes 260px, crops to 1024×1024 with the right pupil at (380,480), white-filling anything outside the source. **Lateral: flip-to-facing-left (re-running bbox+landmark detection on the flipped image) → pad canvas ×1.5 → `LateralCropper.Crop`** (fits the full landmark vertical band into 1024px with padding, anchored horizontally to the right pupil).

5. **Lateral only — `LateralAnalyzer.Analyze`, called internally by `FacePipeline.RunLateral`** — runs U²-Net background removal (the 3rd network) on the cropped image, extracts the sagittal (silhouette) profile, and runs it through a from-scratch Savitzky-Golay filter + peak/corner detector (`Lateral/` folder) to find 6 anatomical landmarks (Glabella/Nasion/NasalTip/Subnasal/MentoLabial/Pogonion). `RunLateral` then composites the sagittal chart onto the crop via `LateralChartRenderer` (the pixel-buffer equivalent of dynaface-lib's matplotlib chart + `_overlay_lateral_analysis`), so the returned crop already contains the chart, with measures drawing on top of it later — same layering as Python.

6. **`FaceMeasureContext` / `Measures/*.cs` / `DynafaceMeasures`** — one context per analysis pass, holding the final crop, landmarks, `Pix2mm` (supplied by the pipeline — see the coordinate/units note below, never recomputed from post-crop landmarks internally), the resolved pose/flipped flags, and (if lateral) the 6 lateral landmarks. Each `FaceMeasureBase` subclass declares named `MeasureItemInfo` sub-fields (mirroring dynaface-lib's `MeasureItem`/`is_enabled` system) and returns a `Dictionary<string, double>` from `Calc()`, in addition to whatever it draws. `DynafaceMeasures.AllMeasures()` is the canonical registry (mirrors `measures.all_measures()`, same order), and `ctx.Analyze()` is the runner (mirrors `AnalyzeFace.analyze()`): it runs every enabled measure and merges the result dicts. Consumers call `ctx.Analyze()` — they do not build their own measure lists or loops.

`FacePipeline` is the single stateless entry point for the whole pipeline, steps 1–5. Call `Initialize(inference)` once at startup before calling `Run` — this also stashes `inference` for step 5's own use. `Run()` decides internally whether to call `LateralAnalyzer.Analyze` + chart overlay (populating `FacePipelineResult.LateralAnalysis`) exactly like dynaface-lib's `AnalyzeFace.load_image()` decides internally whether to call `analyze_lateral()` — the pose-gated branch lives inside the library in both ports, not in the caller. `Run(crop: false)` mirrors `load_image(crop=False)`: it skips the frontal StyleGAN crop (returning original-space landmarks; the lateral branch always crops, same quirk as Python). Do not reintroduce caller-side `if (result.IsLateral) ...` analysis/overlay branches or hand-built measure lists in `FacialDllConsole` or any other consumer; those were past architectural mismatches (library decisions leaking outside the DLL) that have since been fixed.

## Coordinate System Contract

| Location | Convention |
|---|---|
| `FaceImage.Pixels` array | Bottom-left (y=0 at bottom) — Unity texture order |
| Landmark/bbox VALUES passed between pipeline stages | Top-left (y=0 at top) — matches dynaface-lib's own cv2/numpy convention, ported formulas transcribe verbatim |
| `FaceMeasureContext` inputs/API | Top-left; `BLY()` converts internally before calling `FaceRenderer` |
| `FaceRenderer` public methods | Bottom-left |

Mixing these up is the most common source of rendering bugs. `StyleGanCropper`/`LateralCropper`/`SagittalProfile` all follow the same pattern internally: flip the working pixel buffer to top-left ONCE at entry via `ImageUtils.FlipVertical` (so the ported Python formulas apply with zero sign changes), do all the math, flip the OUTPUT buffer back to bottom-left ONCE before constructing the returned `FaceImage`. Landmark VALUES never need a flip in that inner section — only pixel buffers do.

`SpigaLandmarkDetector`'s own internal crop-affine math is the one exception: it works in the source image's raw array-index space throughout (matching the bbox's own space), with the top-left conversion happening only once, right at its own return boundary.

## Units note: Pix2mm is not always what it looks like

`FacePipelineResult.Pix2mm` (frontal case) is computed from the **pre-crop** pupil distance in the original source image — not re-derived from the post-crop landmarks that `FaceMeasureContext.Landmarks` actually holds, even though those end up close to (but not exactly) a 260px pupil distance by construction of `StyleGanCropper`. This exactly matches dynaface-lib's own `calc_pd()` timing (it runs before `crop_stylegan()`).

`MeasurePosition` is the one exception: like dynaface-lib's `AnalyzePosition`, it recomputes its own `pd`/`px2mm` fresh from the **current** (post-crop) landmarks rather than using `ctx.Pix2mm` — these two numbers are deliberately different fields with different meanings. Getting this backwards produces a `px2mm` off by roughly the crop's own scale factor (a mistake this port made once and caught by actually running the pipeline against real images rather than reasoning about it in the abstract — see git history).

## WFLW 98 Landmark Index Reference

| Range | Region |
|---|---|
| 0–32 | Face contour |
| 33–41 | Image-left eyebrow |
| 42–50 | Image-right eyebrow |
| 51–54 | Nose bridge/tip |
| 55–59 | Nose base |
| 60–67 | Image-left eye |
| 68–75 | Image-right eye |
| 76–87 | Outer lip |
| 88–95 | Inner lip |
| 96–97 | Pupil centres (96=right, 97=left) |

Indices 96 and 97 are the inter-pupil baseline for `Pix2mm`. The assumed IOD is `DynafaceConfig.PupilDistMm` — runtime-settable (mirrors dynaface-lib's mutable `AnalyzeFace.pd`, which the desktop app sets from user preferences), defaulting to the 63mm population average (`DynafaceConstants.StdPupilDistMm`).

## Adding a New Measurement

Subclass `FaceMeasureBase` in `facial_dll/Measures/`. In the constructor: add one `MeasureItemInfo` per named output field, set `IsFrontal`/`IsLateral`, call `SyncItems()`. Implement `Label` and `Calc(FaceMeasureContext ctx, bool render = true)`, gating each rendered piece by `IsEnabled("item-name")` (matching dynaface-lib's own per-field enable granularity — `Calc` should still return every declared item's value regardless of whether it's individually enabled, only rendering is gated). Use `ctx.Measure`/`ctx.MeasurePolygon` for geometry, `ctx.AddHeader`/`ctx.AddValue`/`ctx.AddSpacer` for the sidebar text, and `ctx.AddValue(key, value)` / `ctx.Values` for structured numeric output that doesn't belong in the sidebar (see `MeasureLandmarks`). Register the new class in `facial_dll/DynafaceMeasures.cs`'s `AllMeasures()` — the library-owned registry mirroring dynaface-lib's `measures.all_measures()`; if dynaface-lib gains the same measure, keep the two lists in the same order.

## Runtime Interface

All 3 networks are defined on one interface, `IDynafaceInference` (`IDynafaceInference.cs`), kept as 3 independent methods (never fused into one call) so each can be swapped/tested on its own:

- `RunBlazeFace` — face bbox detector. Input: `float[1×128×128×3]` NHWC in [-1,1]. Output: `(regressors float[896×stride], scores float[896])`.
- `RunSpiga` — WFLW-98 landmark + headpose model. Input: `float[1×3×256×256]` NCHW in [0,1]. Output: `(landmarks float[98×2] normalized [0,1], pose float[6] raw [yaw,pitch,roll,tx,ty,tz])`.
- `RunU2Net` — background/saliency segmentation. Input: `float[1×3×320×320]` NCHW, ImageNet-normalized. Output: `float[320×320]` raw sigmoid mask.

`FacialDll.Onnx/OnnxDynafaceInference.cs` (NuGet package `Dynaface.Onnx`) implements it via `Microsoft.ML.OnnxRuntime`, auto-detecting tensor names from model metadata (pass explicit overrides to the constructor, or use `--list-tensors`, if a new export uses unexpected names). The Unity adapter for the same interface lives in `DynafaceRuntime`, not here.
