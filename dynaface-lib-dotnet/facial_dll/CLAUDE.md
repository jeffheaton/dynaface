# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a pure C# face-processing pipeline with no Unity dependencies (`noEngineReferences: true` in `Dynaface.asmdef`). It targets `netstandard2.1` so it works both as a Unity package and as a regular .NET library. It's a contract-faithful port of the Python `dynaface-lib` package (see `../../dynaface-lib-python/`), adapted to .NET idioms (nullable types instead of sentinel tuples, PascalCase, typed DTOs instead of loose dicts) where Python's own idioms didn't fit.

**Solution layout** (rooted at `dynaface-lib-dotnet/`):
- `facial_dll/` — the core library (`FacialDll.csproj`, `netstandard2.1`), NuGet package `Dynaface`
- `FacialDll.Onnx/` — ONNX Runtime inference backend (`netstandard2.1`), NuGet package `Dynaface.Onnx`; holds `OnnxDynafaceInference`
- `FacialDllConsole/` — standalone console app / example (`net10.0`), uses OnnxRuntime (via `FacialDll.Onnx`) + SkiaSharp
- `DynafaceTests/` — xUnit tests (`net10.0`); pure unit tests always run, model-gated integration tests are reported as Skipped without the `.onnx` files, or fail under `DYNAFACE_REQUIRE_MODELS` (see `DynafaceTests/ModelFixture.cs` for the env vars)
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

   Both detectors, and `SagittalProfile` for U²-Net, check each model call's output (`ModelOutputCheck`): null from a ready backend, the wrong size, NaN or infinite values, a constant output, or all landmarks within one crop pixel throw `ModelOutputException` (mirroring `dynaface_onnx.ModelOutputError`), since a backend that computes garbage would otherwise look like "no face" or a crop error.

3. **`PoseClassifier`** — classifies the view, as `load_image()`'s pose block does: with `DynafaceConfig.AutoQuarter` on (off by default; it also needs `AutoLateral`), `IsQuarter` first (`Cheek.ClassifyQuarter`: a 35 to 55 degree landmark turn with the model's calibrated turn agreeing, and the nose tip not past the far contour); then `IsLateral` from headpose yaw + a nose-asymmetry ratio, gated by `DynafaceConfig.AutoLateral`; and a turn in the quarter window that the profile test declined is still a quarter face (`requireMargin: false`), never a frontal one. `Run(previousPose:)` holds a video in the quarter view a few degrees past the window.

4. **Frontal: `StyleGanCropper.Crop`** — optional tilt-threshold rotation correction (off by default), yaw-based foreshortening correction, scales so pupil distance becomes 260px, crops to 1024×1024 with the right pupil at (380,480), white-filling anything outside the source. `StyleGanCropper.CropLandmarks` is its landmark-only twin, for consumers that keep uncropped landmarks and need the cropped ones without the image (the Unity app's video frames); both run the same private `Plan` step, so they always agree to the pixel — keep any change to the crop's math inside `Plan`. **Lateral: flip-to-facing-left (re-running bbox+landmark detection on the flipped image, while the result keeps the head pose found in the photo as given, as `load_image` does) → pad canvas ×1.5 → `LateralCropper.Crop`** (fits the full landmark vertical band into 1024px with padding, horizontally placing the profile's leading edge a fixed margin from the left). **Quarter: `LateralCropper.Crop(center: true)`** — never flipped or straightened; the landmarks' band fills the 1024px crop with its pads (the band-fit scale) and is centered horizontally; `Pix2mm` is the pupil scale measured on the cropped landmarks (foreshortened by the turn, so the millimeter measures don't suit the view).

5. **Lateral only — `LateralAnalyzer.Analyze`, called internally by `FacePipeline.RunLateral`** — runs U²-Net background removal (the 3rd network) on the cropped image, extracts the sagittal (silhouette) profile, and runs it through a from-scratch Savitzky-Golay filter + peak/corner detector (`Lateral/` folder) to find 6 anatomical landmarks (Glabella/Nasion/NasalTip/Subnasal/MentoLabial/Pogonion; their display names are `LateralLandmarkFinder.LandmarkNames`, dynaface-lib's `LATERAL_LANDMARK_NAMES`). `RunLateral` then composites the sagittal chart onto the crop via `LateralChartRenderer` (the pixel-buffer equivalent of dynaface-lib's matplotlib chart + `_overlay_lateral_analysis`), so the returned crop already contains the chart, with measures drawing on top of it later — same layering as Python. **Quarter only:** `FacePipeline.RunQuarter` runs U²-Net on the crop through `SagittalProfile.ForegroundMask` (the alpha truncated to bytes, thresholded at `MaskAlpha` 32 and closed with the 10x10 kernel, as dynaface-lib's `lateral.foreground_mask()`), and `CheekOutline.TraceFarCheek` traces the far cheek's outline from it; the result carries `FarLeft` and the outline (`CheekOutlineX/Y/Src`), and `FaceMeasureContext.FromResult` builds a context with that view set. `Cheek.cs` and `CheekOutline.cs` are line-by-line ports of `cheek.py` and `cheek_outline.py` in double precision; `CheekOutlineTests` checks them exactly against the shared `cheek_outlines.json`, and `CheekTests` checks their constants against the Python source.

6. **`FaceMeasureContext` / `Measures/*.cs` / `DynafaceMeasures`** — one context per analysis pass, holding the final crop, landmarks, `Pix2mm` (measured on those landmarks from a pupillary distance in mm — see the units note below), the resolved pose/flipped flags, and (if lateral) the 6 lateral landmarks. Each `FaceMeasureBase` subclass declares named `MeasureItemInfo` sub-fields (mirroring dynaface-lib's `MeasureItem`/`is_enabled` system) and returns a `Dictionary<string, double>` from `Calc()`, in addition to whatever it draws. `DynafaceMeasures.AllMeasures()` is the canonical registry (mirrors `measures.all_measures()`, same order), and `ctx.Analyze()` is the runner (mirrors `AnalyzeFace.analyze()`): it runs every enabled measure and merges the result dicts. Consumers call `ctx.Analyze()` — they do not build their own measure lists or loops. `MeasureSelection` (mirroring `measures.MeasureSelection`) holds a user's choice of measures and items apart from their `Enabled` flags: `Apply(measures, lateral, quarter)` sets the flags for one face (chosen and suited to its view, the measure's and, for an item limited to some views, the item's) without losing a choice, which `UpdateForType` does, and `ToEntries`/`FromEntries` save it by label and item name.

7. **`FrameSeries`** — a video's outputs frame by frame (one row per frame, one column per enabled output), mirroring dynaface-lib's `dynaface.video.FrameSeries`: `Add` takes each frame's `Analyze()` results, `FromStates` measures saved `FaceAnalysisState`s again with `render: false`, and the series writes its CSV (`ToCsv`/`WriteCsv`: same text as Python's, values to 4 decimals, NaN empty), finds a peak frame (`Peak`), and computes the apps' Evaluation report numbers (`Evaluate`). Apps keep only the presentation (chart, report wording). `FromStates` also takes states saved without their image (the Unity app's landmark-only video frames), leaving out measures with `NeedsImage` (skin tone); that's .NET-only, since Python's states always hold their image. `VideoAnalyzer` (mirroring `dynaface.video.VideoAnalyzer`) is the import loop that makes those states: it skips frames without a face but counts them (`Frames` holds each kept frame's number in the video, passed on as `FromStates`' `frameNumbers`), smooths the cropped landmarks with `LandmarkSmoother` (truncated to whole pixels, like the desktop app's data smoothing), re-measures the scale on them, and stops after `DefaultMaxFrames`. Its .NET-only `keepImages: false` analyzes each frame uncropped and crops just the landmarks, the same landmarks for no image work.

`FacePipeline` is the single stateless entry point for the whole pipeline, steps 1–5. Call `Initialize(inference)` once at startup before calling `Run` — this also stashes `inference` for step 5's own use. `Run()` decides internally whether to call `LateralAnalyzer.Analyze` + chart overlay (populating `FacePipelineResult.LateralAnalysis`) exactly like dynaface-lib's `AnalyzeFace.load_image()` decides internally whether to call `analyze_lateral()` — the pose-gated branch lives inside the library in both ports, not in the caller. `Run(crop: false)` mirrors `load_image(crop=False)`: it skips the frontal StyleGAN crop (returning original-space landmarks; the lateral branch always crops, same quirk as Python). Do not reintroduce caller-side `if (result.IsLateral) ...` analysis/overlay branches or hand-built measure lists in `FacialDllConsole` or any other consumer; those were past architectural mismatches (library decisions leaking outside the DLL) that have since been fixed.

`FaceImage` has the image helpers apps need at the boundary: `FromRgba`/`ToRgba` (RGBA bytes, rows from the bottom or, with `topDown: true`, from the top, copied in bulk), `Rotate(quarterTurns)` (clockwise as seen) and `FlipHorizontal`/`FlipVertical`, each returning a copy.

## Coordinate System Contract

| Location | Convention |
|---|---|
| `FaceImage.Pixels` array | Bottom-left (y=0 at bottom) — Unity texture order |
| Landmark/bbox VALUES passed between pipeline stages | Top-left (y=0 at top) — matches dynaface-lib's own cv2/numpy convention, ported formulas transcribe verbatim |
| `FaceMeasureContext` inputs/API | Top-left; `BLY()` converts internally before calling `FaceRenderer` |
| `FaceRenderer` public methods | Bottom-left |

Mixing these up is the most common source of rendering bugs. `StyleGanCropper`/`LateralCropper`/`SagittalProfile` all follow the same pattern internally: flip the working pixel buffer to top-left ONCE at entry via `ImageUtils.FlipVertical` (so the ported Python formulas apply with zero sign changes), do all the math, flip the OUTPUT buffer back to bottom-left ONCE before constructing the returned `FaceImage`. Landmark VALUES never need a flip in that inner section — only pixel buffers do.

`SpigaLandmarkDetector`'s own internal crop-affine math is the one exception: it works in the source image's raw array-index space throughout (matching the bbox's own space), with the top-left conversion happening only once, right at its own return boundary.

## Units note: Pix2mm

A frontal millimeter value is a pixel distance times `Pix2mm`: the pupillary distance in mm over the distance between the pupils in pixels. That scale is only right for the landmarks it was measured on, so it is never carried from one set of landmarks to another. `FaceMeasureContext` measures it on its own `Landmarks` with `FaceMeasureContext.CalcPd` (the single copy of dynaface-lib's `util_calc_pd`), from the `pupilDistMm` it's given. Pass `FacePipelineResult.PupilDistMm`, or the copy saved with the landmarks, and the context gets the right scale however its landmarks got there: cropped by the pipeline, analyzed with `crop: false` and cropped later by `StyleGanCropper.CropLandmarks`, or restored by `FaceAnalysisPersistence.Restore`. Profiles use the fixed `DynafaceConstants.LateralPix2mm`. dynaface-lib does the same: `analyze()` re-measures `pix2mm` on the current landmarks before every run.

Don't reintroduce a constructor or code path that pairs a ready-made `Pix2mm` with landmarks it wasn't measured on. That pairing has broken three times:
- The scale was measured before the crop, from the source image's pupils, from July 2024 until October 2026 (about 2.3× too large on `img1-512.jpg`).
- The Unity app applied an uncropped run's scale to landmarks it cropped later.
- Saved analyses kept a scale from the old rule.

The explicit-scale constructor survives only as `[Obsolete]`, for existing callers. `ScaleConsistencyTests` (and dynaface-lib-python's `tests/test_scale.py`) measure one face every way and require the same millimeters.

The pupillary distance belongs to the analysis, not to today's setting. `FacePipelineResult.PupilDistMm` records the `DynafaceConfig.PupilDistMm` a run assumed. `Restore` recovers a saved one from the pair a state holds (`PupillaryDistance × Pix2mm`, which every version computed together), and that also corrects analyses saved under the old rule; dynaface-lib's `load_state` does the same.

`PupillaryDistance` is the post-crop pixel distance (close to, but not exactly, 260px by construction of `StyleGanCropper`; less when its yaw correction kicks in). `MeasurePosition`, like dynaface-lib's `AnalyzePosition`, recomputes its own `pd`/`px2mm` from the current landmarks with the context's `PupilDistMm`, so it always agrees with `ctx.Pix2mm`; `FacialPipelineIntegrationTests` asserts that.

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
| 96–97 | Pupil centers (96=right, 97=left) |

Indices 96 and 97 are the inter-pupil baseline for `Pix2mm`. The assumed IOD is `DynafaceConfig.PupilDistMm` — runtime-settable (mirrors dynaface-lib's mutable `AnalyzeFace.pd`, which the desktop app sets from user preferences), defaulting to the 63mm population average (`DynafaceConstants.StdPupilDistMm`).

## Adding a New Measurement

Subclass `FaceMeasureBase` in `facial_dll/Measures/`. In the constructor: add one `MeasureItemInfo` per named output field (or one group item, `outputs:`, standing for several outputs it draws together, as `MeasureCheekMeasures` and `MeasureLandmarks`' "Cheek Landmarks" do; a group's name is a label, never an output), set `IsFrontal`/`IsLateral`/`IsQuarter` (and an item's `views:` when it suits only some of them), call `SyncItems()`. Implement `Label` and `Calc(FaceMeasureContext ctx, bool render = true)`, gating each rendered piece by `IsEnabled("item-name")` (matching dynaface-lib's own per-field enable granularity — `Calc` should still return every declared item's value regardless of whether it's individually enabled or `render` is false; only rendering is gated, and choosing which outputs to show or export is the caller's job via `DynafaceMeasures.GetAllItems`). An item that is only a drawing switch, with no value of its own, is declared `displayOnly: true` (like `MeasureLateral`'s `tip_proj`): it is never returned or listed as an output. `DeclaredOutputsTests` checks every registered measure against this. Use `ctx.Measure`/`ctx.MeasurePolygon` for geometry, `ctx.AddHeader`/`ctx.AddValue`/`ctx.AddSpacer` for the sidebar text, and `ctx.AddValue(key, value)` / `ctx.Values` for structured numeric output that doesn't belong in the sidebar (see `MeasureLandmarks`). Register the new class in `facial_dll/DynafaceMeasures.cs`'s `AllMeasures()` — the library-owned registry mirroring dynaface-lib's `measures.all_measures()`; if dynaface-lib gains the same measure, keep the two lists in the same order.

## Runtime Interface

All 3 networks are defined on one interface, `IDynafaceInference` (`IDynafaceInference.cs`), kept as 3 independent methods (never fused into one call) so each can be swapped/tested on its own:

- `RunBlazeFace` — face bbox detector. Input: `float[1×128×128×3]` NHWC in [-1,1]. Output: `(regressors float[896×stride], scores float[896])`.
- `RunSpiga` — WFLW-98 landmark + headpose model. Input: `float[1×3×256×256]` NCHW in [0,1]. Output: `(landmarks float[98×2] normalized [0,1], pose float[6] raw [yaw,pitch,roll,tx,ty,tz])`. The model also takes two fixed inputs, `model3d` `[1×98×3]` and `cam_matrix` `[1×3×3]`; every backend feeds them from `SpigaModelInputs` (dynaface-lib's `_MODEL3D`/`_CAM_MATRIX`, checked by `SharedConstantsTests`) rather than keeping its own copy.
- `RunU2Net` — background/saliency segmentation. Input: `float[1×3×320×320]` NCHW, ImageNet-normalized. Output: `float[320×320]` raw sigmoid mask.

`FacialDll.Onnx/OnnxDynafaceInference.cs` (NuGet package `Dynaface.Onnx`) implements it via `Microsoft.ML.OnnxRuntime`, auto-detecting tensor names from model metadata (pass explicit overrides to the constructor, or use `--list-tensors`, if a new export uses unexpected names). The Unity adapter for the same interface lives in `DynafaceRuntime`, not here.
