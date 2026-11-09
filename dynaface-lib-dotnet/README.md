# DynaFace .NET

A C# face-processing pipeline — a contract-faithful port of the Python `dynaface-lib` package: BlazeFace detection, SPIGA WFLW-98 landmark + headpose estimation, U²-Net background removal, frontal (StyleGAN-style) and lateral analysis, and the full frontal + lateral + skin-tone measurement set.

## Projects

| Project | Target | NuGet package | Description |
|---|---|---|---|
| `facial_dll` | `netstandard2.1` | **`Dynaface`** | Core library — no ONNX or Unity dependencies |
| `FacialDll.Onnx` | `netstandard2.1` | **`Dynaface.Onnx`** | ONNX Runtime inference backend (`OnnxDynafaceInference`); depends on `Dynaface` + `Microsoft.ML.OnnxRuntime` |
| `FacialDllConsole` | `net10.0` | — (example only) | CLI harness; references both packages + SkiaSharp |
| `DynafaceTests` | `net10.0` | — | xUnit tests — unit tests always run; integration tests need local `.onnx` models (see below) |

The two published packages ship in **lockstep at the same version**: installing `Dynaface.Onnx` pulls in the matching `Dynaface`.

## Inference Backends

The core library runs no neural networks itself. All 3 networks (BlazeFace, SPIGA, U²-Net) are reached through a single interface, `IDynafaceInference` (in `facial_dll/`), which keeps them as 3 independent methods — never fused — so each can be swapped or tested on its own. `facial_dll` therefore has **no ONNX or Unity dependency**; the backend is a plug-in.

This seam exists because Dynaface ships with **two different inference backends** for two different runtimes:

| Backend | Runtime | Implementation | Location |
|---|---|---|---|
| ONNX | `Microsoft.ML.OnnxRuntime` | `OnnxDynafaceInference` | `FacialDll.Onnx/` (this repo) → NuGet package `Dynaface.Onnx` |
| Unity Inference Engine | Unity Sentis | `DynafaceRuntime` adapter | Separate `DynafaceRuntime` assembly (outside this repo) |

Both implement the same `IDynafaceInference` and hand it to the library via `FacePipeline.Initialize(...)`; everything downstream (detection, landmarks, pose, cropping, lateral analysis, measurements) is backend-agnostic and only ever talks to the interface. Most .NET consumers just install `Dynaface.Onnx` and use the ready-made ONNX backend; to add another backend (as the Unity adapter does), implement the interface's 3 methods — see [Using the Library from Another App](#using-the-library-from-another-app) below.

## Build

```bash
# Build all 4 projects
dotnet build FacialDll.sln -c Release

# Build just the core library (e.g. to produce a DLL for Unity)
dotnet build facial_dll/FacialDll.csproj -c Release
# Output: facial_dll/bin/Release/netstandard2.1/Dynaface.dll

# Build just the ONNX backend
dotnet build FacialDll.Onnx/FacialDll.Onnx.csproj -c Release

# Build just the console harness
dotnet build FacialDllConsole/FacialDllConsole.csproj -c Release
```

### Pack the NuGet packages

```bash
dotnet pack facial_dll/FacialDll.csproj       -c Release -o ./nupkgs   # Dynaface.<version>.nupkg
dotnet pack FacialDll.Onnx/FacialDll.Onnx.csproj -c Release -o ./nupkgs # Dynaface.Onnx.<version>.nupkg
```

Both take `<Version>` from their own `.csproj`; keep the two in lockstep. Each also
writes a `.snupkg` of symbols beside its `.nupkg`.

### Publishing to NuGet (maintainers)

Releases take two manually dispatched (`workflow_dispatch`) workflows, the same way
the Python library goes to PyPI. [RELEASING.md](../RELEASING.md) has the whole
sequence for both languages.

1. [Build Library (.NET)](../.github/workflows/build-lib-dotnet.yml) builds and runs
   the full test suite (it downloads the ONNX models and requires the integration
   tests to run). It then packs both packages, each `.nupkg` with a `.snupkg` of
   symbols (Source Link points a debugger at the commit's sources on GitHub). It
   stages `Dynaface.<version>.nupkg`, `Dynaface.Onnx.<version>.nupkg`, their
   `.snupkg` files and a build record, `Dynaface-<version>-build.json` (the commit,
   run and build number), on `s3://data.heatonresearch.com/library/`. It does not
   publish, and rerunning it at the same version replaces what is staged.
2. [Deploy Library to NuGet (.NET)](../.github/workflows/deploy-lib-dotnet.yml), given
   that version, downloads the staged files from S3, without rebuilding anything.
   - It pushes both packages to NuGet.org, and each push carries its `.snupkg` to the
     symbol server.
   - It publishes via **Trusted Publishing**: the `NuGet/login` action exchanges
     GitHub's OIDC token for a short-lived key, so there is **no long-lived API key
     stored in the repo**.
   - It then tags `v<version>` at the commit in the build record. The Python
     library's deploy shares the tag. If the tag already stands at another commit,
     the deploy refuses before pushing anything.

One-time setup on NuGet.org (Account → **Trusted Publishing**): a policy bound to
this repository (`jeffheaton/dynaface`) and the workflow file
`deploy-lib-dotnet.yml`, which is already in place. A new policy is provisional and
expires if it isn't used within about a week.

To cut a release:
1. Bump `<Version>` in both `.csproj` files, keeping them in lockstep with the
   Python library's `setup.cfg`.
2. Run the build.
3. Run the deploy with that version.

Pushes use `--skip-duplicate`, so deploying an already-published version is a no-op
rather than an error.

## Test

```bash
dotnet test DynafaceTests/DynafaceTests.csproj
```

Unit tests (geometry helpers, the Savitzky-Golay filter and peak/corner finder against scipy reference vectors, NMS, the measure item-enable system, etc.) have no external dependency and always run. The model-gated integration tests (full pipeline runs against the bundled `DynafaceTests/TestData/*.jpg` images, persistence round-trips) need the 3 real ONNX models locally — they're too large to commit (~400MB total, already gitignored). Point at a directory containing `blaze_face_short_range.onnx`, `spiga_wflw.onnx`, and `u2net.onnx` via:

```bash
export DYNAFACE_ONNX_MODELS=/path/to/models
dotnet test DynafaceTests/DynafaceTests.csproj
```

or place them at `~/.dynaface/models` (dynaface-lib's own default download location). When `DYNAFACE_ONNX_MODELS` is set it is the only place looked. Without the models, those tests are reported as Skipped; set `DYNAFACE_REQUIRE_MODELS=1` to make them fail instead (CI does).

Every push or pull request that touches `dynaface-lib-dotnet/` runs the [Test Library (.NET)](../.github/workflows/test-lib-dotnet.yml) workflow: the full suite with the models required, with coverage (excluding the `FacialDllConsole` example) uploaded to [Codecov](https://app.codecov.io/gh/jeffheaton/dynaface) for the coverage badges.

## Run

```bash
dotnet run --project FacialDllConsole -- <image> <blazeface.onnx> <spiga.onnx> <u2net.onnx>
```

The console app detects the face, classifies frontal vs. lateral pose, runs the appropriate crop + measurements (including lateral-only nasofrontal/nasolabial angle and tip-projection measures when applicable), and writes an annotated PNG alongside the source image. The image argument may also be an http(s) URL (mirroring dynaface-lib's `load_face_image`); the annotated PNG then lands in the current directory.

```bash
# Example
dotnet run --project FacialDllConsole -- photo.jpg \
  models/blaze_face_short_range.onnx models/spiga_wflw.onnx models/u2net.onnx
```

## Using the Library from Another App

### NuGet packages (recommended)

```bash
# Core pipeline + the ready-made ONNX Runtime backend (pulls in Dynaface + OnnxRuntime)
dotnet add package Dynaface.Onnx

# ...or just the backend-agnostic core, if you supply your own inference backend
dotnet add package Dynaface
```

With `Dynaface.Onnx` you skip step 1 and step 2 below entirely — construct `OnnxDynafaceInference` directly (see step 3). Only implement `IDynafaceInference` yourself (steps 1–2) if you need a different backend, e.g. the Unity Inference Engine adapter.

### Prebuilt DLL

Each CI run of the [Build Library (.NET)](../.github/workflows/build-lib-dotnet.yml)
workflow publishes a zipped `netstandard2.1` build to:

    https://data.heatonresearch.com/library/dynaface-dotnet-<VERSION>.zip

where `<VERSION>` is the `<Version>` in `facial_dll/FacialDll.csproj` (currently `2.1.0`).
The zip contains:

| File | Needed? | Purpose |
|---|---|---|
| `Dynaface.dll` | Required | The library — reference this to use it |
| `Dynaface.pdb` | Optional | Debug symbols (source file/line info for stack traces and debugging) |

Zips prior to `2.0.2` shipped the same assembly under its old name, `FacialDll.dll`.

Unzip and reference `Dynaface.dll` directly instead of building from source or
using a `ProjectReference`. Note the S3 filename is version-only (no build-number
suffix), so re-running the workflow at the same version overwrites the same URL —
there is no stable `latest` alias.

### 1. Reference the core library (custom-backend path only)

Skip this if you installed `Dynaface.Onnx` above. To build your own backend, reference just the core package (or the project) plus whatever inference runtime you're wrapping:

```xml
<!-- In your .csproj -->
<ItemGroup>
  <PackageReference Include="Dynaface" Version="2.1.0" />
  <!-- Plus your inference runtime, e.g.: -->
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.19.2" />
</ItemGroup>
```

### 2. Implement the inference interface (custom-backend path only)

`facial_dll` has no ONNX dependency — you supply inference by implementing `IDynafaceInference`'s 3 methods (kept independent on purpose, so each network can be swapped/tested on its own). This is exactly what the `Dynaface.Onnx` package's `OnnxDynafaceInference` already does, so implement it yourself only for a non-ONNX runtime:

```csharp
public class MyDynafaceInference : IDynafaceInference
{
    public bool IsReady => true;

    // BlazeFace short-range
    //   in:  NHWC float[1×128×128×3], values in [-1, 1]
    //   out: (regressors float[896×stride], scores float[896]), or null on failure
    public (float[] regressors, float[] scores)? RunBlazeFace(float[] tensor) { ... }

    // SPIGA WFLW-98
    //   in:  NCHW float[1×3×256×256], values in [0, 1], y=0 at top
    //        (the model's two fixed inputs, model3d [1×98×3] and cam_matrix [1×3×3],
    //        come from SpigaModelInputs.Model3d() and SpigaModelInputs.CameraMatrix())
    //   out: (landmarks flat float[98×2] normalized to [0,1],
    //         pose float[6] raw [yaw, pitch, roll, tx, ty, tz]), or null on failure
    public (float[] landmarks, float[] pose)? RunSpiga(float[] imageTensor) { ... }

    // U^2-Net background/saliency segmentation
    //   in:  NCHW float[1×3×320×320], ImageNet-normalized ((px/255 - mean) / std)
    //   out: flat float[320×320] raw sigmoid saliency mask (first of the model's
    //        multi-scale outputs), or null on failure
    public float[] RunU2Net(float[] imageTensor) { ... }

    public void Dispose() { /* release sessions */ }
}
```

See `FacialDll.Onnx/OnnxDynafaceInference.cs` (the `Dynaface.Onnx` package) for a complete `Microsoft.ML.OnnxRuntime` implementation.

### 3. Initialize and run the pipeline

```csharp
// One-time setup. With the Dynaface.Onnx package, use the ready-made backend:
FacePipeline.Initialize(new OnnxDynafaceInference(blazeFacePath, spigaPath, u2netPath));
// ...or, with a custom backend from steps 1–2:
// FacePipeline.Initialize(new MyDynafaceInference());

// Build a FaceImage from your pixel data
// Pixels must be a flat Rgba32[] in bottom-left row-major order (y=0 at bottom)
var image = new FaceImage(pixels, width, height);

// ...or from 8-bit RGBA bytes: rows from the bottom by default (Unity's
// Texture2D.GetRawTextureData for RGBA32), or from the top, as most decoders give them.
// image.ToRgba(topDown: true) gives the bytes back. Rotate(quarterTurns) turns a copy
// clockwise as the image is seen, and FlipHorizontal/FlipVertical mirror a copy.
image = FaceImage.FromRgba(rgbaBytes, width, height, topDown: true);
FaceImage upright = image.Rotate(1);

// Run detection + pose classification + crop
FacePipelineResult? result = FacePipeline.Run(image, rotationAngle: 0, flipHorizontal: false);
// flipHorizontal: true for live camera frames on iOS; false for file/gallery images

if (result == null)
{
    // No face detected — optionally retry at 90°
    result = FacePipeline.Run(image, rotationAngle: 90, flipHorizontal: false);
}

// A backend that runs the models but computes them wrongly (as GPU compute on some
// virtual and software graphics adapters does) gives output no working model gives:
// none, NaN values, the same value everywhere, or landmarks all at one point. Run()
// throws a ModelOutputException for it, rather than reporting no face; run the models
// on the CPU instead.

// Run() already performed lateral background-removal analysis internally when the
// pose came back lateral (mirroring dynaface-lib's load_image) — just read it back.
Vec2[] lateralLandmarks = result?.LateralAnalysis?.LateralLandmarks;
```

### 4. Run measurements

```csharp
// Optional: calibrate the pupillary distance (mm) used for px→mm conversion.
// Mirrors dynaface-lib's settable AnalyzeFace.pd; defaults to the 63mm average. It must
// be from 40 to 85 mm (DynafaceConstants.MinPupilDistMm/MaxPupilDistMm): no person's
// pupils are closer or farther apart, so another value throws
// ArgumentOutOfRangeException. A distance passed straight to FaceMeasureContext, or
// restored with a saved state, isn't checked: DynafaceConfig.CheckPupilDistMm checks one.
DynafaceConfig.PupilDistMm = 63f;

// Optional: the size of the text the measures draw, as dynaface-lib's
// AnalyzeFace.text_size: 0.75 by default, and 1.5 draws it twice as large. It sets
// FaceRenderer's glyph scales for the measures' labels and the landmark numbers.
DynafaceConfig.TextSize = 0.75f;

// The context measures its millimeters-per-pixel scale on the landmarks it's given,
// assuming the pupillary distance the run did.
var ctx = new FaceMeasureContext(
    result.Value.AlignedCrop, result.Value.Wflw98,
    isLateral: result.Value.IsLateral, lateralLandmarks: lateralLandmarks,
    headPose: result.Value.HeadPose,
    pose: result.Value.Pose, flipped: result.Value.Flipped,
    pupilDistMm: result.Value.PupilDistMm);

// Runs the full library measure registry (DynafaceMeasures.AllMeasures()) and
// merges every measure's results — mirrors dynaface-lib's AnalyzeFace.analyze().
Dictionary<string, double> results = ctx.Analyze();

// Or pass an explicit subset:
// ctx.Analyze(new FaceMeasureBase[] { new MeasureIntercanthalDistance(), new MeasureEyeArea() });

// Or run only the measures a user chose that fit this face. A MeasureSelection keeps the
// choices apart from the measures' Enabled flags (mirrors dynaface-lib's
// measures.MeasureSelection), so applying it to frontal, three-quarter and profile faces
// in turn never loses one, as FaceMeasureBase.UpdateForType does. ToEntries/FromEntries
// save and load it by the measures' labels and the items' names.
var measures = DynafaceMeasures.AllMeasures();
var selection = MeasureSelection.Of(measures);   // take it while the flags hold the choices
selection.Apply(measures, result.Value.IsLateral, quarter: result.Value.IsQuarter);
Dictionary<string, double> chosenResults = ctx.Analyze(measures);

// Annotated pixel buffer
FaceImage annotated = ctx.ToImage();

// Sidebar text lines (label + value pairs)
foreach (string line in ctx.TextLines)
    Console.WriteLine(line);

// Structured numeric output (e.g. MeasureLandmarks' flat coordinate dump)
foreach (var kv in ctx.Values)
    Console.WriteLine($"{kv.Key}={kv.Value}");
```

### 5. Persistence

```csharp
// Snapshot after a pipeline run (clones the pixel array, so later drawing on a
// live FaceMeasureContext can't retroactively mutate the snapshot)
FaceAnalysisState state = FaceAnalysisPersistence.Capture(result.Value);

// ...later, rebuild a context and re-run measurements against it. It keeps the
// pupillary distance the analysis assumed, even if DynafaceConfig.PupilDistMm has
// changed since, and measures its scale on the saved landmarks. The state keeps the
// view too (pose, flip, and a profile's lateral landmarks and sagittal outline), so a
// restored profile measures as the original did, and the context draws on a copy of
// the saved image, leaving the state as it was.
FaceMeasureContext restored = FaceAnalysisPersistence.Restore(state);

// Saving landmarks your own way (e.g. a video's frames, cropped later with
// StyleGanCropper.CropLandmarks)? Save result.PupilDistMm with them and pass it back
// as pupilDistMm, never a saved Pix2mm: a scale only fits the landmarks it was
// measured on.
```

### 6. Video: measuring frame by frame

```csharp
// Import a video the way the apps do: frames without a face are skipped but counted,
// landmarks can be averaged over the last frames, and only the first 5000 frames are
// analyzed. keepImages: false keeps each frame's landmarks but not its 1024x1024 crop.
var analyzer = new VideoAnalyzer(smoothing: 2, keepImages: false);
foreach (FaceImage frame in frames)   // upright, in order; analyzer.Skip() for one you can't read
    if (!analyzer.IsFull) analyzer.Add(frame);
FrameSeries fromVideo = analyzer.Series(fps: 30f);   // rows numbered as in the video
// Peak and Evaluate report frames by those numbers; StateIndex finds a frame's state
// (once a frame is skipped, the numbers and the positions in analyzer.States differ).
FaceAnalysisState smile = analyzer.States[analyzer.StateIndex(fromVideo.Peak("dental_area").Value)];

// Or any saved states (FaceAnalysisPersistence.Capture), measured again without
// drawing: a row per frame, a column per output. A state saved without its image is
// measured on its landmarks alone, leaving out skin tone, which needs the pixels.
FrameSeries series = FrameSeries.FromStates(states, fps: 30f);
File.WriteAllText("video.csv", series.ToCsv());     // frame,time,<outputs...>

int? widestSmile = series.Peak("dental_area");
int? widestEyes = series.Peak("eye.left", "eye.right");

// The eyes at the widest smile and the mouth at the widest eye opening, as the apps'
// Evaluation report shows them (null without the dental and eye outputs).
FrameSeries.Evaluation evaluation = FrameSeries
    .FromStates(states, new FaceMeasureBase[] { new MeasureDentalArea(), new MeasureEyeArea() })
    .Evaluate();
```

### Options

| Flag | Description |
|---|---|
| `--list-tensors` | Print the auto-detected input/output tensor names for all 3 models and exit. Useful when first using a new ONNX export. |

```bash
dotnet run --project FacialDllConsole -- photo.jpg blaze_face_short_range.onnx spiga_wflw.onnx u2net.onnx --list-tensors
```
