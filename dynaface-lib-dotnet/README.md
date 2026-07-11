# DynaFace .NET

A C# face-processing pipeline — a contract-faithful port of the Python `dynaface-lib` package: BlazeFace detection, SPIGA WFLW-98 landmark + headpose estimation, U²-Net background removal, frontal (StyleGAN-style) and lateral analysis, and the full frontal + lateral + skin-tone measurement set.

## Projects

| Project | Target | Description |
|---|---|---|
| `facial_dll` | `netstandard2.1` | Core library — no ONNX or Unity dependencies |
| `FacialDllConsole` | `net10.0` | CLI harness using OnnxRuntime + SkiaSharp |
| `DynafaceTests` | `net10.0` | xUnit tests — unit tests always run; integration tests need local `.onnx` models (see below) |

## Build

```bash
# Build all 3 projects
dotnet build FacialDll.sln -c Release

# Build just the library (e.g. to produce a DLL for Unity)
dotnet build facial_dll/FacialDll.csproj -c Release
# Output: facial_dll/bin/Release/netstandard2.1/FacialDll.dll

# Build just the console harness
dotnet build FacialDllConsole/FacialDllConsole.csproj -c Release
```

## Test

```bash
dotnet test DynafaceTests/DynafaceTests.csproj
```

Unit tests (geometry helpers, the Savitzky-Golay filter and peak/corner finder against scipy reference vectors, NMS, the measure item-enable system, etc.) have no external dependency and always run. The model-gated integration tests (full pipeline runs against the bundled `DynafaceTests/TestData/*.jpg` images, persistence round-trips) need the 3 real ONNX models locally — they're too large to commit (~400MB total, already gitignored). Point at a directory containing `blaze_face_short_range.onnx`, `spiga_wflw.onnx`, and `u2net.onnx` via:

```bash
export DYNAFACE_ONNX_MODELS=/path/to/models
dotnet test DynafaceTests/DynafaceTests.csproj
```

or place them at `~/.dynaface/models` (dynaface-lib's own default download location). Without a resolvable directory, those tests no-op (pass without exercising the pipeline) rather than fail.

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

### 1. Reference the project (or build a NuGet package)

```xml
<!-- In your .csproj -->
<ItemGroup>
  <ProjectReference Include="path/to/facial_dll/FacialDll.csproj" />
  <!-- Plus your ONNX runtime, e.g.: -->
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.19.2" />
</ItemGroup>
```

### 2. Implement the inference interface

`facial_dll` has no ONNX dependency — you supply inference by implementing `IDynafaceInference`'s 3 methods (kept independent on purpose, so each network can be swapped/tested on its own):

```csharp
public class MyDynafaceInference : IDynafaceInference
{
    public bool IsReady => true;

    // BlazeFace short-range (128×128 NHWC, values in [-1, 1])
    public (float[] regressors, float[] scores)? RunBlazeFace(float[] tensor) { ... }

    // SPIGA WFLW-98 (256×256 NCHW, values in [0, 1])
    // Returns 98 normalized (x,y) landmark pairs + a raw 6-value
    // [yaw, pitch, roll, tx, ty, tz] headpose.
    public (float[] landmarks, float[] pose)? RunSpiga(float[] imageTensor) { ... }

    // U^2-Net background/saliency segmentation (320×320 NCHW, ImageNet-normalized)
    public float[] RunU2Net(float[] imageTensor) { ... }

    public void Dispose() { /* release sessions */ }
}
```

See `FacialDllConsole/OnnxDynafaceInference.cs` for a complete `Microsoft.ML.OnnxRuntime` implementation.

### 3. Initialize and run the pipeline

```csharp
// One-time setup
FacePipeline.Initialize(new MyDynafaceInference());

// Build a FaceImage from your pixel data
// Pixels must be a flat Rgba32[] in bottom-left row-major order (y=0 at bottom)
var image = new FaceImage(pixels, width, height);

// Run detection + pose classification + crop
FacePipelineResult? result = FacePipeline.Run(image, rotationAngle: 0, flipHorizontal: false);
// flipHorizontal: true for live camera frames on iOS; false for file/gallery images

if (result == null)
{
    // No face detected — optionally retry at 90°
    result = FacePipeline.Run(image, rotationAngle: 90, flipHorizontal: false);
}

// Run() already performed lateral background-removal analysis internally when the
// pose came back lateral (mirroring dynaface-lib's load_image) — just read it back.
Vec2[] lateralLandmarks = result?.LateralAnalysis?.LateralLandmarks;
```

### 4. Run measurements

```csharp
// Optional: calibrate the pupillary distance (mm) used for px→mm conversion.
// Mirrors dynaface-lib's settable AnalyzeFace.pd; defaults to the 63mm average.
DynafaceConfig.PupilDistMm = 63f;

var ctx = new FaceMeasureContext(
    result.Value.AlignedCrop, result.Value.Wflw98, result.Value.Pix2mm,
    isLateral: result.Value.IsLateral, lateralLandmarks: lateralLandmarks,
    headPose: result.Value.HeadPose,
    pose: result.Value.Pose, flipped: result.Value.Flipped);

// Runs the full library measure registry (DynafaceMeasures.AllMeasures()) and
// merges every measure's results — mirrors dynaface-lib's AnalyzeFace.analyze().
Dictionary<string, double> results = ctx.Analyze();

// Or pass an explicit subset:
// ctx.Analyze(new FaceMeasureBase[] { new MeasureIntercanthalDistance(), new MeasureEyeArea() });

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

// ...later, rebuild a context and re-run measurements against it
FaceMeasureContext restored = FaceAnalysisPersistence.Restore(state);
```

### Options

| Flag | Description |
|---|---|
| `--list-tensors` | Print the auto-detected input/output tensor names for all 3 models and exit. Useful when first using a new ONNX export. |

```bash
dotnet run --project FacialDllConsole -- photo.jpg blaze_face_short_range.onnx spiga_wflw.onnx u2net.onnx --list-tensors
```
