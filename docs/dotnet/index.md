# .NET library

The .NET library is a C# port of the Python library, with the same steps and formulas. It targets .NET Standard 2.1, so it runs in .NET apps and in Unity.

| Package | What it is |
|---|---|
| [`Dynaface`](https://www.nuget.org/packages/Dynaface) | The pipeline and the measures. It runs no neural networks itself; it reaches all three through the `IDynafaceInference` interface. |
| [`Dynaface.Onnx`](https://www.nuget.org/packages/Dynaface.Onnx) | `OnnxDynafaceInference`, a ready-made `IDynafaceInference` that runs the models with ONNX Runtime. |

## Install

```bash
dotnet add package Dynaface.Onnx
```

This brings in `Dynaface` and ONNX Runtime. The three `.onnx` model files are the same ones the Python library uses; see [Getting started](../getting-started.md#net).

## Analyze a face

```csharp
using var inference = new OnnxDynafaceInference(
    "blaze_face_short_range.onnx",
    "spiga_wflw.onnx",
    "u2net.onnx");
FacePipeline.Initialize(inference);

var image  = new FaceImage(pixels, width, height); // flat Rgba32[], y=0 at bottom
var result = FacePipeline.Run(image, rotationAngle: 0, flipHorizontal: false);

var ctx = new FaceMeasureContext(
    result.Value.AlignedCrop, result.Value.Wflw98,
    isLateral: result.Value.IsLateral,
    lateralLandmarks: result.Value.LateralAnalysis?.LateralLandmarks,
    headPose: result.Value.HeadPose,
    pose: result.Value.Pose, flipped: result.Value.Flipped,
    pupilDistMm: result.Value.PupilDistMm);

Dictionary<string, double> measurements = ctx.Analyze();
```

`Analyze()` returns the same outputs, by the same names, as the Python library's `analyze()`, the [Cheek Measures](../measures.md#cheek-measures) and the Landmarks measure's Cheek Landmarks item included; see [Measures](../measures.md). A three-quarter face takes the same path as in Python: set `DynafaceConfig.AutoQuarter` (off by default, like `config.AUTO_QUARTER`) to detect one, or pass `forcePose: FacePose.Quarter` to `FacePipeline.Run`, and build the context with `FaceMeasureContext.FromResult(result)` so it carries the far cheek's outline.

For a video, `FrameSeries` measures every frame's saved analysis and gives the same CSV, peak frames and evaluation as the Python library's [`FrameSeries`](../guide/video.md).

The context measures its own millimeters-per-pixel scale on the landmarks you give it, from a pupillary distance in millimeters. If you save landmarks to measure later, save `PupilDistMm` with them and pass it back, rather than a saved `Pix2mm`: a scale is only right for the landmarks it was measured on. See [Units and scale](../guide/units.md).

For the full walkthrough, including supplying your own inference backend, see the [.NET README](https://github.com/jeffheaton/dynaface/tree/main/dynaface-lib-dotnet#using-the-library-from-another-app).
