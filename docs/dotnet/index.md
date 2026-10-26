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
    result.Value.AlignedCrop, result.Value.Wflw98, result.Value.Pix2mm,
    isLateral: result.Value.IsLateral,
    lateralLandmarks: result.Value.LateralAnalysis?.LateralLandmarks,
    headPose: result.Value.HeadPose,
    pose: result.Value.Pose, flipped: result.Value.Flipped);

Dictionary<string, double> measurements = ctx.Analyze();
```

`Analyze()` returns the same outputs, by the same names, as the Python library's `analyze()`; see [Measures](../measures.md).

For the full walkthrough, including supplying your own inference backend, see the [.NET README](https://github.com/jeffheaton/dynaface/tree/main/dynaface-lib-dotnet#using-the-library-from-another-app).
