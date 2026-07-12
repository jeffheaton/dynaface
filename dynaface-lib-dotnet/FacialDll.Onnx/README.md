# Dynaface.Onnx

The **ONNX Runtime inference backend** for [Dynaface](https://www.nuget.org/packages/Dynaface) — a C# facial-analysis pipeline ported from the Python [`dynaface-lib`](https://github.com/jeffheaton/dynaface) package.

The core `Dynaface` package runs no neural networks itself; it reaches all 3 models (BlazeFace, SPIGA, U²-Net) through the `IDynafaceInference` interface. This package provides `OnnxDynafaceInference`, a ready-made implementation over `Microsoft.ML.OnnxRuntime`, so you don't have to write one.

## Install

```bash
dotnet add package Dynaface.Onnx
```

This pulls in the core `Dynaface` package and `Microsoft.ML.OnnxRuntime` automatically.

## Usage

```csharp
// 3 separate ONNX models — one per network.
using var inference = new OnnxDynafaceInference(
    "blaze_face_short_range.onnx",
    "spiga_wflw.onnx",
    "u2net.onnx");

FacePipeline.Initialize(inference);
// ...then run FacePipeline.Run(...) + FaceMeasureContext.Analyze() (see the Dynaface package).
```

`OnnxDynafaceInference` auto-detects each model's input/output tensor names from its metadata; pass explicit overrides to the constructor if a custom export uses unexpected names.

The 3 `.onnx` model files are distributed separately (they're large) — see the [project repository](https://github.com/jeffheaton/dynaface) for where to obtain them.

Licensed under Apache-2.0.
