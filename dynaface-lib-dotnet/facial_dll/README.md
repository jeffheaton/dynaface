# Dynaface

A C# facial-analysis pipeline — a contract-faithful port of the Python [`dynaface-lib`](https://github.com/jeffheaton/dynaface) package. It runs BlazeFace detection, SPIGA WFLW-98 landmark + headpose estimation, U²-Net background removal, frontal (StyleGAN-style) and lateral analysis, and the full frontal + lateral + skin-tone measurement set used to assess facial symmetry and movement.

This is the **backend-agnostic core**: it runs no neural networks itself. All 3 networks are reached through a single interface, `IDynafaceInference`, kept as 3 independent methods so each can be swapped or tested on its own.

## Install

```bash
# Core + a ready-made ONNX Runtime backend (recommended for most .NET apps)
dotnet add package Dynaface.Onnx

# ...or the core alone, if you supply your own inference backend
dotnet add package Dynaface
```

`Dynaface` and `Dynaface.Onnx` ship in lockstep at the same version; installing `Dynaface.Onnx` pulls in the matching `Dynaface`.

## Usage

```csharp
// Supply an IDynafaceInference implementation (see the Dynaface.Onnx package),
// then run the pipeline and measurements:
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

## Backends

| Backend | Runtime | Package |
|---|---|---|
| ONNX | `Microsoft.ML.OnnxRuntime` | `Dynaface.Onnx` |
| Unity Inference Engine | Unity Sentis | external `DynafaceRuntime` assembly |

To add your own backend, implement `IDynafaceInference`'s 3 methods (`RunBlazeFace` / `RunSpiga` / `RunU2Net`).

See the [project repository](https://github.com/jeffheaton/dynaface) for full documentation. Licensed under Apache-2.0.
