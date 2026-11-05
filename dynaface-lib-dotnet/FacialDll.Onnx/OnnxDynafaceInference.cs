using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Implements IDynafaceInference using Microsoft.ML.OnnxRuntime.
// Takes separate ONNX model paths for BlazeFace, SPIGA, and U^2-Net at construction
// time — 3 independent InferenceSessions, one per network, never fused.
// Tensor names are auto-detected from model metadata; pass explicit overrides if needed.
//
// SPIGA requires three inputs per inference:
//   image      [1, 3, 256, 256]  — NCHW float32 in [0, 1]
//   model3d    [1, 98, 3]        — fixed WFLW 3D face template  (SpigaModelInputs.Model3d)
//   cam_matrix [1, 3, 3]         — fixed camera intrinsics      (SpigaModelInputs.CameraMatrix)
// and two outputs: "landmarks" and "pose".
//
// U^2-Net has 7 outputs (d0..d6, multi-scale side outputs from training); the first
// declared output (auto-detected, not hardcoded) is the final merged saliency map.
public class OnnxDynafaceInference : IDynafaceInference
{
    readonly InferenceSession _blazeFaceSession;
    readonly InferenceSession _spigaSession;
    readonly InferenceSession _u2netSession;
    readonly string _blazeInputName;
    readonly string _blazeRegressorsName;
    readonly string _blazeScoresName;
    readonly string _u2netInputName;
    readonly string _u2netOutputName;

    // SPIGA's two fixed inputs, the same on every run, from the core library.
    static readonly float[] _model3d = SpigaModelInputs.Model3d();
    static readonly float[] _camMatrix = SpigaModelInputs.CameraMatrix();

    public bool IsReady => _blazeFaceSession != null && _spigaSession != null && _u2netSession != null;

    public OnnxDynafaceInference(string blazeFacePath, string spigaPath, string u2netPath,
        string blazeInputName = null,
        string blazeRegressorsName = null,
        string blazeScoresName = null)
    {
        _blazeFaceSession = new InferenceSession(blazeFacePath);
        _spigaSession = new InferenceSession(spigaPath);
        _u2netSession = new InferenceSession(u2netPath);
        _blazeInputName = blazeInputName ?? _blazeFaceSession.InputMetadata.Keys.First();
        string[] outs = _blazeFaceSession.OutputMetadata.Keys.ToArray();
        _blazeRegressorsName = blazeRegressorsName ?? outs[0];
        _blazeScoresName = blazeScoresName ?? outs[1];
        _u2netInputName = _u2netSession.InputMetadata.Keys.First();
        _u2netOutputName = _u2netSession.OutputMetadata.Keys.First();
    }

    public string DescribeTensors()
    {
        var spigaIns = string.Join(", ", _spigaSession.InputMetadata.Keys);
        var spigaOuts = string.Join(", ", _spigaSession.OutputMetadata.Keys);
        return $"BlazeFace: input={_blazeInputName}  regressors={_blazeRegressorsName}  scores={_blazeScoresName}\n" +
               $"SPIGA:     inputs=[{spigaIns}]  outputs=[{spigaOuts}]\n" +
               $"U2Net:     input={_u2netInputName}  output={_u2netOutputName}";
    }

    public (float[] regressors, float[] scores)? RunBlazeFace(float[] tensor)
    {
        try
        {
            var t = new DenseTensor<float>(tensor, new[] { 1, 128, 128, 3 });
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_blazeInputName, t) };
            using var results = _blazeFaceSession.Run(inputs);
            float[] regs = results.First(r => r.Name == _blazeRegressorsName).AsTensor<float>().ToArray();
            float[] scores = results.First(r => r.Name == _blazeScoresName).AsTensor<float>().ToArray();
            return (regs, scores);
        }
        catch { return null; }
    }

    public (float[] landmarks, float[] pose)? RunSpiga(float[] imageTensor)
    {
        try
        {
            var imgT = new DenseTensor<float>(imageTensor, new[] { 1, 3, 256, 256 });
            var m3dT = new DenseTensor<float>(_model3d, new[] { 1, 98, 3 });
            var camT = new DenseTensor<float>(_camMatrix, new[] { 1, 3, 3 });

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("image",      imgT),
                NamedOnnxValue.CreateFromTensor("model3d",    m3dT),
                NamedOnnxValue.CreateFromTensor("cam_matrix", camT),
            };

            using var results = _spigaSession.Run(inputs);
            float[] landmarks = results.First(r => r.Name == "landmarks").AsTensor<float>().ToArray();
            float[] pose = results.First(r => r.Name == "pose").AsTensor<float>().ToArray();
            return (landmarks, pose);
        }
        catch { return null; }
    }

    public float[] RunU2Net(float[] imageTensor)
    {
        try
        {
            var t = new DenseTensor<float>(imageTensor, new[] { 1, 3, 320, 320 });
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_u2netInputName, t) };
            using var results = _u2netSession.Run(inputs);
            return results.First(r => r.Name == _u2netOutputName).AsTensor<float>().ToArray();
        }
        catch { return null; }
    }

    public void Dispose()
    {
        _blazeFaceSession?.Dispose();
        _spigaSession?.Dispose();
        _u2netSession?.Dispose();
    }
}
