using System;

// Low-level inference contract for the Dynaface pipeline: 3 separate model calls,
// kept independent so each can be swapped/tested on its own.
// One implementation wraps the ONNX runtime; another wraps Unity Inference Engine (in DynafaceRuntime).
public interface IDynafaceInference : IDisposable
{
    bool IsReady { get; }

    // Input:  NHWC float[1×128×128×3], values in [-1, 1]
    // Output: (regressors float[896×stride], scores float[896]), or null on failure
    (float[] regressors, float[] scores)? RunBlazeFace(float[] tensor);

    // Input:  NCHW float[1×3×256×256], values in [0, 1], y=0 at top
    // Output: (landmarks flat float[98×2] normalized to [0,1],
    //          pose float[6] raw [yaw, pitch, roll, tx, ty, tz]), or null on failure
    (float[] landmarks, float[] pose)? RunSpiga(float[] imageTensor);

    // Input:  NCHW float[1×3×320×320], ImageNet-normalized ((px/255 - mean) / std)
    // Output: flat float[320×320] raw sigmoid saliency mask (first of the model's
    //         multi-scale outputs), or null on failure
    float[] RunU2Net(float[] imageTensor);
}
