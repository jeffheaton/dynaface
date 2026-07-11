using System.Collections.Generic;
using System.IO;

namespace DynafaceTests;

// Model-gated end-to-end pipeline tests, mirroring dynaface-lib's test_facial.py
// (test_frontal/test_right_lateral/test_left_lateral). Skips (no-op pass) when the 3
// real ONNX models aren't found locally — see ModelPathResolver — since they can't
// be committed to the repo (~400MB, gitignored).
//
// Unlike test_facial.py, these assert sane RANGES rather than dynaface-lib's exact
// hardcoded values: the frontal alignment now matches dynaface-lib's crop_stylegan
// algorithm, but BlazeFace/SPIGA are still two independently-run ONNX sessions, so
// bit-for-bit numeric parity with Python isn't guaranteed even with faithfully
// ported formulas. Tighten these once real vs. Python numbers have been compared
// side by side.
[Collection("ModelTests")]
public class FacialPipelineIntegrationTests
{
    readonly ModelFixture _fixture;

    public FacialPipelineIntegrationTests(ModelFixture fixture) => _fixture = fixture;

    static string TestImage(string name) => Path.Combine(AppContext.BaseDirectory, "TestData", name);

    [Fact]
    public void FrontalImage_DetectsFrontalPoseAndProducesSensibleMeasurements()
    {
        if (!_fixture.Available) return;

        var photo = ImageLoader.Load(TestImage("img1-512.jpg"));
        var result = RunWithRotationRetry(photo);

        Assert.NotNull(result);
        Assert.False(result.Value.IsLateral);
        Assert.Equal(1024, result.Value.AlignedCrop.Width);
        Assert.Equal(1024, result.Value.AlignedCrop.Height);
        Assert.Equal(98, result.Value.Wflw98.Length);

        var ctx = BuildContext(result.Value);
        var all = RunAllFrontalMeasures(ctx);

        Assert.True(all["fai"] >= 0);
        Assert.InRange(all["pd"], 100, 400);
        Assert.InRange(all["px2mm"], 0.1, 1.0);
        Assert.True(all["eye.left"] > 0);
        Assert.True(all["eye.right"] > 0);
        Assert.True(all["id"] > 0);
        Assert.True(all["ml"] > 0);
    }

    [Fact]
    public void RightLateralImage_ClassifiesLateralAndProducesLateralLandmarks()
    {
        if (!_fixture.Available) return;

        var photo = ImageLoader.Load(TestImage("img2-1024-right-lateral.jpg"));
        var result = RunWithRotationRetry(photo);

        Assert.NotNull(result);
        Assert.True(result.Value.IsLateral);

        // FacePipeline.Run already performed lateral analysis internally.
        var lateral = result.Value.LateralAnalysis;
        Assert.NotNull(lateral);
        Assert.Equal(6, lateral.Value.LateralLandmarks.Length);
    }

    [Fact]
    public void LeftLateralImage_ClassifiesLateralAndProducesLateralLandmarks()
    {
        if (!_fixture.Available) return;

        var photo = ImageLoader.Load(TestImage("img3-1024-left-lateral.jpg"));
        var result = RunWithRotationRetry(photo);

        Assert.NotNull(result);
        Assert.True(result.Value.IsLateral);

        // FacePipeline.Run already performed lateral analysis internally.
        var lateral = result.Value.LateralAnalysis;
        Assert.NotNull(lateral);
        Assert.Equal(6, lateral.Value.LateralLandmarks.Length);
    }

    internal static FacePipelineResult? RunWithRotationRetry(FaceImage photo)
    {
        FacePipelineResult? best = null;
        foreach (int rot in new[] { 0, 90, 180, 270 })
        {
            var candidate = FacePipeline.Run(photo, rot, flipHorizontal: false);
            if (candidate == null) continue;
            if (best == null) best = candidate;
            if (FacePipeline.LastDetectionEyesOk)
            {
                best = candidate;
                break;
            }
        }
        return best;
    }

    internal static FaceMeasureContext BuildContext(FacePipelineResult result) =>
        new FaceMeasureContext(
            result.AlignedCrop, result.Wflw98, result.Pix2mm,
            isLateral: result.IsLateral, lateralLandmarks: null, headPose: result.HeadPose,
            pose: result.Pose, flipped: result.Flipped);

    // Runs the library's own registry + runner (DynafaceMeasures.AllMeasures via
    // ctx.Analyze), the same path dynaface-lib's AnalyzeFace.analyze() takes.
    internal static Dictionary<string, double> RunAllFrontalMeasures(FaceMeasureContext ctx) =>
        ctx.Analyze();
}
