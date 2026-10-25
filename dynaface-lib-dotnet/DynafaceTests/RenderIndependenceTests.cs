using System.IO;

namespace DynafaceTests;

// Every measure must return the same values whether or not it draws: rendering only shows
// results, it must not decide whether they are computed. (Tilt used to be 0 unless its
// label was drawn, and skin tone was only computed while drawing; the Unity app's video
// chart and CSV run every measure with render: false.) Mirrors dynaface-lib-python's
// tests/test_render_independence.py. Model-gated, like FacialPipelineIntegrationTests.
[Collection("ModelTests")]
public class RenderIndependenceTests
{
    readonly ModelFixture _fixture;

    public RenderIndependenceTests(ModelFixture fixture) => _fixture = fixture;

    static FaceMeasureContext Context(FacePipelineResult r) =>
        new FaceMeasureContext(
            new FaceImage((Rgba32[])r.AlignedCrop.Pixels.Clone(), r.AlignedCrop.Width, r.AlignedCrop.Height),
            r.Wflw98, r.Pix2mm, isLateral: r.IsLateral,
            lateralLandmarks: r.LateralAnalysis?.LateralLandmarks, headPose: r.HeadPose,
            pose: r.Pose, flipped: r.Flipped);

    [SkippableTheory]
    [InlineData("img1-512.jpg")]
    [InlineData("img2-1024-right-lateral.jpg")]
    public void MeasuresReturnTheSameValuesWithAndWithoutRendering(string image)
    {
        _fixture.RequireModels();

        var photo = ImageLoader.Load(Path.Combine(AppContext.BaseDirectory, "TestData", image));
        var result = FacialPipelineIntegrationTests.RunWithRotationRetry(photo);
        Assert.NotNull(result);

        var off = Context(result.Value).Analyze(render: false);
        var on = Context(result.Value).Analyze(render: true);

        Assert.Equal(on.Keys.OrderBy(k => k), off.Keys.OrderBy(k => k));
        var mismatches = on.Where(kv => !kv.Value.Equals(off[kv.Key]))
            .Select(kv => $"{kv.Key}: on {kv.Value} != off {off[kv.Key]}").ToList();
        Assert.Empty(mismatches);
        if (!result.Value.IsLateral)
        {
            Assert.NotEqual(0.0, off["tilt"]);
            Assert.True(off.ContainsKey("hue"));
        }
    }
}
