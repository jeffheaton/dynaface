using System.IO;

namespace DynafaceTests;

// Every registered measure returns exactly its declared outputs (its items that aren't
// DisplayOnly), whether or not those items are enabled and whether or not it renders.
// Display-only items (drawing switches such as tip_proj) are never returned or listed by
// GetAllItems. Mirrors dynaface-lib-python's tests/test_declared_outputs.py. Model-gated.
[Collection("ModelTests")]
public class DeclaredOutputsTests
{
    readonly ModelFixture _fixture;

    public DeclaredOutputsTests(ModelFixture fixture) => _fixture = fixture;

    static FacePipelineResult Run(string image)
    {
        var photo = ImageLoader.Load(Path.Combine(AppContext.BaseDirectory, "TestData", image));
        var result = FacialPipelineIntegrationTests.RunWithRotationRetry(photo);
        Assert.NotNull(result);
        return result.Value;
    }

    static FaceMeasureContext Context(FacePipelineResult r) =>
        new FaceMeasureContext(
            new FaceImage((Rgba32[])r.AlignedCrop.Pixels.Clone(), r.AlignedCrop.Width, r.AlignedCrop.Height),
            r.Wflw98, r.Pix2mm, isLateral: r.IsLateral,
            lateralLandmarks: r.LateralAnalysis?.LateralLandmarks, headPose: r.HeadPose,
            pose: r.Pose, flipped: r.Flipped);

    [Fact]
    public void EveryMeasureReturnsExactlyItsDeclaredOutputs()
    {
        if (!_fixture.Available) return;

        var views = new[] { Run("img1-512.jpg"), Run("img2-1024-right-lateral.jpg") };
        var problems = new List<string>();
        foreach (var measure in DynafaceMeasures.AllMeasures())
        {
            var declared = measure.Items.Where(i => !i.DisplayOnly).Select(i => i.Name).OrderBy(n => n).ToList();
            foreach (var item in measure.Items) item.Enabled = false; // disabled items are still returned
            foreach (var view in views)
            {
                if (!(view.IsLateral ? measure.IsLateral : measure.IsFrontal)) continue;
                foreach (bool render in new[] { false, true })
                {
                    var keys = measure.Calc(Context(view), render).Keys.OrderBy(k => k).ToList();
                    if (!keys.SequenceEqual(declared))
                        problems.Add($"{measure.Label} lateral={view.IsLateral} render={render}: " +
                                     $"returned [{string.Join(",", keys)}], declared [{string.Join(",", declared)}]");
                }
            }
        }
        Assert.Empty(problems);
    }

    [Fact]
    public void DisplayOnlyItemsAreNotOutputs()
    {
        var measure = new MeasureLateral();
        Assert.True(measure.Items.Single(i => i.Name == "tip_proj").DisplayOnly);
        var items = DynafaceMeasures.GetAllItems(new FaceMeasureBase[] { measure });
        Assert.DoesNotContain("tip_proj", items);
        Assert.Contains("at_nt", items);
    }
}
