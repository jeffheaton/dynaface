using System.IO;

namespace DynafaceTests;

// Every registered measure returns exactly its declared outputs (measure.Outputs(): its
// items' output names, a group's outputs listed in its place, DisplayOnly items none),
// whether or not those items are enabled and whether or not it renders, on each view it
// suits: a frontal photo, a profile and a three-quarter (quarter) face. Display-only
// items (drawing switches such as tip_proj) are never returned or listed by GetAllItems.
// Mirrors dynaface-lib-python's tests/test_declared_outputs.py. Model-gated.
[Collection("ModelTests")]
public class DeclaredOutputsTests
{
    readonly ModelFixture _fixture;

    public DeclaredOutputsTests(ModelFixture fixture) => _fixture = fixture;

    static FaceImage Load(string image) =>
        ImageLoader.Load(Path.Combine(AppContext.BaseDirectory, "TestData", image));

    static FacePipelineResult Run(string image)
    {
        var result = FacialPipelineIntegrationTests.RunWithRotationRetry(Load(image));
        Assert.NotNull(result);
        return result.Value;
    }

    // The frontal photo forced to the quarter view, as dynaface-lib forces its
    // three-quarter sketch to it: the quarter path's band crop and far-cheek trace, so
    // every measure that suits the view runs on a real quarter face.
    static FacePipelineResult RunQuarter(string image)
    {
        var result = FacePipeline.Run(Load(image), 0, flipHorizontal: false, forcePose: FacePose.Quarter);
        Assert.NotNull(result);
        return result.Value;
    }

    static FaceMeasureContext Context(FacePipelineResult r) => TestHelpers.ContextFor(r);

    // The three views, each with the result to measure on.
    static (string view, FacePipelineResult result)[] Views() => new[]
    {
        ("frontal", Run("img1-512.jpg")),
        ("lateral", Run("img2-1024-right-lateral.jpg")),
        ("quarter", RunQuarter("img1-512.jpg")),
    };

    [SkippableFact]
    public void TheViews()
    {
        _fixture.RequireModels();

        var views = Views().ToDictionary(v => v.view, v => Context(v.result));
        Assert.False(views["frontal"].IsLateral || views["frontal"].IsQuarter);
        Assert.True(views["lateral"].IsLateral);
        Assert.True(views["quarter"].IsQuarter);
        Assert.Equal(FacePose.Quarter, views["quarter"].Pose);
        Assert.False(views["quarter"].IsLateral);
    }

    // The quarter view's measures: the ones that need no millimeter.
    [Fact]
    public void TheQuarterViewsMeasures() =>
        Assert.Equal(new[] { "POSITION", "POSE", "LANDMARKS", "CHEEK MEASURES" },
            DynafaceMeasures.AllMeasures().Where(m => m.Suits(false, quarter: true)).Select(m => m.Label));

    [SkippableFact]
    public void EveryMeasureReturnsExactlyItsDeclaredOutputs()
    {
        _fixture.RequireModels();

        var views = Views();
        var problems = new List<string>();
        int checkedQuarter = 0;
        foreach (var measure in DynafaceMeasures.AllMeasures())
        {
            var declared = measure.Outputs().OrderBy(n => n, StringComparer.Ordinal).ToList();
            foreach (var item in measure.Items) item.Enabled = false; // disabled items are still returned
            foreach (var (view, result) in views)
            {
                var ctx = Context(result);
                if (!measure.Suits(ctx.IsLateral, ctx.IsQuarter)) continue;
                if (view == "quarter") checkedQuarter++;
                foreach (bool render in new[] { false, true })
                {
                    var returned = measure.Calc(Context(result), render).Keys.ToList();
                    var keys = returned.OrderBy(k => k, StringComparer.Ordinal).ToList();
                    if (!keys.SequenceEqual(declared))
                        problems.Add($"{measure.Label} view={view} render={render}: " +
                                     $"returned [{string.Join(",", keys)}], declared [{string.Join(",", declared)}]");
                    // In declared order, a group's outputs together (a Dictionary that's
                    // only added to enumerates in the order its keys were added).
                    else if (!returned.SequenceEqual(measure.Outputs()))
                        problems.Add($"{measure.Label} view={view} render={render}: " +
                                     $"returned [{string.Join(",", returned)}] out of declared order");
                }
            }
        }
        Assert.Empty(problems);
        Assert.Equal(4, checkedQuarter);
    }

    [Fact]
    public void DisplayOnlyItemsAreNotOutputs()
    {
        var measure = new MeasureLateral();
        Assert.True(measure.Items.Single(i => i.Name == "tip_proj").DisplayOnly);
        Assert.DoesNotContain("tip_proj", measure.Outputs());
        var items = DynafaceMeasures.GetAllItems(new FaceMeasureBase[] { measure });
        Assert.DoesNotContain("tip_proj", items);
        Assert.Contains("at_nt", items);
    }
}
