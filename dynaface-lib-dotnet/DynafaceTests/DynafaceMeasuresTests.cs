namespace DynafaceTests;

// Unit tests for the library-owned measure registry (DynafaceMeasures.AllMeasures,
// mirroring dynaface-lib's measures.all_measures) and the FaceMeasureContext.Analyze
// runner (mirroring AnalyzeFace.analyze). No models required.
public class DynafaceMeasuresTests
{
    [Fact]
    public void AllMeasures_MatchesPythonRegistrySizeAndOrder()
    {
        var measures = DynafaceMeasures.AllMeasures();

        // Same count and order as dynaface-lib's all_measures().
        Assert.Equal(14, measures.Length);
        Assert.IsType<MeasureFAI>(measures[0]);
        Assert.IsType<MeasureOCE>(measures[1]);
        Assert.IsType<MeasureLateral>(measures[9]);
        Assert.IsType<MeasureLandmarks>(measures[13]);
        Assert.All(measures, m => Assert.True(m.Enabled));
    }

    [Fact]
    public void GetAllItems_SkipsDisabledMeasuresAndItems()
    {
        var measures = DynafaceMeasures.AllMeasures();
        var allItems = DynafaceMeasures.GetAllItems(measures);
        Assert.Contains("fai", allItems);
        Assert.Contains("nfa", allItems);

        foreach (var m in measures)
            if (m is MeasureLateral) m.SetEnabled(false);
        var withoutLateral = DynafaceMeasures.GetAllItems(measures);
        Assert.DoesNotContain("nfa", withoutLateral);
        Assert.Contains("fai", withoutLateral);
    }

    [Fact]
    public void Analyze_ReturnsNullWithoutLandmarks()
    {
        var photo = new FaceImage(new Rgba32[16 * 16], 16, 16);
        var ctx = new FaceMeasureContext(photo, new Vec2[0], 0.24f);
        Assert.Null(ctx.Analyze());
    }
}
