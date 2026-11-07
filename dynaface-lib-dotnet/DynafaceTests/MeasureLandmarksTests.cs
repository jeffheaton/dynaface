namespace DynafaceTests;

// Ports dynaface-lib's test_measures_landmarks.py TestAnalyzeLandmarksStructure,
// TestAnalyzeLandmarksCalc and TestCheekLandmarksItem (the non-integration classes — no
// model required). The Cheek Landmarks item on fixture faces, and its drawing, are in
// MeasureCheekMeasuresTests.CheekLandmarksItem.
public class MeasureLandmarksTests
{
    static readonly string[] CheekOutputs = Cheek.CheekLandmarkOutputs.ToArray();

    [Fact]
    public void ItemCount_Is98TimesTwo_ThenTheCheekLandmarksGroup()
    {
        var measure = new MeasureLandmarks();
        Assert.Equal(98, MeasureLandmarks.NumLandmarksDumped);
        Assert.Equal(MeasureLandmarks.NumLandmarksDumped * 2 + 1, measure.Items.Count);
        Assert.Equal(MeasureLandmarks.NumLandmarksDumped * 2 + 20, measure.Outputs().Count);
    }

    [Fact]
    public void CheekLandmarksItem()
    {
        var measure = new MeasureLandmarks();
        var item = measure.Items[^1];
        Assert.Equal("Cheek Landmarks", item.Name);
        Assert.Equal(MeasureLandmarks.CheekItem, item.Name);
        Assert.False(item.DisplayOnly);
        Assert.Equal(CheekOutputs, item.OutputNames());
        Assert.Equal(CheekOutputs, measure.Outputs().TakeLast(20));
        Assert.True(item.IsFrontal && item.IsQuarter);
        Assert.False(item.IsLateral);
        Assert.True(item.Enabled);
    }

    [Fact]
    public void CheekLandmarkOutputs()
    {
        Assert.Equal(20, CheekOutputs.Length);
        Assert.Equal(new[] { "zy.r.x", "zy.r.y", "zy.l.x", "zy.l.y", "cr.r.x", "cr.r.y" }, CheekOutputs.Take(6));
        Assert.Equal(new[] { "sub.l.x", "sub.l.y" }, CheekOutputs.TakeLast(2));
        for (int i = 0; i < Cheek.CheekLandmarkCodes.Count; i++)
        {
            string code = Cheek.CheekLandmarkCodes[i];
            Assert.Equal(new[] { $"{code}.x", $"{code}.y" }, CheekOutputs.Skip(2 * i).Take(2));
        }
    }

    [Fact]
    public void ItemNames_InterleaveXThenY_ForEachLandmark()
    {
        var measure = new MeasureLandmarks();
        Assert.Equal("landmark-1-x", measure.Items[0].Name);
        Assert.Equal("landmark-1-y", measure.Items[1].Name);
        Assert.Equal($"landmark-{MeasureLandmarks.NumLandmarksDumped}-x", measure.Items[^3].Name);
        Assert.Equal($"landmark-{MeasureLandmarks.NumLandmarksDumped}-y", measure.Items[^2].Name);
        Assert.Equal("Cheek Landmarks", measure.Items[^1].Name);
    }

    [Fact]
    public void AllItems_EnabledByDefault_FrontalAndQuarter()
    {
        var measure = new MeasureLandmarks();
        Assert.True(measure.Enabled);
        Assert.True(measure.IsFrontal);
        Assert.True(measure.IsQuarter);
        Assert.False(measure.IsLateral);
        foreach (var item in measure.Items)
        {
            Assert.True(item.Enabled);
            Assert.True(item.IsFrontal);
            Assert.True(item.IsQuarter);
            Assert.False(item.IsLateral);
        }
    }

    [Fact]
    public void Calc_ReturnsCorrectValuesForEveryDumpedLandmark()
    {
        var measure = new MeasureLandmarks();
        var ctx = TestHelpers.BuildContext(landmarks: TestHelpers.MakeLandmarks(98));

        var result = measure.Calc(ctx, render: false);

        Assert.Equal(MeasureLandmarks.NumLandmarksDumped * 2 + 20, result.Count);
        Assert.Equal(measure.Outputs(), result.Keys);
        for (int i = 0; i < MeasureLandmarks.NumLandmarksDumped; i++)
        {
            int n = i + 1;
            Assert.Equal(i * 2, result[$"landmark-{n}-x"]);
            Assert.Equal(i * 2 + 1, result[$"landmark-{n}-y"]);
        }
    }

    [Fact]
    public void Calc_IncludesAllWflwLandmarks_IncludingTheLastOne()
    {
        // WFLW has 98 points (indices 0..97). dynaface-lib's AnalyzeLandmarks used
        // to only emit 97 (1..97), silently dropping index 97 (the left pupil) —
        // an off-by-one, fixed upstream; all 98 are now included.
        var measure = new MeasureLandmarks();
        var ctx = TestHelpers.BuildContext(landmarks: TestHelpers.MakeLandmarks(98));

        var result = measure.Calc(ctx, render: false);

        Assert.True(result.ContainsKey("landmark-98-x"));
        Assert.True(result.ContainsKey("landmark-98-y"));
        Assert.Equal(97 * 2, result["landmark-98-x"]);
        Assert.Equal(97 * 2 + 1, result["landmark-98-y"]);
    }

    [Fact]
    public void SetItemEnabled_OnlyAffectsThatItem()
    {
        var measure = new MeasureLandmarks();
        measure.SetItemEnabled("landmark-1-x", false);

        Assert.False(measure.IsEnabled("landmark-1-x"));
        Assert.True(measure.IsEnabled("landmark-1-y"));
    }

    // The Cheek Landmarks item's values are the context's cheek landmarks: Cheek.Locate's
    // rule on its landmarks and view, NaN where absent, as doubles.
    [Fact]
    public void CheekValuesFollowCheekLocate()
    {
        var ctx = TestHelpers.BuildContext(landmarks: TestHelpers.MakeLandmarks(98));
        var result = new MeasureLandmarks().Calc(ctx, render: false);
        Vec2d[] expected = Cheek.LocateAll(ctx.Landmarks, FacePose.Frontal, false, null, null, null);
        for (int i = 0; i < expected.Length; i++)
        {
            string code = Cheek.CheekLandmarkCodes[i];
            Assert.True(MeasureCheekMeasuresTests.Same(expected[i].X, result[$"{code}.x"]), code);
            Assert.True(MeasureCheekMeasuresTests.Same(expected[i].Y, result[$"{code}.y"]), code);
            Assert.Equal(result[$"{code}.x"], ctx.Values[$"{code}.x"]);
        }
    }

    [Fact]
    public void CheekDrawingGatedByTheItem()
    {
        var orange = MeasureLandmarks.CheekColor;
        var ctx = CheekFaces.Frontal();
        new MeasureLandmarks().Calc(ctx, render: true);
        Assert.Contains(ctx.Pixels, p => p == orange);

        var measure = new MeasureLandmarks();
        measure.SetItemEnabled("Cheek Landmarks", false);
        ctx = CheekFaces.Frontal();
        var result = measure.Calc(ctx, render: true);
        Assert.DoesNotContain(ctx.Pixels, p => p == orange);
        Assert.Contains(ctx.Pixels, p => p == new Rgba32(255, 255, 255, 255)); // the 98 still drawn
        Assert.False(double.IsNaN(result["zy.r.x"])); // still returned
        Assert.DoesNotContain("zy.r.x", measure.Outputs().Take(MeasureLandmarks.NumLandmarksDumped * 2));

        ctx = CheekFaces.Frontal();
        new MeasureLandmarks().Calc(ctx, render: false);
        Assert.True(ctx.Pixels.All(p => p == default(Rgba32)));
    }
}
