namespace DynafaceTests;

// Ports dynaface-lib's test_measures_landmarks.py TestAnalyzeLandmarksStructure and
// TestAnalyzeLandmarksCalc (the non-integration classes — no model required).
public class MeasureLandmarksTests
{
    [Fact]
    public void ItemCount_Is98TimesTwo()
    {
        var measure = new MeasureLandmarks();
        Assert.Equal(98, MeasureLandmarks.NumLandmarksDumped);
        Assert.Equal(MeasureLandmarks.NumLandmarksDumped * 2, measure.Items.Count);
    }

    [Fact]
    public void ItemNames_InterleaveXThenY_ForEachLandmark()
    {
        var measure = new MeasureLandmarks();
        Assert.Equal("landmark-1-x", measure.Items[0].Name);
        Assert.Equal("landmark-1-y", measure.Items[1].Name);
        Assert.Equal($"landmark-{MeasureLandmarks.NumLandmarksDumped}-x", measure.Items[^2].Name);
        Assert.Equal($"landmark-{MeasureLandmarks.NumLandmarksDumped}-y", measure.Items[^1].Name);
    }

    [Fact]
    public void AllItems_EnabledByDefault_AndFrontalOnly()
    {
        var measure = new MeasureLandmarks();
        Assert.True(measure.Enabled);
        Assert.True(measure.IsFrontal);
        Assert.False(measure.IsLateral);
        foreach (var item in measure.Items)
        {
            Assert.True(item.Enabled);
            Assert.True(item.IsFrontal);
            Assert.False(item.IsLateral);
        }
    }

    [Fact]
    public void Calc_ReturnsCorrectValuesForEveryDumpedLandmark()
    {
        var measure = new MeasureLandmarks();
        var ctx = TestHelpers.BuildContext(landmarks: TestHelpers.MakeLandmarks(98));

        var result = measure.Calc(ctx, render: false);

        Assert.Equal(MeasureLandmarks.NumLandmarksDumped * 2, result.Count);
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
}
