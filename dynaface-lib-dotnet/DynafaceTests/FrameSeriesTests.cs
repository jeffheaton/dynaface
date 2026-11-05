namespace DynafaceTests;

// FrameSeries: a video's outputs frame by frame, and the CSV, peaks and evaluation built
// from them. Mirrors dynaface-lib-python's tests/test_video.py, with the same expectations.
public class FrameSeriesTests
{
    static FrameSeries SeriesOf(IEnumerable<FaceMeasureBase> measures, float fps,
        params (int frame, Dictionary<string, double>? results)[] rows)
    {
        var series = new FrameSeries(measures, fps);
        foreach (var (frame, results) in rows) series.Add(frame, results);
        return series;
    }

    static Dictionary<string, double> Results(params (string name, double value)[] values) =>
        values.ToDictionary(v => v.name, v => v.value);

    [Fact]
    public void ColumnsAreTheEnabledOutputs()
    {
        var eyes = new MeasureEyeArea();
        eyes.SetItemEnabled("eye.diff", false);
        var mouth = new MeasureMouthLength { Enabled = false };
        var series = new FrameSeries(new FaceMeasureBase[] { eyes, mouth });
        Assert.Equal(new[] { "eye.left", "eye.right", "eye.ratio" }, series.Names);
    }

    [Fact]
    public void FramesWithoutAFaceAreNaN()
    {
        var series = SeriesOf(new[] { new MeasureFAI() }, 0f, (0, Results(("fai", 1.5))), (1, null));
        Assert.Equal(2, series.Count);
        Assert.Equal(1.5, series["fai"][0]);
        Assert.True(double.IsNaN(series["fai"][1]));
    }

    [Fact]
    public void Csv()
    {
        var series = SeriesOf(new FaceMeasureBase[] { new MeasureFAI(), new MeasureMouthLength() }, 30f,
            (0, Results(("fai", 1.5), ("ml", 50.0))),
            (1, null),
            (2, Results(("fai", double.NaN), ("ml", 52.123456))),
            (3, Results(("fai", -0.00001), ("ml", double.PositiveInfinity))));
        // Values to at most 4 decimal places; no face, NaN and infinity are empty.
        Assert.Equal("frame,time,fai,ml\n0,0.00,1.5,50\n1,0.03,,\n2,0.07,,52.1235\n3,0.10,0,\n", series.ToCsv());
    }

    [Fact]
    public void CsvWithoutAFrameRate()
    {
        var series = SeriesOf(new[] { new MeasureFAI() }, 0f, (4, Results(("fai", 2.0))));
        Assert.Equal("frame,time,fai\n4,,2\n", series.ToCsv());
        Assert.True(double.IsNaN(series.Time(4)));
    }

    [Fact]
    public void Peak()
    {
        var series = SeriesOf(new[] { new MeasureEyeArea() }, 0f,
            (10, Results(("eye.left", 5.0), ("eye.right", 5.0))),
            (11, Results(("eye.left", 8.0), ("eye.right", double.NaN))),
            (12, Results(("eye.left", 6.0), ("eye.right", 4.0))),
            (13, Results(("eye.left", 2.0), ("eye.right", 1.0))));
        // Frame 11 has no right eye, so it can't count for both; 10 and 12 tie, and the
        // first wins.
        Assert.Equal(10, series.Peak("eye.left", "eye.right"));
        Assert.Equal(11, series.Peak("eye.left"));
        Assert.Throws<KeyNotFoundException>(() => series.Peak("dental_area"));
        Assert.Throws<ArgumentException>(() => series.Peak());
        Assert.Null(new FrameSeries(new[] { new MeasureEyeArea() }).Peak("eye.left"));
    }

    [Fact]
    public void Evaluation()
    {
        var both = new FaceMeasureBase[] { new MeasureDentalArea(), new MeasureEyeArea() };
        var series = SeriesOf(both, 10f,
            (0, Results(("dental_area", 100.0), ("eye.left", 60.0), ("eye.right", 50.0))),
            (1, Results(("dental_area", 400.0), ("eye.left", 30.0), ("eye.right", 40.0))),
            (2, Results(("dental_area", 50.0), ("eye.left", 70.0), ("eye.right", 65.0))));
        var evaluation = series.Evaluate();
        Assert.NotNull(evaluation);
        var smile = evaluation!.WidestSmile;
        var eyes = evaluation.WidestEyes;
        Assert.Equal((1, 400.0), (smile.Frame, smile.DentalArea));
        Assert.Equal(0.1, smile.Time, 10);
        Assert.Equal((30.0, 40.0), (smile.EyeLeft, smile.EyeRight));
        Assert.Equal(0.75, smile.EyeLeftOverRight, 10);
        Assert.Equal(40.0 / 30.0, smile.EyeRightOverLeft, 10);
        Assert.Equal((2, 50.0, 135.0), (eyes.Frame, eyes.DentalArea, eyes.EyeTotal));

        // Without the outputs, or without a single value, there's nothing to compare.
        Assert.Null(SeriesOf(new[] { new MeasureDentalArea() }, 0f, (0, Results(("dental_area", 1.0)))).Evaluate());
        Assert.Null(SeriesOf(both, 0f, (0, null)).Evaluate());
    }

    [Fact]
    public void RatioOfAClosedEye()
    {
        var closed = new FrameSeries.Moment(0, double.NaN, 1.0, 5.0, 0.0);
        Assert.True(double.IsNaN(closed.EyeLeftOverRight));
        Assert.Equal(0.0, closed.EyeRightOverLeft);
    }
}

// Measuring saved analyses again, as the apps do for a loaded video.
public class FrameSeriesFromStatesTests
{
    static FaceImage Gray(int width, int height) =>
        new(Enumerable.Repeat(new Rgba32(128, 128, 128, 255), width * height).ToArray(), width, height);

    // A made-up face in a video-sized frame, with pupils 230 + spread pixels apart, so each
    // spread measures differently.
    static FacePipelineResult Analyze(float spread)
    {
        var right = new Vec2(530f, 320f);
        var left = new Vec2(760f + spread, 330f);
        var landmarks = new Vec2[98];
        for (int i = 0; i < landmarks.Length; i++)
            landmarks[i] = new Vec2(right.X - 80f + (i * 37) % 380, right.Y - 120f + (i * 53) % 400);
        landmarks[DynafaceConstants.LmRightPupil] = right;
        landmarks[DynafaceConstants.LmLeftPupil] = left;
        return FacePipeline.RunFrontal(Gray(1280, 720), landmarks, new[] { 8f, 1f, -2f, 0f, 0f, 0f },
            FacePose.Frontal, DynafaceConstants.DefaultTiltThreshold, crop: true)!.Value;
    }

    static readonly FacePipelineResult[] Results = { Analyze(0f), Analyze(40f), Analyze(80f) };
    static readonly FaceAnalysisState[] States = Results.Select(FaceAnalysisPersistence.Capture).ToArray();

    static bool Same(double a, double b) =>
        (double.IsNaN(a) && double.IsNaN(b)) || Math.Abs(a - b) <= 1e-6 * Math.Max(1.0, Math.Abs(a));

    [Fact]
    public void SameValuesAsTheLiveFaces()
    {
        var series = FrameSeries.FromStates(States, fps: 30f);
        Assert.Equal(new[] { 0, 1, 2 }, series.Frames);
        Assert.Equal(DynafaceMeasures.GetAllItems(DynafaceMeasures.AllMeasures()), series.Names);
        for (int row = 0; row < Results.Length; row++)
        {
            var live = TestHelpers.ContextFor(Results[row]).Analyze(render: false)!;
            foreach (string name in series.Names)
            {
                double expected = live.TryGetValue(name, out double value) ? value : double.NaN;
                Assert.True(Same(expected, series[name][row]), $"{name}, {row}: {expected} vs {series[name][row]}");
            }
        }
        // The frames differ, so the series isn't one state repeated.
        Assert.NotEqual(series["ml"][0], series["ml"][2]);
    }

    [Fact]
    public void RangesKeepEachFrameNumber()
    {
        Assert.Equal(new[] { 1, 2 }, FrameSeries.FromStates(States, start: 1).Frames);
        Assert.Equal(new[] { 0, 1 }, FrameSeries.FromStates(States, stop: 2).Frames);
        Assert.Equal(new[] { 0, 2 }, FrameSeries.FromStates(States, step: 2).Frames);
        Assert.Throws<ArgumentOutOfRangeException>(() => FrameSeries.FromStates(States, step: 0));
    }

    [Fact]
    public void FramesSavedWithoutTheirImage()
    {
        // Landmarks alone, like the Unity app's video frames: skin tone needs the image, so
        // it's NaN, and everything else measures the same.
        var withImage = FrameSeries.FromStates(States);
        var landmarksOnly = FrameSeries.FromStates(States.Select(s => new FaceAnalysisState
        {
            HeadPose = s.HeadPose,
            Landmarks = s.Landmarks,
            PupillaryDistance = s.PupillaryDistance,
            Pix2mm = s.Pix2mm,
        }).ToArray());

        Assert.Equal(withImage.Names, landmarksOnly.Names);
        foreach (string name in withImage.Names)
            for (int row = 0; row < States.Length; row++)
            {
                if (name is "hue" or "sat" or "brightness")
                    Assert.True(double.IsNaN(landmarksOnly[name][row]), name);
                else
                    Assert.True(Same(withImage[name][row], landmarksOnly[name][row]), $"{name}, {row}");
            }
        Assert.False(double.IsNaN(withImage["hue"][0]));
    }

    [Fact]
    public void ChosenMeasures()
    {
        var series = FrameSeries.FromStates(States, new FaceMeasureBase[] { new MeasureDentalArea(), new MeasureEyeArea() });
        var evaluation = series.Evaluate();
        Assert.NotNull(evaluation);
        Assert.Equal(series.Peak("dental_area"), evaluation!.WidestSmile.Frame);
        Assert.Equal(series.Peak("eye.left", "eye.right"), evaluation.WidestEyes.Frame);
    }
}
