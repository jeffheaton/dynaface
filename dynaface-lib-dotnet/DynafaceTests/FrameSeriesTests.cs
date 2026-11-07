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

    // A profile's lateral data, as SavedViewTests makes it up.
    static readonly Vec2[] LateralLandmarks =
    {
        new(300, 200), new(290, 300), new(200, 450), new(260, 520), new(280, 640), new(270, 760),
    };

    // The frontal state with its view rewritten as a quarter or lateral one. Restore
    // gives the view a state says, so neither needs a model: a quarter view, and a
    // lateral one with made-up lateral landmarks and sagittal outline.
    static FaceAnalysisState InView(FaceAnalysisState state, FacePose pose)
    {
        var turned = new FaceAnalysisState
        {
            PostCropImage = state.PostCropImage,
            HeadPose = state.HeadPose,
            Landmarks = state.Landmarks,
            PupillaryDistance = state.PupillaryDistance,
            Pix2mm = state.Pix2mm,
            FaceRotationRad = state.FaceRotationRad,
            Pose = pose,
        };
        if (pose == FacePose.Lateral)
        {
            turned.IsLateral = true;
            turned.Pix2mm = DynafaceConstants.LateralPix2mm;
            turned.LateralLandmarks = (Vec2[])LateralLandmarks.Clone();
            turned.SagittalX = Enumerable.Range(0, 1024).Select(y => 300f + y % 50).ToArray();
            turned.SagittalY = Enumerable.Range(0, 1024).Select(y => (float)y).ToArray();
        }
        return turned;
    }

    // A mixed-view video: the same three frames as quarter frames (0 to 2), profiles (3
    // to 5) and frontal frames (6 to 8). A measure's outputs are NaN on the frames whose
    // view it doesn't suit, so the frontal measures' numbers from a turned head (which
    // Analyze computes all the same) are never reported, and the peaks and the
    // evaluation come from the frontal frames, although the turned copies come first and
    // would win a tie. Mirrors dynaface-lib's test_a_measure_is_nan_off_its_view.
    [Fact]
    public void AMeasureIsNaNOffItsView()
    {
        var quarter = States.Select(s => InView(s, FacePose.Quarter)).ToArray();
        var lateral = States.Select(s => InView(s, FacePose.Lateral)).ToArray();
        var series = FrameSeries.FromStates(quarter.Concat(lateral).Concat(States).ToArray());
        Assert.Equal(Enumerable.Range(0, 9), series.Frames);
        var measures = DynafaceMeasures.AllMeasures();
        Assert.Equal(DynafaceMeasures.GetAllItems(measures), series.Names);

        // Analyze itself still measures a turned head in millimeters.
        var turned = FaceAnalysisPersistence.Restore(quarter[0]);
        Assert.True(turned.IsQuarter);
        Assert.True(double.IsFinite(turned.Analyze(render: false)!["dental_area"]));

        var live = Results.Select(r => TestHelpers.ContextFor(r).Analyze(render: false)!).ToArray();
        var rows = new[] { (pose: FacePose.Quarter, first: 0), (pose: FacePose.Lateral, first: 3) };
        foreach (var measure in measures)
        {
            foreach (string name in DynafaceMeasures.GetAllItems(new[] { measure }))
            {
                var values = series[name];
                foreach (var (pose, first) in rows)
                {
                    if (measure.Suits(pose == FacePose.Lateral, quarter: pose == FacePose.Quarter)) continue;
                    for (int row = first; row < first + 3; row++)
                        Assert.True(double.IsNaN(values[row]), $"{name} {row}");
                }
                for (int row = 6; row < 9; row++)
                {
                    if (measure.Suits(false))
                    {
                        double expected = live[row - 6].TryGetValue(name, out double value) ? value : double.NaN;
                        Assert.True(Same(expected, values[row]), $"{name} {row}");
                    }
                    else
                        Assert.True(double.IsNaN(values[row]), $"{name} {row}");
                }
            }
        }
        // The measures that suit a view keep their values there, except a quarter face's
        // scale, which isn't defined.
        foreach (string name in new[] { "yaw", "landmark-1-x", "pd", "tilt" })
            Assert.All(series[name].Take(3), v => Assert.True(double.IsFinite(v), name));
        Assert.All(series["px2mm"].Take(3), v => Assert.True(double.IsNaN(v)));
        foreach (string name in new[] { "yaw", "nfa", "px2mm" })
            Assert.All(series[name].Skip(3).Take(3), v => Assert.True(double.IsFinite(v), name));
        Assert.All(series["nfa"].Take(3), v => Assert.True(double.IsNaN(v)));
        Assert.All(series["fai"].Take(6), v => Assert.True(double.IsNaN(v)));

        // The peaks and the evaluation skip the turned frames.
        int smile = 6 + ArgMax(live.Select(r => r["dental_area"]));
        int eyes = 6 + ArgMax(live.Select(r => r["eye.left"] + r["eye.right"]));
        Assert.Equal(smile, series.Peak("dental_area"));
        Assert.Equal(eyes, series.Peak("eye.left", "eye.right"));
        var evaluation = series.Evaluate();
        Assert.NotNull(evaluation);
        Assert.Equal(smile, evaluation!.WidestSmile.Frame);
        Assert.Equal(eyes, evaluation.WidestEyes.Frame);
    }

    // The first of the largest values' position, as Python's max() picks it.
    static int ArgMax(IEnumerable<double> values)
    {
        int best = -1, index = 0;
        double bestValue = double.NegativeInfinity;
        foreach (double value in values)
        {
            if (value > bestValue) { best = index; bestValue = value; }
            index++;
        }
        return best;
    }
}
