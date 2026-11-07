namespace DynafaceTests;

// A saved analysis keeps its view: the pose, whether a profile was flipped, a
// profile's lateral landmarks and sagittal outline, and a three-quarter face's far side
// and far-cheek outline, so a restored profile or quarter face measures as the original
// did. Mirrors dynaface-lib's TestSavedView in tests/test_persist.py and TestQuarterState
// in tests/test_pose_quarter.py; the model-run profile is in PersistenceIntegrationTests
// and the model-run quarter face in QuarterPipelineTests.
public class SavedViewTests
{
    static readonly Vec2[] LateralLandmarks =
    {
        new(300, 200), new(290, 300), new(200, 450), new(260, 520), new(280, 640), new(270, 760),
    };

    static FaceImage Gray() => new(
        Enumerable.Repeat(new Rgba32(128, 128, 128, 255), 1024 * 1024).ToArray(), 1024, 1024);

    // What the pipeline gives for a profile facing right, made up instead of run.
    internal static FacePipelineResult Profile() => new()
    {
        AlignedCrop = Gray(),
        Wflw98 = TestHelpers.MakeLandmarks(),
        HeadPose = new[] { 60f, -4f, 2f, 0f, 0f, 0f },
        Pose = FacePose.Lateral,
        IsLateral = true,
        Flipped = true,
        Pix2mm = DynafaceConstants.LateralPix2mm,
        LateralAnalysis = new LateralAnalyzer.Result
        {
            LateralLandmarks = (Vec2[])LateralLandmarks.Clone(),
            SagittalX = Enumerable.Range(0, 1024).Select(y => 300f + y % 50).ToArray(),
            SagittalY = Enumerable.Range(0, 1024).Select(y => (float)y).ToArray(),
        },
    };

    static FaceMeasureBase[] Measures() => new FaceMeasureBase[] { new MeasureLateral(), new MeasurePose() };

    [Fact]
    public void AProfileSurvivesSaving()
    {
        var result = Profile();
        var expected = TestHelpers.ContextFor(result).Analyze(Measures(), render: false);
        Assert.Contains("nfa", expected.Keys);

        var state = FaceAnalysisPersistence.Capture(result);
        Assert.True(state.IsLateral);
        Assert.True(state.Flipped);
        Assert.Equal(FacePose.Lateral, state.Pose);
        Assert.Equal(LateralLandmarks, state.LateralLandmarks);
        Assert.NotSame(result.LateralAnalysis!.Value.LateralLandmarks, state.LateralLandmarks);
        Assert.Equal(result.LateralAnalysis.Value.SagittalX, state.SagittalX);
        Assert.Equal(result.LateralAnalysis.Value.SagittalY, state.SagittalY);

        var restored = FaceAnalysisPersistence.Restore(state);
        Assert.True(restored.IsLateral);
        Assert.True(restored.Flipped);
        Assert.Equal(FacePose.Lateral, restored.Pose);
        Assert.Equal(DynafaceConstants.LateralPix2mm, restored.Pix2mm);
        Assert.Equal(expected, restored.Analyze(Measures(), render: false));
    }

    [Fact]
    public void AFrontalFaceSavesNoLateralData()
    {
        var result = Profile();
        result.IsLateral = false;
        result.Flipped = false;
        result.Pose = FacePose.Frontal;
        var state = FaceAnalysisPersistence.Capture(result);
        Assert.False(state.IsLateral);
        Assert.Null(state.LateralLandmarks);
        Assert.Null(state.SagittalX);
        Assert.Empty(FaceAnalysisPersistence.Restore(state).Analyze(new[] { new MeasureLateral() }));
        // Nor a quarter view.
        Assert.Null(state.FarLeft);
        Assert.Null(state.CheekOutlineX);
        Assert.Null(state.CheekOutlineY);
        Assert.Null(state.CheekOutlineSrc);
        Assert.False(FaceAnalysisPersistence.Restore(state).IsQuarter);
    }

    // Up to 2.0.4 a state held no view, and it restores as a frontal face, as it did.
    [Fact]
    public void AStateWithoutAViewRestoresAsAFrontalFace()
    {
        var saved = FaceAnalysisPersistence.Capture(Profile());
        var state = new FaceAnalysisState
        {
            PostCropImage = saved.PostCropImage,
            HeadPose = saved.HeadPose,
            Landmarks = saved.Landmarks,
            PupillaryDistance = saved.PupillaryDistance,
            Pix2mm = saved.Pix2mm,
        };
        var restored = FaceAnalysisPersistence.Restore(state);
        Assert.False(restored.IsLateral);
        Assert.False(restored.IsQuarter);
        Assert.Equal(FacePose.Frontal, restored.Pose);
        Assert.Empty(restored.Analyze(new[] { new MeasureLateral() }));
    }

    // Measures draw on the context's image. Up to 2.0.4 Restore gave the context the
    // state's own pixels, so drawing on a restored analysis changed the saved image.
    [Fact]
    public void DrawingOnARestoredStateLeavesItAlone()
    {
        var state = FaceAnalysisPersistence.Capture(Profile());
        var saved = (Rgba32[])state.PostCropImage.Pixels.Clone();

        var restored = FaceAnalysisPersistence.Restore(state);
        restored.Analyze(Measures(), render: true);
        Assert.NotEqual(saved, restored.Pixels);   // it drew
        Assert.Equal(saved, state.PostCropImage.Pixels);
    }

    [Fact]
    public void FromStatesMeasuresAProfile()
    {
        var state = FaceAnalysisPersistence.Capture(Profile());
        double nfa = FaceAnalysisPersistence.Restore(state).Analyze(Measures(), render: false)["nfa"];

        var series = FrameSeries.FromStates(new[] { state, state.WithImage(default) }, Measures());
        Assert.Contains("nfa", series.Names);
        Assert.Equal(new[] { nfa, nfa }, series["nfa"]);   // with its image, and without
    }

    [Fact]
    public void VideoAnalyzerKeepsAProfilesView()
    {
        var analyzer = new VideoAnalyzer(1, VideoAnalyzer.DefaultMaxFrames, keepImages: true,
            DynafaceConstants.DefaultTiltThreshold, (frame, rotationAngle, flipHorizontal, crop) => Profile());
        Assert.True(analyzer.Add(Gray()));

        var state = Assert.Single(analyzer.States);
        Assert.True(state.IsLateral);
        Assert.True(state.Flipped);
        Assert.Equal(LateralLandmarks, state.LateralLandmarks);
        Assert.Contains("nfa", analyzer.Series(Measures()).Names);
    }

    // The 45 degree render's quarter face, cropped and traced without the models (see
    // QuarterFixtures).
    static readonly Lazy<FacePipelineResult> Yaw45 = new(() => QuarterFixtures.Quarter(QuarterFixtures.WideFrame("yaw45")));

    // The measures that suit the quarter view.
    static FaceMeasureBase[] QuarterMeasures() =>
        new FaceMeasureBase[] { new MeasurePosition(), new MeasurePose(), new MeasureLandmarks() };

    static void AssertSameResults(Dictionary<string, double> expected, Dictionary<string, double> actual)
    {
        Assert.Equal(expected.Keys.OrderBy(k => k, StringComparer.Ordinal), actual.Keys.OrderBy(k => k, StringComparer.Ordinal));
        foreach (var kv in expected)
            Assert.True(kv.Value.Equals(actual[kv.Key]), $"{kv.Key}: {kv.Value} != {actual[kv.Key]}");
    }

    [Fact]
    public void AQuarterFaceSurvivesSaving()
    {
        var result = Yaw45.Value;
        var original = FaceMeasureContext.FromResult(result);
        var expected = original.Analyze(QuarterMeasures(), render: false)!;

        // The state holds the far side and copies of the outline, never the mask.
        var state = FaceAnalysisPersistence.Capture(result);
        Assert.Equal(FacePose.Quarter, state.Pose);
        Assert.False(state.IsLateral || state.Flipped);
        Assert.True(state.FarLeft);
        Assert.Equal(result.CheekOutlineX, state.CheekOutlineX);
        Assert.Equal(result.CheekOutlineY, state.CheekOutlineY);
        Assert.Equal(result.CheekOutlineSrc, state.CheekOutlineSrc);
        Assert.NotSame(result.CheekOutlineX, state.CheekOutlineX);
        Assert.NotSame(result.CheekOutlineY, state.CheekOutlineY);
        Assert.NotSame(result.CheekOutlineSrc, state.CheekOutlineSrc);
        Assert.Null(state.LateralLandmarks);

        // Restored, it is the quarter face it was: the far side, the outline (a copy, so
        // the context and the state don't share it), the cheek landmarks and the values.
        var restored = FaceAnalysisPersistence.Restore(state);
        Assert.True(restored.IsQuarter);
        Assert.False(restored.IsLateral);
        Assert.True(restored.FarLeft);
        Assert.Equal(state.CheekOutlineX, restored.CheekOutlineX);
        Assert.NotSame(state.CheekOutlineX, restored.CheekOutlineX);
        Assert.Equal(result.Pix2mm, restored.Pix2mm);
        Assert.Equal(original.CheekLandmarks, restored.CheekLandmarks);
        QuarterFixtures.AssertRows(restored.CheekLandmarks, Cheek.CheekLmApexL, Cheek.CheekLmSubmalarL);
        Assert.Equal(original.ObliqueCheek()!.ApexH, restored.ObliqueCheek()!.ApexH);
        AssertSameResults(expected, restored.Analyze(QuarterMeasures(), render: false)!);
        // And so does one saved without its image.
        AssertSameResults(expected, FaceAnalysisPersistence.Restore(state.WithImage(default)).Analyze(QuarterMeasures(), render: false)!);
    }

    [Fact]
    public void AQuarterStateSavedWithoutItsFarSideTakesItFromItsLandmarks()
    {
        // A quarter state saved before the quarter view was kept: the pose alone. Its far
        // side comes from its landmarks and the landmark contour stands in for the outline.
        var state = FaceAnalysisPersistence.Capture(Yaw45.Value);
        state.FarLeft = null;
        state.CheekOutlineX = null;
        state.CheekOutlineY = null;
        state.CheekOutlineSrc = null;
        var restored = FaceAnalysisPersistence.Restore(state);
        Assert.True(restored.IsQuarter);
        Assert.True(restored.FarLeft);
        Assert.Null(restored.CheekOutlineX);
        Assert.Equal(Cheek.LocateAll(state.Landmarks, FacePose.Quarter, true, null, null, null), restored.CheekLandmarks);
        QuarterFixtures.AssertRows(restored.CheekLandmarks, Cheek.CheekLmApexL);

        // The mirrored turn's far side is the right.
        var mirrored = FaceAnalysisPersistence.Capture(
            QuarterFixtures.Quarter(QuarterFixtures.Mirrored(QuarterFixtures.WideFrame("yaw45"))));
        Assert.False(mirrored.FarLeft);
        mirrored.FarLeft = null;
        Assert.False(FaceAnalysisPersistence.Restore(mirrored).FarLeft);
    }

    [Fact]
    public void AQuarterPoseWithoutLandmarksIsNoQuarterFace()
    {
        // The quarter view follows the pose and needs a face, as load_image() sets it.
        var state = FaceAnalysisPersistence.Capture(Yaw45.Value);
        state.Landmarks = System.Array.Empty<Vec2>();
        var restored = FaceAnalysisPersistence.Restore(state);
        Assert.Equal(FacePose.Quarter, restored.Pose);
        Assert.False(restored.IsQuarter);
        Assert.False(restored.FarLeft);
        Assert.Null(restored.CheekOutlineX);
    }

    [Fact]
    public void VideoAnalyzerKeepsAQuarterView()
    {
        var analyzer = new VideoAnalyzer(1, VideoAnalyzer.DefaultMaxFrames, keepImages: true,
            DynafaceConstants.DefaultTiltThreshold, (frame, rotationAngle, flipHorizontal, crop) => Yaw45.Value);
        Assert.True(analyzer.Add(Gray()));
        var state = Assert.Single(analyzer.States);
        Assert.Equal(FacePose.Quarter, state.Pose);
        Assert.True(state.FarLeft);
        Assert.Equal(Yaw45.Value.CheekOutlineX, state.CheekOutlineX);
        Assert.True(FaceAnalysisPersistence.Restore(state).IsQuarter);
    }
}
