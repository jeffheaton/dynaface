namespace DynafaceTests;

// A saved analysis keeps its view: the pose, whether a profile was flipped, and a
// profile's lateral landmarks and sagittal outline, so a restored profile measures as
// the original did. Mirrors dynaface-lib's TestSavedView in tests/test_persist.py; the
// model-run profile is in PersistenceIntegrationTests.
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
}
