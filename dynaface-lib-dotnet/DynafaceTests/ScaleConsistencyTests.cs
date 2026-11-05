using System.IO;

namespace DynafaceTests;

// Every millimeter value depends on Pix2mm matching the landmarks it's applied to,
// and that pairing has broken more than once: the scale was measured before the crop
// (from July 2024 until October 2026), an app kept an uncropped run's scale for
// landmarks it cropped later, and saved analyses kept a scale from the old rule.
// These measure one made-up face every way an app can reach its landmarks and require
// the same millimeters each time. Mirrors dynaface-lib-python's tests/test_scale.py;
// ScaleConsistencyModelTests repeats the first check on a photo.
public class ScaleConsistencyTests
{
    const int FrameWidth = 1280, FrameHeight = 720;
    static readonly float[] HeadPose = { 8f, 0f, 0f, 0f, 0f, 0f };

    static FaceImage Gray(int width, int height) =>
        new(Enumerable.Repeat(new Rgba32(128, 128, 128, 255), width * height).ToArray(), width, height);

    // A made-up frontal face in a video-sized frame: 98 landmarks scattered around
    // pupils 230 pixels apart, in top-left pixels.
    static Vec2[] Face()
    {
        var right = new Vec2(530.4f, 320.6f);
        var left = new Vec2(760.2f, 330.1f);
        var landmarks = new Vec2[98];
        for (int i = 0; i < landmarks.Length; i++)
            landmarks[i] = new Vec2(
                right.X - 80f + (i * 37) % 380 + 0.37f,
                right.Y - 120f + (i * 53) % 400 + 0.71f);
        landmarks[DynafaceConstants.LmRightPupil] = right;
        landmarks[DynafaceConstants.LmLeftPupil] = left;
        return landmarks;
    }

    static FacePipelineResult Run(bool crop) =>
        FacePipeline.RunFrontal(Gray(FrameWidth, FrameHeight), Face(), HeadPose, FacePose.Frontal,
            DynafaceConstants.DefaultTiltThreshold, crop)!.Value;

    internal static FaceImage Copy(FaceImage image) =>
        new((Rgba32[])image.Pixels.Clone(), image.Width, image.Height);

    // Every output of every measure, without drawing.
    internal static Dictionary<string, double> Outputs(FaceMeasureContext ctx) => ctx.Analyze(render: false)!;

    internal static void AssertSameOutputs(Dictionary<string, double> expected, Dictionary<string, double> actual)
    {
        Assert.Equal(expected.Keys.OrderBy(k => k), actual.Keys.OrderBy(k => k));
        foreach (var (key, value) in expected)
        {
            double other = actual[key];
            if (double.IsNaN(value))
                Assert.True(double.IsNaN(other), $"{key}: expected NaN, got {other}");
            else
                Assert.True(Math.Abs(value - other) <= 1e-5 * Math.Max(1.0, Math.Abs(value)),
                    $"{key}: expected {value}, got {other}");
        }
    }

    [Fact]
    public void SameFaceSameMillimeters()
    {
        // All three measure on a copy of the same cropped image, so only the scale
        // could make them differ.
        var still = Run(crop: true);
        var expected = Outputs(TestHelpers.ContextFor(still));

        // Analyzed without the crop and cropped afterward, like a video's frames.
        var uncropped = Run(crop: false);
        Vec2[] croppedLater = StyleGanCropper.CropLandmarks(
            FrameWidth, FrameHeight, uncropped.Wflw98, uncropped.HeadPose[0]);
        AssertSameOutputs(expected, Outputs(new FaceMeasureContext(
            Copy(still.AlignedCrop), croppedLater, headPose: uncropped.HeadPose, pupilDistMm: uncropped.PupilDistMm)));

        // Saved, then restored.
        AssertSameOutputs(expected,
            Outputs(FaceAnalysisPersistence.Restore(FaceAnalysisPersistence.Capture(still))));
    }

    [Fact]
    public void AnalysisSavedWithTheOldRuleIsCorrected()
    {
        // Before October 2026 the scale was measured on the uncropped pupils, and that
        // is the pair a saved analysis holds. Restoring measures it again.
        var still = Run(crop: true);
        var uncropped = Run(crop: false);
        Assert.True(Math.Abs(uncropped.Pix2mm - still.Pix2mm) > 0.01f, "the old rule's scale should differ");

        var state = FaceAnalysisPersistence.Capture(still);
        state.PupillaryDistance = uncropped.PupillaryDistance;
        state.Pix2mm = uncropped.Pix2mm;
        AssertSameOutputs(Outputs(TestHelpers.ContextFor(still)), Outputs(FaceAnalysisPersistence.Restore(state)));
    }

    [Fact]
    public void RestoredAnalysisKeepsItsPupillaryDistance()
    {
        // An analysis saved with a 58 mm pupillary distance restores with 58 mm,
        // whatever DynafaceConfig.PupilDistMm says today.
        var still = Run(crop: true);
        var state = FaceAnalysisPersistence.Capture(still);
        state.Pix2mm = 58f / state.PupillaryDistance;

        var restored = FaceAnalysisPersistence.Restore(state);
        Assert.Equal(58.0, restored.PupilDistMm, 3);
        Assert.Equal(58.0 / still.PupilDistMm, restored.Pix2mm / still.Pix2mm, 4);
    }

    [Fact]
    public void ProfileKeepsItsFixedScale()
    {
        // A profile's saved pair holds no pupillary distance (its pupils give none),
        // so its fixed scale comes back as it was.
        var state = new FaceAnalysisState
        {
            PostCropImage = Gray(1024, 1024),
            Landmarks = Face(),
            PupillaryDistance = 0f,
            Pix2mm = DynafaceConstants.LateralPix2mm,
        };
        Assert.Equal(DynafaceConstants.LateralPix2mm, FaceAnalysisPersistence.Restore(state).Pix2mm);
        Assert.Equal(DynafaceConstants.LateralPix2mm,
            new FaceMeasureContext(Gray(1024, 1024), Face(), isLateral: true).Pix2mm);
    }

    [Fact]
    public void TheOldConstructorStillPinsItsScale()
    {
#pragma warning disable CS0618 // kept for existing callers until they move to pupilDistMm
        var ctx = new FaceMeasureContext(Gray(1024, 1024), Face(), 0.5f);
#pragma warning restore CS0618
        Assert.Equal(0.5f, ctx.Pix2mm);
        Assert.Equal(DynafaceConfig.PupilDistMm, ctx.PupilDistMm);
    }
}

// The first check on a photo. img1-512.jpg is where the scale measured before the
// crop was furthest off, about 2.3 times too large.
[Collection("ModelTests")]
public class ScaleConsistencyModelTests
{
    readonly ModelFixture _fixture;

    public ScaleConsistencyModelTests(ModelFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void SamePhotoSameMillimeters()
    {
        _fixture.RequireModels();
        var photo = ImageLoader.Load(Path.Combine(AppContext.BaseDirectory, "TestData", "img1-512.jpg"));
        var still = FacePipeline.Run(photo, 0, false)!.Value;
        var expected = ScaleConsistencyTests.Outputs(TestHelpers.ContextFor(still));

        var uncropped = FacePipeline.Run(photo, 0, false, crop: false)!.Value;
        Vec2[] croppedLater = StyleGanCropper.CropLandmarks(
            photo.Width, photo.Height, uncropped.Wflw98, uncropped.HeadPose[0]);
        ScaleConsistencyTests.AssertSameOutputs(expected, ScaleConsistencyTests.Outputs(new FaceMeasureContext(
            ScaleConsistencyTests.Copy(still.AlignedCrop), croppedLater,
            headPose: uncropped.HeadPose, pupilDistMm: uncropped.PupilDistMm)));

        ScaleConsistencyTests.AssertSameOutputs(expected, ScaleConsistencyTests.Outputs(
            FaceAnalysisPersistence.Restore(FaceAnalysisPersistence.Capture(still))));
    }
}
