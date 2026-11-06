namespace DynafaceTests;

// LandmarkSmoother and VideoAnalyzer, which import a video. Mirrors dynaface-lib-python's
// tests/test_video.py, with a stand-in for the pipeline instead of the models.
public class LandmarkSmootherTests
{
    static Vec2[] Points(params (float x, float y)[] points) => points.Select(p => new Vec2(p.x, p.y)).ToArray();

    [Fact]
    public void AveragesTheLastFrames()
    {
        var smoother = new LandmarkSmoother(2);
        Assert.Equal(Points((10, 20), (30, 41)), smoother.Add(Points((10, 20), (30, 41))));
        // Averages are truncated to whole pixels: 11.5 is 11, 40.5 is 40.
        Assert.Equal(Points((11, 22), (30, 40)), smoother.Add(Points((13, 25), (31, 40))));
        // Only the last 2 frames count.
        Assert.Equal(Points((16, 27), (30, 40)), smoother.Add(Points((20, 30), (30, 40))));
        smoother.Reset();
        Assert.Equal(Points((0, 0), (1, 1)), smoother.Add(Points((0, 0), (1, 1))));
    }

    [Fact]
    public void SizeOneIsOff()
    {
        var smoother = new LandmarkSmoother(1);
        smoother.Add(Points((10, 20)));
        Assert.Equal(Points((13, 25)), smoother.Add(Points((13, 25))));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LandmarkSmoother(0));
    }
}

public class VideoAnalyzerTests
{
    static readonly FaceImage Frame = new(
        Enumerable.Repeat(new Rgba32(128, 128, 128, 255), 1280 * 720).ToArray(), 1280, 720);
    static readonly float[] HeadPose = { 8f, 1f, -2f, 0f, 0f, 0f };

    // What the models would find in a made-up frame: pupils 230 + spread pixels apart, so
    // each spread measures differently.
    static Vec2[] Detected(float spread)
    {
        var right = new Vec2(530f, 320f);
        var left = new Vec2(760f + spread, 330f);
        var landmarks = new Vec2[98];
        for (int i = 0; i < landmarks.Length; i++)
            landmarks[i] = new Vec2(right.X - 80f + (i * 37) % 380, right.Y - 120f + (i * 53) % 400);
        landmarks[DynafaceConstants.LmRightPupil] = right;
        landmarks[DynafaceConstants.LmLeftPupil] = left;
        return landmarks;
    }

    static FacePipelineResult Cropped(float spread) =>
        FacePipeline.RunFrontal(Frame, Detected(spread), HeadPose, FacePose.Frontal,
            DynafaceConstants.DefaultTiltThreshold, crop: true)!.Value;

    // A stand-in for FacePipeline.Run that takes the next made-up detection on each call
    // (null for a frame without a face), running the frontal branch as the pipeline would.
    static VideoAnalyzer Analyzer(float?[] spreads, int smoothing = 1, int maxFrames = 5000, bool keepImages = true)
    {
        int call = 0;
        return new VideoAnalyzer(smoothing, maxFrames, keepImages, DynafaceConstants.DefaultTiltThreshold,
            (frame, rotationAngle, flipHorizontal, crop) =>
            {
                float? spread = spreads[call++];
                return spread == null
                    ? null
                    : FacePipeline.RunFrontal(frame, Detected(spread.Value), HeadPose, FacePose.Frontal,
                        DynafaceConstants.DefaultTiltThreshold, crop);
            });
    }

    static VideoAnalyzer Import(float?[] spreads, int smoothing = 1, int maxFrames = 5000, bool keepImages = true)
    {
        var analyzer = Analyzer(spreads, smoothing, maxFrames, keepImages);
        foreach (var _ in spreads)
            if (!analyzer.IsFull) analyzer.Add(Frame);
        return analyzer;
    }

    [Fact]
    public void FramesWithoutAFaceAreSkippedButCounted()
    {
        var analyzer = Import(new float?[] { 0f, null, 40f });
        Assert.Equal(new[] { 0, 2 }, analyzer.Frames);
        Assert.Equal(2, analyzer.States.Count);
        Assert.Same(analyzer.States[0].PostCropImage.Pixels, analyzer.Thumbnail!.Value.Pixels);
        // The series keeps the frames' numbers, so their times are right too.
        var series = analyzer.Series(new[] { new MeasureFAI() }, fps: 30f);
        Assert.Equal(new[] { 0, 2 }, series.Frames);
        Assert.Equal(2.0 / 30.0, series.Time(2), 10);
    }

    [Fact]
    public void FramesThatCouldntBeRead()
    {
        var analyzer = Analyzer(new float?[] { 0f, 0f });
        analyzer.Add(Frame);
        analyzer.Skip();
        analyzer.Add(Frame);
        Assert.Equal(new[] { 0, 2 }, analyzer.Frames);
    }

    [Fact]
    public void OnlyTheFirstFrames()
    {
        var analyzer = Import(new float?[] { 0f, 0f, 0f }, maxFrames: 2);
        Assert.Equal(new[] { 0, 1 }, analyzer.Frames);
        Assert.True(analyzer.IsFull);
        Assert.False(analyzer.Add(Frame));
        Assert.Null(new VideoAnalyzer().Thumbnail);
    }

    [Fact]
    public void WithoutSmoothingEachFrameIsAsAnalyzed()
    {
        var analyzer = Import(new float?[] { 0f, 80f });
        for (int i = 0; i < 2; i++)
        {
            var expected = Cropped(i == 0 ? 0f : 80f);
            Assert.Equal(expected.Wflw98, analyzer.States[i].Landmarks);
            Assert.Equal(expected.Pix2mm, analyzer.States[i].Pix2mm);
        }
    }

    [Fact]
    public void Smoothing()
    {
        var analyzer = Import(new float?[] { 0f, 80f }, smoothing: 2);
        var first = Cropped(0f);
        var second = Cropped(80f);
        var smoother = new LandmarkSmoother(2);
        smoother.Add(first.Wflw98);
        Vec2[] averaged = smoother.Add(second.Wflw98);
        Assert.Equal(first.Wflw98, analyzer.States[0].Landmarks);
        Assert.Equal(averaged, analyzer.States[1].Landmarks);
        Assert.NotEqual(second.Wflw98, averaged);
        // The saved scale is the averaged landmarks' own.
        var (pd, pix2mm) = FaceMeasureContext.CalcPd(averaged, second.PupilDistMm);
        Assert.Equal((pd, pix2mm), (analyzer.States[1].PupillaryDistance, analyzer.States[1].Pix2mm));
    }

    [Fact]
    public void KeepingOnlyLandmarks()
    {
        // Analyzed uncropped with only the landmarks cropped: the same landmarks and
        // scale as cropping, no images, and the same thumbnail.
        var withImages = Import(new float?[] { 0f, 80f }, smoothing: 2);
        var landmarksOnly = Import(new float?[] { 0f, 80f }, smoothing: 2, keepImages: false);
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(withImages.States[i].Landmarks, landmarksOnly.States[i].Landmarks);
            Assert.Equal(withImages.States[i].Pix2mm, landmarksOnly.States[i].Pix2mm);
            Assert.False(landmarksOnly.States[i].PostCropImage.IsValid);
        }
        Assert.Equal(withImages.Thumbnail!.Value.Pixels, landmarksOnly.Thumbnail!.Value.Pixels);
        // Measured on their landmarks alone, the frames give the same values, skin tone
        // aside.
        var full = withImages.Series();
        var lean = landmarksOnly.Series();
        Assert.Equal(full["ml"], lean["ml"]);
        Assert.Equal(full["eye.left"], lean["eye.left"]);
        Assert.True(double.IsNaN(lean["hue"][0]));
    }

    [Fact]
    public void FrameNumbersMustMatchTheStates()
    {
        var analyzer = Import(new float?[] { 0f, 40f });
        Assert.Throws<ArgumentException>(() => FrameSeries.FromStates(analyzer.States, frameNumbers: new[] { 0 }));
    }
}
