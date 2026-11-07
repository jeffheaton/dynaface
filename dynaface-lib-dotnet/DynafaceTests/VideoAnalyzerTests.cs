namespace DynafaceTests;

// LandmarkSmoother and VideoAnalyzer, which import a video. Mirrors dynaface-lib-python's
// tests/test_video.py and TestQuarterVideo in tests/test_pose_quarter.py, with a stand-in
// for the pipeline instead of the models.
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
    public void StateIndexFindsAFramesState()
    {
        // Peak and Evaluate give frame numbers; after a skipped frame they aren't the
        // positions in States.
        var analyzer = Import(new float?[] { 0f, null, 10f });
        Assert.Equal(new[] { 0, 2 }, analyzer.Frames);
        Assert.Equal(0, analyzer.StateIndex(0));
        Assert.Equal(1, analyzer.StateIndex(2));
        Assert.Equal(-1, analyzer.StateIndex(1));   // skipped
        Assert.Equal(-1, analyzer.StateIndex(7));
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

    // A video that turns, from the synthetic-head renders (QuarterFixtures): each frame's
    // result as the pipeline gives it, frontal or quarter, from a stand-in for
    // FacePipeline.Run that records the previous pose each frame is analyzed with.
    static (VideoAnalyzer Analyzer, List<FacePose?> Previous) TurningVideo(
        (FaceImage Frame, Vec2[] Landmarks, float[] Pose)[] frames, bool[] quarter, int smoothing,
        bool keepImages = true)
    {
        int call = 0;
        var previous = new List<FacePose?>();
        var analyzer = new VideoAnalyzer(smoothing, VideoAnalyzer.DefaultMaxFrames, keepImages,
            DynafaceConstants.DefaultTiltThreshold,
            (frame, rotationAngle, flipHorizontal, crop, previousPose) =>
            {
                previous.Add(previousPose);
                int i = call++;
                return quarter[i]
                    ? QuarterFixtures.Quarter(frames[i])   // always cropped, crop or not
                    : FacePipeline.RunFrontal(frames[i].Frame, frames[i].Landmarks, frames[i].Pose,
                        FacePose.Frontal, DynafaceConstants.DefaultTiltThreshold, crop);
            });
        foreach (var f in frames) Assert.True(analyzer.Add(f.Frame));
        return (analyzer, previous);
    }

    [Fact]
    public void ASmoothedVideoOverBothViews()
    {
        var frames = new[] { "yaw0", "yaw45", "yaw45_hollow3", "yaw0" }.Select(n => QuarterFixtures.WideFrame(n)).ToArray();
        var quarter = new[] { false, true, true, false };
        var alone = frames.Select((f, i) => quarter[i] ? QuarterFixtures.Quarter(f) : QuarterFixtures.Frontal(f)).ToArray();
        var (analyzer, previous) = TurningVideo(frames, quarter, smoothing: 2);

        // Each frame is analyzed with the previous kept frame's pose.
        Assert.Equal(new FacePose?[] { null, FacePose.Frontal, FacePose.Quarter, FacePose.Quarter }, previous);
        var states = analyzer.States;
        Assert.Equal(new[] { 0, 1, 2, 3 }, analyzer.Frames);
        Assert.Equal(new[] { FacePose.Frontal, FacePose.Quarter, FacePose.Quarter, FacePose.Frontal },
            states.Select(s => s.Pose));

        // The smoothing starts afresh at both view changes: frames 1 and 3 keep their own
        // landmarks, and only frame 2 averages with the frame before.
        Assert.Equal(alone[0].Wflw98, states[0].Landmarks);
        Assert.Equal(alone[1].Wflw98, states[1].Landmarks);
        Assert.Equal(alone[3].Wflw98, states[3].Landmarks);
        var smoother = new LandmarkSmoother(2);
        smoother.Add(alone[1].Wflw98);
        Vec2[] averaged = smoother.Add(alone[2].Wflw98);
        Assert.Equal(averaged, states[2].Landmarks);
        Assert.NotEqual(alone[2].Wflw98, averaged);
        // The saved scale is the averaged landmarks' own pupil scale.
        var (pd, pix2mm) = FaceMeasureContext.CalcPd(averaged, alone[2].PupilDistMm);
        Assert.Equal((pd, pix2mm), (states[2].PupillaryDistance, states[2].Pix2mm));

        // Each quarter frame keeps an outline traced from its own mask on the landmarks
        // its state saves: frame 1's on its own detection, since the smoothing started
        // afresh there, and frame 2's traced again on the averaged landmarks, so its rows
        // are the averaged band's, not its detection's; the frontal frames keep none.
        Assert.True(states[1].FarLeft);
        Assert.True(states[2].FarLeft);
        Assert.Equal(alone[1].CheekOutlineX, states[1].CheekOutlineX);
        Assert.Equal(alone[1].CheekOutlineY, states[1].CheekOutlineY);
        Assert.Equal(alone[1].CheekOutlineSrc, states[1].CheekOutlineSrc);
        var (fg, alpha) = QuarterFixtures.BackgroundMask(alone[2].AlignedCrop);
        var retraced = CheekOutline.TraceFarCheek(fg, alpha, averaged, true, gray: CheekOutline.GrayImage(alone[2].AlignedCrop));
        Assert.Equal(retraced.X, states[2].CheekOutlineX);
        Assert.Equal(retraced.Y, states[2].CheekOutlineY);
        Assert.Equal(retraced.Src, states[2].CheekOutlineSrc);
        Assert.Equal(((int)Math.Min(averaged[72].Y, averaged[32].Y), (int)averaged[16].Y),
            (states[2].CheekOutlineY[0], states[2].CheekOutlineY[^1]));
        Assert.NotEqual(alone[2].CheekOutlineY, states[2].CheekOutlineY);
        foreach (int i in new[] { 0, 3 })
        {
            Assert.Null(states[i].FarLeft);
            Assert.Null(states[i].CheekOutlineX);
        }

        // The second quarter frame's apex comes from the averaged landmarks on that frame's
        // own outline.
        var restored = FaceAnalysisPersistence.Restore(states[2]);
        Assert.Equal(averaged, restored.Landmarks);
        var expected = CheekOutline.ObservedPoints(averaged, retraced.X, retraced.Y, retraced.Src, true);
        Assert.NotNull(expected);
        Assert.Equal(expected!.ApexH, restored.ObliqueCheek()!.ApexH);
        Assert.Equal(expected.Apex, restored.CheekLandmarks[Cheek.CheekLmApexL]);
    }

    [Fact]
    public void AFarSideFlipResetsTheSmoothing()
    {
        // A quarter video whose far cheek flips (a cut between the two mirrored turns): the
        // two band crops are mirrored, so the smoothing starts afresh at the flip.
        var frames = new[] { QuarterFixtures.WideFrame("yaw45"), QuarterFixtures.WideFrame("yaw45_hollow3") };
        frames = frames.Concat(frames.Select(QuarterFixtures.Mirrored)).ToArray();
        var alone = frames.Select(QuarterFixtures.Quarter).ToArray();
        Assert.Equal(new[] { true, true, false, false }, alone.Select(r => r.FarLeft));
        var (analyzer, previous) = TurningVideo(frames, new[] { true, true, true, true }, smoothing: 2);
        var states = analyzer.States;
        Assert.Equal(new FacePose?[] { null, FacePose.Quarter, FacePose.Quarter, FacePose.Quarter }, previous);
        Assert.Equal(new bool?[] { true, true, false, false }, states.Select(s => s.FarLeft));

        // Frames 0 and 2 start an average; 1 and 3 average with the frame before.
        var smoother = new LandmarkSmoother(2);
        Assert.Equal(alone[0].Wflw98, states[0].Landmarks);
        smoother.Add(alone[0].Wflw98);
        Assert.Equal(smoother.Add(alone[1].Wflw98), states[1].Landmarks);
        Assert.Equal(alone[2].Wflw98, states[2].Landmarks);
        smoother.Reset();
        smoother.Add(alone[2].Wflw98);
        Assert.Equal(smoother.Add(alone[3].Wflw98), states[3].Landmarks);
        Assert.NotEqual(alone[3].Wflw98, states[3].Landmarks);

        // Flip-free frames keep their own outline; the averaged ones are traced on the
        // averaged landmarks.
        foreach (int i in new[] { 0, 2 })
            Assert.Equal(alone[i].CheekOutlineX, states[i].CheekOutlineX);
        foreach (int i in new[] { 1, 3 })
        {
            var expected = alone[i].RetraceFarCheek(states[i].Landmarks)!.Value;
            Assert.Equal(expected.X, states[i].CheekOutlineX);
            Assert.Equal(expected.Y, states[i].CheekOutlineY);
        }
        // Each turn's rows fill its own side's cheek landmarks.
        Assert.True(FaceAnalysisPersistence.Restore(states[1]).CheekLandmarks[Cheek.CheekLmApexL].IsFinite);
        Assert.True(FaceAnalysisPersistence.Restore(states[3]).CheekLandmarks[Cheek.CheekLmApexR].IsFinite);
    }

    [Fact]
    public void AProfileFacingTheOtherWayResetsTheSmoothing()
    {
        // The view includes which way a profile faces: the flipped one's crop is mirrored.
        var profile = SavedViewTests.Profile();
        var other = SavedViewTests.Profile();
        other.Flipped = false;
        other.Wflw98 = other.Wflw98.Select(p => new Vec2(p.X + 40, p.Y + 20)).ToArray();
        var results = new[] { profile, other, other };
        int call = 0;
        var analyzer = new VideoAnalyzer(2, VideoAnalyzer.DefaultMaxFrames, keepImages: true,
            DynafaceConstants.DefaultTiltThreshold, (frame, rotationAngle, flipHorizontal, crop) => results[call++]);
        foreach (var _ in results) analyzer.Add(Frame);
        Assert.Equal(other.Wflw98, analyzer.States[1].Landmarks);   // afresh at the flip
        Assert.Equal(other.Wflw98, analyzer.States[2].Landmarks);   // the same frame twice
    }

    [Fact]
    public void KeepingOnlyLandmarksOfAQuarterVideo()
    {
        // A quarter frame is always cropped, image and all, so keepImages: false keeps the
        // same landmarks, scale and outline, without the images.
        var frames = new[] { "yaw45", "yaw45_hollow3" }.Select(n => QuarterFixtures.WideFrame(n)).ToArray();
        var quarter = new[] { true, true };
        var withImages = TurningVideo(frames, quarter, smoothing: 2).Analyzer;
        var landmarksOnly = TurningVideo(frames, quarter, smoothing: 2, keepImages: false).Analyzer;
        for (int i = 0; i < 2; i++)
        {
            var a = withImages.States[i];
            var b = landmarksOnly.States[i];
            Assert.Equal(a.Landmarks, b.Landmarks);
            Assert.Equal(a.Pix2mm, b.Pix2mm);
            Assert.Equal(a.CheekOutlineX, b.CheekOutlineX);
            Assert.Equal(a.CheekOutlineY, b.CheekOutlineY);
            Assert.Equal(a.FarLeft, b.FarLeft);
            Assert.False(b.PostCropImage.IsValid);
        }
        Assert.Equal(withImages.Thumbnail!.Value.Pixels, landmarksOnly.Thumbnail!.Value.Pixels);
        Assert.Equal(FaceAnalysisPersistence.Restore(withImages.States[1]).CheekLandmarks,
            FaceAnalysisPersistence.Restore(landmarksOnly.States[1]).CheekLandmarks);
    }
}
