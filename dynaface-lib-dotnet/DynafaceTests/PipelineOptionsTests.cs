using System.IO;

namespace DynafaceTests;

// The pipeline's options and guards on made-up landmarks, without the models:
// straightening a tilted face, the turned-head correction, the croppers' edge
// cases (the centered quarter crop's too), forcing a profile or a quarter face and the
// image helpers they use. Mirrors
// dynaface-lib-python's tests/test_pipeline_options.py, with the same expectations;
// PipelineOptionsModelTests runs the options that need the models.
public class PipelineOptionsTests
{
    static FaceImage Gray(int width, int height) =>
        new(Enumerable.Repeat(new Rgba32(128, 128, 128, 255), width * height).ToArray(), width, height);

    // 98 landmarks in top-left pixels, all in the middle except the pupils.
    static Vec2[] MadeUpLandmarks(Vec2 rightPupil, Vec2 leftPupil)
    {
        var landmarks = Enumerable.Repeat(new Vec2(512, 512), 98).ToArray();
        landmarks[DynafaceConstants.LmRightPupil] = rightPupil;
        landmarks[DynafaceConstants.LmLeftPupil] = leftPupil;
        return landmarks;
    }

    static Vec2[] Face() => MadeUpLandmarks(new(400, 500), new(600, 500)); // 200 px apart

    static float PupilDistance(Vec2[] landmarks) =>
        Vec2.Distance(landmarks[DynafaceConstants.LmRightPupil], landmarks[DynafaceConstants.LmLeftPupil]);

    [Fact]
    public void TurnedHeadCorrection()
    {
        // The crop puts the pupils 260 pixels apart, but when the head is turned more
        // than 5 degrees, it first corrects their distance, making the face smaller.
        var straight = StyleGanCropper.Crop(Gray(1024, 1024), Face(), yaw: 0f);
        var turned = StyleGanCropper.Crop(Gray(1024, 1024), Face(), yaw: 10f);
        Assert.InRange(PupilDistance(straight.landmarks), 258f, 262f);
        Assert.True(PupilDistance(turned.landmarks) < 240f);
    }

    [Theory]
    [InlineData(8f)]
    [InlineData(-8f)]
    public void StraighteningATiltedFace(float degrees)
    {
        float rad = degrees * MathF.PI / 180f;
        var tilted = MadeUpLandmarks(new(400, 500), new(400 + 200 * MathF.Cos(rad), 500 + 200 * MathF.Sin(rad)));

        // Kept as it is by default, straightened above a threshold.
        Assert.Null(StyleGanCropper.Crop(Gray(1024, 1024), tilted, 0f).faceRotationRad);
        var straightened = StyleGanCropper.Crop(Gray(1024, 1024), tilted, 0f, tiltThreshold: 2f);
        Assert.NotNull(straightened.faceRotationRad);
        Vec2 right = straightened.landmarks[DynafaceConstants.LmRightPupil];
        Vec2 left = straightened.landmarks[DynafaceConstants.LmLeftPupil];
        Assert.InRange(MathF.Abs(left.Y - right.Y), 0f, 2f);
    }

    [Fact]
    public void PupilsInOnePlace() =>
        Assert.Throws<InvalidOperationException>(() =>
            StyleGanCropper.Crop(Gray(1024, 1024), MadeUpLandmarks(new(500, 500), new(500, 500)), 0f));

    [Fact]
    public void GivenPupilsSetTheScale()
    {
        // Pupils 300 pixels apart scale the crop by 260/300, so the landmarks' own
        // pupils, 200 apart, end up about 173 apart.
        var crop = StyleGanCropper.Crop(Gray(1024, 1024), Face(), 0f,
            pupilLeftOverride: new Vec2(650, 500), pupilRightOverride: new Vec2(350, 500));
        Assert.InRange(PupilDistance(crop.landmarks), 170f, 177f);
    }

    [Fact]
    public void FrontalWithoutAHeadPose()
    {
        // A missing head pose counts as a yaw of 0, and too few landmarks to find
        // the pupils give no scale.
        var cropped = FacePipeline.RunFrontal(
            Gray(1024, 1024), Face(), null!, FacePose.Frontal, DynafaceConstants.DefaultTiltThreshold, crop: true);
        Assert.InRange(cropped!.Value.PupillaryDistance, 258f, 262f);

        var few = FacePipeline.RunFrontal(
            Gray(64, 64), new Vec2[10], new float[3], FacePose.Frontal, DynafaceConstants.DefaultTiltThreshold, crop: false);
        Assert.Equal(0f, few!.Value.PupillaryDistance);
        Assert.True(float.IsNaN(few.Value.Pix2mm));
    }

    [Fact]
    public void ProfileCropEdges()
    {
        // No landmarks: the image is centered.
        var (centered, none) = LateralCropper.Crop(Gray(400, 600), null!);
        Assert.Equal((1024, 1024), (centered.Width, centered.Height));
        Assert.Empty(none);
        Assert.Equal(1024, LateralCropper.Crop(Gray(400, 600), Array.Empty<Vec2>()).crop.Width);

        // Landmarks spanning a whole square image: it's shrunk to leave room for the
        // padding, and padded out to the full size.
        var (_, spanning) = LateralCropper.Crop(Gray(1000, 1000), new Vec2[] { new(300, 0), new(200, 500), new(400, 999) });
        Assert.All(spanning, p => Assert.InRange(p.Y, 0f, 1023f));

        // Landmarks at the very bottom of a tall image: the crop can't go lower, but
        // keeps them in view.
        var (_, bottom) = LateralCropper.Crop(Gray(1500, 3000), new Vec2[] { new(700, 2990), new(750, 2999) });
        Assert.All(bottom, p => Assert.InRange(p.Y, 0f, 1023f));
    }

    [Fact]
    public void QuarterCropEdges()
    {
        // The centered band crop of a quarter face: without landmarks, the image centered.
        var (centered, none) = LateralCropper.Crop(Gray(400, 600), null!, center: true);
        Assert.Equal((1024, 1024), (centered.Width, centered.Height));
        Assert.Empty(none);

        // The band and its pads fill the crop whatever the frame: the same landmarks in a
        // frame and in one ten times as large crop to the same pixels, where a profile's
        // crop never scales up past filling the frame.
        var band = new Vec2[] { new(500, 200), new(900, 600), new(700, 1000) };
        var (_, crop) = LateralCropper.Crop(Gray(1400, 1400), band, center: true);
        var (_, large) = LateralCropper.Crop(Gray(14000, 14000), band, center: true);
        Assert.Equal(crop, large);
        int padTop = (int)Math.Round(1024 * DynafaceConstants.LateralPadTopRatio);
        int padBottom = (int)Math.Round(1024 * DynafaceConstants.LateralPadBottomRatio);
        Assert.InRange(crop.Min(p => p.Y) - padTop, -1f, 1f);
        Assert.InRange(crop.Max(p => p.Y) - (1023 - padBottom), -1f, 1f);
        // The landmarks' middle (as pixel centers) at the crop's middle.
        Assert.InRange((crop.Min(p => p.X) + crop.Max(p => p.X) + 1) / 2 - 512f, -1f, 1f);
        var (_, profile) = LateralCropper.Crop(Gray(14000, 14000), band);
        Assert.True(profile.Max(p => p.Y) - profile.Min(p => p.Y) < crop.Max(p => p.Y) - crop.Min(p => p.Y));

        // Mirrored landmarks in a mirrored frame crop to mirrored pixels.
        var mirrored = band.Select(p => new Vec2(1400 - 1 - p.X, p.Y)).ToArray();
        var (_, back) = LateralCropper.Crop(Gray(1400, 1400), mirrored, center: true);
        for (int i = 0; i < band.Length; i++)
        {
            Assert.InRange(back[i].X - (1023 - crop[i].X), -2f, 2f);
            Assert.Equal(crop[i].Y, back[i].Y);
        }

        // A frame narrower than the crop at the band's scale is padded with white at both
        // sides, the face between the pads.
        var narrow = new Vec2[] { new(200, 200), new(400, 600), new(300, 1000) };
        var (padded, inside) = LateralCropper.Crop(Gray(600, 1400), narrow, center: true);
        Assert.Equal(DynafaceConstants.FillColorWhite, padded.Pixels[0]);
        Assert.Equal(DynafaceConstants.FillColorWhite, padded.Pixels[1023]);
        Assert.Equal(new Rgba32(128, 128, 128, 255), padded.Pixels[512]);
        Assert.InRange((inside.Min(p => p.X) + inside.Max(p => p.X) + 1) / 2 - 512f, -1f, 1f);
    }

    [Fact]
    public void ForcingAProfile()
    {
        Assert.Equal((true, false), PoseClassifier.ForceLateral(new[] { 0f, 0f, 0f }, Array.Empty<Vec2>(), 1024));
        Assert.Equal((true, true), PoseClassifier.ForceLateral(new[] { -30f, 0f, 0f }, Face(), 1024));
        Assert.Equal((true, false), PoseClassifier.ForceLateral(new[] { 30f, 0f, 0f }, Face(), 1024));
        // Without a yaw, the nose's position decides which way the face looks.
        Assert.True(PoseClassifier.ForceLateral(new[] { float.NaN, 0f, 0f }, Face(), 1024).isLateral);
        // Detecting rather than forcing: no landmarks is no profile.
        Assert.Equal((false, false), PoseClassifier.IsLateral(new[] { 40f, 0f, 0f }, Array.Empty<Vec2>(), 1024));
    }

    [Fact]
    public void ForcingAQuarterFace()
    {
        // A quarter face whenever there are landmarks, the far side from them, and the
        // model's yaw only when they give no frame.
        Assert.Equal((false, false), PoseClassifier.ForceQuarter(new[] { 30f, 0f, 0f }, Array.Empty<Vec2>()));
        Assert.Equal((false, false), PoseClassifier.ForceQuarter(new[] { 30f, 0f, 0f }, null!));
        Assert.Equal((true, true), PoseClassifier.ForceQuarter(new[] { 30f, 0f, 0f }, new Vec2[98]));
        Assert.Equal((true, false), PoseClassifier.ForceQuarter(null!, new Vec2[98]));
        var (resolved, isLateral, _, isQuarter, _) = FacePipeline.ResolvePose(
            FacePose.Quarter, Face(), new[] { 0f, 0f, 0f }, 1024, previousPose: null);
        Assert.Equal((FacePose.Quarter, false, true), (resolved, isLateral, isQuarter));
    }

    [Fact]
    public void StraightenFillsTheCornersWithTheAverageColor()
    {
        // Half black, half white: the corners uncovered by turning it become gray.
        const int size = 40;
        var pixels = Enumerable.Range(0, size * size)
            .Select(i => i % size < size / 2 ? new Rgba32(0, 0, 0, 255) : new Rgba32(254, 254, 254, 255))
            .ToArray();
        Rgba32 average = ImageUtils.AverageColor(pixels);
        Assert.Equal(new Rgba32(127, 127, 127, 255), average);

        float tenDegrees = 10f * MathF.PI / 180f;
        Rgba32[] turned = ImageUtils.Straighten(pixels, size, size, tenDegrees);
        Assert.Equal(average, turned[0]);
        // Angles are taken modulo a half turn, toward the nearest upright.
        Assert.Equal(turned, ImageUtils.Straighten(pixels, size, size, tenDegrees - MathF.PI));
        Assert.Equal(ImageUtils.Straighten(pixels, size, size, -tenDegrees),
                     ImageUtils.Straighten(pixels, size, size, MathF.PI - tenDegrees));
        Assert.Equal(new Rgba32(0, 0, 0, 255), ImageUtils.AverageColor(Array.Empty<Rgba32>()));
    }

    [Fact]
    public void SingularTransforms()
    {
        var red = new Rgba32(255, 0, 0, 255);
        var pixels = new Rgba32[16];
        Rgba32[] warped = ImageUtils.WarpAffine(pixels, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, bilinear: true, fillColor: red);
        Assert.All(warped, p => Assert.Equal(red, p)); // nothing to sample from
        Assert.Equal((1f, 0f, 0f, 0f, 1f, 0f), ImageUtils.InvertAffine(0, 0, 0, 0, 0, 0)); // identity
    }
}

// The options that need the models, on the sample images.
[Collection("ModelTests")]
public class PipelineOptionsModelTests
{
    readonly ModelFixture _fixture;

    public PipelineOptionsModelTests(ModelFixture fixture) => _fixture = fixture;

    static FaceImage Load(string name) => ImageLoader.Load(Path.Combine(AppContext.BaseDirectory, "TestData", name));

    [SkippableFact]
    public void ForcingThePose()
    {
        _fixture.RequireModels();
        Assert.False(FacePipeline.Run(Load("img2-1024-right-lateral.jpg"), 0, false, FacePose.Frontal)!.Value.IsLateral);
        Assert.True(FacePipeline.Run(Load("img4-1024-frontal.jpg"), 0, false, FacePose.Lateral)!.Value.IsLateral);

        // A forced quarter pose takes the quarter path, even on a frontal photo: the
        // centered band crop, never flipped or straightened, the far side from the
        // landmarks (img4's nose sits a hair to the subject's right of the canthi, so the
        // far cheek is the right) and the far cheek traced from the background model.
        var quarter = FacePipeline.Run(Load("img4-1024-frontal.jpg"), 0, false, FacePose.Quarter)!.Value;
        Assert.Equal(FacePose.Quarter, quarter.Pose);
        Assert.True(quarter.IsQuarter);
        Assert.False(quarter.IsLateral || quarter.Flipped || quarter.FarLeft);
        Assert.Null(quarter.LateralAnalysis);
        Assert.Null(quarter.FaceRotationRad);
        Assert.Equal((1024, 1024), (quarter.AlignedCrop.Width, quarter.AlignedCrop.Height));
        Assert.NotNull(quarter.CheekOutlineX);
        Assert.True(quarter.CheekOutlineY.Length > 0);
        // The band crop's scale, not the frontal crop's 260 px between the pupils.
        Assert.True(quarter.PupillaryDistance > 300f);
        var (pd, pix2mm) = FaceMeasureContext.CalcPd(quarter.Wflw98, quarter.PupilDistMm);
        Assert.Equal((pd, pix2mm), (quarter.PupillaryDistance, quarter.Pix2mm));
        var ctx = FaceMeasureContext.FromResult(quarter);
        Assert.True(ctx.IsQuarter);
        Assert.Null(ctx.FrontalCheek());
        Assert.NotNull(ctx.ObliqueCheek());
        Assert.True(ctx.CheekLandmarks[Cheek.CheekLmApexR].IsFinite);
    }

    [SkippableFact]
    public void ProfileDetectionOff()
    {
        _fixture.RequireModels();
        // Safe to change here: the other tests that depend on it share this collection,
        // so they never run at the same time.
        try
        {
            DynafaceConfig.AutoLateral = false;
            Assert.False(FacePipeline.Run(Load("img2-1024-right-lateral.jpg"), 0, false)!.Value.IsLateral);
        }
        finally
        {
            DynafaceConfig.AutoLateral = true;
        }
    }

    [SkippableFact]
    public void MirroredInput()
    {
        _fixture.RequireModels();
        var photo = Load("img4-1024-frontal.jpg");
        var plain = FacePipeline.Run(photo, 0, false)!.Value;
        var mirrored = FacePipeline.Run(photo, 0, flipHorizontal: true)!.Value;
        // The same face, mirrored, is cropped to about the same pupil distance.
        Assert.InRange(mirrored.PupillaryDistance, plain.PupillaryDistance - 10, plain.PupillaryDistance + 10);
    }

    [SkippableFact]
    public void NoFace()
    {
        _fixture.RequireModels();
        var blank = new FaceImage(Enumerable.Repeat(new Rgba32(255, 255, 255, 255), 300 * 300).ToArray(), 300, 300);
        Assert.Null(FacePipeline.Run(blank, 0, false));
        Assert.False(FacePipeline.LastDetectionEyesOk);
        Assert.Equal(90, FacePipeline.SuggestRotation(blank)); // no upright face found
        Assert.Equal(0, FacePipeline.SuggestRotation(Load("img4-1024-frontal.jpg")));
    }
}
