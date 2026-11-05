namespace DynafaceTests;

// StyleGanCropper.CropLandmarks: Crop's landmarks without building the image, for a
// consumer that keeps a frame's uncropped landmarks and needs the cropped ones only
// for measures (a video whose crops are rebuilt just for display). It has to agree
// with Crop to the pixel, or landmarks drawn on a rebuilt crop would miss the face.
// .NET-only: dynaface-lib's crop_stylegan always builds the image.
public class CropLandmarksTests
{
    static FaceImage Gray(int width, int height) =>
        new(Enumerable.Repeat(new Rgba32(128, 128, 128, 255), width * height).ToArray(), width, height);

    // 98 landmarks scattered around the given pupils, in top-left pixels. Their
    // fractional parts make the crop's truncation matter.
    static Vec2[] Scattered(Vec2 rightPupil, Vec2 leftPupil)
    {
        var landmarks = new Vec2[98];
        for (int i = 0; i < landmarks.Length; i++)
            landmarks[i] = new Vec2(
                rightPupil.X - 80f + (i * 37) % 380 + 0.37f,
                rightPupil.Y - 120f + (i * 53) % 400 + 0.71f);
        landmarks[DynafaceConstants.LmRightPupil] = rightPupil;
        landmarks[DynafaceConstants.LmLeftPupil] = leftPupil;
        return landmarks;
    }

    // A face near the middle of the image, pupils 220 pixels apart, tilted by the
    // given angle.
    static Vec2[] Face(int width, int height, float tiltDegrees = 0f)
    {
        float rad = tiltDegrees * MathF.PI / 180f;
        var right = new Vec2(width / 2f - 110.4f, height / 2f - 40.6f);
        var left = new Vec2(right.X + 220f * MathF.Cos(rad), right.Y + 220f * MathF.Sin(rad));
        return Scattered(right, left);
    }

    [Theory]
    [InlineData(1024, 1024, 0f, -1f, 0f)]  // straight on
    [InlineData(1024, 1024, 12f, -1f, 0f)] // turned head: the distance is corrected first
    [InlineData(1280, 720, 0f, -1f, 0f)]   // a landscape video frame
    [InlineData(721, 1279, 3f, -1f, 0f)]   // an odd-sized portrait frame
    [InlineData(1024, 1024, 0f, 2f, 8f)]   // tilted, and straightened
    [InlineData(1281, 719, 9f, 2f, -6f)]   // all of it, turned about an odd center
    public void MatchesCrop(int width, int height, float yaw, float tiltThreshold, float tiltDegrees)
    {
        Vec2[] landmarks = Face(width, height, tiltDegrees);
        var cropped = StyleGanCropper.Crop(Gray(width, height), landmarks, yaw, tiltThreshold);
        Assert.Equal(cropped.landmarks,
            StyleGanCropper.CropLandmarks(width, height, landmarks, yaw, tiltThreshold));
    }

    [Fact]
    public void MatchesCropWithGivenPupils()
    {
        Vec2[] landmarks = Face(1024, 1024);
        var left = new Vec2(655.2f, 507.9f);
        var right = new Vec2(352.6f, 498.1f);
        var cropped = StyleGanCropper.Crop(Gray(1024, 1024), landmarks, 7f, 1f,
            pupilLeftOverride: left, pupilRightOverride: right);
        Assert.Equal(cropped.landmarks, StyleGanCropper.CropLandmarks(1024, 1024, landmarks, 7f, 1f,
            pupilLeftOverride: left, pupilRightOverride: right));
    }

    [Fact]
    public void RealignsAnUncroppedRun()
    {
        // A frame analyzed with crop: false keeps the frame's own landmarks; cropping
        // them afterward gives exactly what a cropped run would have.
        FaceImage frame = Gray(1280, 720);
        Vec2[] landmarks = Face(frame.Width, frame.Height);
        float[] headPose = { 8f, 0f, 0f, 0f, 0f, 0f };
        var uncropped = FacePipeline.RunFrontal(
            frame, landmarks, headPose, FacePose.Frontal, DynafaceConstants.DefaultTiltThreshold, crop: false)!.Value;
        var cropped = FacePipeline.RunFrontal(
            frame, landmarks, headPose, FacePose.Frontal, DynafaceConstants.DefaultTiltThreshold, crop: true)!.Value;

        Vec2[] realigned = StyleGanCropper.CropLandmarks(
            frame.Width, frame.Height, uncropped.Wflw98, uncropped.HeadPose[0]);
        Assert.Equal(cropped.Wflw98, realigned);

        // But the uncropped run's Pix2mm doesn't carry over: it was measured on the
        // frame's own pupils, which the crop moves closer together or farther apart.
        float realignedPd = Vec2.Distance(
            realigned[DynafaceConstants.LmRightPupil], realigned[DynafaceConstants.LmLeftPupil]);
        Assert.Equal(cropped.PupillaryDistance, realignedPd);
        Assert.NotEqual(uncropped.PupillaryDistance, realignedPd);
    }

    [Fact]
    public void LeavesTheLandmarksAlone()
    {
        Vec2[] landmarks = Face(1024, 1024, tiltDegrees: 8f);
        Vec2[] before = (Vec2[])landmarks.Clone();
        StyleGanCropper.CropLandmarks(1024, 1024, landmarks, 12f, tiltThreshold: 2f);
        Assert.Equal(before, landmarks);
    }

    [Fact]
    public void PupilsInOnePlace() =>
        // No scale to crop by, the same as Crop.
        Assert.Throws<InvalidOperationException>(() =>
            StyleGanCropper.CropLandmarks(1024, 1024, Scattered(new(500, 500), new(500, 500)), 0f));

    [Fact]
    public void NoImageSize()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => StyleGanCropper.CropLandmarks(0, 1024, Face(1024, 1024), 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => StyleGanCropper.CropLandmarks(1024, -1, Face(1024, 1024), 0f));
    }
}
