using System.IO;
using static DynafaceTests.CheekFixtures;

namespace DynafaceTests;

// The synthetic-head renders as photos for the quarter path, as dynaface-lib-python's
// tests/test_pose_quarter.py uses them: each render's raw landmarks and head pose in
// tests_data/cheek_landmarks.json stand in for the models' detection, and the
// foreground mask is every pixel differing from the background color
// (tests_data/cheek/cheek_renders.json), in place of the background model.
internal static class QuarterFixtures
{
    public const int Standard = 1024;

    // The landmark each WFLW landmark becomes when the image is mirrored: the outline
    // reverses, each side's brow, eye, nose base and lip points swap with the other
    // side's, and the midline points stay (test_pose_quarter.wflw_mirror_index()).
    public static readonly int[] Mirror = MirrorIndex();

    static int[] MirrorIndex()
    {
        var index = Enumerable.Range(0, 98).ToArray();
        void Swap(int a, int b) => (index[a], index[b]) = (index[b], index[a]);
        for (int i = 0; i < 16; i++) Swap(i, 32 - i);
        for (int k = 0; k < 5; k++) Swap(33 + k, 46 - k);
        for (int k = 0; k < 4; k++) Swap(38 + k, 50 - k);
        Swap(55, 59);
        Swap(56, 58);
        for (int k = 0; k < 5; k++) Swap(60 + k, 72 - k);
        for (int k = 0; k < 3; k++) Swap(65 + k, 75 - k);
        for (int k = 0; k < 3; k++) Swap(76 + k, 82 - k);
        for (int k = 0; k < 2; k++) Swap(83 + k, 87 - k);
        for (int k = 0; k < 2; k++) Swap(88 + k, 92 - k);
        Swap(93, 95);
        Swap(96, 97);
        return index;
    }

    public static Rgba32 Background
    {
        get
        {
            var bg = Renders.GetProperty("background");
            return new Rgba32(bg[0].GetByte(), bg[1].GetByte(), bg[2].GetByte(), 255);
        }
    }

    public static string FileOf(string name) => Entry(name).GetProperty("file").GetString()!;

    public static FaceImage Image(string name) => ImageLoader.Load(DataPath(FileOf(name).Split('/')));

    // What the models found in the image: its raw landmarks, shifted by (dx, dy).
    public static Vec2[] Detection(string name, int dx = 0, int dy = 0) =>
        RawLandmarks(name).Select(p => new Vec2((float)p.X + dx, (float)p.Y + dy)).ToArray();

    public static float[] HeadPose(string name)
    {
        var raw = Entry(name).GetProperty("raw");
        return new[] { raw.GetProperty("yaw").GetSingle(), raw.GetProperty("pitch").GetSingle(),
                       raw.GetProperty("roll").GetSingle(), 0f, 0f, 0f };
    }

    public static int WidthOf(string name) => Entry(name).GetProperty("size")[0].GetInt32();

    // The render in the middle of a wider, background-colored frame, like a video frame,
    // with its detection shifted to match (test_pose_quarter.wide_frame()).
    public static (FaceImage Frame, Vec2[] Landmarks, float[] Pose) WideFrame(
        string name, int width = 1600, int height = 1024)
    {
        FaceImage image = Image(name);
        int dx = (width - image.Width) / 2, dy = (height - image.Height) / 2;
        var pixels = Enumerable.Repeat(Background, width * height).ToArray();
        for (int y = 0; y < image.Height; y++)
            Array.Copy(image.Pixels, (image.Height - 1 - y) * image.Width,
                pixels, (height - 1 - (dy + y)) * width + dx, image.Width);
        return (new FaceImage(pixels, width, height), Detection(name, dx, dy), HeadPose(name));
    }

    // The frame and its detection mirrored left to right: the photo of the other turn,
    // with the landmarks renumbered for the swapped sides and the yaw and roll negated.
    public static (FaceImage Frame, Vec2[] Landmarks, float[] Pose) Mirrored(
        (FaceImage Frame, Vec2[] Landmarks, float[] Pose) found)
    {
        var (frame, landmarks, pose) = found;
        var flipped = new Vec2[98];
        for (int i = 0; i < 98; i++)
            flipped[Mirror[i]] = new Vec2(frame.Width - 1 - landmarks[i].X, landmarks[i].Y);
        var negated = (float[])pose.Clone();
        negated[0] = -negated[0];
        negated[2] = -negated[2];
        return (frame.FlipHorizontal(), flipped, negated);
    }

    // What SagittalProfile.ForegroundMask would give for a render: every pixel that
    // differs from the background color is foreground, with a full alpha.
    public static (bool[,] Foreground, byte[,] Alpha) BackgroundMask(FaceImage image)
    {
        Rgba32 bg = Background;
        var fg = new bool[image.Height, image.Width];
        var alpha = new byte[image.Height, image.Width];
        for (int y = 0; y < image.Height; y++)
        {
            int row = (image.Height - 1 - y) * image.Width;
            for (int x = 0; x < image.Width; x++)
            {
                Rgba32 p = image.Pixels[row + x];
                bool on = p.R != bg.R || p.G != bg.G || p.B != bg.B;
                fg[y, x] = on;
                alpha[y, x] = on ? (byte)255 : (byte)0;
            }
        }
        return (fg, alpha);
    }

    // The quarter path on a frame and its detection, with the background-color mask in
    // place of the background model: what FacePipeline.Run gives a forced quarter face.
    public static FacePipelineResult Quarter((FaceImage Frame, Vec2[] Landmarks, float[] Pose) found)
    {
        var (frame, landmarks, pose) = found;
        var (isQuarter, farLeft) = PoseClassifier.ForceQuarter(pose, landmarks);
        Assert.True(isQuarter);
        return FacePipeline.RunQuarter(frame, landmarks, pose, farLeft, FacePose.Quarter, BackgroundMask);
    }

    public static FacePipelineResult Frontal((FaceImage Frame, Vec2[] Landmarks, float[] Pose) found) =>
        FacePipeline.RunFrontal(found.Frame, found.Landmarks, found.Pose, FacePose.Frontal,
            DynafaceConstants.DefaultTiltThreshold, crop: true)!.Value;

    // The cheek landmarks' rows that exist are exactly these, the rest NaN.
    public static void AssertRows(Vec2d[] points, params int[] finite)
    {
        Assert.Equal(10, points.Length);
        var present = Enumerable.Range(0, 10).Where(i => points[i].IsFinite).ToArray();
        Assert.Equal(finite, present);
        Assert.All(Enumerable.Range(0, 10).Except(finite),
            i => Assert.True(double.IsNaN(points[i].X) && double.IsNaN(points[i].Y), $"row {i}"));
    }

    // Cheek.LocateAll on a context's own landmarks and view, which its CheekLandmarks
    // must equal exactly.
    public static Vec2d[] Located(FaceMeasureContext ctx) =>
        Cheek.LocateAll(ctx.Landmarks, ctx.Pose, ctx.FarLeft, ctx.CheekOutlineX, ctx.CheekOutlineY, ctx.CheekOutlineSrc);

    public static int Count(byte[] src, byte source) => src.Count(s => s == source);
}

// The quarter path without the models: the centered band crop, the scale, the far
// cheek's trace from the mask and its retrace, the context built from the result, the
// pose caption and the mask the trace reads. Mirrors dynaface-lib-python's
// tests/test_pose_quarter.py (TestQuarterDetection's crop and trace, TestRetrace).
public class QuarterPathTests
{
    const int Standard = QuarterFixtures.Standard;

    static readonly Lazy<FacePipelineResult> Yaw45 = new(() => QuarterFixtures.Quarter(QuarterFixtures.WideFrame("yaw45")));

    [Fact]
    public void The45DegreeRenderIsCroppedAndTraced()
    {
        var found = QuarterFixtures.WideFrame("yaw45");
        var r = Yaw45.Value;
        Assert.True(r.IsQuarter);
        Assert.Equal(FacePose.Quarter, r.Pose);
        // The nose points to the image's right, so the far cheek is the subject's left:
        // the photo was taken from the subject's right.
        Assert.True(r.FarLeft);
        Assert.False(r.Flipped);
        Assert.False(r.IsLateral);
        Assert.Null(r.LateralAnalysis);
        Assert.Null(r.FaceRotationRad);
        Assert.Equal((Standard, Standard), (r.AlignedCrop.Width, r.AlignedCrop.Height));

        // The centered band crop: every landmark inside, the chin at least 10% above the
        // bottom, the landmarks' middle (as pixel centers) within two pixels of the
        // crop's middle, and the band scaled to fill the crop with its pads whatever the
        // frame: here the 800 px render's 1024-tall frame is scaled up.
        var lm = r.Wflw98;
        float minX = lm.Min(p => p.X), maxX = lm.Max(p => p.X);
        float minY = lm.Min(p => p.Y), maxY = lm.Max(p => p.Y);
        Assert.True(minX >= 0 && maxX < Standard && minY >= 0 && maxY < Standard);
        Assert.True(lm[Cheek.LmChin].Y <= 0.9f * Standard);
        Assert.InRange((minX + maxX + 1) / 2 - Standard / 2f, -2f, 2f);
        int padTop = (int)Math.Round(Standard * DynafaceConstants.LateralPadTopRatio);
        int padBottom = (int)Math.Round(Standard * DynafaceConstants.LateralPadBottomRatio);
        Assert.InRange(minY - padTop, -2f, 2f);
        Assert.InRange(maxY - (Standard - 1 - padBottom), -2f, 2f);
        float rawBand = found.Landmarks.Max(p => p.Y) - found.Landmarks.Min(p => p.Y);
        Assert.True((maxY - minY) / rawBand > 1.5f);
        // Cropped landmarks are whole pixels, as in dynaface-lib.
        Assert.All(lm, p => Assert.True(p.X == MathF.Floor(p.X) && p.Y == MathF.Floor(p.Y)));

        // The scale: the pupil scale measured on the cropped landmarks, as for a frontal
        // face, so a context measures it again the same way.
        var (pd, pix2mm) = FaceMeasureContext.CalcPd(lm, r.PupilDistMm);
        Assert.Equal((pd, pix2mm), (r.PupillaryDistance, r.Pix2mm));
        Assert.Equal(DynafaceConfig.PupilDistMm, r.PupilDistMm);

        // The far cheek's outline: one row from the far outer eye corner's row (or the far
        // contour's top) to the chin row, every x inside the crop, nearly every row from
        // the mask on a plain render.
        int top = (int)Math.Min(lm[72].Y, lm[32].Y);
        Assert.Equal(Enumerable.Range(top, (int)lm[Cheek.LmChin].Y - top + 1), r.CheekOutlineY);
        Assert.Equal(r.CheekOutlineY.Length, r.CheekOutlineX.Length);
        Assert.Equal(r.CheekOutlineY.Length, r.CheekOutlineSrc.Length);
        Assert.All(r.CheekOutlineX, x => Assert.InRange(x, 0f, Standard - 1f));
        Assert.True(QuarterFixtures.Count(r.CheekOutlineSrc, CheekOutline.OutlineSrcMask) > 0.9 * r.CheekOutlineY.Length);

        // The outline is the trace of the crop's own mask on the cropped landmarks.
        var (fg, alpha) = QuarterFixtures.BackgroundMask(r.AlignedCrop);
        var expected = CheekOutline.TraceFarCheek(fg, alpha, lm, true, gray: CheekOutline.GrayImage(r.AlignedCrop));
        Assert.Equal(expected.X, r.CheekOutlineX);
        Assert.Equal(expected.Y, r.CheekOutlineY);
        Assert.Equal(expected.Src, r.CheekOutlineSrc);
    }

    [Fact]
    public void TheContextFromTheResultFindsTheFarCheek()
    {
        var r = Yaw45.Value;
        var ctx = FaceMeasureContext.FromResult(r);
        Assert.True(ctx.IsQuarter);
        Assert.False(ctx.IsLateral);
        Assert.True(ctx.FarLeft);
        Assert.Same(r.CheekOutlineX, ctx.CheekOutlineX);
        Assert.Equal(r.Pix2mm, ctx.Pix2mm);
        Assert.Equal(r.PupilDistMm, ctx.PupilDistMm);
        // It draws on a copy unless told otherwise.
        Assert.NotSame(r.AlignedCrop.Pixels, ctx.Pixels);
        Assert.Same(r.AlignedCrop.Pixels, FaceMeasureContext.FromResult(r, copyImage: false).Pixels);

        // The cheek landmarks: the far cheek's observed points, nothing frontal.
        QuarterFixtures.AssertRows(ctx.CheekLandmarks, Cheek.CheekLmApexL, Cheek.CheekLmSubmalarL);
        Assert.Equal(QuarterFixtures.Located(ctx), ctx.CheekLandmarks);
        var observed = ctx.ObliqueCheek();
        Assert.NotNull(observed);
        Assert.True(observed!.FarLeft);
        Assert.Equal(1.0, observed.MaskFraction);
        Assert.Equal(observed.Apex, ctx.CheekLandmarks[Cheek.CheekLmApexL]);
        Assert.Null(ctx.FrontalCheek());

        // A frontal result gives a frontal context, with no quarter view.
        var frontal = FaceMeasureContext.FromResult(QuarterFixtures.Frontal(QuarterFixtures.WideFrame("yaw0")));
        Assert.False(frontal.IsQuarter || frontal.FarLeft);
        Assert.Null(frontal.CheekOutlineX);
        QuarterFixtures.AssertRows(frontal.CheekLandmarks, 0, 1, 2, 3, 4, 5);
    }

    [Fact]
    public void TheMirroredTurnGetsTheMirroredCrop()
    {
        var plus = Yaw45.Value;
        var minus = QuarterFixtures.Quarter(QuarterFixtures.Mirrored(QuarterFixtures.WideFrame("yaw45")));
        Assert.True(minus.IsQuarter);
        Assert.False(minus.FarLeft);
        Assert.False(minus.Flipped);
        for (int i = 0; i < 98; i++)
        {
            Vec2 p = plus.Wflw98[i], m = minus.Wflw98[QuarterFixtures.Mirror[i]];
            Assert.InRange(m.X - (Standard - 1 - p.X), -2f, 2f);
            Assert.InRange(m.Y - p.Y, -2f, 2f);
        }
        // So is the outline: the same rows, the edge mirrored, a few rows at the chin
        // allowed to change their source.
        Assert.Equal(plus.CheekOutlineY, minus.CheekOutlineY);
        int sameSource = plus.CheekOutlineSrc.Zip(minus.CheekOutlineSrc).Count(t => t.First == t.Second);
        Assert.True(sameSource >= 0.98 * plus.CheekOutlineSrc.Length);
        int both = 0;
        for (int i = 0; i < plus.CheekOutlineY.Length; i++)
        {
            if (plus.CheekOutlineSrc[i] != CheekOutline.OutlineSrcMask || minus.CheekOutlineSrc[i] != CheekOutline.OutlineSrcMask)
                continue;
            both++;
            Assert.InRange(minus.CheekOutlineX[i] - (Standard - 1 - plus.CheekOutlineX[i]), -2f, 2f);
        }
        Assert.True(both > 0);

        var plusCtx = FaceMeasureContext.FromResult(plus);
        var minusCtx = FaceMeasureContext.FromResult(minus);
        QuarterFixtures.AssertRows(minusCtx.CheekLandmarks, Cheek.CheekLmApexR, Cheek.CheekLmSubmalarR);
        Assert.Equal(plusCtx.ObliqueCheek()!.ApexH, minusCtx.ObliqueCheek()!.ApexH, 5e-4);
        Assert.Equal(plusCtx.ObliqueCheek()!.SubmalarDepth, minusCtx.ObliqueCheek()!.SubmalarDepth, 1e-4);

        // And the crops themselves are mirror images, to the rounding of the scaled
        // frame's width.
        var mirroredMinus = minus.AlignedCrop.FlipHorizontal();
        double total = 0;
        for (int i = 0; i < plus.AlignedCrop.Pixels.Length; i++)
        {
            Rgba32 a = plus.AlignedCrop.Pixels[i], b = mirroredMinus.Pixels[i];
            total += Math.Abs(a.R - b.R) + Math.Abs(a.G - b.G) + Math.Abs(a.B - b.B);
        }
        Assert.True(total / (3.0 * plus.AlignedCrop.Pixels.Length) < 1.0);
    }

    [Fact]
    public void APortraitFrameCropsAlike()
    {
        // The band crop scales the landmarks' band to fill the crop whatever the frame,
        // so one face reads alike from a phone frame and a wide one.
        var wide = Yaw45.Value;
        var portrait = QuarterFixtures.Quarter(QuarterFixtures.WideFrame("yaw45", width: 800, height: 1600));
        Assert.True(portrait.FarLeft);
        Assert.Equal((Standard, Standard), (portrait.AlignedCrop.Width, portrait.AlignedCrop.Height));
        for (int i = 0; i < 98; i++)
        {
            Assert.InRange(portrait.Wflw98[i].X - wide.Wflw98[i].X, -2f, 2f);
            Assert.InRange(portrait.Wflw98[i].Y - wide.Wflw98[i].Y, -2f, 2f);
        }
        Assert.InRange(portrait.PupillaryDistance - wide.PupillaryDistance, -2f, 2f);
        Assert.Equal(wide.CheekOutlineY, portrait.CheekOutlineY);
        var a = FaceMeasureContext.FromResult(wide).ObliqueCheek()!;
        var b = FaceMeasureContext.FromResult(portrait).ObliqueCheek()!;
        Assert.InRange(b.Scale - a.Scale, -2.0, 2.0);
        Assert.Equal(a.ApexH, b.ApexH, 1e-3);
    }

    [Fact]
    public void TheOtherTurnsRender()
    {
        // The -45 degree render, with its own landmarks, is the mirrored view.
        var r = QuarterFixtures.Quarter(QuarterFixtures.WideFrame("yaw-45"));
        Assert.True(r.IsQuarter);
        Assert.False(r.FarLeft);
        Assert.False(r.Flipped);
        float minX = r.Wflw98.Min(p => p.X), maxX = r.Wflw98.Max(p => p.X);
        Assert.InRange((minX + maxX + 1) / 2 - Standard / 2f, -2f, 2f);
        Assert.True(r.Wflw98[Cheek.LmChin].Y <= 0.9f * Standard);
        var ctx = FaceMeasureContext.FromResult(r);
        QuarterFixtures.AssertRows(ctx.CheekLandmarks, Cheek.CheekLmApexR, Cheek.CheekLmSubmalarR);
        Assert.Equal(QuarterFixtures.Located(ctx), ctx.CheekLandmarks);
        Assert.False(ctx.ObliqueCheek()!.FarLeft);
    }

    [Fact]
    public void TheCropMatchesDynafaceLib()
    {
        // The forced quarter sketch's raw landmarks (the app's 512 px three-quarter
        // sketch) cropped by the band crop give dynaface-lib's pipeline landmarks to the
        // pixel, and its far side: the crop's landmark math needs only the frame's size.
        const string name = "quarter_sketch_forced";
        var pipeline = Entry(name).GetProperty("pipeline");
        Assert.Equal("quarter", pipeline.GetProperty("pose").GetString());
        int size = QuarterFixtures.WidthOf(name);
        var blank = new FaceImage(new Rgba32[size * size], size, size);
        var raw = QuarterFixtures.Detection(name);
        var (_, cropped) = LateralCropper.Crop(blank, raw, center: true);
        var expected = PipelineLandmarks(name);
        for (int i = 0; i < 98; i++)
            Assert.Equal((expected[i].X, expected[i].Y), ((double)cropped[i].X, (double)cropped[i].Y));
        var (isQuarter, farLeft) = PoseClassifier.ForceQuarter(QuarterFixtures.HeadPose(name), raw);
        Assert.True(isQuarter);
        Assert.Equal(pipeline.GetProperty("far_left").GetBoolean(), farLeft);
        var (pd, pix2mm) = FaceMeasureContext.CalcPd(cropped, 63f);
        Assert.Equal(pipeline.GetProperty("pupillary_distance").GetDouble(), pd, 0.01);
        Assert.Equal(pipeline.GetProperty("pix2mm").GetDouble(), pix2mm, 1e-5);
    }

    [Fact]
    public void RetracingFollowsTheLandmarksWithoutAModelPass()
    {
        var r = Yaw45.Value;
        // On its own landmarks, the same outline.
        var same = r.RetraceFarCheek(r.Wflw98);
        Assert.NotNull(same);
        Assert.Equal(r.CheekOutlineX, same!.Value.X);
        Assert.Equal(r.CheekOutlineY, same.Value.Y);
        Assert.Equal(r.CheekOutlineSrc, same.Value.Src);

        // Three rows lower, the band follows by the 3 rows, and a row both traces read
        // from the mask has the same edge: the mask is the frame's.
        var lower = r.Wflw98.Select(p => new Vec2(p.X, p.Y + 3)).ToArray();
        var moved = r.RetraceFarCheek(lower)!.Value;
        Assert.Equal(r.CheekOutlineY.Select(y => y + 3), moved.Y);
        var before = ByRow(r.CheekOutlineY);
        for (int i = 0; i < moved.Y.Length; i++)
        {
            if (!before.TryGetValue(moved.Y[i], out int j)) continue;
            if (moved.Src[i] == CheekOutline.OutlineSrcMask && r.CheekOutlineSrc[j] == CheekOutline.OutlineSrcMask)
                Assert.Equal(r.CheekOutlineX[j], moved.X[i]);
        }
        // The result itself is left as it was.
        Assert.Equal(same.Value.Y, r.CheekOutlineY);

        // A frontal face, or a quarter result without its mask, has nothing to trace from.
        Assert.Null(QuarterFixtures.Frontal(QuarterFixtures.WideFrame("yaw0")).RetraceFarCheek(lower));
        var handMade = r;
        handMade.FarCheekMask = null;
        Assert.Null(handMade.RetraceFarCheek(lower));
    }

    [Theory]
    [InlineData(FacePose.Frontal, false, false, "Frontal")]
    [InlineData(FacePose.Quarter, true, false, "Quarter (left cheek)")]
    [InlineData(FacePose.Quarter, false, false, "Quarter (right cheek)")]
    [InlineData(FacePose.Lateral, false, false, "Lateral (left)")]
    [InlineData(FacePose.Lateral, false, true, "Lateral (right)")]
    public void TheCaptionNamesTheView(FacePose pose, bool farLeft, bool flipped, string caption)
    {
        // draw_static()'s captions, a quarter face's naming the cheek measured.
        static FaceImage Blank() => new(new Rgba32[400 * 300], 400, 300);
        var landmarks = TestHelpers.MakeLandmarks();
        var ctx = new FaceMeasureContext(Blank(), landmarks, isLateral: pose == FacePose.Lateral,
            pose: pose, flipped: flipped);
        ctx.SetQuarterView(farLeft, null!, null!, null!);
        ctx.DrawStatic();

        var expected = new FaceMeasureContext(Blank(), landmarks);
        var size = FaceRenderer.GetTextSize(caption, FaceRenderer.TEXT_SIZE_MEASURE);
        expected.DrawImageText(new Vec2(10, 300 - 20 - size.Y), caption, new Rgba32(255, 255, 255, 255));
        Assert.Equal(expected.Pixels, ctx.Pixels);
    }

    // The mask the trace reads is lateral.py's exactly: the alpha truncated to bytes,
    // above MASK_ALPHA (32) foreground, closed with OpenCV's 10x10 kernel, whose anchor
    // at (5, 5) shifts the closed mask one pixel right and down. The expected pixels were
    // read from dynaface-lib's _close_foreground() (cv2.morphologyEx) on the same masks.
    // A 320 px image takes the 320 px saliency map as it is.
    static (bool[,] Foreground, byte[,] Alpha) Mask(Action<float[]> paint)
    {
        var mask = new float[320 * 320];
        paint(mask);
        var inference = new ModelOutputTests.FakeInference { Mask = mask };
        var photo = new FaceImage(Enumerable.Repeat(new Rgba32(128, 128, 128, 255), 320 * 320).ToArray(), 320, 320);
        var result = SagittalProfile.ForegroundMask(inference, photo);
        // The profile's silhouette is the same mask, inverted.
        byte[] silhouette = SagittalProfile.ProcessImage(inference, photo);
        for (int y = 0; y < 320; y++)
            for (int x = 0; x < 320; x++)
                Assert.Equal(result.Foreground[y, x], silhouette[y * 320 + x] == 0);
        return result;
    }

    static float Alpha(double value) => (float)((value + 0.5) / 255.0);

    [Fact]
    public void TheMaskThresholdIsLateralPys()
    {
        // lateral.MASK_ALPHA, which cheek_outline.MASK_ALPHA names again for the tracer.
        string source = File.ReadAllText(DataPath("python", "lateral.py"));
        var match = System.Text.RegularExpressions.Regex.Match(source, @"^MASK_ALPHA: int = (\d+)$",
            System.Text.RegularExpressions.RegexOptions.Multiline);
        Assert.True(match.Success);
        Assert.Equal(SagittalProfile.MaskAlpha, int.Parse(match.Groups[1].Value));
        Assert.Equal(CheekOutline.MaskAlpha, SagittalProfile.MaskAlpha);
    }

    static IEnumerable<(int Y, int X)> On(bool[,] fg) =>
        from y in Enumerable.Range(0, fg.GetLength(0))
        from x in Enumerable.Range(0, fg.GetLength(1))
        where fg[y, x]
        select (y, x);

    [Fact]
    public void TheMaskThresholdsTheTruncatedAlpha()
    {
        // An alpha of 32.5 is the byte 32: background, as in dynaface-lib, where the
        // saliency compared before truncating would have been foreground.
        var (fg, alpha) = Mask(m => { for (int i = 0; i < m.Length; i++) if (i % 320 >= 20) m[i] = Alpha(32); });
        Assert.Equal(32, alpha[5, 100]);
        Assert.Equal(0, alpha[5, 5]);
        Assert.Empty(On(fg));

        // 33 is foreground; the half plane's edge moves from column 20 to 21.
        (fg, alpha) = Mask(m => { for (int i = 0; i < m.Length; i++) if (i % 320 >= 20) m[i] = Alpha(33); });
        Assert.Equal(33, alpha[5, 100]);
        for (int y = 0; y < 320; y++)
            for (int x = 0; x < 320; x++)
                Assert.Equal(x >= 21, fg[y, x]);
    }

    [Fact]
    public void TheMaskIsClosedAsOpenCVClosesIt()
    {
        // One pixel: one pixel, one right and one down.
        var (fg, _) = Mask(m => m[20 * 320 + 20] = Alpha(200));
        Assert.Equal(new[] { (21, 21) }, On(fg));

        // Two pixels 9 or 10 apart: joined, the joint shifted the same way.
        (fg, _) = Mask(m => { m[20 * 320 + 10] = Alpha(200); m[20 * 320 + 19] = Alpha(200); });
        Assert.Equal(Enumerable.Range(11, 10).Select(x => (21, x)), On(fg));
        (fg, _) = Mask(m => { m[20 * 320 + 10] = Alpha(200); m[20 * 320 + 20] = Alpha(200); });
        Assert.Equal(Enumerable.Range(11, 11).Select(x => (21, x)), On(fg));

        // A gap 9 columns wide is filled; one 10 wide stays, shifted a column right.
        static void Gap(float[] m, int from, int width)
        {
            for (int i = 0; i < m.Length; i++) m[i] = Alpha(200);
            for (int y = 0; y < 320; y++)
                for (int x = from; x < from + width; x++) m[y * 320 + x] = 0f;
        }
        (fg, _) = Mask(m => Gap(m, 15, 9));
        Assert.All(Enumerable.Range(0, 320), x => Assert.True(fg[100, x]));
        (fg, _) = Mask(m => Gap(m, 15, 10));
        Assert.Equal(Enumerable.Range(16, 10), Enumerable.Range(0, 320).Where(x => !fg[100, x]));
    }
}

// The pose decision on the renders' recorded detections, without the models: quarter
// detection under DynafaceConfig.AutoQuarter, the fall-through from the profile test,
// the video hold and the forced poses. Mirrors TestQuarterDetection in dynaface-lib's
// tests/test_pose_quarter.py. In the model collection, since it switches the global
// detection settings, which the model tests read.
[Collection("ModelTests")]
public class QuarterPoseTests
{
    static (FacePose Pose, bool IsLateral, bool FacingLeft, bool IsQuarter, bool FarLeft) Resolve(
        string name, FacePose force = FacePose.Detect, FacePose? previous = null, Vec2[]? landmarks = null) =>
        FacePipeline.ResolvePose(force, landmarks ?? QuarterFixtures.Detection(name), QuarterFixtures.HeadPose(name),
            QuarterFixtures.WidthOf(name), previous);

    static void WithDetection(bool autoQuarter, bool autoLateral, Action test)
    {
        try
        {
            DynafaceConfig.AutoQuarter = autoQuarter;
            DynafaceConfig.AutoLateral = autoLateral;
            test();
        }
        finally
        {
            DynafaceConfig.AutoQuarter = false;
            DynafaceConfig.AutoLateral = true;
        }
    }

    [Fact]
    public void DetectionNeedsBothSwitches()
    {
        // Shipped off: a 45 degree turn is a frontal face, as before.
        Assert.False(DynafaceConfig.AutoQuarter);
        Assert.True(DynafaceConfig.AutoLateral);
        Assert.Equal((FacePose.Frontal, false, false, false, false), Resolve("yaw45"));
        Assert.Equal((false, false), PoseClassifier.IsQuarter(QuarterFixtures.HeadPose("yaw45"), QuarterFixtures.Detection("yaw45")));
        // AutoLateral off turns all detection off, quarter detection included.
        WithDetection(true, false, () => Assert.Equal(FacePose.Frontal, Resolve("yaw45").Pose));
        // Both on: detected, each turn with its own far side.
        WithDetection(true, true, () =>
        {
            Assert.Equal((FacePose.Quarter, false, false, true, true), Resolve("yaw45"));
            Assert.Equal((FacePose.Quarter, false, false, true, false), Resolve("yaw-45"));
            // A level render and a frontal photo are frontal either way, and the profiles
            // (their noses past the far contour, turned past the window) stay profiles.
            Assert.Equal(FacePose.Frontal, Resolve("yaw0").Pose);
            Assert.Equal(FacePose.Frontal, Resolve("img4").Pose);
            Assert.Equal(FacePose.Frontal, Resolve("img1").Pose);
            Assert.Equal((FacePose.Lateral, true, false, false, false), Resolve("img2"));
            Assert.Equal((FacePose.Lateral, true, true, false, false), Resolve("img3"));
            // No landmarks: no quarter face.
            Assert.Equal((false, false), PoseClassifier.IsQuarter(new[] { 30f, 0f, 0f }, Array.Empty<Vec2>()));
        });
    }

    [Fact]
    public void ATurnTheProfileTestDeclinesIsAQuarterFace()
    {
        // The 45 degree render with the far contour's top (32) moved past the nose tip:
        // the margin rule sends it to the profile test, which declines it (the cheek's
        // contour, 26, is still outside the nose), so it's a quarter face (its cheek.nose
        // flags it), never a frontal one.
        var landmarks = QuarterFixtures.Detection("yaw45");
        float width = landmarks[32].X - landmarks[0].X;
        landmarks[32] = new Vec2(landmarks[54].X - 0.1f * width, landmarks[32].Y);
        var pose = QuarterFixtures.HeadPose("yaw45");
        WithDetection(true, true, () =>
        {
            Assert.Equal((false, false), PoseClassifier.IsQuarter(pose, landmarks));
            Assert.False(PoseClassifier.IsLateral(pose, landmarks, 800).isLateral);
            Assert.Equal((FacePose.Quarter, false, false, true, true), Resolve("yaw45", landmarks: landmarks));

            // With the cheek's contour past the nose too, the profile test takes it.
            landmarks[26] = new Vec2(landmarks[54].X - 0.1f * width, landmarks[26].Y);
            Assert.Equal(FacePose.Lateral, Resolve("yaw45", landmarks: landmarks).Pose);
        });
    }

    [Fact]
    public void ThePreviousPoseHoldsAVideoInTheView()
    {
        // The 35 degree render with its nose tip moved toward the middle, so the landmark
        // yaw reads about 33.5 degrees: below the window, inside it widened by the hold.
        var landmarks = QuarterFixtures.Detection("yaw35");
        var frame = Cheek.PupilFrame(landmarks)!;
        for (int step = 1; Cheek.LandmarkYaw(landmarks, frame) > 33.5 && step < 100; step++)
            landmarks[54] = new Vec2(landmarks[54].X - 0.25f, landmarks[54].Y);
        double yaw = Cheek.LandmarkYaw(landmarks, frame);
        Assert.InRange(yaw, 32.5, 34.5);
        WithDetection(true, true, () =>
        {
            Assert.Equal(FacePose.Frontal, Resolve("yaw35", landmarks: landmarks).Pose);
            Assert.Equal(FacePose.Frontal, Resolve("yaw35", previous: FacePose.Frontal, landmarks: landmarks).Pose);
            Assert.Equal(FacePose.Frontal, Resolve("yaw35", previous: FacePose.Lateral, landmarks: landmarks).Pose);
            Assert.Equal((FacePose.Quarter, false, false, true, true),
                Resolve("yaw35", previous: FacePose.Quarter, landmarks: landmarks));
        });
    }

    [Fact]
    public void TheForcedPoses()
    {
        WithDetection(true, true, () =>
        {
            // A forced quarter pose on a frontal photo takes the quarter path, the far side
            // from the landmarks: img4's nose sits a hair to the subject's right of the
            // canthi, so the far cheek is the right.
            Assert.Equal((FacePose.Quarter, false, false, true, false), Resolve("img4", FacePose.Quarter));
            // A forced frontal pose never takes the quarter path, and a forced profile is one.
            Assert.Equal((FacePose.Frontal, false, false, false, false), Resolve("yaw45", FacePose.Frontal));
            Assert.Equal((FacePose.Lateral, true, false, false, false), Resolve("yaw45", FacePose.Lateral));
        });
        // Forcing needs no detection switch.
        Assert.Equal((FacePose.Quarter, false, false, true, true), Resolve("yaw45", FacePose.Quarter));
        Assert.Equal((false, false), PoseClassifier.ForceQuarter(new[] { 30f, 0f, 0f }, Array.Empty<Vec2>()));
        // Without a frame the far side falls back to the model's yaw.
        var noFrame = new Vec2[98];
        Assert.Equal((true, true), PoseClassifier.ForceQuarter(new[] { 30f, 0f, 0f }, noFrame));
        Assert.Equal((true, false), PoseClassifier.ForceQuarter(new[] { -30f, 0f, 0f }, noFrame));
    }
}

// The quarter path through the real models, on the 45 degree renders, the frontal photo
// and the profiles. Model-gated, like FacialPipelineIntegrationTests. Mirrors
// dynaface-lib-python's tests/test_cheek_photos.py for the pipeline's part.
//
// What dynaface-lib gave on CPU for cheek/yaw45.png forced to the quarter view, with
// these .NET numbers beside it (2026-11-07): Python 709 outline rows (204 to 912), 652
// from the mask, 46 from the contour and 11 from the intensity edge, the apex the prior
// at h 0.7935, H (cheek.scale) 665.4 px, pd 251.05 px; .NET 712 rows (202 to 913), 642,
// 52 and 18, apex h 0.7924, H 666.9 px, pd 249.02 px. The two ports' SPIGA landmarks
// differ (the raw yaw reads 28.28 and 28.31, the cropped landmarks a mean 2.0 px apart;
// dynaface-lib also rounds its raw landmarks to whole pixels), which accounts for the
// difference: on dynaface-lib's own raw landmarks (tests_data/cheek_landmarks.json) the
// .NET quarter path crops all 98 to the same pixels, traces the same 709 rows with 651
// from the mask (the two ports' image resizers differ by up to 2 levels) and reads the
// apex at h 0.7935.
[Collection("ModelTests")]
public class QuarterPipelineTests
{
    readonly ModelFixture _fixture;

    public QuarterPipelineTests(ModelFixture fixture) => _fixture = fixture;

    const int Standard = QuarterFixtures.Standard;

    static FaceImage Load(string name) => ImageLoader.Load(DataPath(name.Split('/')));

    static FacePipelineResult Run(string name, FacePose pose = FacePose.Detect, bool autoQuarter = false,
        bool crop = true, FacePose? previous = null)
    {
        try
        {
            DynafaceConfig.AutoQuarter = autoQuarter;
            var result = FacePipeline.Run(Load(name), 0, false, pose, crop: crop, previousPose: previous);
            Assert.NotNull(result);
            return result.Value;
        }
        finally
        {
            DynafaceConfig.AutoQuarter = false;
        }
    }

    // A quarter face of the pipeline: the band crop, the far side, the trace, the
    // cheek landmarks.
    static void AssertQuarterFace(FacePipelineResult r, bool farLeft)
    {
        Assert.True(r.IsQuarter);
        Assert.Equal(FacePose.Quarter, r.Pose);
        Assert.Equal(farLeft, r.FarLeft);
        Assert.False(r.IsLateral || r.Flipped);
        Assert.Null(r.LateralAnalysis);
        Assert.Null(r.FaceRotationRad);
        Assert.Equal((Standard, Standard), (r.AlignedCrop.Width, r.AlignedCrop.Height));

        // The band-fit crop: every landmark inside, the band filling the crop with its
        // pads, the landmarks centered.
        var lm = r.Wflw98;
        float minX = lm.Min(p => p.X), maxX = lm.Max(p => p.X);
        float minY = lm.Min(p => p.Y), maxY = lm.Max(p => p.Y);
        Assert.True(minX >= 0 && maxX < Standard && minY >= 0 && maxY < Standard);
        Assert.InRange((minX + maxX + 1) / 2 - Standard / 2f, -2f, 2f);
        int padTop = (int)Math.Round(Standard * DynafaceConstants.LateralPadTopRatio);
        int padBottom = (int)Math.Round(Standard * DynafaceConstants.LateralPadBottomRatio);
        Assert.InRange(minY - padTop, -2f, 2f);
        Assert.InRange(maxY - (Standard - 1 - padBottom), -2f, 2f);
        var (pd, pix2mm) = FaceMeasureContext.CalcPd(lm, r.PupilDistMm);
        Assert.Equal((pd, pix2mm), (r.PupillaryDistance, r.Pix2mm));

        // The outline: the band's rows, most from the mask.
        int top = (int)Math.Min(lm[farLeft ? 72 : 60].Y, lm[farLeft ? 32 : 0].Y);
        Assert.Equal(Enumerable.Range(top, (int)lm[Cheek.LmChin].Y - top + 1), r.CheekOutlineY);
        Assert.Equal(r.CheekOutlineY.Length, r.CheekOutlineX.Length);
        Assert.Equal(r.CheekOutlineY.Length, r.CheekOutlineSrc.Length);
        Assert.True(QuarterFixtures.Count(r.CheekOutlineSrc, CheekOutline.OutlineSrcMask) > 0.85 * r.CheekOutlineY.Length);

        // The far cheek's observed points: the apex and the submalar point, finite.
        var ctx = FaceMeasureContext.FromResult(r);
        int apex = farLeft ? Cheek.CheekLmApexL : Cheek.CheekLmApexR;
        int submalar = farLeft ? Cheek.CheekLmSubmalarL : Cheek.CheekLmSubmalarR;
        QuarterFixtures.AssertRows(ctx.CheekLandmarks, apex, submalar);
        var observed = ctx.ObliqueCheek()!;
        Assert.Equal(farLeft, observed.FarLeft);
        Assert.InRange(observed.ApexH, CheekOutline.ApexHWindow.Low, CheekOutline.ApexHWindow.High);
        Assert.InRange(observed.Scale, 600.0, 720.0);
    }

    [SkippableFact]
    public void A45DegreeRenderForcedToTheQuarterView()
    {
        _fixture.RequireModels();
        var r = Run("cheek/yaw45.png", FacePose.Quarter);
        AssertQuarterFace(r, farLeft: true);
        // Close to dynaface-lib's numbers (see the class comment).
        Assert.InRange(r.CheekOutlineY.Length, 699, 719);
        Assert.InRange(QuarterFixtures.Count(r.CheekOutlineSrc, CheekOutline.OutlineSrcMask), 620, 680);
        Assert.Equal(0.7935, FaceMeasureContext.FromResult(r).ObliqueCheek()!.ApexH, 0.005);
        Assert.Equal(1.0, FaceMeasureContext.FromResult(r).Analyze(render: false)!["cheek.far"]);
    }

    [SkippableFact]
    public void A45DegreeRenderIsDetectedWithAutoQuarterOn()
    {
        _fixture.RequireModels();
        var forced = Run("cheek/yaw45.png", FacePose.Quarter);
        var detected = Run("cheek/yaw45.png", autoQuarter: true);
        AssertQuarterFace(detected, farLeft: true);
        Assert.Equal(forced.Wflw98, detected.Wflw98);
        Assert.Equal(forced.CheekOutlineX, detected.CheekOutlineX);
        Assert.Equal(forced.CheekOutlineSrc, detected.CheekOutlineSrc);
        // Held by a previous quarter frame, it's a quarter face too.
        Assert.True(Run("cheek/yaw45.png", autoQuarter: true, previous: FacePose.Quarter).IsQuarter);
    }

    [SkippableFact]
    public void TheMirroredTurnsFarCheekIsTheRight()
    {
        _fixture.RequireModels();
        AssertQuarterFace(Run("cheek/yaw-45.png", FacePose.Quarter), farLeft: false);
        AssertQuarterFace(Run("cheek/yaw-45.png", autoQuarter: true), farLeft: false);
    }

    [SkippableFact]
    public void AFrontalPhotoStaysFrontal()
    {
        _fixture.RequireModels();
        var r = Run("img4-1024-frontal.jpg", autoQuarter: true);
        Assert.Equal(FacePose.Frontal, r.Pose);
        Assert.False(r.IsQuarter || r.IsLateral || r.FarLeft);
        Assert.Null(r.CheekOutlineX);
        Assert.Null(r.RetraceFarCheek(r.Wflw98));
        Assert.InRange(r.PupillaryDistance, 258f, 262f);   // the StyleGAN crop
        QuarterFixtures.AssertRows(FaceMeasureContext.FromResult(r).CheekLandmarks, 0, 1, 2, 3, 4, 5);
    }

    [SkippableFact]
    public void ProfilesStayLateral()
    {
        _fixture.RequireModels();
        foreach (var (name, flipped) in new[] { ("img2-1024-right-lateral.jpg", true), ("img3-1024-left-lateral.jpg", false) })
        {
            var r = Run(name, autoQuarter: true);
            Assert.True(r.IsLateral, name);
            Assert.Equal(FacePose.Lateral, r.Pose);
            Assert.Equal(flipped, r.Flipped);
            Assert.False(r.IsQuarter || r.FarLeft);
            Assert.Null(r.CheekOutlineX);
            Assert.NotNull(r.LateralAnalysis);
        }
    }

    [SkippableFact]
    public void AutoQuarterOffGivesTheOldBehavior()
    {
        _fixture.RequireModels();
        // A 45 degree turn is a frontal face with the frontal crop, as before 2.1.
        var r = Run("cheek/yaw45.png");
        Assert.Equal(FacePose.Frontal, r.Pose);
        Assert.False(r.IsQuarter || r.IsLateral);
        Assert.Null(r.CheekOutlineX);
        Assert.Null(r.CheekOutlineY);
        Assert.Null(r.CheekOutlineSrc);
        // The frontal crop's yaw correction: the pupils less than 260 px apart.
        Assert.True(r.PupillaryDistance < 255f);
        // AutoLateral off turns quarter detection off as well.
        try
        {
            DynafaceConfig.AutoLateral = false;
            Assert.Equal(FacePose.Frontal, Run("cheek/yaw45.png", autoQuarter: true).Pose);
        }
        finally
        {
            DynafaceConfig.AutoLateral = true;
        }
    }

    [SkippableFact]
    public void AQuarterFaceIsAlwaysCropped()
    {
        _fixture.RequireModels();
        // crop: false does nothing to a quarter face, forced or detected, as for a profile.
        var cropped = Run("cheek/yaw45.png", FacePose.Quarter);
        foreach (var r in new[] { Run("cheek/yaw45.png", FacePose.Quarter, crop: false),
                                  Run("cheek/yaw45.png", autoQuarter: true, crop: false) })
        {
            Assert.True(r.IsQuarter);
            Assert.Equal((Standard, Standard), (r.AlignedCrop.Width, r.AlignedCrop.Height));
            Assert.Equal(cropped.Wflw98, r.Wflw98);
            Assert.Equal(cropped.CheekOutlineX, r.CheekOutlineX);
        }
    }

    [SkippableFact]
    public void OnDynafaceLibsLandmarksTheQuarterPathMatchesIt()
    {
        _fixture.RequireModels();
        // dynaface-lib's raw landmarks for the render (tests_data/cheek_landmarks.json)
        // through the .NET crop, background model and trace: the same 709 rows as
        // dynaface-lib's 204 to 912, about as many from the mask (652 there), and the apex
        // where dynaface-lib reads it (see the class comment).
        var found = (QuarterFixtures.Image("yaw45"), QuarterFixtures.Detection("yaw45"), QuarterFixtures.HeadPose("yaw45"));
        var r = FacePipeline.RunQuarter(found.Item1, found.Item2, found.Item3, true, FacePose.Quarter,
            image => SagittalProfile.ForegroundMask(_fixture.Inference, image));
        Assert.Equal(Enumerable.Range(204, 709), r.CheekOutlineY);
        Assert.InRange(QuarterFixtures.Count(r.CheekOutlineSrc, CheekOutline.OutlineSrcMask), 647, 657);
        var observed = FaceMeasureContext.FromResult(r).ObliqueCheek()!;
        Assert.Equal(0.79351, observed.ApexH, 1e-3);
        Assert.Equal(665.44, observed.Scale, 0.01);
    }

    [SkippableFact]
    public void ARestoredQuarterFaceMeasuresAsTheOriginal()
    {
        _fixture.RequireModels();
        var r = Run("cheek/yaw45.png", FacePose.Quarter);
        var state = FaceAnalysisPersistence.Capture(r);
        Assert.Equal(FacePose.Quarter, state.Pose);
        Assert.True(state.FarLeft);
        Assert.Equal(r.CheekOutlineX, state.CheekOutlineX);

        var original = FaceMeasureContext.FromResult(r);
        var restored = FaceAnalysisPersistence.Restore(state);
        Assert.True(restored.IsQuarter);
        Assert.True(restored.FarLeft);
        Assert.Equal(original.CheekLandmarks, restored.CheekLandmarks);
        var expected = original.Analyze(render: false)!;
        var actual = restored.Analyze(render: false)!;
        Assert.Equal(expected.Keys, actual.Keys);
        foreach (var kv in expected)
            Assert.True(kv.Value.Equals(actual[kv.Key]), $"{kv.Key}: {kv.Value} != {actual[kv.Key]}");
    }
}
