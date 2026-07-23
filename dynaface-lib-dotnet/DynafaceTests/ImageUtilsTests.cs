namespace DynafaceTests;

// Ports the safe_clip cases from dynaface-lib's test_util.py/test_image.py, plus
// ScaleCropPoints/RotateCropPoints and new C#-only WarpAffine coverage (Python just
// calls cv2.warpAffine directly and trusts OpenCV; this hand-rolled port needs its
// own correctness net).
public class ImageUtilsTests
{
    static Rgba32[] SolidImage(int w, int h, Rgba32 color)
    {
        var px = new Rgba32[w * h];
        for (int i = 0; i < px.Length; i++) px[i] = color;
        return px;
    }

    [Fact]
    public void SafeClip_NoClipNeeded_ReturnsRegionUnchanged()
    {
        var black = new Rgba32(0, 0, 0, 255);
        var src = SolidImage(10, 10, black);
        var (clipped, offsetX, offsetY) = ImageUtils.SafeClip(src, 10, 10, 2, 2, 5, 5, new Rgba32(255, 255, 255, 255));

        Assert.Equal(25, clipped.Length);
        Assert.Equal(0, offsetX);
        Assert.Equal(0, offsetY);
        Assert.All(clipped, p => Assert.Equal(black, p));
    }

    [Fact]
    public void SafeClip_PartiallyOutOfBounds_FillsExposedCorner()
    {
        var black = new Rgba32(0, 0, 0, 255);
        var white = new Rgba32(255, 255, 255, 255);
        var src = SolidImage(10, 10, black);
        var (clipped, offsetX, offsetY) = ImageUtils.SafeClip(src, 10, 10, -2, -2, 5, 5, white);

        Assert.Equal(2, offsetX);
        Assert.Equal(2, offsetY);
        Assert.Equal(white, clipped[0 * 5 + 0]);   // top-left corner: background
        Assert.Equal(black, clipped[4 * 5 + 4]);   // bottom-right corner: original image
    }

    [Fact]
    public void SafeClip_FullyOutsideSource_AllBackground()
    {
        var white = new Rgba32(255, 255, 255, 255);
        var black = new Rgba32(0, 0, 0, 255);
        var src = SolidImage(500, 500, white);
        var (clipped, offsetX, offsetY) = ImageUtils.SafeClip(src, 500, 500, 600, 600, 100, 100, black);

        Assert.Equal(100 * 100, clipped.Length);
        Assert.Equal(0, offsetX);
        Assert.Equal(0, offsetY);
        Assert.All(clipped, p => Assert.Equal(black, p));
    }

    [Fact]
    public void SafeClip_NegativeOrigin_ReportsClampOffset()
    {
        var src = SolidImage(1000, 1000, default);
        var (clipped, offsetX, offsetY) = ImageUtils.SafeClip(src, 1000, 1000, -100, -100, 1024, 1024, new Rgba32(255, 0, 0, 255));

        Assert.Equal(1024 * 1024, clipped.Length);
        Assert.Equal(100, offsetX);
        Assert.Equal(100, offsetY);
    }

    [Fact]
    public void ScaleCropPoints_UnitScale_SubtractsCropOrigin()
    {
        var pts = new[] { new Vec2(10, 10), new Vec2(20, 20) };
        var result = ImageUtils.ScaleCropPoints(pts, 10, 10, 1.0f);
        Assert.Equal(new Vec2(0, 0), result[0]);
        Assert.Equal(new Vec2(10, 10), result[1]);
    }

    [Fact]
    public void ScaleCropPoints_DoubleScale_ScalesThenSubtracts()
    {
        var pts = new[] { new Vec2(10, 10), new Vec2(20, 20) };
        var result = ImageUtils.ScaleCropPoints(pts, 10, 10, 2.0f);
        Assert.Equal(new Vec2(10, 10), result[0]);
        Assert.Equal(new Vec2(30, 30), result[1]);
    }

    [Fact]
    public void RotateCropPoints_Rotate90_MatchesExpectedDirection()
    {
        var pts = new[] { new Vec2(0, 0), new Vec2(10, 0) };
        var rotated = ImageUtils.RotateCropPoints(pts, new Vec2(0, 0), 90f);
        Assert.Equal(0, rotated[0].X, 3);
        Assert.Equal(0, rotated[0].Y, 3);
        Assert.Equal(0, rotated[1].X, 3);
        Assert.Equal(-10, rotated[1].Y, 3);
    }

    [Fact]
    public void WarpAffine_Identity_ReproducesSource()
    {
        var src = new Rgba32[4 * 4];
        for (int i = 0; i < src.Length; i++) src[i] = new Rgba32((byte)(i * 10), 0, 0, 255);

        var dst = ImageUtils.WarpAffine(src, 4, 4, 1, 0, 0, 0, 1, 0, 4, 4, bilinear: false, fillColor: default);

        Assert.Equal(src, dst);
    }

    [Fact]
    public void WarpAffine_Translation_ShiftsPixelsAndFillsExposedArea()
    {
        // 4x4 source, distinct per-pixel value; forward transform shifts +1 in x and y,
        // so dst(dx,dy) should sample src(dx-1, dy-1).
        var src = new Rgba32[4 * 4];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                src[y * 4 + x] = new Rgba32((byte)(y * 4 + x), 0, 0, 255);

        var fill = new Rgba32(9, 9, 9, 9);
        var dst = ImageUtils.WarpAffine(src, 4, 4, 1, 0, 1, 0, 1, 1, 4, 4, bilinear: false, fillColor: fill);

        Assert.Equal(fill, dst[0 * 4 + 0]);          // top-left row/col exposed by the shift
        Assert.Equal(src[0 * 4 + 0], dst[1 * 4 + 1]); // dst(1,1) <- src(0,0)
        Assert.Equal(src[2 * 4 + 2], dst[3 * 4 + 3]); // dst(3,3) <- src(2,2)
    }

    [Fact]
    public void WarpAffine_OutOfBoundsBilinearTap_UsesFillColor()
    {
        var src = new Rgba32[2 * 2];
        for (int i = 0; i < src.Length; i++) src[i] = new Rgba32(200, 200, 200, 255);
        var fill = new Rgba32(0, 0, 0, 255);

        // Identity, but request a 4x4 destination from a 2x2 source: the far corner
        // has no source data at all and must come back as pure fill.
        var dst = ImageUtils.WarpAffine(src, 2, 2, 1, 0, 0, 0, 1, 0, 4, 4, bilinear: true, fillColor: fill);

        Assert.Equal(fill, dst[3 * 4 + 3]);
    }
}
