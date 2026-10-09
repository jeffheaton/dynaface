using System;

namespace DynafaceTests;

// brow.d must cancel head tilt the way dynaface-lib's AnalyzeBrows does: both brow
// lines run at the tilt angle and are compared at the same x. The expected values
// match AnalyzeBrows.calc run on the same landmarks (see dynaface-lib-python's
// tests/test_measures_brows.py, which pins the Python side of these cases).
public class MeasureBrowsTests
{
    const float Pix2mm = 0.25f;

    // Symmetric face (pupils at the StyleGAN crop targets, brows 80px above them)
    // rotated by deg about the crop center. raiseLeftPx raises the image-right brow
    // (44) in the face's own frame before rotating.
    static Vec2[] RotatedFace(float deg, float raiseLeftPx = 0f)
    {
        var lm = TestHelpers.MakeLandmarks();
        lm[96] = new Vec2(380, 480);
        lm[97] = new Vec2(640, 480);
        lm[35] = new Vec2(380, 400);
        lm[36] = new Vec2(410, 398);
        lm[44] = new Vec2(640, 400 - raiseLeftPx);

        float rad = deg * MathF.PI / 180f;
        float c = MathF.Cos(rad), s = MathF.Sin(rad);
        foreach (int i in new[] { 35, 36, 44, 96, 97 })
        {
            float x = lm[i].X - 512f, y = lm[i].Y - 512f;
            lm[i] = new Vec2(512f + x * c - y * s, 512f + x * s + y * c);
        }
        return lm;
    }

    // Integer landmarks on a tilt of slope 13/260 = 0.05 (~2.86 degrees), so Python's
    // int truncation of the edge points doesn't change its result.
    static Vec2[] SlopedFace(float leftBrowY)
    {
        var lm = TestHelpers.MakeLandmarks();
        lm[96] = new Vec2(380, 480);
        lm[97] = new Vec2(640, 493);
        lm[35] = new Vec2(380, 400);
        lm[36] = new Vec2(410, 398);
        lm[44] = new Vec2(640, leftBrowY);
        return lm;
    }

    static double BrowD(Vec2[] landmarks, bool render = false)
    {
        var ctx = TestHelpers.BuildContext(landmarks: landmarks, pix2mm: Pix2mm);
        return new MeasureBrows().Calc(ctx, render)["brow.d"];
    }

    [Theory]
    [InlineData(2f)]
    [InlineData(3.5f)]
    [InlineData(5f)]
    [InlineData(-4f)]
    [InlineData(0f)]
    public void TiltedSymmetricFace_HasNoBrowDifference(float deg)
    {
        Assert.Equal(0.0, BrowD(RotatedFace(deg)), 3);
        Assert.Equal(0.0, BrowD(RotatedFace(deg), render: true), 3);
    }

    [Fact]
    public void TiltedSymmetricFace_IntegerLandmarks_MatchesPython()
    {
        // 44 sits on the tilted line through 35: 400 + 0.05 * 260 = 413.
        Assert.Equal(0.0, BrowD(SlopedFace(413)), 3);
    }

    [Fact]
    public void TiltedAsymmetricFace_MatchesPython()
    {
        // 44 is 12px above the tilted line through 35; AnalyzeBrows gives 12 * 0.25 = 3.0.
        Assert.Equal(3.0, BrowD(SlopedFace(401)), 3);
    }

    [Theory]
    [InlineData(3.5f)]
    [InlineData(-4f)]
    public void RotatedAsymmetricFace_ReportsVerticalGapAtSameX(float deg)
    {
        // A 12px raise in the face frame is a 12/cos(tilt) px vertical gap between the
        // tilted lines (Python reports the same, modulo its <=1px int truncation).
        double expected = 12.0 / Math.Cos(deg * Math.PI / 180.0) * Pix2mm;
        Assert.Equal(expected, BrowD(RotatedFace(deg, raiseLeftPx: 12f)), 3);
    }

    [Fact]
    public void LevelFace_StillReportsBrowDifference()
    {
        var lm = RotatedFace(0f, raiseLeftPx: 12f);
        Assert.Equal(3.0, BrowD(lm), 3);
    }
}
