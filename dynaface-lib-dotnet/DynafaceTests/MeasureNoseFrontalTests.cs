using System;
using System.Linq;

namespace DynafaceTests;

// Mirrors dynaface-lib-python's tests/test_measures_nose.py: the same lower nose outline
// (WFLW 55-59) with alar edges 100 px apart and mid-nostril points 50 px apart.
public class MeasureNoseFrontalTests
{
    const float Pix2mm = 0.25f;

    static Vec2[] Nose(float deg = 0f)
    {
        var lm = TestHelpers.MakeLandmarks();
        lm[55] = new Vec2(460, 700);
        lm[56] = new Vec2(485, 706);
        lm[57] = new Vec2(510, 710);
        lm[58] = new Vec2(535, 706);
        lm[59] = new Vec2(560, 700);
        float rad = deg * MathF.PI / 180f;
        float c = MathF.Cos(rad), s = MathF.Sin(rad);
        foreach (int i in new[] { 55, 56, 57, 58, 59 })
        {
            float x = lm[i].X - 512f, y = lm[i].Y - 512f;
            lm[i] = new Vec2(512f + x * c - y * s, 512f + x * s + y * c);
        }
        return lm;
    }

    [Theory]
    [InlineData(false, 0f)]
    [InlineData(true, 0f)]
    [InlineData(false, 5f)]
    [InlineData(false, -8f)] // head roll doesn't change the widths
    public void AlarAndNostrilWidths(bool render, float deg)
    {
        var ctx = TestHelpers.BuildContext(landmarks: Nose(deg), pix2mm: Pix2mm);
        var r = new MeasureNoseFrontal().Calc(ctx, render);
        Assert.Equal(new[] { "nose.alar", "nose.nostril" }, r.Keys.OrderBy(k => k));
        Assert.Equal(100 * Pix2mm, r["nose.alar"], 3);
        Assert.Equal(50 * Pix2mm, r["nose.nostril"], 3);
    }

    [Fact]
    public void OldColumnsAreRetired()
    {
        var names = new MeasureNoseFrontal().Items.Select(i => i.Name).ToArray();
        Assert.Equal(new[] { "nose.alar", "nose.nostril" }, names);
    }
}
