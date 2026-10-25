using System;

namespace DynafaceTests;

// Undefined geometry must come back as NaN, never as a plausible-looking number: lateral
// angles and tip projection when a landmark is missing or a side has zero length, and the
// mm scale when the pupils coincide. Mirrors dynaface-lib-python's
// tests/test_degenerate_geometry.py (same profile, same expected values).
public class DegenerateGeometryTests
{
    // A missing lateral landmark: (-1, -1) with X shifted into image space (shift 100).
    static readonly Vec2 Missing = new(99, -1);

    // Glabella, nasion, nasal tip, subnasal, mentolabial, pogonion. NFA at the nasion is
    // 135°, NLA at the subnasal point acos(-2/sqrt(5)) = 153.435°, and AT = NT = 50.
    static Vec2[] Profile() => new Vec2[]
    {
        new(300, 300), new(300, 350), new(350, 400), new(320, 460), new(310, 540), new(320, 600),
    };

    static readonly double Nla = Math.Acos(-2 / Math.Sqrt(5)) * 180 / Math.PI;

    static System.Collections.Generic.Dictionary<string, double> Lateral(Vec2[] points, bool render = false)
    {
        var photo = new FaceImage(new Rgba32[1024 * 1024], 1024, 1024);
        var ctx = new FaceMeasureContext(photo, TestHelpers.MakeLandmarks(), 0.24f,
            isLateral: true, lateralLandmarks: points);
        return new MeasureLateral().Calc(ctx, render);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteProfile(bool render)
    {
        var r = Lateral(Profile(), render);
        Assert.Equal(135.0, r["nfa"], 3);
        Assert.Equal(Nla, r["nla"], 3);
        Assert.Equal(50.0, r["at"], 3);
        Assert.Equal(50.0, r["nt"], 3);
        Assert.Equal(1.0, r["at_nt"], 3);
    }

    [Fact]
    public void MissingGlabellaOnlyAffectsNfa()
    {
        var p = Profile();
        p[0] = Missing;
        var r = Lateral(p, render: true);
        Assert.True(double.IsNaN(r["nfa"]));
        Assert.Equal(Nla, r["nla"], 3);
        Assert.Equal(1.0, r["at_nt"], 3);
    }

    [Fact]
    public void MissingNasalTipMakesEverythingUsingItNaN()
    {
        var p = Profile();
        p[2] = Missing;
        var r = Lateral(p, render: true);
        foreach (var key in new[] { "nfa", "nla", "at", "nt", "at_nt" })
            Assert.True(double.IsNaN(r[key]), key);
    }

    [Fact]
    public void ZeroLengthSideIsNaN()
    {
        var p = Profile();
        p[0] = p[1]; // glabella on the nasion
        Assert.True(double.IsNaN(Lateral(p)["nfa"]));
    }

    [Fact]
    public void TwoMissingPointsAreNaNNotZero()
    {
        var p = Profile();
        p[0] = Missing;
        p[1] = Missing;
        var r = Lateral(p);
        Assert.True(double.IsNaN(r["nfa"]));
        Assert.True(double.IsNaN(r["at_nt"]));
    }

    static Vec2[] CoincidentPupils()
    {
        var lm = TestHelpers.MakeLandmarks();
        lm[96] = new Vec2(500, 480);
        lm[97] = new Vec2(500, 480);
        return lm;
    }

    [Fact]
    public void PositionReportsNaNPx2mm()
    {
        var ctx = TestHelpers.BuildContext(landmarks: CoincidentPupils());
        var r = new MeasurePosition().Calc(ctx, render: false);
        Assert.Equal(0.0, r["pd"]);
        Assert.True(double.IsNaN(r["px2mm"]));
    }

    [Fact]
    public void ContextWithoutDistinctPupilsHasNaNScale()
    {
        var photo = new FaceImage(new Rgba32[64 * 64], 64, 64);
        Assert.True(float.IsNaN(new FaceMeasureContext(photo, CoincidentPupils()).Pix2mm));
    }

    [Fact]
    public void UncroppedPipelineGivesNaNScale()
    {
        var photo = new FaceImage(new Rgba32[1024 * 1024], 1024, 1024);
        var result = FacePipeline.RunFrontal(photo, CoincidentPupils(), new float[6], FacePose.Frontal,
            DynafaceConstants.DefaultTiltThreshold, crop: false);
        Assert.True(float.IsNaN(result.Value.Pix2mm));
    }
}
