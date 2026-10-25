namespace DynafaceTests;

// Mirrors dynaface-lib-python's tests/test_measures_dental.py. The parity cases share the
// same inner lips, pupils and expected values (PARITY_CASES there), so both ports must
// split the mouth the same way: clipped at the tilted facial midline, with dental_right
// the person's right (the image-left half).
public class MeasureDentalAreaTests
{
    const float Pix2mm = 0.25f;

    // Closed mouth: the lower inner lip (93-95) retraces the upper (89-91), so the outline
    // has no area. Every closed-mouth video frame looks like this.
    static readonly Vec2[] Closed =
    {
        new(440, 700), new(480, 710), new(510, 712), new(540, 710),
        new(580, 700), new(540, 710), new(510, 712), new(480, 710),
    };

    // Open mouth: a 120 x 30 px rectangle, x 450-570, y 690-720.
    static readonly Vec2[] Open =
    {
        new(450, 690), new(480, 690), new(530, 690), new(570, 690),
        new(570, 720), new(530, 720), new(480, 720), new(450, 720),
    };

    static System.Collections.Generic.Dictionary<string, double> Calc(Vec2[] lip, float leftPupilX, float leftPupilY)
    {
        var lm = TestHelpers.MakeLandmarks();
        for (int i = 0; i < lip.Length; i++) lm[88 + i] = lip[i];
        lm[96] = new Vec2(380, 480);
        lm[97] = new Vec2(leftPupilX, leftPupilY);
        var ctx = TestHelpers.BuildContext(landmarks: lm, pix2mm: Pix2mm);
        return new MeasureDentalArea().Calc(ctx, render: true);
    }

    [Theory]
    // case, left pupil, expected area, right (image-left), left (image-right), diff, ratio
    [InlineData("closed", 640, 480, 0.0, 0.0, 0.0, 0.0, 1.0)]
    [InlineData("open", 650, 480, 225.0, 121.290894, 103.709106, 17.581787, 0.855044)]
    [InlineData("tilted", 640, 506, 225.0, 71.412964, 153.587036, 82.174072, 0.464967)]
    public void SharedParityCases(string name, float leftPupilX, float leftPupilY,
        double area, double right, double left, double diff, double ratio)
    {
        var lip = name == "closed" ? Closed : Open;
        var r = Calc(lip, leftPupilX, leftPupilY);
        Assert.Equal(area, r["dental_area"], 3);
        Assert.Equal(right, r["dental_right"], 3);
        Assert.Equal(left, r["dental_left"], 3);
        Assert.Equal(diff, r["dental_diff"], 3);
        Assert.Equal(ratio, r["dental_ratio"], 3);
    }

    [Fact]
    public void ClosedMouthMeasuresExactlyZero()
    {
        var r = Calc(Closed, 640, 480);
        Assert.Equal(0.0, r["dental_area"]);
        Assert.Equal(1.0, r["dental_ratio"]);
    }
}
