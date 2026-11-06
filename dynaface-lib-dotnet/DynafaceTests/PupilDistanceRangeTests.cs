namespace DynafaceTests;

// A pupillary distance outside 40–85 mm is a mistake, such as centimeters for
// millimeters, so it is refused where a scale is measured from it. Mirrors
// dynaface-lib's TestPupilDistanceRange in tests/test_scale.py.
public class PupilDistanceRangeTests
{
    // Landmarks with pupils 230 pixels apart.
    static Vec2[] Face()
    {
        var landmarks = TestHelpers.MakeLandmarks();
        landmarks[DynafaceConstants.LmRightPupil] = new Vec2(530, 320);
        landmarks[DynafaceConstants.LmLeftPupil] = new Vec2(760, 330);
        return landmarks;
    }

    static FaceImage Gray() => new(new Rgba32[1024 * 1024], 1024, 1024);

    [Theory]
    [InlineData(40f)]
    [InlineData(63f)]
    [InlineData(85f)]
    [InlineData(58.5f)]
    public void Accepts(float pd) => Assert.Equal(pd, DynafaceConfig.CheckPupilDistMm(pd));

    [Theory]
    [InlineData(39.9f)]
    [InlineData(85.1f)]
    [InlineData(6.3f)]
    [InlineData(630f)]
    [InlineData(0f)]
    [InlineData(-63f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void Refuses(float pd)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DynafaceConfig.CheckPupilDistMm(pd));
        Assert.Throws<ArgumentOutOfRangeException>(() => FaceMeasureContext.CalcPd(Face(), pd));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FaceMeasureContext(Gray(), Face(), pupilDistMm: pd));

        float before = DynafaceConfig.PupilDistMm;
        Assert.Throws<ArgumentOutOfRangeException>(() => DynafaceConfig.PupilDistMm = pd);
        Assert.Equal(before, DynafaceConfig.PupilDistMm);
    }

    // A state saved before 2.1 could hold any distance. It restores with the scale it
    // was saved with, rather than failing.
    [Fact]
    public void AStateWithADistanceOutsideKeepsItsScale()
    {
        var landmarks = Face();
        var (pupils, _) = FaceMeasureContext.CalcPd(landmarks, 63f);
        var state = new FaceAnalysisState
        {
            PostCropImage = Gray(), Landmarks = landmarks, PupillaryDistance = pupils, Pix2mm = 30f / pupils,
        };
        Assert.Equal(state.Pix2mm, FaceAnalysisPersistence.Restore(state).Pix2mm);
    }

    // The product of the saved pair has float error; the context gets 63 back.
    [Fact]
    public void TheRestoredDistanceIsExact()
    {
        var landmarks = Face();
        var (pupils, pix2mm) = FaceMeasureContext.CalcPd(landmarks, 63f);
        var state = new FaceAnalysisState
        {
            PostCropImage = Gray(), Landmarks = landmarks, PupillaryDistance = pupils, Pix2mm = pix2mm,
        };
        Assert.Equal(63f, FaceAnalysisPersistence.Restore(state).PupilDistMm);
    }
}
