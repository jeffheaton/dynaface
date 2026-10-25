namespace DynafaceTests;

// Ports the geometry-helper cases from dynaface-lib's test_util.py that have a clean
// 1:1 correspondence in the C# port (SymmetryRatio, MeasurePolygon's shoelace area,
// SplitPolygonByX, ClipPolygonToSide, and CalcBisect against calc_bisect's exact end
// points). ctx.LineToEdge is deliberately NOT tested against Python's
// util.line_to_edge expectations — it's a directional ray-cast, a different
// algorithm in kind from Python's undirected slope-vs-4-edges test (see
// MeasureBrows.cs's header comment) — so it gets its own, self-consistent tests
// instead of a mechanical port of test_line_to_edge.
public class GeometryTests
{
    [Fact]
    public void SymmetryRatio_EqualValues_IsOne()
    {
        Assert.Equal(1.0f, FaceMeasureContext.SymmetryRatio(0, 0));
        Assert.Equal(1.0f, FaceMeasureContext.SymmetryRatio(5, 5));
    }

    [Fact]
    public void SymmetryRatio_UnequalValues_IsMinOverMax()
    {
        Assert.Equal(0.5f, FaceMeasureContext.SymmetryRatio(2, 4));
        Assert.Equal(0.2f, FaceMeasureContext.SymmetryRatio(10, 2), 3);
    }

    [Fact]
    public void MeasurePolygon_Square_ReturnsExpectedArea()
    {
        var ctx = TestHelpers.BuildContext(pix2mm: 1.0f);
        var square = new[]
        {
            new Vec2(10, 10), new Vec2(40, 10), new Vec2(40, 40), new Vec2(10, 40),
        };
        float area = ctx.MeasurePolygon(square, render: false);
        Assert.Equal(900f, area, 1);
    }

    [Fact]
    public void MeasurePolygon_Triangle_MatchesShoelaceFormula()
    {
        var ctx = TestHelpers.BuildContext(pix2mm: 1.0f);
        var triangle = new[] { new Vec2(0, 0), new Vec2(4, 0), new Vec2(0, 3) };
        float area = ctx.MeasurePolygon(triangle, render: false);
        Assert.Equal(6f, area, 3);
    }

    [Fact]
    public void SplitPolygonByX_Square_SplitsIntoTwoHalves()
    {
        var square = new[]
        {
            new Vec2(0, 0), new Vec2(10, 0), new Vec2(10, 10), new Vec2(0, 10),
        };
        var (left, right) = FaceMeasureContext.SplitPolygonByX(square, 5f);

        Assert.True(left.Length >= 3);
        Assert.True(right.Length >= 3);
        Assert.All(left, p => Assert.True(p.X <= 5f + 1e-3f));
        Assert.All(right, p => Assert.True(p.X >= 5f - 1e-3f));
    }

    // Expected end points come from dynaface-lib's AnalyzeFace.calc_bisect on the same
    // pupils, including its integer truncation (level pupils give 509 -> 510, not 510).
    [Theory]
    [InlineData(640, 480, 509, 0, 510, 1024)]
    [InlineData(650, 480, 514, 0, 515, 1024)]
    [InlineData(640, 506, 456, 1024, 559, 0)]
    public void CalcBisect_MatchesDynafaceLib(float leftX, float leftY, float ax, float ay, float bx, float by)
    {
        var lm = TestHelpers.MakeLandmarks();
        lm[96] = new Vec2(380, 480); // right pupil
        lm[97] = new Vec2(leftX, leftY); // left pupil
        var ctx = TestHelpers.BuildContext(landmarks: lm);

        var (p1, p2) = ctx.CalcBisect();

        Assert.Equal(new[] { ax, ay, bx, by }, new[] { p1.X, p1.Y, p2.X, p2.Y });
    }

    [Theory]
    [InlineData(4, -5, 4, 15)]
    [InlineData(4, 15, 4, -5)] // either line direction keeps the same side
    public void ClipPolygonToSide_SplitsSquareAtLine(float x1, float y1, float x2, float y2)
    {
        var square = new[] { new Vec2(0, 0), new Vec2(10, 0), new Vec2(10, 10), new Vec2(0, 10) };
        var a = new Vec2(x1, y1);
        var b = new Vec2(x2, y2);

        var left = FaceMeasureContext.ClipPolygonToSide(square, a, b, imageLeft: true);
        var right = FaceMeasureContext.ClipPolygonToSide(square, a, b, imageLeft: false);

        Assert.Equal(40.0, FaceMeasureContext.PolygonAreaPx(left), 6);
        Assert.Equal(60.0, FaceMeasureContext.PolygonAreaPx(right), 6);
        Assert.All(left, p => Assert.True(p.X <= 4f));
        Assert.All(right, p => Assert.True(p.X >= 4f));
    }

    [Fact]
    public void ClipPolygonToSide_PolygonWhollyOnOtherSideIsEmpty()
    {
        var square = new[] { new Vec2(0, 0), new Vec2(10, 0), new Vec2(10, 10), new Vec2(0, 10) };
        Assert.Empty(FaceMeasureContext.ClipPolygonToSide(square, new Vec2(20, 0), new Vec2(20, 10), imageLeft: false));
    }

    [Fact]
    public void LineToEdge_RayPointingRight_HitsRightEdge()
    {
        var ctx = TestHelpers.BuildContext(width: 100, height: 100);
        Vec2? hit = ctx.LineToEdge(new Vec2(10, 50), angleDeg: 0f);

        Assert.NotNull(hit);
        Assert.Equal(99f, hit.Value.X, 1);
        Assert.Equal(50f, hit.Value.Y, 1);
    }

    [Fact]
    public void LineToEdge_StartAlreadyAtEdge_ReturnsNull()
    {
        var ctx = TestHelpers.BuildContext(width: 100, height: 100);
        Vec2? hit = ctx.LineToEdge(new Vec2(99, 50), angleDeg: 0f);
        Assert.Null(hit);
    }
}
