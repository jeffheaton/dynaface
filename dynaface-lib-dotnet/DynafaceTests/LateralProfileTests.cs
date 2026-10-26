using System.Text.Json;
using static LateralLandmarkFinder;

namespace DynafaceTests;

// The synthetic side-profile outlines in dynaface-lib-python/tests_data/
// lateral_profiles.json, linked into TestData. Their expected extrema, corners and
// landmarks come from dynaface-lib-python, the reference implementation, whose
// tests/test_lateral.py checks the same file, so passing here means the two ports
// find the same landmarks. scripts/make_lateral_profiles.py writes the file. The
// helper tests mirror test_lateral.py's TestSearchHelpers, with the same inputs.
public class LateralProfileTests
{
    static readonly JsonElement Profiles = JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "TestData", "lateral_profiles.json"))).RootElement;

    public static IEnumerable<object[]> CaseNames() =>
        Profiles.GetProperty("cases").EnumerateArray()
            .Select(c => new object[] { c.GetProperty("name").GetString()! });

    static JsonElement Case(string name) =>
        Profiles.GetProperty("cases").EnumerateArray()
            .Single(c => c.GetProperty("name").GetString() == name);

    static JsonElement Outline(JsonElement c) =>
        Profiles.GetProperty("outlines").GetProperty(c.GetProperty("outline").GetString()!);

    static float[] Floats(JsonElement a) => a.EnumerateArray().Select(e => e.GetSingle()).ToArray();
    static int[] Ints(JsonElement a) => a.EnumerateArray().Select(e => e.GetInt32()).ToArray();
    static Vec2[] Points(JsonElement a) =>
        a.EnumerateArray().Select(p => new Vec2(p[0].GetSingle(), p[1].GetSingle())).ToArray();
    static (int, int)[] Pixels(IEnumerable<Vec2> points) =>
        points.Select(p => ((int)p.X, (int)p.Y)).ToArray();

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void FindsThePythonLandmarks(string name)
    {
        JsonElement c = Case(name);
        JsonElement outline = Outline(c);
        float[] xs = Floats(outline.GetProperty("sagittal_x"));
        float[] ys = Floats(outline.GetProperty("sagittal_y"));
        Vec2[] frontal = Points(c.GetProperty("frontal"));
        int shiftX = c.GetProperty("shift_x").GetInt32();
        int[] maxIndices = Ints(c.GetProperty("max_indices"));
        int[] minIndices = Ints(c.GetProperty("min_indices"));
        int[] cornerIdxs = Ints(c.GetProperty("corner_idxs"));

        Vec2[] landmarks;
        if (c.GetProperty("search_only").GetBoolean())
        {
            landmarks = FindLateralLandmarks(xs, ys, maxIndices, minIndices, cornerIdxs, shiftX, frontal);
        }
        else
        {
            var found = LateralAnalyzer.FindProfileLandmarks(xs, ys, shiftX, frontal);
            Assert.Equal(maxIndices, found.maxIndices);
            Assert.Equal(minIndices, found.minIndices);
            Assert.Equal(cornerIdxs, found.cornerIdxs);
            landmarks = found.landmarks;
        }
        Assert.Equal(Pixels(Points(c.GetProperty("landmarks"))), Pixels(landmarks));
    }

    [Fact]
    public void PogonionNotFound()
    {
        // With no usable x across the chin, the pogonion isn't found, so the
        // mentolabial point has nothing to overlap.
        JsonElement c = Case("face");
        JsonElement outline = Outline(c);
        float[] xs = Floats(outline.GetProperty("sagittal_x"));
        float[] ys = Floats(outline.GetProperty("sagittal_y"));
        for (int i = 0; i < ys.Length; i++)
            if (ys[i] >= 380 && ys[i] <= 410) xs[i] = float.NaN; // landmarks 14 and 16
        int shiftX = c.GetProperty("shift_x").GetInt32();

        Vec2[] landmarks = FindLateralLandmarks(
            xs, ys, Ints(c.GetProperty("max_indices")), Ints(c.GetProperty("min_indices")),
            Ints(c.GetProperty("corner_idxs")), shiftX, Points(c.GetProperty("frontal")));

        Assert.Equal((shiftX - 1, -1), Pixels(landmarks)[Pogonion]);
        Assert.Equal(Pixels(Points(c.GetProperty("landmarks"))).Take(4), Pixels(landmarks).Take(4));
    }

    // A small outline: x by row, rows 10 to 19.
    static readonly float[] Xs = { 5, 3, 1, 0, 2, 4, 6, 4, 2, 3 };
    static readonly float[] Rows = Enumerable.Range(10, 10).Select(i => (float)i).ToArray();
    static readonly Vec2[] TwoFrontal = { new(0, 12), new(0, 17) };

    static float[] WithNaN(float[] xs, int start, int stop)
    {
        var copy = (float[])xs.Clone();
        for (int i = start; i < stop; i++) copy[i] = float.NaN;
        return copy;
    }

    [Fact]
    public void FindLateralLandmarkModes()
    {
        Vec2? Find(SearchMode mode, bool? yForward = null, float y = 14.5f) =>
            FindLateralLandmark(Xs, Rows, new[] { 6 }, new[] { 3 }, new[] { 2, 8 }, y, mode, yForward);

        Assert.Equal(new Vec2(6, 16), Find(SearchMode.Max));
        Assert.Equal(new Vec2(2, 18), Find(SearchMode.Corner, yForward: true));
        Assert.Equal(new Vec2(1, 12), Find(SearchMode.Corner, yForward: false));
        Assert.Null(Find(SearchMode.Min, yForward: true, y: 15));
        Assert.Null(Find((SearchMode)99)); // not a search mode
    }

    [Fact]
    public void CornerLandmarkInRange()
    {
        Vec2? Find(float[]? xs = null, float[]? rows = null, int[]? corners = null,
                   int? lo = null, int? hi = null, Vec2[]? frontal = null, int? frontalHi = null) =>
            FindCornerLandmarkInRange(xs ?? Xs, rows ?? Rows, corners ?? Array.Empty<int>(),
                lo, hi, frontal!, frontal == null ? null : 0, frontalHi);

        Assert.Null(Find(xs: Array.Empty<float>(), rows: Array.Empty<float>()));
        Assert.Null(Find(rows: Rows.Take(5).ToArray())); // lengths differ
        Assert.Null(Find()); // neither a range nor frontal landmarks
        Assert.Null(Find(frontal: TwoFrontal, frontalHi: 2)); // landmark 2 doesn't exist
        Assert.Equal(new Vec2(6, 16), Find(lo: 7, hi: 3)); // bounds in either order
        float[] nanWindow = WithNaN(Xs, 2, 8);
        Assert.Null(Find(xs: nanWindow, lo: 3, hi: 6));
        // With no usable x in the window, the first corner in it is chosen.
        Vec2? corner = Find(xs: nanWindow, corners: new[] { 5 }, lo: 3, hi: 6);
        Assert.True(float.IsNaN(corner!.Value.X));
        Assert.Equal(15f, corner.Value.Y);
        Assert.Null(Find(xs: nanWindow, corners: new[] { 9 }, lo: 3, hi: 6));
    }

    [Fact]
    public void LandmarkInRange()
    {
        Vec2? Find(float[]? xs = null, float[]? rows = null, int? lo = null, int? hi = null,
                   ExtremePick pick = ExtremePick.Min, Vec2[]? frontal = null, int? frontalHi = null) =>
            FindLateralLandmarkInRange(xs ?? Xs, rows ?? Rows, lo, hi, pick,
                frontal!, frontal == null ? null : 0, frontalHi);

        Assert.Null(Find(xs: Array.Empty<float>(), rows: Array.Empty<float>()));
        Assert.Null(Find()); // neither a range nor frontal landmarks
        Assert.Null(Find(frontal: TwoFrontal, frontalHi: 5)); // landmark 5 doesn't exist
        Assert.Equal(new Vec2(0, 13), Find(lo: 8, hi: 1));
        Assert.Equal(new Vec2(6, 16), Find(lo: 4, hi: 8, pick: ExtremePick.Max));
        Assert.Null(Find(xs: WithNaN(Xs, 0, 10), lo: 0, hi: 9));
    }

    [Fact]
    public void LandmarkMinMax()
    {
        Assert.Null(FindLateralLandmarkMinMax(Array.Empty<float>(), Array.Empty<float>(), null, null, ExtremePick.Max));
        Assert.Equal(new Vec2(6, 16), FindLateralLandmarkMinMax(Xs, Rows, null, null, ExtremePick.Max)); // the whole outline
        Assert.Equal(new Vec2(0, 13), FindLateralLandmarkMinMax(Xs, Rows, 8, 1, ExtremePick.Min));
        Assert.Null(FindLateralLandmarkMinMax(WithNaN(Xs, 0, 10), Rows, null, null, ExtremePick.Max));
    }

    [Fact]
    public void CornersNearExtremaAreExcluded()
    {
        // The same face and settings as test_lateral.py's test_corner_finder_edges.
        double[] face = Enumerable.Range(0, 200).Select(i => 40 + 30 * Math.Sin(i / 15.0)).ToArray();
        int[] Corners(IEnumerable<int> exclude) => MonotonicCornerFinder.FindMonotonicCorners(
            face, scales: new[] { 9 }, polyOrder: 2, dxTol: 0.03, minRun: 8, distancePx: 28,
            anglePercentile: 92.0, angleMinDeg: 14.0, kappaPercentile: 90.0, mixWeightAngle: 0.7,
            excludeExtrema: exclude);

        Assert.NotEmpty(Corners(Array.Empty<int>()));
        Assert.Empty(Corners(Enumerable.Range(0, 200)));
    }
}
