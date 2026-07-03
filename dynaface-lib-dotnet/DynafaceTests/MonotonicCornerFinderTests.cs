using System.Collections.Generic;

namespace DynafaceTests;

// Golden-path test for MonotonicCornerFinder: a synthetic sagittal profile shaped
// like a real lateral face silhouette (forehead slope, nasal bump, subnasal dip,
// lip bump, mentolabial dip, chin bump), run through the exact same parameters
// LateralAnalyzer uses, compared against dynaface-lib's own _find_monotonic_corners
// executed once in Python (scipy 1.17.1) on the identical input array. This is the
// single most valuable test in the lateral module, given how failure-prone this
// function's transcription is.
public class MonotonicCornerFinderTests
{
    // y=0..219; x = 40 + 0.15*y, plus gaussian bumps/dips at y=70 (nasal tip),
    // y=95 (subnasal dip), y=115 (lip bump), y=135 (mentolabial dip), y=150 (chin),
    // shifted so min(x)=0. Deterministic — no random noise.
    static double[] BuildSyntheticProfile()
    {
        const int n = 220;
        var x = new double[n];
        for (int y = 0; y < n; y++)
        {
            double v = 40 + 0.15 * y;
            v += 30 * System.Math.Exp(-0.5 * System.Math.Pow((y - 70) / 8.0, 2));
            v -= 12 * System.Math.Exp(-0.5 * System.Math.Pow((y - 95) / 6.0, 2));
            v += 18 * System.Math.Exp(-0.5 * System.Math.Pow((y - 115) / 7.0, 2));
            v -= 10 * System.Math.Exp(-0.5 * System.Math.Pow((y - 135) / 6.0, 2));
            v += 14 * System.Math.Exp(-0.5 * System.Math.Pow((y - 150) / 8.0, 2));
            x[y] = v;
        }
        double min = x[0];
        foreach (var v in x) if (v < min) min = v;
        for (int i = 0; i < n; i++) x[i] -= min;
        return x;
    }

    [Fact]
    public void FindMonotonicCorners_SyntheticProfile_MatchesPythonReference()
    {
        double[] x = BuildSyntheticProfile();

        var (maxIndices, _) = PeakFinder.FindPeaks(x);
        int[] minIndices = PeakFinder.FindMinimaIndices(x);

        // Reference max/min indices from the identical Python-side computation —
        // asserted first since a mismatch here would misattribute a corner-finding
        // failure to the wrong function.
        Assert.Equal(new[] { 70, 115, 151 }, maxIndices);
        Assert.Equal(new[] { 95, 134, 171 }, minIndices);

        var extremaSet = new HashSet<int>(maxIndices);
        foreach (int m in minIndices) extremaSet.Add(m);

        int[] corners = MonotonicCornerFinder.FindMonotonicCorners(
            x, new[] { 9, 13, 17 }, 2,
            dxTol: 0.035, minRun: 10, distancePx: 32,
            anglePercentile: 93.0, angleMinDeg: 16.0,
            kappaPercentile: 92.0, mixWeightAngle: 0.75,
            excludeExtrema: extremaSet);

        Assert.Equal(new[] { 50 }, corners);
    }
}
