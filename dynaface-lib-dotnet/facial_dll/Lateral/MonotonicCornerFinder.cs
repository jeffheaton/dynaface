using System;
using System.Collections.Generic;

// Direct transcription of dynaface-lib's lateral.py _find_monotonic_corners and its
// helpers (_ensure_odd, _turning_angle, _angle_change, _exclude_near, _nms_keep_best).
// Deliberately mirrors Python's structure/variable names closely (rather than a
// "cleaner" refactor) so it can be diffed line-by-line against lateral.py if a
// numeric mismatch ever needs tracking down.
public static class MonotonicCornerFinder
{
    public static int[] FindMonotonicCorners(
        double[] sagittalX,
        int[] scales, int polyOrder,
        double dxTol, int minRun, int distancePx,
        double anglePercentile, double angleMinDeg,
        double kappaPercentile, double mixWeightAngle,
        IEnumerable<int> excludeExtrema)
    {
        double[] x = sagittalX;
        int n = x.Length;
        if (n < 7) return Array.Empty<int>();

        var allIdx    = new List<int>();
        var allScores = new List<double>();
        int halfNeigh = Math.Max(1, minRun / 2);

        foreach (int rawW in scales)
        {
            int w = EnsureOdd(rawW);
            if (w >= n) continue;

            double[] dx  = SavitzkyGolay.Filter(x, w, polyOrder, 1);
            double[] ddx = SavitzkyGolay.Filter(x, w, polyOrder, 2);

            var kappa = new double[n];
            for (int i = 0; i < n; i++)
                kappa[i] = Math.Abs(ddx[i]) / Math.Pow(1.0 + dx[i] * dx[i], 1.5);

            var theta = new double[n];
            for (int i = 0; i < n; i++) theta[i] = Math.Atan(dx[i]);

            int halfwin = w / 2;
            double[] dtheta = AngleChange(theta, halfwin);

            var signDx = new int[n];
            for (int i = 0; i < n; i++) signDx[i] = Math.Sign(dx[i]);

            var strong = new bool[n];
            for (int i = 0; i < n; i++) strong[i] = Math.Abs(dx[i]) >= dxTol;

            var same = new bool[n];
            for (int i = 0; i < n; i++) same[i] = true;
            for (int off = 1; off <= halfNeigh; off++)
            {
                for (int i = 0; i < n; i++)
                {
                    int leftIdx  = ((i - off) % n + n) % n;
                    int rightIdx = ((i + off) % n + n) % n;
                    same[i] = same[i] && signDx[i] == signDx[leftIdx] && signDx[i] == signDx[rightIdx];
                }
            }

            var mono = new bool[n];
            for (int i = 0; i < n; i++) mono[i] = strong[i] && same[i];

            var monoVals = new List<double>();
            for (int i = 0; i < n; i++) if (mono[i]) monoVals.Add(dtheta[i]);
            if (monoVals.Count == 0) continue;

            double thAngle = Percentile(monoVals.ToArray(), anglePercentile);
            thAngle = Math.Max(thAngle, angleMinDeg * MathHelpers.Deg2Rad);

            var dthetaPeaks = new double[n];
            for (int i = 0; i < n; i++) dthetaPeaks[i] = mono[i] ? dtheta[i] : 0.0;
            var (peaksA, heightsA) = PeakFinder.FindPeaks(dthetaPeaks, thAngle, distancePx);

            var monoKappa = new List<double>();
            for (int i = 0; i < n; i++) if (mono[i]) monoKappa.Add(kappa[i]);
            double thKappa = monoKappa.Count > 0 ? Percentile(monoKappa.ToArray(), kappaPercentile) : 0.0;

            var kappaPeaksArr = new double[n];
            for (int i = 0; i < n; i++) kappaPeaksArr[i] = mono[i] ? kappa[i] : 0.0;
            var (peaksK, heightsK) = thKappa > 0
                ? PeakFinder.FindPeaks(kappaPeaksArr, thKappa, distancePx)
                : PeakFinder.FindPeaks(kappaPeaksArr, null, distancePx);

            var idxSet = new SortedSet<int>();
            foreach (int p in peaksA) idxSet.Add(p);
            foreach (int p in peaksK) idxSet.Add(p);
            if (idxSet.Count == 0) continue;

            var scoreA = new Dictionary<int, double>();
            for (int i = 0; i < peaksA.Length; i++) scoreA[peaksA[i]] = heightsA[i];
            var scoreK = new Dictionary<int, double>();
            for (int i = 0; i < peaksK.Length; i++) scoreK[peaksK[i]] = heightsK[i];

            double maxA  = heightsA.Length > 0 ? MaxOf(heightsA) : 0.0;
            double maxKv = heightsK.Length > 0 ? MaxOf(heightsK) : 0.0;

            foreach (int i in idxSet)
            {
                double sa = Norm(scoreA.TryGetValue(i, out double va) ? va : 0.0, maxA);
                double sk = Norm(scoreK.TryGetValue(i, out double vk) ? vk : 0.0, maxKv);
                double s  = mixWeightAngle * sa + (1.0 - mixWeightAngle) * sk;
                allIdx.Add(i);
                allScores.Add(s);
            }
        }

        if (allIdx.Count == 0) return Array.Empty<int>();

        var excludeSet = new HashSet<int>(excludeExtrema ?? Array.Empty<int>());
        var survivingIdx    = new List<int>();
        var survivingScores = new List<double>();
        for (int i = 0; i < allIdx.Count; i++)
        {
            bool excluded = false;
            foreach (int b in excludeSet)
                if (Math.Abs(allIdx[i] - b) <= 6) { excluded = true; break; }
            if (!excluded)
            {
                survivingIdx.Add(allIdx[i]);
                survivingScores.Add(allScores[i]);
            }
        }
        if (survivingIdx.Count == 0) return Array.Empty<int>();

        int nmsRadius = Math.Max(8, distancePx / 2);
        return NmsKeepBest(survivingIdx.ToArray(), survivingScores.ToArray(), nmsRadius);
    }

    static int EnsureOdd(int k)
    {
        k = Math.Max(3, k);
        return (k % 2 == 1) ? k : k + 1;
    }

    static double[] AngleChange(double[] theta, int halfwin)
    {
        int n = theta.Length;
        var outArr = new double[n];
        if (halfwin <= 0 || n < 2 * halfwin + 1) return outArr;

        for (int j = halfwin; j < n - halfwin; j++)
            outArr[j] = Math.Abs(theta[j + halfwin] - theta[j - halfwin]);

        return outArr;
    }

    // scipy/numpy's default percentile method ("linear" interpolation between the
    // two nearest ranks).
    static double Percentile(double[] values, double percentile)
    {
        if (values.Length == 0) return 0.0;
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        if (sorted.Length == 1) return sorted[0];

        double h = (percentile / 100.0) * (sorted.Length - 1);
        int lo = Math.Max(0, Math.Min((int)Math.Floor(h), sorted.Length - 1));
        int hi = Math.Max(0, Math.Min((int)Math.Ceiling(h), sorted.Length - 1));
        double frac = h - Math.Floor(h);
        return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
    }

    static double Norm(double v, double vmax) => vmax <= 0 ? 0.0 : v / vmax;

    static double MaxOf(double[] arr)
    {
        double m = double.NegativeInfinity;
        foreach (double v in arr) if (v > m) m = v;
        return m;
    }

    // Matches lateral.py's _nms_keep_best: full pairwise (not contiguous) suppression,
    // since candidate corner indices (pooled across scales) aren't necessarily sorted.
    static int[] NmsKeepBest(int[] idxs, double[] scores, int radius)
    {
        int n = Math.Min(idxs.Length, scores.Length);
        if (n == 0) return Array.Empty<int>();
        if (n == 1) return new[] { idxs[0] };

        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) => scores[b].CompareTo(scores[a])); // descending

        var taken = new bool[n];
        var kept  = new List<int>();
        foreach (int o in order)
        {
            if (taken[o]) continue;
            int i = idxs[o];
            kept.Add(i);
            for (int k = 0; k < n; k++)
                if (Math.Abs(idxs[k] - i) <= radius) taken[k] = true;
        }
        kept.Sort();
        return kept.ToArray();
    }
}
