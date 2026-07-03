using System;
using System.Collections.Generic;

// Port of the subset of scipy.signal.find_peaks that lateral.py actually exercises:
// plain local-maxima detection (_find_local_max_min), and local maxima with an
// optional height threshold and/or minimum distance between peaks
// (_find_monotonic_corners). Matches scipy's plateau-aware local-maxima definition
// and its greedy, highest-priority-first distance-based suppression exactly.
public static class PeakFinder
{
    // heightThreshold: inclusive minimum (matches scipy's height filter, `>=`).
    // minDistance: greedy suppression keeping the highest peak in any cluster
    // closer together than this many samples (matches scipy's distance filter).
    public static (int[] indices, double[] heights) FindPeaks(
        double[] x, double? heightThreshold = null, int? minDistance = null)
    {
        var peaks = LocalMaxima(x);

        if (heightThreshold.HasValue)
        {
            var filtered = new List<int>();
            foreach (int i in peaks)
                if (x[i] >= heightThreshold.Value) filtered.Add(i);
            peaks = filtered;
        }

        if (minDistance.HasValue && peaks.Count > 0)
            peaks = SelectByPriority(peaks, x, minDistance.Value);

        var heights = new double[peaks.Count];
        for (int i = 0; i < peaks.Count; i++) heights[i] = x[peaks[i]];
        return (peaks.ToArray(), heights);
    }

    // Convenience for the negated-signal minima case used by _find_local_max_min.
    public static int[] FindMinimaIndices(double[] x)
    {
        var negated = new double[x.Length];
        for (int i = 0; i < x.Length; i++) negated[i] = -x[i];
        return LocalMaxima(negated).ToArray();
    }

    // Plateau-aware local maxima: a peak is any maximal flat run whose immediate
    // outer neighbors are both strictly lower; reported at the run's midpoint.
    // Matches scipy's _local_maxima_1d exactly, including that index 0 and the last
    // index can never be peaks (both need a real neighbor on each side).
    static List<int> LocalMaxima(double[] x)
    {
        var peaks = new List<int>();
        int n = x.Length;
        if (n < 3) return peaks;

        int i = 1;
        int iMax = n - 1;
        while (i < iMax)
        {
            if (x[i - 1] < x[i])
            {
                int iAhead = i + 1;
                while (iAhead < iMax && x[iAhead] == x[i]) iAhead++;

                if (x[iAhead] < x[i])
                {
                    int leftEdge = i, rightEdge = iAhead - 1;
                    peaks.Add((leftEdge + rightEdge) / 2);
                    i = iAhead;
                }
            }
            i++;
        }
        return peaks;
    }

    // Matches scipy's _select_by_peak_distance: process peaks from highest to
    // lowest value; for each still-kept peak, suppress every other peak within
    // `distance` samples, expanding contiguously outward (valid because `peaks`
    // is position-sorted, so once a neighbor is far enough away every peak
    // farther out in that direction is too).
    static List<int> SelectByPriority(List<int> peaks, double[] x, int distance)
    {
        int size = peaks.Count;
        var priority = new int[size];
        for (int i = 0; i < size; i++) priority[i] = i;
        Array.Sort(priority, (a, b) => x[peaks[a]].CompareTo(x[peaks[b]]));

        var keep = new bool[size];
        for (int i = 0; i < size; i++) keep[i] = true;

        for (int pi = size - 1; pi >= 0; pi--)
        {
            int j = priority[pi];
            if (!keep[j]) continue;

            int k = j - 1;
            while (k >= 0 && peaks[j] - peaks[k] < distance)
            {
                keep[k] = false;
                k--;
            }
            k = j + 1;
            while (k < size && peaks[k] - peaks[j] < distance)
            {
                keep[k] = false;
                k++;
            }
        }

        var result = new List<int>();
        for (int i = 0; i < size; i++)
            if (keep[i]) result.Add(peaks[i]);
        return result;
    }
}
