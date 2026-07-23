using System;
using System.Collections.Generic;

// Direct transcription of dynaface-lib's lateral.py _find_lateral_landmark,
// _find_corner_landmark_in_range, _find_lateral_landmark_in_range,
// _find_lateral_landmark_minmax, and the main _find_lateral_landmarks.
//
// dynaface-lib uses [-1,-1] as a "not found" sentinel throughout this module; here
// that's replaced with a nullable Vec2? (the .NET-idiomatic equivalent) internally,
// converting to the public (-1,-1)-sentineled Vec2[] contract only at the very end
// of FindLateralLandmarks, matching what downstream consumers (MeasureLateral) expect.
// The near-duplicate range-search helpers are kept separate (not consolidated),
// matching dynaface-lib's own structure, to keep this diffable against lateral.py.
public static class LateralLandmarkFinder
{
    public enum SearchMode { Max, Min, Corner, Nearest }
    enum ExtremePick { Max, Min }

    // Output layout: 0=Glabella, 1=Nasion, 2=NasalTip, 3=Subnasal, 4=MentoLabial, 5=Pogonion.
    public const int Glabella = 0, Nasion = 1, NasalTip = 2, Subnasal = 3, MentoLabial = 4, Pogonion = 5;

    public static Vec2[] FindLateralLandmarks(
        float[] sagittalX, float[] sagittalY,
        int[] maxIndices, int[] minIndices, int[] cornerIdxs,
        int shiftX, Vec2[] landmarksFrontal)
    {
        if (landmarksFrontal == null || landmarksFrontal.Length == 0)
        {
            var allMissing = new Vec2[6];
            for (int i = 0; i < 6; i++) allMissing[i] = new Vec2(-1, -1);
            return allMissing;
        }

        int nSag = sagittalX.Length;
        var landmarks = new Vec2?[6];

        // Pogonion via frontal pair (14..16), MIN X.
        if (landmarksFrontal.Length > 16)
            landmarks[Pogonion] = FindLateralLandmarkInRange(
                sagittalX, sagittalY, null, null, ExtremePick.Min, landmarksFrontal, 14, 16);

        // Subnasal via frontal pair (57..79), corner-or-max.
        if (landmarksFrontal.Length > 79)
            landmarks[Subnasal] = FindCornerLandmarkInRange(
                sagittalX, sagittalY, cornerIdxs, null, null, landmarksFrontal, 57, 79);

        // Mento-labial via frontal pair (8..14), MAX X via the minmax finder.
        if (landmarksFrontal.Length > 14)
        {
            float yUpperBound = landmarksFrontal[8].Y;
            float yLowerBound = landmarksFrontal[14].Y;
            int idxUpperBound = ArgMinAbsDiff(sagittalY, yUpperBound);
            int idxLowerBound = ArgMinAbsDiff(sagittalY, yLowerBound);
            if (idxUpperBound == idxLowerBound)
            {
                const int widen = 2;
                idxUpperBound = Math.Max(0, idxUpperBound - widen);
                idxLowerBound = Math.Min(nSag - 1, idxLowerBound + widen);
            }
            int searchLo = Math.Min(idxUpperBound, idxLowerBound);
            int searchHi = Math.Max(idxUpperBound, idxLowerBound);
            landmarks[MentoLabial] = FindLateralLandmarkMinMax(sagittalX, sagittalY, searchLo, searchHi, ExtremePick.Max);
        }

        // Overlap guard: if Mento-labial ~ Pogonion, recompute Mento-labial via
        // CORNER search at frontal idx 14 instead.
        if (landmarks[MentoLabial].HasValue && landmarks[Pogonion].HasValue)
        {
            Vec2 ml = landmarks[MentoLabial].Value, pog = landmarks[Pogonion].Value;
            float dx = MathF.Abs(ml.X - pog.X);
            float dy = MathF.Abs(ml.Y - pog.Y);
            float dEuclid = MathF.Sqrt(dx * dx + dy * dy);

            float faceHeight = nSag > 0 ? (MaxOf(sagittalY) - MinOf(sagittalY)) : 0f;
            int xTolPx = Math.Max(3, MathHelpers.RoundToInt(0.004f * faceHeight));
            int yTolPx = Math.Max(5, MathHelpers.RoundToInt(0.008f * faceHeight));
            int euclidTolPx = Math.Max(6, MathHelpers.RoundToInt(0.008f * faceHeight));

            bool condAxis = dx <= xTolPx && dy <= yTolPx;
            bool condEuclid = dEuclid <= euclidTolPx;

            if (condAxis || condEuclid)
            {
                float targetYForCorner = landmarksFrontal[14].Y;
                Vec2? mlCornerPt = FindLateralLandmark(
                    sagittalX, sagittalY, maxIndices, minIndices, cornerIdxs,
                    targetYForCorner, SearchMode.Corner, false);
                if (mlCornerPt.HasValue) landmarks[MentoLabial] = mlCornerPt;
            }
        }

        // Highest frontal landmark (smallest Y): dynamic Glabella anchor.
        int highestFrontalIdx = ArgMinY(landmarksFrontal);

        var mapping = new (int outIdx, int frontalIdx, SearchMode mode, bool? yForward)[]
        {
            (Glabella, highestFrontalIdx, SearchMode.Nearest, (bool?)null),
            (Nasion,   51,                SearchMode.Nearest, false),
            (NasalTip, 54,                SearchMode.Min,     (bool?)null),
        };
        foreach (var (outIdx, frontalIdx, mode, yForward) in mapping)
        {
            float yTarget = landmarksFrontal[frontalIdx].Y;
            landmarks[outIdx] = FindLateralLandmark(
                sagittalX, sagittalY, maxIndices, minIndices, cornerIdxs, yTarget, mode, yForward);
        }

        var result = new Vec2[6];
        for (int i = 0; i < 6; i++)
        {
            // Python adds shift_x to the WHOLE array unconditionally, sentinels
            // included — replicated here rather than special-cased away.
            Vec2 v = landmarks[i] ?? new Vec2(-1, -1);
            result[i] = new Vec2((int)(v.X + shiftX), (int)v.Y);
        }
        return result;
    }

    static Vec2? FindLateralLandmark(
        float[] sagittalX, float[] sagittalY,
        int[] maxIndices, int[] minIndices, int[] cornerIdxs,
        float yCoord, SearchMode mode, bool? yForward)
    {
        int[] DirFilter(int[] idxs)
        {
            if (yForward == true)
            {
                var kept = new List<int>();
                foreach (int i in idxs) if (sagittalY[i] >= yCoord) kept.Add(i);
                return kept.ToArray();
            }
            if (yForward == false)
            {
                var kept = new List<int>();
                foreach (int i in idxs) if (sagittalY[i] <= yCoord) kept.Add(i);
                return kept.ToArray();
            }
            return idxs;
        }

        int[] candidates;
        switch (mode)
        {
            case SearchMode.Max: candidates = DirFilter(maxIndices); break;
            case SearchMode.Min: candidates = DirFilter(minIndices); break;
            case SearchMode.Corner: candidates = DirFilter(cornerIdxs); break;
            case SearchMode.Nearest:
                var all = new int[sagittalY.Length];
                for (int i = 0; i < all.Length; i++) all[i] = i;
                candidates = DirFilter(all);
                break;
            default: candidates = Array.Empty<int>(); break;
        }
        if (candidates.Length == 0) return null;

        int closest = 0;
        float bestDiff = MathF.Abs(sagittalY[candidates[0]] - yCoord);
        for (int k = 1; k < candidates.Length; k++)
        {
            float diff = MathF.Abs(sagittalY[candidates[k]] - yCoord);
            if (diff < bestDiff) { bestDiff = diff; closest = k; }
        }
        int idx = candidates[closest];
        return new Vec2(sagittalX[idx], sagittalY[idx]);
    }

    static Vec2? FindCornerLandmarkInRange(
        float[] sagittalX, float[] sagittalY, int[] cornerIdxs,
        int? idxLow, int? idxHigh,
        Vec2[] landmarksFrontal, int? frontalLoIdx, int? frontalHiIdx, int widenIfSame = 2)
    {
        int n = sagittalX.Length;
        if (n == 0 || n != sagittalY.Length) return null;

        int lo, hi;
        if (idxLow == null || idxHigh == null)
        {
            if (landmarksFrontal == null || frontalLoIdx == null || frontalHiIdx == null ||
                Math.Max(frontalLoIdx.Value, frontalHiIdx.Value) >= landmarksFrontal.Length)
                return null;

            float yLo = landmarksFrontal[frontalLoIdx.Value].Y;
            float yHi = landmarksFrontal[frontalHiIdx.Value].Y;
            int computedLow = ArgMinAbsDiff(sagittalY, yLo);
            int computedHigh = ArgMinAbsDiff(sagittalY, yHi);
            if (computedLow == computedHigh && widenIfSame > 0)
            {
                computedLow = Math.Max(0, computedLow - widenIfSame);
                computedHigh = Math.Min(n - 1, computedHigh + widenIfSame);
            }
            lo = computedLow;
            hi = computedHigh;
        }
        else
        {
            lo = idxLow.Value;
            hi = idxHigh.Value;
        }

        lo = Math.Max(0, Math.Min(lo, n - 1));
        hi = Math.Max(0, Math.Min(hi, n - 1));
        if (lo > hi) (lo, hi) = (hi, lo);
        if (lo >= n || hi < lo) return null;

        if (cornerIdxs.Length == 0)
        {
            int? maxIdxNoCorners = ArgMaxFiniteInRange(sagittalX, lo, hi);
            if (maxIdxNoCorners == null) return null;
            return new Vec2(sagittalX[maxIdxNoCorners.Value], sagittalY[maxIdxNoCorners.Value]);
        }

        var cornerSet = new HashSet<int>();
        foreach (int c in cornerIdxs) if (c >= 0 && c < n) cornerSet.Add(c);

        int? maxIdx = ArgMaxFiniteInRange(sagittalX, lo, hi);
        int? firstCornerIdx = FirstInRange(cornerSet, lo, hi);

        if (maxIdx == null)
        {
            if (firstCornerIdx == null) return null;
            int fallback = firstCornerIdx.Value;
            return new Vec2(sagittalX[fallback], sagittalY[fallback]);
        }

        int chosen = (firstCornerIdx != null && firstCornerIdx.Value <= maxIdx.Value)
            ? firstCornerIdx.Value
            : maxIdx.Value;
        return new Vec2(sagittalX[chosen], sagittalY[chosen]);
    }

    static Vec2? FindLateralLandmarkInRange(
        float[] sagittalX, float[] sagittalY,
        int? idxLow, int? idxHigh, ExtremePick pick,
        Vec2[] landmarksFrontal, int? frontalLoIdx, int? frontalHiIdx, int widenIfSame = 2)
    {
        int n = sagittalX.Length;
        if (n == 0 || n != sagittalY.Length) return null;

        int lo, hi;
        if (idxLow == null || idxHigh == null)
        {
            if (landmarksFrontal == null || frontalLoIdx == null || frontalHiIdx == null ||
                Math.Max(frontalLoIdx.Value, frontalHiIdx.Value) >= landmarksFrontal.Length)
                return null;

            float yLo = landmarksFrontal[frontalLoIdx.Value].Y;
            float yHi = landmarksFrontal[frontalHiIdx.Value].Y;
            int computedLow = ArgMinAbsDiff(sagittalY, yLo);
            int computedHigh = ArgMinAbsDiff(sagittalY, yHi);
            if (computedLow == computedHigh && widenIfSame > 0)
            {
                computedLow = Math.Max(0, computedLow - widenIfSame);
                computedHigh = Math.Min(n - 1, computedHigh + widenIfSame);
            }
            lo = computedLow;
            hi = computedHigh;
        }
        else
        {
            lo = idxLow.Value;
            hi = idxHigh.Value;
        }

        lo = Math.Max(0, Math.Min(lo, n - 1));
        hi = Math.Max(0, Math.Min(hi, n - 1));
        if (lo > hi) (lo, hi) = (hi, lo);
        if (hi < lo || lo >= n) return null;

        return PickExtremeInRange(sagittalX, sagittalY, lo, hi, pick);
    }

    static Vec2? FindLateralLandmarkMinMax(float[] sagittalX, float[] sagittalY, int? idxLow, int? idxHigh, ExtremePick pick)
    {
        int n = sagittalX.Length;
        if (n == 0 || n != sagittalY.Length) return null;

        int lo = idxLow ?? 0;
        int hi = idxHigh ?? (n - 1);
        lo = Math.Max(0, Math.Min(lo, n - 1));
        hi = Math.Max(0, Math.Min(hi, n - 1));
        if (lo > hi) (lo, hi) = (hi, lo);

        return PickExtremeInRange(sagittalX, sagittalY, lo, hi, pick);
    }

    // Shared by FindLateralLandmarkInRange and FindLateralLandmarkMinMax: find the
    // MIN/MAX x within [lo,hi], breaking ties by the candidate whose y is closest
    // to the median y among the tied candidates.
    static Vec2? PickExtremeInRange(float[] sagittalX, float[] sagittalY, int lo, int hi, ExtremePick pick)
    {
        float extremeVal = pick == ExtremePick.Max ? float.NegativeInfinity : float.PositiveInfinity;
        for (int i = lo; i <= hi; i++)
        {
            float v = sagittalX[i];
            if (pick == ExtremePick.Max ? v > extremeVal : v < extremeVal) extremeVal = v;
        }

        var relCandidates = new List<int>();
        for (int i = lo; i <= hi; i++)
            if (sagittalX[i] == extremeVal) relCandidates.Add(i - lo);
        if (relCandidates.Count == 0) return null;

        int relIdx;
        if (relCandidates.Count > 1)
        {
            var candYs = new float[relCandidates.Count];
            for (int k = 0; k < relCandidates.Count; k++) candYs[k] = sagittalY[lo + relCandidates[k]];
            float medianY = Median(candYs);

            int best = 0;
            float bestDiff = MathF.Abs(candYs[0] - medianY);
            for (int k = 1; k < relCandidates.Count; k++)
            {
                float diff = MathF.Abs(candYs[k] - medianY);
                if (diff < bestDiff) { bestDiff = diff; best = k; }
            }
            relIdx = relCandidates[best];
        }
        else
        {
            relIdx = relCandidates[0];
        }

        int idx = lo + relIdx;
        return new Vec2(sagittalX[idx], sagittalY[idx]);
    }

    static int ArgMinAbsDiff(float[] arr, float target)
    {
        int best = 0;
        float bestDiff = MathF.Abs(arr[0] - target);
        for (int i = 1; i < arr.Length; i++)
        {
            float diff = MathF.Abs(arr[i] - target);
            if (diff < bestDiff) { bestDiff = diff; best = i; }
        }
        return best;
    }

    static int? ArgMaxFiniteInRange(float[] arr, int lo, int hi)
    {
        int? best = null;
        float bestVal = float.NegativeInfinity;
        for (int i = lo; i <= hi; i++)
        {
            float v = float.IsFinite(arr[i]) ? arr[i] : float.NegativeInfinity;
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    static int? FirstInRange(HashSet<int> set, int lo, int hi)
    {
        for (int i = lo; i <= hi; i++)
            if (set.Contains(i)) return i;
        return null;
    }

    static float Median(float[] values)
    {
        var sorted = (float[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        if (n % 2 == 1) return sorted[n / 2];
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2f;
    }

    static int ArgMinY(Vec2[] points)
    {
        int best = 0;
        float bestVal = points[0].Y;
        for (int i = 1; i < points.Length; i++)
            if (points[i].Y < bestVal) { bestVal = points[i].Y; best = i; }
        return best;
    }

    static float MaxOf(float[] arr)
    {
        float m = float.NegativeInfinity;
        foreach (float v in arr) if (v > m) m = v;
        return m;
    }

    static float MinOf(float[] arr)
    {
        float m = float.PositiveInfinity;
        foreach (float v in arr) if (v < m) m = v;
        return m;
    }
}
