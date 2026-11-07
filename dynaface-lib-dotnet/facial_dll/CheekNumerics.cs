using System;
using System.Collections.Generic;

// The NumPy and libm operations the cheek geometry (Cheek, CheekOutline) leans on,
// written to give dynaface-lib's results bit for bit where that can be pinned down:
// argmin/argmax with NumPy's NaN rule, np.interp, np.median, np.hypot, Python's
// round() and the box filter's np.convolve. Where NumPy's result depends on the
// compiled library under it (np.hypot is the C library's hypot, np.interp is C that
// the compiler may fuse, np.convolve is a BLAS dot product), these follow the build
// the shared fixtures were written with (macOS arm64, Accelerate): that hypot is
// sqrt(fma(small, small, big * big)), np.interp's step is fused and the dot product
// accumulates with a fused multiply-add from the first term to the last, each matched
// exactly on thousands of random inputs. Another build (Linux, OpenBLAS, glibc) can
// differ in the last bit; dynaface-lib's own results do too, which is why the shared
// tests compare derived heights and deviations to 1e-9 and only the selections exactly.
internal static class CheekNumerics
{
    /// <summary>np.argmin: the first index of the least value, or of the first NaN.</summary>
    internal static int ArgMin(IReadOnlyList<double> values)
    {
        if (values.Count == 0) throw new ArgumentException("argmin of an empty sequence", nameof(values));
        int best = 0;
        for (int i = 0; i < values.Count; i++)
        {
            double v = values[i];
            if (double.IsNaN(v)) return i;
            if (v < values[best]) best = i;
        }
        return best;
    }

    /// <summary>np.argmax: the first index of the greatest value, or of the first NaN.</summary>
    internal static int ArgMax(IReadOnlyList<double> values)
    {
        if (values.Count == 0) throw new ArgumentException("argmax of an empty sequence", nameof(values));
        int best = 0;
        for (int i = 0; i < values.Count; i++)
        {
            double v = values[i];
            if (double.IsNaN(v)) return i;
            if (v > values[best]) best = i;
        }
        return best;
    }

    /// <summary>
    /// np.interp(x, xp, fp) with its default ends (fp[0] below the table, fp[^1] above)
    /// and its step slope * (x - xp[j]) + fp[j] fused, as the reference build computes
    /// it; xp strictly increasing.
    /// </summary>
    internal static double Interp(double x, IReadOnlyList<double> xp, IReadOnlyList<double> fp)
    {
        if (double.IsNaN(x)) return x;
        int len = xp.Count;
        if (x > xp[len - 1]) return fp[len - 1];
        if (x < xp[0]) return fp[0];
        // xp[j] <= x < xp[j + 1], or j = len - 1 when x is the last point.
        int lo = 0, hi = len;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (xp[mid] <= x) lo = mid; else hi = mid;
        }
        int j = lo;
        if (j == len - 1 || xp[j] == x) return fp[j];
        double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
        double r = Fma(slope, x - xp[j], fp[j]);
        if (double.IsNaN(r))
        {
            r = Fma(slope, x - xp[j + 1], fp[j + 1]);
            if (double.IsNaN(r) && fp[j] == fp[j + 1]) r = fp[j];
        }
        return r;
    }

    /// <summary>
    /// np.median of the values: the middle one, or the mean of the two middle ones
    /// ((lower + upper) / 2); NaN when any is NaN.
    /// </summary>
    internal static double Median(List<double> values)
    {
        if (values.Count == 0) return double.NaN;
        var sorted = new List<double>(values);
        foreach (double v in sorted)
            if (double.IsNaN(v)) return double.NaN;
        sorted.Sort();
        int n = sorted.Count;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    // 2^600 and 2^-600 exactly, from their bits.
    static readonly double TwoPow600 = BitConverter.Int64BitsToDouble((1023L + 600) << 52);
    static readonly double TwoPowMinus600 = BitConverter.Int64BitsToDouble((1023L - 600) << 52);

    /// <summary>
    /// np.hypot as the reference build's C library computes it: sqrt(fma(small, small,
    /// big * big)) of the two magnitudes, scaled by a power of two at the ends of the
    /// double range so neither square overflows nor underflows. Infinite when either
    /// is, else NaN when either is.
    /// </summary>
    internal static double Hypot(double x, double y)
    {
        double ax = Math.Abs(x), ay = Math.Abs(y);
        if (double.IsInfinity(ax) || double.IsInfinity(ay)) return double.PositiveInfinity;
        if (double.IsNaN(ax) || double.IsNaN(ay)) return double.NaN;
        double big = ax >= ay ? ax : ay, small = ax >= ay ? ay : ax;
        double scale = 1.0;
        if (big > 1e150)
        {
            big *= TwoPowMinus600; small *= TwoPowMinus600; scale = TwoPow600;
        }
        else if (big < 1e-150 && big > 0.0)
        {
            big *= TwoPow600; small *= TwoPow600; scale = TwoPowMinus600;
        }
        return scale * Math.Sqrt(Fma(small, small, big * big));
    }

    /// <summary>
    /// fma(a, b, c): a * b + c with one rounding, in software, because netstandard2.1 has
    /// no Math.FusedMultiplyAdd. Boldo and Melquiond's emulation through rounding to odd
    /// ("Emulation of FMA and correctly-rounded sums", IEEE Trans. Computers, 2008):
    /// a * b = ph + pl and c + ph = th + tl exactly (Dekker's product, Knuth's sum), then
    /// th + RO(tl + pl), the inner sum rounded to odd, rounded once. Correctly rounded,
    /// so it equals the hardware instruction, while every value stays inside the normal
    /// range (here |a * b| and |c| between 1e-250 and 1e300, far beyond any pixel or
    /// ratio this code sees); outside it, and for zeros, infinities and NaN, it falls back
    /// to a * b + c. The same on every runtime, which a reflected Math.FusedMultiplyAdd
    /// (absent from Unity's class libraries) would not be.
    /// </summary>
    internal static double Fma(double a, double b, double c)
    {
        double p = a * b;
        double ap = Math.Abs(p), ac = Math.Abs(c);
        if (!(ap > 1e-250 && ap < 1e300) || !(ac < 1e300) || (c != 0.0 && ac < 1e-250) ||
            Math.Abs(a) > 1e290 || Math.Abs(b) > 1e290)
            return p + c;
        if (c == 0.0) return p;
        // a * b = p + pl exactly (Dekker's product through Veltkamp's split).
        Split(a, out double ah, out double al);
        Split(b, out double bh, out double bl);
        double pl = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
        // c + p = th + tl exactly (Knuth's two-sum).
        double th = c + p;
        double bb = th - c;
        double tl = (c - (th - bb)) + (p - bb);
        return th + AddRoundToOdd(tl, pl);
    }

    /// <summary>Veltkamp's split: a = hi + lo exactly, each with at most 26 significant bits.</summary>
    static void Split(double a, out double hi, out double lo)
    {
        double t = 134217729.0 * a; // 2^27 + 1
        hi = t - (t - a);
        lo = a - hi;
    }

    /// <summary>x + y rounded to odd: the nearest sum, moved one ulp toward the exact sum when it was inexact and its last bit is even.</summary>
    static double AddRoundToOdd(double x, double y)
    {
        double s = x + y;
        double bb = s - x;
        double err = (x - (s - bb)) + (y - bb);
        if (err == 0.0 || double.IsInfinity(s) || double.IsNaN(s)) return s;
        long bits = BitConverter.DoubleToInt64Bits(s);
        if ((bits & 1L) != 0) return s;
        // Away from zero when the error has s's sign, else toward it.
        bool away = (err > 0.0) == (s > 0.0);
        return BitConverter.Int64BitsToDouble(away ? bits + 1 : bits - 1);
    }

    /// <summary>Python's round(): to the nearest integer, halves to even.</summary>
    internal static double RoundHalfEven(double x) => Math.Round(x, MidpointRounding.ToEven);

    /// <summary>Whether x is neither NaN nor infinite (math.isfinite).</summary>
    internal static bool IsFinite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);
}
