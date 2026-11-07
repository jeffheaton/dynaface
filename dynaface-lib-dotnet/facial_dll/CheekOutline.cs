#nullable enable
using System;
using System.Collections.Generic;
using static CheekNumerics;

/// <summary>
/// The far-cheek outline of a three-quarter (quarter) face and the observed cheek
/// points on it: the malar apex and the submalar point. A transcription of
/// dynaface-lib's <c>dynaface/cheek_outline.py</c>, the reference implementation, name
/// for name in PascalCase; its numbers are float64 there and <c>double</c> here, with
/// the same operation order, so the shared fixture (<c>tests_data/cheek_outlines.json</c>,
/// checked by <c>CheekOutlineTests</c>) holds both ports to the same outlines and points.
///
/// <para><see cref="TraceFarCheek(bool[,], byte[,], Vec2d[], bool, double, byte[,])"/>
/// follows the background model's silhouette row by row from the far canthus to the
/// chin, seeded and bounded by the far WFLW contour, keeping the contour where the mask
/// has no edge (or, given the gray image, the strongest intensity edge beside the seed)
/// and marking the mask rows the nose disturbs: those in the nose band that depart
/// outward from the contour by <see cref="NoseStep"/> more than the band's reference,
/// the median departure of the mask rows between the stomion and canthus levels.
/// <see cref="ContourOutline(Vec2d[], bool)"/> is the same band from the contour alone,
/// the fallback. <see cref="ObliqueFrameOf(Vec2d[])"/>, <see cref="LevelThrough"/> and
/// <see cref="HeightRatio"/> draw the canthus, stomion and chin levels from the vanishing
/// point of the canthi and alar-base lines and read Kaptein's height ratio h along the
/// roll-corrected vertical, and <see cref="ObservedPoints(Vec2d[], float[], int[], byte[], bool)"/>
/// finds the apex and the submalar point with every flag the measure reports.
/// <see cref="GrayImage(byte[,,])"/> is the gray conversion the second edge source reads.</para>
///
/// <para>Coordinates: landmarks are top-left pixel values (y down), as everywhere in the
/// library. Images and masks are top-left, row-major 2D arrays indexed
/// <c>[row, column]</c> (row 0 at the top), as NumPy holds them, not
/// <see cref="FaceImage"/>'s bottom-left pixels: a caller converts at the boundary
/// (<see cref="GrayImage(FaceImage)"/> does for the gray image).</para>
/// </summary>
public static class CheekOutline
{
    /// <summary>How many WFLW landmarks a face has; fewer give no frame and no point.</summary>
    public const int NumLandmarks = 98;

    // The WFLW landmarks the cheek landmarks read (Cheek re-exports them): the chin,
    // the lateral canthi, the alar bases, the nose tip, the subnasale and the inner-lip
    // midpoints whose mean is the stomion.
    public const int LmChin = 16;
    public const int LmCanthusR = 60, LmCanthusL = 72;
    public const int LmAlaR = 55, LmAlaL = 59;
    public const int LmNoseTip = 54, LmSubnasale = 57;
    /// <summary>The inner-lip midpoints (90 upper, 94 lower) whose mean is the stomion.</summary>
    public static readonly (int Upper, int Lower) LmStomion = (90, 94);

    /// <summary>
    /// The right half contour in polyline order, from the contour's top point to the
    /// chin (0..16): the seed polyline when the far cheek is the right.
    /// </summary>
    public static readonly IReadOnlyList<int> RightFarContour = Range(0, 17, 1);
    /// <summary>The left half contour in polyline order (32..16): the seed polyline when the far cheek is the left.</summary>
    public static readonly IReadOnlyList<int> LeftFarContour = Range(32, 15, -1);

    /// <summary>The far lateral canthus: 72 when the far cheek is the subject's left, else 60 (Python's FAR_CANTHUS).</summary>
    public static int FarCanthus(bool farLeft) => farLeft ? LmCanthusL : LmCanthusR;

    /// <summary>Kaptein 2015: chin-to-apex over chin-to-canthus.</summary>
    public const double ApexHPrior = 0.793;
    /// <summary>Kaptein's observed range: the apex search window.</summary>
    public static readonly (double Low, double High) ApexHWindow = (0.71, 0.83);
    /// <summary>Of H: a bulge smaller than this is not an observed apex.</summary>
    public const double ApexMinDev = 0.01;
    /// <summary>The definitions' flag, unconditional: apex.*.near_nose within this h of the tip.</summary>
    public const double ApexNearNoseH = 0.02;
    /// <summary>The tip's and the subnasale's levels bound the nose rows (<see cref="OutlineSrcNose"/>).</summary>
    public static readonly (int Tip, int Subnasale) NoseRows = (LmNoseTip, LmSubnasale);
    /// <summary>
    /// Of the tracer's span (the chin-to-far-canthus distance along the roll-corrected
    /// vertical read between those two landmarks, not <see cref="ObliqueCheek.Scale"/>):
    /// a mask row in the nose band whose outward departure from the seed exceeds the
    /// band's reference by this is the nose. The reference is the median departure of
    /// the <see cref="OutlineSrcMask"/> rows between the stomion level and the canthus
    /// level outside the band, 0.0 when there are none.
    /// </summary>
    public const double NoseStep = 0.01;
    /// <summary>Of the tracer's span, like <see cref="NoseStep"/>: how far the per-row walk may go from the seed.</summary>
    public const double OutlineTol = 0.05;
    /// <summary>Fraction of mask rows a window needs for apex.ok / sub.ok.</summary>
    public const double OutlineMinMask = 0.8;
    /// <summary>Box filter on the deviation profiles before extrema.</summary>
    public const int SmoothRows = 9;
    /// <summary>
    /// The rows on each side of a row the mask didn't give where no local maximum or
    /// deepest point is read: the box filter's halo and the one row beyond it.
    /// </summary>
    internal const int HaloRows = SmoothRows / 2 + 1;
    /// <summary>Of H, positive inward: the depth at which a submalar hollow is present.</summary>
    public const double SubmalarMinDepth = 0.003;
    /// <summary>|sin| of the canthi-alar angle below which the vanishing-point solve is singular: a numerical guard only.</summary>
    public const double VanishMinSin = 1e-6;
    /// <summary>lateral._process_image's foreground threshold; the edge is interpolated at this alpha.</summary>
    public const int MaskAlpha = 32;
    /// <summary>Px on either side of the closed mask's edge pixel where the pair that brackets <see cref="MaskAlpha"/> is looked for.</summary>
    public const int AlphaSearch = 2;
    /// <summary>
    /// The second edge source: where the mask walk finds no edge, the strongest horizontal
    /// intensity edge in the same window stands in when the raw 3 x 3 Sobel response of the
    /// binomially blurred 8-bit gray image reaches this.
    /// </summary>
    public const double EdgeGradMin = 40.0;
    /// <summary>The edge source is off on rows whose plain row ratio (chin row to canthus row) is above this.</summary>
    public const double EdgeMaxH = 0.95;
    /// <summary>The gray image's weights: ITU-R BT.601 luma in OpenCV's fixed point, R, G and B over 2^<see cref="GrayShift"/>.</summary>
    internal static readonly (int R, int G, int B) GrayWeights = (4899, 9617, 1868);
    internal const int GrayShift = 14;
    /// <summary>A row kept at the contour seed.</summary>
    public const byte OutlineSrcContour = 0;
    /// <summary>A row traced on the foreground mask.</summary>
    public const byte OutlineSrcMask = 1;
    /// <summary>A mask row the nose disturbed, kept at the contour seed.</summary>
    public const byte OutlineSrcNose = 2;
    /// <summary>
    /// The intensity edge stood in where the mask had none: drawn and flagged, but never a
    /// mask row for the gates and never a row an extremum is read on.
    /// </summary>
    public const byte OutlineSrcEdge = 3;
    /// <summary>The separable 5-tap binomial blur and the 3 x 3 Sobel's row weights.</summary>
    internal static readonly double[] Binomial5 = { 1.0, 4.0, 6.0, 4.0, 1.0 };
    internal static readonly double[] SobelRows = { 1.0, 2.0, 1.0 };

    static IReadOnlyList<int> Range(int start, int stop, int step)
    {
        var list = new List<int>();
        for (int i = start; step > 0 ? i < stop : i > stop; i += step) list.Add(i);
        return list.AsReadOnly();
    }

    // ------------------------------------------------------------------ landmarks

    /// <summary>
    /// The landmarks as doubles, or null when there are fewer than
    /// <see cref="NumLandmarks"/> or any coordinate isn't finite: the one check every
    /// cheek construction and the oblique search start with, so a NaN chin or an
    /// infinite canthus gives no point. A double array that passes is returned as is.
    /// </summary>
    public static Vec2d[]? AsLandmarks(Vec2d[]? landmarks)
    {
        if (landmarks == null || landmarks.Length < NumLandmarks) return null;
        foreach (var p in landmarks)
            if (!p.IsFinite) return null;
        return landmarks;
    }

    /// <summary>The float landmarks widened to doubles (exactly), checked as <see cref="AsLandmarks(Vec2d[])"/> checks them.</summary>
    public static Vec2d[]? AsLandmarks(Vec2[]? landmarks) => AsLandmarks(ToVec2d(landmarks));

    /// <summary>The float landmarks widened to doubles, exactly; null stays null.</summary>
    internal static Vec2d[]? ToVec2d(Vec2[]? landmarks)
    {
        if (landmarks == null) return null;
        var pts = new Vec2d[landmarks.Length];
        for (int i = 0; i < landmarks.Length; i++) pts[i] = landmarks[i];
        return pts;
    }

    /// <summary>The unit vector along v, or null when v is zero or not finite.</summary>
    internal static Vec2d? Unit(Vec2d v)
    {
        double n = Hypot(v.X, v.Y);
        if (n == 0.0 || !IsFinite(n)) return null;
        return v / n;
    }

    /// <summary>The z component of a x b.</summary>
    internal static double Cross(Vec2d a, Vec2d b) => Vec2d.Cross(a, b);

    static (float[] X, int[] Y, byte[] Src) EmptyOutline() =>
        (Array.Empty<float>(), Array.Empty<int>(), Array.Empty<byte>());

    /// <summary>
    /// The flagged rows widened by width rows on each side: where the box filter's
    /// window reaches a flagged row, and the row beyond.
    /// </summary>
    internal static bool[] Halo(bool[] rows, int width = HaloRows)
    {
        var output = new bool[rows.Length];
        for (int i = 0; i < rows.Length; i++)
        {
            if (!rows[i]) continue;
            int from = Math.Max(i - width, 0), to = Math.Min(i + width + 1, rows.Length);
            for (int k = from; k < to; k++) output[k] = true;
        }
        return output;
    }

    // ------------------------------------------------------------------ gray image

    /// <summary>
    /// The 8-bit gray image the second edge source reads (<c>gray</c> of
    /// <see cref="TraceFarCheek(bool[,], byte[,], Vec2d[], bool, double, byte[,])"/>):
    /// ITU-R BT.601 luma with OpenCV's fixed-point weights,
    /// (R * 4899 + G * 9617 + B * 1868 + 8192) >> 14. The library defines it itself so
    /// both ports trace the same edge rows from one rule.
    /// </summary>
    /// <param name="rgb">The image, <c>[row, column, channel]</c> top-left, channels R, G, B (a fourth, alpha, is ignored).</param>
    /// <returns>The gray image, <c>[row, column]</c> top-left.</returns>
    public static byte[,] GrayImage(byte[,,] rgb)
    {
        if (rgb == null) throw new ArgumentNullException(nameof(rgb));
        if (rgb.GetLength(2) < 3) throw new ArgumentException("The image needs R, G and B channels.", nameof(rgb));
        int h = rgb.GetLength(0), w = rgb.GetLength(1);
        var gray = new byte[h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                gray[y, x] = Luma(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2]);
        return gray;
    }

    /// <summary>
    /// <see cref="GrayImage(byte[,,])"/> of a <see cref="FaceImage"/>, whose bottom-left
    /// pixels it turns over: the result is <c>[row, column]</c> top-left, row 0 the
    /// image's top, as the tracer reads it.
    /// </summary>
    public static byte[,] GrayImage(FaceImage image)
    {
        if (!image.IsValid) throw new ArgumentException("The image has no pixels.", nameof(image));
        int h = image.Height, w = image.Width;
        var gray = new byte[h, w];
        for (int y = 0; y < h; y++)
        {
            int source = (h - 1 - y) * w;
            for (int x = 0; x < w; x++)
            {
                Rgba32 p = image.Pixels[source + x];
                gray[y, x] = Luma(p.R, p.G, p.B);
            }
        }
        return gray;
    }

    static byte Luma(byte r, byte g, byte b) =>
        (byte)((r * GrayWeights.R + g * GrayWeights.G + b * GrayWeights.B + (1 << (GrayShift - 1))) >> GrayShift);

    // ------------------------------------------------------------------ smoothing

    /// <summary>
    /// A box filter of n rows on an edge-padded array (np.convolve), the same length as
    /// the input. Each output accumulates its n products with a fused multiply-add from
    /// the first to the last, as the BLAS dot product under the reference build's
    /// np.convolve does.
    /// </summary>
    internal static double[] Smooth(double[] values, int n = SmoothRows)
    {
        if (n <= 1 || values.Length == 0) return (double[])values.Clone();
        int left = n / 2, len = values.Length;
        double k = 1.0 / n;
        var output = new double[len];
        for (int i = 0; i < len; i++)
        {
            double s = 0.0;
            for (int j = 0; j < n; j++)
            {
                int src = i + j - left;
                double v = values[src < 0 ? 0 : src >= len ? len - 1 : src];
                s = Fma(v, k, s);
            }
            output[i] = s;
        }
        return output;
    }

    // ------------------------------------------------------------------ the seed

    /// <summary>
    /// The far half contour's x in each row, built segment by segment in polyline order
    /// (the cross_polyline rule the frontal core uses): a row crossed by several segments
    /// takes the outward-most crossing; rows above the first point extrapolate the first
    /// segment. No sort and no interpolation table: integer landmarks give duplicate rows
    /// and the chin reverses, where a sort's order would be implementation-defined.
    /// A flat segment (two points on one row) offers both its ends to that row; the first
    /// and last segments that are not flat extrapolate the rows above the first point and
    /// below the last. <see cref="ContourOutline(Vec2d[], bool)"/> uses this same routine.
    /// </summary>
    /// <param name="cont">The far half contour, (17) px, in polyline order (<see cref="RightFarContour"/> or <see cref="LeftFarContour"/>).</param>
    /// <param name="rows">The rows to seed, top to bottom.</param>
    /// <param name="farLeft">Whether the far cheek is the subject's left, so outward is +x; null reads it from the polyline (its first point lies lateral of its last, the chin).</param>
    /// <returns>The seed x of each row.</returns>
    public static double[] ContourSeed(Vec2d[] cont, double[] rows, bool? farLeft = null)
    {
        if (cont == null) throw new ArgumentNullException(nameof(cont));
        if (rows == null) throw new ArgumentNullException(nameof(rows));
        int n = rows.Length;
        var xs = new double[n];
        for (int i = 0; i < n; i++) xs[i] = double.NaN;
        if (n == 0 || cont.Length == 0) return xs;
        if (cont.Length == 1)
        {
            for (int i = 0; i < n; i++) xs[i] = cont[0].X;
            return xs;
        }
        double outward = farLeft == null
            ? (cont[0].X >= cont[cont.Length - 1].X ? 1.0 : -1.0)
            : (farLeft.Value ? 1.0 : -1.0);
        int? first = null, last = null;
        for (int k = 0; k < cont.Length - 1; k++)
        {
            double x0 = cont[k].X, y0 = cont[k].Y, x1 = cont[k + 1].X, y1 = cont[k + 1].Y;
            if (y0 == y1)
            {
                // A flat segment: the row takes the outward-most of its two ends.
                double xFlat = (x0 - x1) * outward >= 0 ? x0 : x1;
                for (int i = 0; i < n; i++)
                {
                    if (rows[i] != y0) continue;
                    double cur = xs[i];
                    if (double.IsNaN(cur) || (xFlat - cur) * outward > 0) xs[i] = xFlat;
                }
                continue;
            }
            first ??= k;
            last = k;
            double lo = y0 < y1 ? y0 : y1, hi = y0 < y1 ? y1 : y0;
            for (int i = 0; i < n; i++)
            {
                double r = rows[i];
                if (!(r >= lo && r <= hi)) continue;
                double x = x0 + (r - y0) * (x1 - x0) / (y1 - y0);
                double cur = xs[i];
                if (double.IsNaN(cur) || (x - cur) * outward > 0) xs[i] = x;
            }
        }
        bool anyMissing = false;
        for (int i = 0; i < n; i++) anyMissing |= double.IsNaN(xs[i]);
        if (!anyMissing) return xs;
        if (first == null || last == null)
        {
            // Every point on one row: that row's outward-most x everywhere.
            var lateral = new double[cont.Length];
            for (int i = 0; i < cont.Length; i++) lateral[i] = (cont[i].X - cont[0].X) * outward;
            double xMost = cont[ArgMax(lateral)].X;
            for (int i = 0; i < n; i++)
                if (double.IsNaN(xs[i])) xs[i] = xMost;
            return xs;
        }
        var missing = new bool[n];
        for (int i = 0; i < n; i++) missing[i] = double.IsNaN(xs[i]);
        for (int pass = 0; pass < 2; pass++)
        {
            int k = pass == 0 ? first.Value : last.Value;
            double x0 = cont[k].X, y0 = cont[k].Y, x1 = cont[k + 1].X, y1 = cont[k + 1].Y;
            for (int i = 0; i < n; i++)
            {
                if (!missing[i]) continue;
                bool above = rows[i] < cont[0].Y;
                if (above != (pass == 0)) continue;
                xs[i] = x0 + (rows[i] - y0) * (x1 - x0) / (y1 - y0);
            }
        }
        return xs;
    }

    /// <summary><see cref="ContourSeed(Vec2d[], double[], bool?)"/> of integer rows.</summary>
    public static double[] ContourSeed(Vec2d[] cont, int[] rows, bool? farLeft = null)
    {
        if (rows == null) throw new ArgumentNullException(nameof(rows));
        var r = new double[rows.Length];
        for (int i = 0; i < rows.Length; i++) r[i] = rows[i];
        return ContourSeed(cont, r, farLeft);
    }

    /// <summary>
    /// The rows of the far-cheek band, from the far canthus row or the far contour's top
    /// row (whichever is higher) to the chin row, and the contour seed of each.
    /// </summary>
    internal static (int[] Rows, double[] Seed) BandRows(Vec2d[] landmarks, bool farLeft)
    {
        var idx = farLeft ? LeftFarContour : RightFarContour;
        var cont = new Vec2d[idx.Count];
        for (int i = 0; i < idx.Count; i++) cont[i] = landmarks[idx[i]];
        int canthusRow = (int)RoundHalfEven(landmarks[FarCanthus(farLeft)].Y);
        int top = Math.Min(canthusRow, (int)RoundHalfEven(cont[0].Y));
        int chinRow = (int)RoundHalfEven(landmarks[LmChin].Y);
        int count = Math.Max(chinRow + 1 - top, 0);
        var rows = new int[count];
        for (int i = 0; i < count; i++) rows[i] = top + i;
        return (rows, ContourSeed(cont, rows, farLeft));
    }

    // ------------------------------------------------------------------ the crossing

    /// <summary>
    /// Where the alpha crosses <see cref="MaskAlpha"/> between a foreground pixel and the
    /// adjacent background pixel, interpolated between their centers; the pixel boundary
    /// when their alphas don't bracket it (a pixel the closing filled in).
    /// </summary>
    internal static double AlphaCrossing(byte[] alphaRow, int xFg, int xBg)
    {
        double aFg = alphaRow[xFg], aBg = alphaRow[xBg];
        double t = aFg > MaskAlpha && MaskAlpha >= aBg ? (aFg - MaskAlpha) / (aFg - aBg) : 0.5;
        return xFg + t * (xBg - xFg);
    }

    /// <summary>
    /// The sub-pixel edge of a row whose closed mask ends at the foreground pixel xFg, the
    /// background beginning at xFg + outward: where the raw alpha crosses
    /// <see cref="MaskAlpha"/> nearest that edge. The mask's own pair is tried first, then
    /// the pairs one and two pixels (<see cref="AlphaSearch"/>) inward and outward of it,
    /// nearest first and inward before outward; the first that brackets MaskAlpha gives
    /// the crossing (<see cref="AlphaCrossing"/>). When none does, the mask's pixel
    /// boundary xFg + outward / 2.
    /// </summary>
    internal static double RefinedCrossing(byte[] alphaRow, int xFg, int outward)
    {
        int width = alphaRow.Length;
        for (int distance = 0; distance <= AlphaSearch; distance++)
        {
            for (int pick = 0; pick < (distance == 0 ? 1 : 2); pick++)
            {
                int k = distance == 0 ? 0 : (pick == 0 ? -distance : distance);
                int p = xFg + k * outward;
                int q = p + outward;
                if (Math.Min(p, q) < 0 || Math.Max(p, q) >= width) continue;
                if (alphaRow[p] > MaskAlpha && MaskAlpha >= alphaRow[q])
                    return AlphaCrossing(alphaRow, p, q);
            }
        }
        return xFg + 0.5 * outward;
    }

    // ------------------------------------------------------------------ the edge source

    /// <summary>The separable 5-tap binomial blur ([1, 4, 6, 4, 1] / 16), edge-padded.</summary>
    internal static double[,] Blur5(double[,] a)
    {
        int h = a.GetLength(0), w = a.GetLength(1);
        // across[r, c] for the padded rows r = -2 .. h + 1 (stored at r + 2).
        var across = new double[h + 4, w];
        for (int r = 0; r < h + 4; r++)
        {
            int sr = Clamp(r - 2, h);
            for (int c = 0; c < w; c++)
            {
                double s = 0.0;
                for (int j = 0; j < 5; j++) s += Binomial5[j] * a[sr, Clamp(c + j - 2, w)];
                across[r, c] = s;
            }
        }
        var down = new double[h, w];
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
            {
                double s = 0.0;
                for (int j = 0; j < 5; j++) s += Binomial5[j] * across[r + j, c];
                down[r, c] = s / 256.0;
            }
        return down;
    }

    /// <summary>The 3 x 3 horizontal Sobel response ([-1, 0, 1] across, [1, 2, 1] down), edge-padded.</summary>
    internal static double[,] SobelX(double[,] a)
    {
        int h = a.GetLength(0), w = a.GetLength(1);
        var d = new double[h + 2, w];
        for (int r = 0; r < h + 2; r++)
        {
            int sr = Clamp(r - 1, h);
            for (int c = 0; c < w; c++) d[r, c] = a[sr, Clamp(c + 1, w)] - a[sr, Clamp(c - 1, w)];
        }
        var g = new double[h, w];
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
            {
                double s = 0.0;
                for (int j = 0; j < 3; j++) s += SobelRows[j] * d[r + j, c];
                g[r, c] = s;
            }
        return g;
    }

    static int Clamp(int i, int length) => i < 0 ? 0 : i >= length ? length - 1 : i;

    /// <summary>
    /// The second edge source: on each failed row whose plain row ratio is at most
    /// <see cref="EdgeMaxH"/>, the strongest horizontal gradient of the blurred gray image
    /// within limit px of the seed, when it reaches <see cref="EdgeGradMin"/>, refined to
    /// sub-pixel x by the parabola through its two neighbors, as <see cref="OutlineSrcEdge"/>.
    /// </summary>
    static void EdgeRows(byte[,] gray, int[] rows, double[] seed, double[] xs, byte[] src,
        List<int> failed, int limit, double[] hPlain)
    {
        int height = gray.GetLength(0), width = gray.GetLength(1);
        var use = new List<int>();
        foreach (int i in failed)
            if (hPlain[i] <= EdgeMaxH && 0 <= rows[i] && rows[i] < height && IsFinite(seed[i]))
                use.Add(i);
        if (use.Count == 0 || limit <= 0) return;
        double sMin = double.PositiveInfinity, sMax = double.NegativeInfinity;
        foreach (int i in use)
        {
            if (seed[i] < sMin) sMin = seed[i];
            if (seed[i] > sMax) sMax = seed[i];
        }
        int rLo = Math.Max(rows[use[0]] - 3, 0);
        int rHi = Math.Min(rows[use[use.Count - 1]] + 3, height - 1);
        // In double until they are known to lie in the image, so a far seed can't overflow.
        double cLoD = Math.Max(Math.Floor(sMin) - limit - 3, 0);
        double cHiD = Math.Min(Math.Ceiling(sMax) + limit + 3, width - 1);
        if (rHi < rLo || cHiD < cLoD) return;
        int cLo = (int)cLoD, cHi = (int)cHiD;
        var band = new double[rHi - rLo + 1, cHi - cLo + 1];
        for (int r = rLo; r <= rHi; r++)
            for (int c = cLo; c <= cHi; c++) band[r - rLo, c - cLo] = gray[r, c];
        double[,] grad = SobelX(Blur5(band));
        foreach (int i in use)
        {
            double x0 = RoundHalfEven(seed[i]);
            // x0 +/- limit clipped to the band's columns, in double so a far seed can't overflow.
            double loD = Math.Max(x0 - limit, cLo), hiD = Math.Min(x0 + limit, cHi);
            if (hiD < loD) continue;
            int lo = (int)loD, hi = (int)hiD;
            int r = rows[i] - rLo;
            int len = hi - lo + 1;
            var g = new double[len];
            for (int k = 0; k < len; k++) g[k] = Math.Abs(grad[r, lo - cLo + k]);
            int j = ArgMax(g);
            if (g[j] < EdgeGradMin) continue;
            double offset = 0.0;
            if (0 < j && j < len - 1)
            {
                double denom = g[j - 1] - 2.0 * g[j] + g[j + 1];
                if (denom < 0.0)
                    offset = Math.Min(Math.Max(0.5 * (g[j - 1] - g[j + 1]) / denom, -0.5), 0.5);
            }
            xs[i] = (lo + j) + offset;
            src[i] = OutlineSrcEdge;
        }
    }

    // ------------------------------------------------------------------ the tracer

    /// <summary>
    /// The far cheek's silhouette row by row, from the far canthus row or the far contour's
    /// top row (landmark 0, or 32), whichever is higher, to the chin row (16), seeded and
    /// bounded by that half contour.
    ///
    /// <para>Key lines (dynaface-lib's trace_far_cheek):
    /// <code>
    /// idx = farLeft ? LeftFarContour (32..16) : RightFarContour (0..16)
    /// cont = L[idx]; top = min(row of L[FarCanthus(farLeft)], row of cont[0])
    /// rows = top .. chin row; seed = ContourSeed(cont, rows)
    /// span = (L[16] - L[FarCanthus(farLeft)]) · ObliqueFrameOf(L).Down   // the tracer's own scale
    /// limit = (int)(tol * span); outward = farLeft ? +1 : -1
    /// per row: start at the seed's pixel (clipped to the image); walk outward while
    ///     foreground, or inward while background, up to limit, never past the border;
    ///     an edge found within limit gives the alpha == MaskAlpha crossing
    ///     (RefinedCrossing) and OutlineSrcMask, else the row keeps the seed
    ///     (OutlineSrcContour), unless gray gives an intensity edge (OutlineSrcEdge)
    /// nose = mask rows between the levels of landmarks 54 and 57 whose outward
    ///     departure from the seed exceeds the median departure of the mask rows with
    ///     h in [sto, 1.0] outside that band by NoseStep * span, extended upward from
    ///     the tip's row while that holds; they keep the seed as OutlineSrcNose
    /// </code></para>
    ///
    /// <para>The levels of 54 and 57 and the stomion's are read as h along the
    /// roll-corrected vertical (<see cref="HeightRatio"/> of each row's seed point), so a
    /// rolled head keeps the same rows. Rows outside the image keep the seed. The edge
    /// source reads its heights on image rows, from its own row span; the nose rows use
    /// the tracer's span along the vertical.</para>
    /// </summary>
    /// <param name="foreground">The closed foreground mask of the standard image, <c>[row, column]</c> top-left (lateral.foreground_mask()).</param>
    /// <param name="alpha">The raw alpha it was thresholded from, <c>[row, column]</c> top-left.</param>
    /// <param name="landmarks">The 98 WFLW landmarks, top-left px of the same image.</param>
    /// <param name="farLeft">Whether the far cheek is the subject's left.</param>
    /// <param name="tol">How far the walk may go from the seed, as a fraction of the tracer's span.</param>
    /// <param name="gray">The gray image, <c>[row, column]</c> top-left, from <see cref="GrayImage(byte[,,])"/>, for the second edge source on the rows the mask walk fails on; null keeps the seed there.</param>
    /// <returns>
    /// The outline's x in each row (sub-pixel, float as NumPy's float32), the rows, and each
    /// row's source (OutlineSrc*), in that order: AnalyzeFace.cheek_outline_x, _y and _src.
    /// Three empty arrays without valid landmarks, a frame or band rows.
    /// </returns>
    public static (float[] X, int[] Y, byte[] Src) TraceFarCheek(
        bool[,] foreground, byte[,] alpha, Vec2d[]? landmarks, bool farLeft,
        double tol = OutlineTol, byte[,]? gray = null)
    {
        if (foreground == null) throw new ArgumentNullException(nameof(foreground));
        if (alpha == null) throw new ArgumentNullException(nameof(alpha));
        Vec2d[]? pts = AsLandmarks(landmarks);
        if (pts == null) return EmptyOutline();
        ObliqueFrame? frame = ObliqueFrameOf(pts);
        if (frame == null) return EmptyOutline();
        var (rows, seed) = BandRows(pts, farLeft);
        int n = rows.Length;
        if (n == 0) return EmptyOutline();
        int height = foreground.GetLength(0), width = foreground.GetLength(1);
        Vec2d chin = pts[LmChin], canthus = pts[FarCanthus(farLeft)];
        // The tracer's span: the two landmarks' distance along the vertical, which
        // NoseStep and OutlineTol are fractions of (not ObliqueCheek.Scale).
        double span = Vec2d.Dot(chin - canthus, frame.Down);
        int limit = span > 0.0 ? (int)(tol * span) : 0;
        int outward = farLeft ? 1 : -1;
        var xs = (double[])seed.Clone();
        var src = new byte[n]; // OutlineSrcContour
        var failed = new List<int>();
        int alphaWidth = alpha.GetLength(1);
        var alphaRow = new byte[alphaWidth];
        for (int i = 0; i < n; i++)
        {
            int y = rows[i];
            if (y < 0 || y >= height || !IsFinite(seed[i])) continue;
            double start = RoundHalfEven(seed[i]);
            int x = start < 0 ? 0 : start > width - 1 ? width - 1 : (int)start;
            bool inside = foreground[y, x];
            int step = inside ? outward : -outward;
            int foundFg = -1;
            bool found = false;
            for (int k = 0; k < limit; k++)
            {
                int nx = x + step;
                if (nx < 0 || nx >= width) break; // the border: keep the seed
                if (foreground[y, nx] != inside)
                {
                    foundFg = inside ? x : nx;
                    found = true;
                    break;
                }
                x = nx;
            }
            if (!found)
            {
                failed.Add(i);
                continue;
            }
            Buffer.BlockCopy(alpha, y * alphaWidth, alphaRow, 0, alphaWidth);
            xs[i] = RefinedCrossing(alphaRow, foundFg, outward);
            src[i] = OutlineSrcMask;
        }
        if (gray != null && failed.Count > 0)
        {
            double chinRow = chin.Y, canthusRow = canthus.Y;
            // The edge source's height is read on image rows, from its own span: the
            // tracer's span above stays what NoseStep is a fraction of below.
            double rowSpan = chinRow - canthusRow;
            var hPlain = new double[n];
            for (int i = 0; i < n; i++)
                hPlain[i] = rowSpan > 0.0 ? (chinRow - rows[i]) / rowSpan : double.PositiveInfinity;
            EdgeRows(gray, rows, seed, xs, src, failed, limit, hPlain);
        }
        // The nose rows: the mask rows between the tip's and the subnasale's levels where
        // the traced edge departs outward from the seed by NoseStep beyond the band's
        // reference, and upward from the tip's row while it keeps doing so.
        var seedPts = new Vec2d[n];
        for (int i = 0; i < n; i++) seedPts[i] = new Vec2d(seed[i], rows[i]);
        var (hRows, _) = HeightRatios(seedPts, chin, canthus, frame);
        double hTip = HeightRatio(pts[NoseRows.Tip], chin, canthus, frame).H;
        double hSub = HeightRatio(pts[NoseRows.Subnasale], chin, canthus, frame).H;
        Vec2d stomion = 0.5 * (pts[LmStomion.Upper] + pts[LmStomion.Lower]);
        double hSto = HeightRatio(stomion, chin, canthus, frame).H;
        if (IsFinite(hTip) && IsFinite(hSub) && span > 0.0)
        {
            bool[] nose = FindNoseRows(xs, seed, src, hRows, outward, span, hTip, hSub, hSto);
            for (int i = 0; i < n; i++)
            {
                if (!nose[i]) continue;
                xs[i] = seed[i];
                src[i] = OutlineSrcNose;
            }
        }
        var xf = new float[n];
        for (int i = 0; i < n; i++) xf[i] = (float)xs[i];
        return (xf, rows, src);
    }

    /// <summary><see cref="TraceFarCheek(bool[,], byte[,], Vec2d[], bool, double, byte[,])"/> of float landmarks.</summary>
    public static (float[] X, int[] Y, byte[] Src) TraceFarCheek(
        bool[,] foreground, byte[,] alpha, Vec2[]? landmarks, bool farLeft,
        double tol = OutlineTol, byte[,]? gray = null) =>
        TraceFarCheek(foreground, alpha, ToVec2d(landmarks), farLeft, tol, gray);

    /// <summary>
    /// The nose rows of a traced outline: mask rows in the nose band (h between the tip's
    /// and the subnasale's levels) whose outward departure from the seed exceeds the
    /// reference by more than NoseStep * span, the band extended upward from its top row
    /// while consecutive rows keep departing so. The reference is the median departure of
    /// the mask rows with h in [hSto, 1.0] outside the band, 0.0 when there are none.
    /// Contour and intensity-edge rows are never nose rows.
    /// </summary>
    static bool[] FindNoseRows(double[] xs, double[] seed, byte[] src, double[] hRows,
        int outward, double span, double hTip, double hSub, double hSto)
    {
        int n = xs.Length;
        double loH = hSub < hTip ? hSub : hTip, hiH = hSub > hTip ? hSub : hTip;
        var isMask = new bool[n];
        var outwardPx = new double[n];
        var inBand = new bool[n];
        var cheek = new List<double>();
        for (int i = 0; i < n; i++)
        {
            isMask[i] = src[i] == OutlineSrcMask;
            outwardPx[i] = (xs[i] - seed[i]) * outward;
            inBand[i] = hRows[i] >= loH && hRows[i] <= hiH;
            if (isMask[i] && !inBand[i] && hRows[i] >= hSto && hRows[i] <= 1.0) cheek.Add(outwardPx[i]);
        }
        double reference = cheek.Count > 0 ? Median(cheek) : 0.0;
        var depart = new bool[n];
        var nose = new bool[n];
        int bandFirst = -1;
        for (int i = 0; i < n; i++)
        {
            depart[i] = isMask[i] && outwardPx[i] - reference > NoseStep * span;
            nose[i] = depart[i] && inBand[i];
            if (inBand[i] && bandFirst < 0) bandFirst = i;
        }
        if (bandFirst >= 0 && depart[bandFirst])
            for (int i = bandFirst - 1; i >= 0 && depart[i]; i--) nose[i] = true;
        return nose;
    }

    /// <summary>
    /// The fallback: <see cref="ContourSeed(Vec2d[], int[], bool?)"/> over the same rows as
    /// <see cref="TraceFarCheek(bool[,], byte[,], Vec2d[], bool, double, byte[,])"/>, all
    /// <see cref="OutlineSrcContour"/>; empty without valid landmarks.
    /// </summary>
    public static (float[] X, int[] Y, byte[] Src) ContourOutline(Vec2d[]? landmarks, bool farLeft)
    {
        Vec2d[]? pts = AsLandmarks(landmarks);
        if (pts == null) return EmptyOutline();
        var (rows, seed) = BandRows(pts, farLeft);
        var xf = new float[rows.Length];
        for (int i = 0; i < rows.Length; i++) xf[i] = (float)seed[i];
        return (xf, rows, new byte[rows.Length]);
    }

    /// <summary><see cref="ContourOutline(Vec2d[], bool)"/> of float landmarks.</summary>
    public static (float[] X, int[] Y, byte[] Src) ContourOutline(Vec2[]? landmarks, bool farLeft) =>
        ContourOutline(ToVec2d(landmarks), farLeft);

    // ------------------------------------------------------------------ the level frame

    /// <summary>The level frame of a quarter face: where its horizontals converge.</summary>
    public sealed class ObliqueFrame
    {
        /// <summary>Unit normal of the canthi line (60 -> 72), toward the chin: the roll-corrected vertical.</summary>
        public Vec2d Down { get; }
        /// <summary>Where the canthi line and the alar-base line (55 -> 59) meet, px, or null when the solve is singular (<see cref="VanishMinSin"/>).</summary>
        public Vec2d? Vanishing { get; }
        /// <summary>The mean direction of the two lines, used only when the solve is singular.</summary>
        public Vec2d MeanDir { get; }

        public ObliqueFrame(Vec2d down, Vec2d? vanishing, Vec2d meanDir)
        {
            Down = down;
            Vanishing = vanishing;
            MeanDir = meanDir;
        }
    }

    /// <summary>
    /// The level frame of the 98 landmarks (dynaface-lib's oblique_frame): down from the
    /// canthi line, the vanishing point of the canthi and alar-base lines (null when |sin|
    /// of their angle is below <see cref="VanishMinSin"/>) and their mean direction.
    /// </summary>
    /// <returns>The frame, or null when the canthi coincide or the landmarks aren't valid.</returns>
    public static ObliqueFrame? ObliqueFrameOf(Vec2d[]? landmarks)
    {
        Vec2d[]? pts = AsLandmarks(landmarks);
        if (pts == null) return null;
        Vec2d canthusR = pts[LmCanthusR], canthusL = pts[LmCanthusL];
        Vec2d? acrossOrNull = Unit(canthusL - canthusR);
        if (acrossOrNull == null) return null;
        Vec2d across = acrossOrNull.Value;
        var down = new Vec2d(-across.Y, across.X);
        if (Vec2d.Dot(pts[LmChin] - 0.5 * (canthusR + canthusL), down) < 0.0) down = -down;
        Vec2d? alarOrNull = Unit(pts[LmAlaL] - pts[LmAlaR]);
        Vec2d? vanishing = null;
        Vec2d meanDir = across;
        if (alarOrNull != null)
        {
            Vec2d alar = alarOrNull.Value;
            if (Vec2d.Dot(across, alar) < 0.0) alar = -alar;
            double sin = Cross(across, alar);
            if (Math.Abs(sin) >= VanishMinSin)
            {
                // canthus_r + s * across = ala_r + t * alar: cross both sides with alar.
                double s = Cross(pts[LmAlaR] - canthusR, alar) / sin;
                vanishing = canthusR + s * across;
            }
            Vec2d? mean = Unit(across + alar);
            if (mean != null) meanDir = mean.Value;
        }
        return new ObliqueFrame(down, vanishing, meanDir);
    }

    /// <summary><see cref="ObliqueFrameOf(Vec2d[])"/> of float landmarks.</summary>
    public static ObliqueFrame? ObliqueFrameOf(Vec2[]? landmarks) => ObliqueFrameOf(ToVec2d(landmarks));

    /// <summary>
    /// The level line through anchor: from the vanishing point through it, else along
    /// <see cref="ObliqueFrame.MeanDir"/>.
    /// </summary>
    /// <returns>The line as a point (the anchor) and a unit direction.</returns>
    public static (Vec2d Point, Vec2d Direction) LevelThrough(ObliqueFrame frame, Vec2d anchor)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        if (frame.Vanishing != null)
        {
            Vec2d? direction = Unit(anchor - frame.Vanishing.Value);
            if (direction != null) return (anchor, direction.Value);
        }
        return (anchor, frame.MeanDir);
    }

    /// <summary>
    /// Where the level meets the vertical (direction frame.Down) through each point: a 2x2
    /// solve per point; NaN when the level runs along the vertical.
    /// </summary>
    internal static Vec2d[] LevelsAtVerticals((Vec2d Point, Vec2d Direction) level, Vec2d[] points, ObliqueFrame frame)
    {
        var (point, direction) = level;
        Vec2d down = frame.Down;
        var met = new Vec2d[points.Length];
        // point + s * direction = p + t * down: cross both sides with down.
        double den = Cross(direction, down);
        if (Math.Abs(den) < 1e-12)
        {
            for (int i = 0; i < met.Length; i++) met[i] = Vec2d.NaN;
            return met;
        }
        for (int i = 0; i < points.Length; i++)
        {
            Vec2d b = points[i] - point;
            double s = (b.X * down.Y - b.Y * down.X) / den;
            met[i] = point + s * direction;
        }
        return met;
    }

    /// <summary>Where that level meets the vertical (direction frame.Down) through p: a 2x2 solve.</summary>
    public static Vec2d LevelAtVertical((Vec2d Point, Vec2d Direction) level, Vec2d p, ObliqueFrame frame) =>
        LevelsAtVerticals(level, new[] { p }, frame)[0];

    /// <summary><see cref="HeightRatio"/> of every point at once: h and H of each, NaN where H is zero.</summary>
    internal static (double[] H, double[] Scale) HeightRatios(Vec2d[] points, Vec2d chin, Vec2d farCanthus, ObliqueFrame frame)
    {
        Vec2d[] chinPts = LevelsAtVerticals(LevelThrough(frame, chin), points, frame);
        Vec2d[] canPts = LevelsAtVerticals(LevelThrough(frame, farCanthus), points, frame);
        Vec2d up = -frame.Down;
        var h = new double[points.Length];
        var scale = new double[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            scale[i] = Vec2d.Dot(canPts[i] - chinPts[i], up);
            double rise = Vec2d.Dot(points[i] - chinPts[i], up);
            h[i] = scale[i] != 0.0 ? rise / scale[i] : double.NaN;
        }
        return (h, scale);
    }

    /// <summary>
    /// Kaptein's h at p and H on p's vertical:
    /// <code>
    /// chin_pt = LevelAtVertical(LevelThrough(frame, chin), p, frame)
    /// can_pt = LevelAtVertical(LevelThrough(frame, farCanthus), p, frame)
    /// H = (can_pt - chin_pt) · -frame.Down;  h = (p - chin_pt) · -frame.Down / H
    /// </code>
    /// </summary>
    /// <returns>h, the chin-to-point over chin-to-canthus distance along the vertical at p, and H, that chin-to-canthus distance, px.</returns>
    public static (double H, double Scale) HeightRatio(Vec2d p, Vec2d chin, Vec2d farCanthus, ObliqueFrame frame)
    {
        var (h, scale) = HeightRatios(new[] { p }, chin, farCanthus, frame);
        return (h[0], scale[0]);
    }

    // ------------------------------------------------------------------ the search

    /// <summary>
    /// The observed points of a quarter face's far cheek, with what the measure reports and
    /// draws. Heights and depths are fractions of H (<see cref="Scale"/>), points are
    /// top-left pixels of the standard image.
    /// </summary>
    public sealed class ObliqueCheek
    {
        /// <summary>Whether the far cheek is the subject's left.</summary>
        public bool FarLeft { get; set; }
        /// <summary>The band rows, px from the canthus level to the stomion level, for drawing.</summary>
        public Vec2d[] Outline { get; set; } = Array.Empty<Vec2d>();
        /// <summary>Each band row's source (OutlineSrc*).</summary>
        public byte[] OutlineSrc { get; set; } = Array.Empty<byte>();
        /// <summary>The outline at the canthus level and at the stomion level.</summary>
        public (Vec2d Canthus, Vec2d Stomion) Chord { get; set; }
        /// <summary>"canthus", "stomion" and "chin", each a (point, direction) line from <see cref="LevelThrough"/>, for drawing.</summary>
        public IReadOnlyDictionary<string, (Vec2d Point, Vec2d Direction)> Levels { get; set; } =
            new Dictionary<string, (Vec2d Point, Vec2d Direction)>();
        /// <summary>The malar apex, px.</summary>
        public Vec2d Apex { get; set; }
        /// <summary>The apex's height ratio.</summary>
        public double ApexH { get; set; }
        /// <summary>The apex's outward deviation from the chord, a fraction of H.</summary>
        public double ApexDev { get; set; }
        /// <summary>Whether the prior stood in for a local maximum.</summary>
        public bool ApexPrior { get; set; }
        /// <summary>Whether the apex's h is within <see cref="ApexNearNoseH"/> of the nose tip's.</summary>
        public bool ApexNearNose { get; set; }
        /// <summary>
        /// The submalar point: the deepest row between the apex and the stomion level against
        /// the chord joining them, never either end; (NaN, NaN) when its depth is NaN.
        /// </summary>
        public Vec2d Submalar { get; set; } = Vec2d.NaN;
        /// <summary>The submalar point's height ratio; NaN when its depth is.</summary>
        public double SubmalarH { get; set; } = double.NaN;
        /// <summary>
        /// Its depth inward from that chord, a fraction of H, positive inward, so a convex
        /// cheek reads about zero or slightly negative; NaN when too few rows came from the
        /// mask.
        /// </summary>
        public double SubmalarDepth { get; set; } = double.NaN;
        /// <summary>The same depth in px.</summary>
        public double SubmalarDepthPx { get; set; } = double.NaN;
        /// <summary>
        /// 1.0 when the depth reaches <see cref="SubmalarMinDepth"/>, 0.0 below, NaN when the
        /// depth is. With 0.0 the point is still reported (wherever the outline comes nearest
        /// the chord), and the measure doesn't draw it.
        /// </summary>
        public double SubmalarPresent { get; set; } = double.NaN;
        /// <summary>Whether the nose joins the far outline.</summary>
        public bool NoseOn { get; set; }
        /// <summary>The signed gap from the nose tip to the outline at the tip's row, a fraction of H, positive inside the face.</summary>
        public double NoseGap { get; set; }
        /// <summary>The fraction of <see cref="OutlineSrcMask"/> rows in the band (an edge row counts as none).</summary>
        public double MaskFraction { get; set; }
        /// <summary>The fraction of mask rows in the apex window.</summary>
        public double WindowMask { get; set; }
        /// <summary>The fraction of mask rows from the apex down.</summary>
        public double BelowMask { get; set; }
        /// <summary>H, the chin-to-far-canthus distance along the far canthus's vertical, px.</summary>
        public double Scale { get; set; }
    }

    /// <summary>The unit normal of the chord a -> b whose x has the sign of outward.</summary>
    internal static Vec2d? OutwardNormal(Vec2d a, Vec2d b, double outward)
    {
        Vec2d? normalOrNull = Unit(new Vec2d(-(b.Y - a.Y), b.X - a.X));
        if (normalOrNull == null) return null;
        Vec2d normal = normalOrNull.Value;
        if (normal.X * outward < 0.0 || (normal.X == 0.0 && normal.Y * outward < 0.0)) normal = -normal;
        return normal;
    }

    /// <summary>
    /// The observed points of the far cheek (dynaface-lib's observed_points), or null when
    /// no frame, band, scale or chord can be formed: fewer than 98 landmarks or one that
    /// isn't finite, coincident canthi (no frame), an empty outline or one whose three
    /// arrays differ in length, fewer than 3 band rows between the canthus and stomion
    /// levels, a non-finite or non-positive H, or chord ends that coincide.
    ///
    /// <para>Key lines:
    /// <code>
    /// outline null: ContourOutline(L, farLeft)
    /// h, Hs = HeightRatio of every row; sto = h of the stomion level at each row's x
    /// band = (h &lt;= 1.0) &amp; (h &gt;= sto); P, S, Hh = rows, sources, h of the band
    /// a, b = P[argmin |Hh - 1.0|], P[argmin |Hh - sto|]     // the chord
    /// n = unit normal of (b - a) pointing outward
    /// dev = Smooth((P - a) · n) / Hs                         // outward, fraction of H
    /// nose_gap = (outline x - tip x) * outward at the tip's level / H
    /// nose_on = any nose row or nose_gap &lt;= 0
    /// unsteady = rows not from the mask widened by SmoothRows / 2 + 1 on each side
    /// excluded = nose rows widened the same way
    /// cand = local maxima of dev in the window [0.71, 0.83], dev &gt;= ApexMinDev, steady
    /// apex = cand nearest h = 0.793, else (the prior) the window's steady mask row
    ///        nearest it, else any row outside the nose rows' halo nearest it
    /// submalar = deepest steady row strictly between the apex and the band's last row
    ///        against that chord, read only when BelowMask &gt;= OutlineMinMask, the
    ///        apex is a mask row and there are more than 4 rows
    /// </code></para>
    /// </summary>
    /// <param name="landmarks">The 98 WFLW landmarks, top-left px of the standard image.</param>
    /// <param name="outlineX">The far cheek's outline (AnalyzeFace.cheek_outline_x), or null for the contour fallback.</param>
    /// <param name="outlineY">Its rows (cheek_outline_y).</param>
    /// <param name="outlineSrc">Each row's source (cheek_outline_src).</param>
    /// <param name="farLeft">Whether the far cheek is the subject's left.</param>
    public static ObliqueCheek? ObservedPoints(Vec2d[]? landmarks, float[]? outlineX, int[]? outlineY,
        byte[]? outlineSrc, bool farLeft)
    {
        Vec2d[]? pts = AsLandmarks(landmarks);
        if (pts == null) return null;
        ObliqueFrame? frame = ObliqueFrameOf(pts);
        if (frame == null) return null;
        if (outlineX == null || outlineY == null || outlineSrc == null)
            (outlineX, outlineY, outlineSrc) = ContourOutline(pts, farLeft);
        int count = outlineX.Length;
        if (count == 0 || count != outlineY.Length || count != outlineSrc.Length) return null;
        Vec2d chin = pts[LmChin], canthus = pts[FarCanthus(farLeft)];
        Vec2d stomion = 0.5 * (pts[LmStomion.Upper] + pts[LmStomion.Lower]);
        double outward = farLeft ? 1.0 : -1.0;
        var points = new Vec2d[count];
        for (int i = 0; i < count; i++) points[i] = new Vec2d(outlineX[i], outlineY[i]);
        var (h, scales) = HeightRatios(points, chin, canthus, frame);
        var levels = new Dictionary<string, (Vec2d Point, Vec2d Direction)>
        {
            ["canthus"] = LevelThrough(frame, canthus),
            ["stomion"] = LevelThrough(frame, stomion),
            ["chin"] = LevelThrough(frame, chin),
        };
        Vec2d[] stoPts = LevelsAtVerticals(levels["stomion"], points, frame);
        var (sto, _) = HeightRatios(stoPts, chin, canthus, frame);
        double scale = HeightRatio(canthus, chin, canthus, frame).Scale;
        var bandIdx = new List<int>();
        for (int i = 0; i < count; i++)
            if (h[i] <= 1.0 && h[i] >= sto[i] && IsFinite(h[i]) && IsFinite(sto[i])) bandIdx.Add(i);
        if (bandIdx.Count < 3 || !IsFinite(scale) || scale <= 0.0) return null;
        int m = bandIdx.Count;
        var P = new Vec2d[m];
        var S = new byte[m];
        var Hh = new double[m];
        var Hs = new double[m];
        var toCanthus = new double[m];
        var toStomion = new double[m];
        for (int k = 0; k < m; k++)
        {
            int i = bandIdx[k];
            P[k] = points[i];
            S[k] = outlineSrc[i];
            Hh[k] = h[i];
            Hs[k] = scales[i];
            toCanthus[k] = Math.Abs(h[i] - 1.0);
            toStomion[k] = Math.Abs(h[i] - sto[i]);
        }
        Vec2d a = P[ArgMin(toCanthus)];
        Vec2d b = P[ArgMin(toStomion)];
        Vec2d? normalOrNull = OutwardNormal(a, b, outward);
        if (normalOrNull == null) return null;
        Vec2d normal = normalOrNull.Value;
        var along = new double[m];
        for (int k = 0; k < m; k++) along[k] = Vec2d.Dot(P[k] - a, normal);
        double[] dev = Smooth(along);
        for (int k = 0; k < m; k++) dev[k] /= Hs[k];
        double hNose = HeightRatio(pts[NoseRows.Tip], chin, canthus, frame).H;
        // The signed tip-to-outline gap at the tip's level, positive with the tip inside
        // the face.
        var toNose = new double[count];
        for (int i = 0; i < count; i++) toNose[i] = Math.Abs(h[i] - hNose);
        int tipRow = ArgMin(toNose);
        double noseGap = ((double)outlineX[tipRow] - pts[NoseRows.Tip].X) * outward / scale;
        var noseRows = new bool[m];
        var isMask = new bool[m];
        var notMask = new bool[m];
        bool anyNose = false;
        for (int k = 0; k < m; k++)
        {
            noseRows[k] = S[k] == OutlineSrcNose;
            isMask[k] = S[k] == OutlineSrcMask;
            notMask[k] = !isMask[k];
            anyNose |= noseRows[k];
        }
        bool noseOn = anyNose || noseGap <= 0.0;
        // No local maximum or deepest point is read within the box filter's halo of a row
        // the mask didn't give; the prior keeps clear of the nose rows and prefers the
        // window's mask rows.
        bool[] unsteady = Halo(notMask);
        bool[] excluded = Halo(noseRows);
        var window = new bool[m];
        for (int k = 0; k < m; k++) window[k] = Hh[k] >= ApexHWindow.Low && Hh[k] <= ApexHWindow.High;
        int iApex = -1;
        double best = double.PositiveInfinity;
        for (int k = 1; k < m - 1; k++)
        {
            if (!(dev[k] > dev[k - 1] && dev[k] >= dev[k + 1] && dev[k] >= ApexMinDev && window[k] && !unsteady[k]))
                continue;
            // min(cand, key=...): the first candidate at the least distance.
            double d = Math.Abs(Hh[k] - ApexHPrior);
            if (iApex < 0 || d < best)
            {
                iApex = k;
                best = d;
            }
        }
        bool apexPrior;
        if (iApex >= 0)
        {
            apexPrior = false;
        }
        else
        {
            // The prior: of the window's mask rows outside the nose rows' halo, the one
            // nearest Kaptein's height; when no row of the window came from the mask, the
            // row nearest that height of any source but a nose row.
            var distance = new double[m];
            bool anyPreferred = false, allExcluded = true;
            for (int k = 0; k < m; k++)
            {
                distance[k] = Math.Abs(Hh[k] - ApexHPrior);
                anyPreferred |= window[k] && isMask[k] && !excluded[k];
                allExcluded &= excluded[k];
            }
            if (anyPreferred)
            {
                for (int k = 0; k < m; k++)
                    if (!(window[k] && isMask[k] && !excluded[k])) distance[k] = double.PositiveInfinity;
            }
            else if (!allExcluded)
            {
                for (int k = 0; k < m; k++)
                    if (excluded[k]) distance[k] = double.PositiveInfinity;
            }
            iApex = ArgMin(distance);
            apexPrior = true;
        }
        bool apexNearNose = Math.Abs(Hh[iApex] - hNose) <= ApexNearNoseH;
        int windowCount = 0, windowMaskCount = 0, maskCount = 0, belowMaskCount = 0;
        for (int k = 0; k < m; k++)
        {
            if (window[k]) { windowCount++; if (isMask[k]) windowMaskCount++; }
            if (isMask[k]) { maskCount++; if (k >= iApex) belowMaskCount++; }
        }
        double windowMask = windowCount > 0 ? (double)windowMaskCount / windowCount : 0.0;
        double belowMask = (double)belowMaskCount / (m - iApex);
        double maskFraction = (double)maskCount / m;
        // The submalar point: the deepest row below the apex against the chord from the
        // apex to the band's last row (the stomion level), between those two rows.
        int q = m - iApex;
        Vec2d submalar = Vec2d.NaN;
        double submalarH = double.NaN, depth = double.NaN, depthPx = double.NaN, present = double.NaN;
        Vec2d? inwardNormal = q > 1 ? OutwardNormal(P[iApex], P[m - 1], -outward) : null;
        // Read only below an apex the mask gave.
        if (belowMask >= OutlineMinMask && isMask[iApex] && q > 4 && inwardNormal != null)
        {
            Vec2d q0 = P[iApex];
            var depthAlong = new double[q];
            for (int k = 0; k < q; k++) depthAlong[k] = Vec2d.Dot(P[iApex + k] - q0, inwardNormal.Value);
            double[] inward = Smooth(depthAlong);
            bool anyFinite = false;
            for (int k = 0; k < q; k++)
            {
                inward[k] /= Hs[iApex + k];
                if (unsteady[iApex + k]) inward[k] = double.NegativeInfinity;
            }
            inward[0] = inward[q - 1] = double.NegativeInfinity;
            for (int k = 0; k < q; k++) anyFinite |= IsFinite(inward[k]);
            if (anyFinite)
            {
                int j = ArgMax(inward);
                depth = inward[j];
                depthPx = depth * scale;
                present = depth >= SubmalarMinDepth ? 1.0 : 0.0;
                submalar = P[iApex + j];
                submalarH = Hh[iApex + j];
            }
        }
        return new ObliqueCheek
        {
            FarLeft = farLeft,
            Outline = P,
            OutlineSrc = S,
            Chord = (a, b),
            Levels = levels,
            Apex = P[iApex],
            ApexH = Hh[iApex],
            ApexDev = dev[iApex],
            ApexPrior = apexPrior,
            ApexNearNose = apexNearNose,
            Submalar = submalar,
            SubmalarH = submalarH,
            SubmalarDepth = depth,
            SubmalarDepthPx = depthPx,
            SubmalarPresent = present,
            NoseOn = noseOn,
            NoseGap = noseGap,
            MaskFraction = maskFraction,
            WindowMask = windowMask,
            BelowMask = belowMask,
            Scale = scale,
        };
    }

    /// <summary><see cref="ObservedPoints(Vec2d[], float[], int[], byte[], bool)"/> of float landmarks.</summary>
    public static ObliqueCheek? ObservedPoints(Vec2[]? landmarks, float[]? outlineX, int[]? outlineY,
        byte[]? outlineSrc, bool farLeft) =>
        ObservedPoints(ToVec2d(landmarks), outlineX, outlineY, outlineSrc, farLeft);
}
