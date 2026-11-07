using System.Reflection;
using System.Text.Json;
using static CheekOutline;

namespace DynafaceTests;

// The cheek fixtures shared with dynaface-lib-python (linked into TestData by the
// project file), and the helpers CheekOutlineTests and CheekTests read them with.
internal static class CheekFixtures
{
    public static string DataPath(params string[] parts) =>
        Path.Combine(new[] { AppContext.BaseDirectory, "TestData" }.Concat(parts).ToArray());

    static JsonElement Parse(params string[] parts) =>
        JsonDocument.Parse(File.ReadAllText(DataPath(parts))).RootElement;

    static readonly Lazy<JsonElement> LandmarksJson = new(() => Parse("cheek_landmarks.json"));
    static readonly Lazy<JsonElement> RendersJson = new(() => Parse("cheek", "cheek_renders.json"));
    static readonly Lazy<JsonElement> OutlinesJson = new(() => Parse("cheek_outlines.json"));

    // tests_data/cheek_landmarks.json's images: raw and pipeline landmarks and yaws.
    public static JsonElement Images => LandmarksJson.Value.GetProperty("images");
    // tests_data/cheek/cheek_renders.json: the renders' silhouette oracles.
    public static JsonElement Renders => RendersJson.Value;
    // tests_data/cheek_outlines.json: the shared outline cases.
    public static JsonElement Outlines => OutlinesJson.Value;

    public static IEnumerable<string> ImageNames => Images.EnumerateObject().Select(p => p.Name);
    public static JsonElement Entry(string name) => Images.GetProperty(name);
    public static Vec2d[] RawLandmarks(string name) => Points(Entry(name).GetProperty("raw").GetProperty("landmarks"));
    public static double RawYaw(string name) => Entry(name).GetProperty("raw").GetProperty("yaw").GetDouble();
    public static Vec2d[] PipelineLandmarks(string name) =>
        Points(Entry(name).GetProperty("pipeline").GetProperty("landmarks"));

    // A stored number, where null stands for NaN.
    public static double Number(JsonElement e) => e.ValueKind == JsonValueKind.Null ? double.NaN : e.GetDouble();
    public static Vec2d Point(JsonElement e) => new(Number(e[0]), Number(e[1]));
    public static Vec2d[] Points(JsonElement e) => e.EnumerateArray().Select(Point).ToArray();

    // A render's foreground, [row, column] from the top: every pixel whose color isn't
    // the background (cheek_renders.json's header), decoded by the console project's
    // SkiaSharp ImageLoader, whose FaceImage rows run from the bottom.
    public static bool[,] RenderForeground(string relativePath)
    {
        FaceImage image = ImageLoader.Load(DataPath(relativePath.Split('/')));
        JsonElement bg = Renders.GetProperty("background");
        byte r = bg[0].GetByte(), g = bg[1].GetByte(), b = bg[2].GetByte();
        var fg = new bool[image.Height, image.Width];
        for (int y = 0; y < image.Height; y++)
        {
            int row = (image.Height - 1 - y) * image.Width;
            for (int x = 0; x < image.Width; x++)
            {
                Rgba32 p = image.Pixels[row + x];
                fg[y, x] = p.R != r || p.G != g || p.B != b;
            }
        }
        return fg;
    }

    // 255 on the foreground, 0 elsewhere.
    public static byte[,] AlphaOf(bool[,] fg)
    {
        var alpha = new byte[fg.GetLength(0), fg.GetLength(1)];
        for (int y = 0; y < fg.GetLength(0); y++)
            for (int x = 0; x < fg.GetLength(1); x++)
                alpha[y, x] = fg[y, x] ? (byte)255 : (byte)0;
        return alpha;
    }

    public static IEnumerable<int> Span(int start, int stopExclusive) =>
        Enumerable.Range(start, stopExclusive - start);

    public static Dictionary<int, int> ByRow(int[] rows)
    {
        var map = new Dictionary<int, int>();
        for (int i = 0; i < rows.Length; i++) map[rows[i]] = i;
        return map;
    }

    // NumPy's assert_allclose: |actual - expected| <= atol + rtol * |expected|, NaN equal to NaN.
    public static void Close(double expected, double actual, double atol = 0.0, double rtol = 1e-7, string? what = null)
    {
        if (double.IsNaN(expected) && double.IsNaN(actual)) return;
        if (expected == actual) return;
        Assert.True(Math.Abs(actual - expected) <= atol + rtol * Math.Abs(expected),
            $"{what ?? "value"}: expected {expected:R}, got {actual:R}");
    }

    public static void Close(Vec2d expected, Vec2d actual, double atol = 0.0, double rtol = 1e-7, string? what = null)
    {
        Close(expected.X, actual.X, atol, rtol, what);
        Close(expected.Y, actual.Y, atol, rtol, what);
    }

    // unittest's assertAlmostEqual(places=places): round(|diff|, places) == 0.
    public static void Near(double expected, double actual, int places = 7, string? what = null)
    {
        if (expected == actual) return;
        Assert.True(Math.Round(Math.Abs(actual - expected), places) == 0.0,
            $"{what ?? "value"}: expected {expected:R}, got {actual:R} to {places} places");
    }

    // unittest's assertAlmostEqual(delta=delta).
    public static void Within(double expected, double actual, double delta, string? what = null) =>
        Assert.True(Math.Abs(actual - expected) <= delta,
            $"{what ?? "value"}: expected {expected:R} +/- {delta}, got {actual:R}");

    public static Vec2d[] Rows(Vec2d[] points, params int[] indices) => indices.Select(i => points[i]).ToArray();
}

// The synthetic face of test_cheek_outline.py: a level head with parallel canthi and
// alar-base lines, so every level is horizontal and h = (600 - row) / 300 at any x.
internal static class SyntheticFace
{
    public const int BaseChinRow = 600;
    public const double BaseH = 300.0;
    public const double BaseX = 540.0; // the straight synthetic outline (the far cheek is the left)

    // The 98 landmarks of the synthetic face, the far cheek the subject's left.
    public static Vec2d[] BaseLandmarks()
    {
        var lm = new Vec2d[98];
        lm[60] = new(300, 300); lm[72] = new(500, 300); // lateral canthi
        lm[96] = new(330, 300); lm[97] = new(470, 300); // pupils
        lm[55] = new(360, 400); lm[59] = new(440, 400); // alar bases, parallel to the canthi
        lm[16] = new(400, BaseChinRow);
        lm[90] = new(400, 470); lm[94] = new(400, 474); // stomion at 472: h 0.4267
        lm[54] = new(480, 394); lm[57] = new(430, 418); // nose tip h 0.6867, subnasale 0.6067
        lm[53] = new(470, 362); // the lower dorsum point, h 0.7933
        lm[76] = new(360, 470); lm[82] = new(440, 470); // mouth corners
        for (int k = 0; k < 17; k++)
        {
            double t = k / 16.0;
            lm[32 - k] = new(Math.Round(BaseX - 140 * t), Math.Round(290 + 310 * t));
            lm[k] = new(Math.Round(260 + 140 * t), Math.Round(290 + 310 * t));
        }
        return lm;
    }

    public static double RowOf(double h) => BaseChinRow - h * BaseH;

    // A synthetic outline at x = BaseX with raised-cosine bumps, each (h, amplitude as a
    // fraction of H, outward positive, half-width in rows), all mask rows.
    public static (float[] X, int[] Y, byte[] Src) Outline(
        (double H, double Amp, double Half)[]? bumps = null, int first = 290, int last = BaseChinRow)
    {
        int n = last - first + 1;
        var rows = new int[n];
        var xs = new double[n];
        for (int i = 0; i < n; i++) { rows[i] = first + i; xs[i] = BaseX; }
        foreach (var (h, amp, half) in bumps ?? Array.Empty<(double, double, double)>())
            for (int i = 0; i < n; i++)
            {
                double d = (rows[i] - RowOf(h)) / half;
                double lobe = Math.Abs(d) < 1.0 ? Math.Pow(Math.Cos(0.5 * Math.PI * d), 2) : 0.0;
                xs[i] += amp * BaseH * lobe;
            }
        var src = Enumerable.Repeat(OutlineSrcMask, n).ToArray();
        return (xs.Select(x => (float)x).ToArray(), rows, src);
    }

    // A projective warp (a keystone converging the horizontals to the right).
    public static Vec2d Warp(Vec2d p, double k = 4e-4)
    {
        double w = 1.0 + k * (p.X - 400.0);
        return new((p.X - 400.0) / w + 400.0, (p.Y - 400.0) / w + 400.0);
    }

    public static Vec2d[] Warp(Vec2d[] points, double k = 4e-4) => points.Select(p => Warp(p, k)).ToArray();

    // An outline through Warp(), resampled on integer rows: x by linear interpolation,
    // each row's source from the nearest warped row.
    public static (float[] X, int[] Y, byte[] Src) WarpOutline(float[] xs, int[] rows, byte[] src, double k = 4e-4)
    {
        var warped = xs.Select((x, i) => Warp(new Vec2d(x, rows[i]), k)).ToArray();
        for (int i = 1; i < warped.Length; i++) Assert.True(warped[i].Y > warped[i - 1].Y);
        double[] wy = warped.Select(p => p.Y).ToArray(), wx = warped.Select(p => p.X).ToArray();
        int lo = (int)Math.Ceiling(wy.Min()), hi = (int)Math.Floor(wy.Max());
        int n = hi - lo + 1;
        var outX = new float[n];
        var outY = new int[n];
        var outSrc = new byte[n];
        for (int i = 0; i < n; i++)
        {
            int r = lo + i;
            outY[i] = r;
            outX[i] = (float)CheekNumerics.Interp(r, wy, wx);
            outSrc[i] = src[CheekNumerics.ArgMin(wy.Select(y => Math.Abs(y - r)).ToArray())];
        }
        return (outX, outY, outSrc);
    }

    // The landmarks of the mirrored image: x flipped and the sides swapped.
    public static Vec2d[] MirrorLandmarks(Vec2d[] lm, int width)
    {
        var perm = Enumerable.Range(0, 98).ToArray();
        for (int i = 0; i < 33; i++) perm[i] = 32 - i;
        (perm[60], perm[72]) = (72, 60);
        (perm[55], perm[59]) = (59, 55);
        (perm[96], perm[97]) = (97, 96);
        var mirrored = new Vec2d[98];
        for (int i = 0; i < 98; i++) mirrored[perm[i]] = new Vec2d(width - 1 - lm[i].X, lm[i].Y);
        return mirrored;
    }

    public static (bool[,], byte[,]) FlipHorizontally(bool[,] fg, byte[,] alpha)
    {
        int h = fg.GetLength(0), w = fg.GetLength(1);
        var f = new bool[h, w];
        var a = new byte[h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) { f[y, x] = fg[y, w - 1 - x]; a[y, x] = alpha[y, w - 1 - x]; }
        return (f, a);
    }

    // A size x size image whose columns at or left of edge[row] are on.
    public static bool[,] LeftOf(double[] edge, int size, Func<int, int, bool>? extra = null)
    {
        var fg = new bool[size, size];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
                fg[y, x] = x <= edge[y] || (extra != null && extra(y, x));
        return fg;
    }
}

// A synthetic head: a disc of radius 200 at (300, 300) in a 600 px image, the far cheek
// the subject's left (the image's right), the far contour inset px (2) inside the disc's
// edge, the chin below the disc. The tracer's span (and H) is 330, so the walk's limit
// is 16 px and NoseStep is 3.3 px. test_cheek_outline.py's DiscScene.
internal sealed class DiscScene
{
    public const int Size = 600;
    public const double CX = 300.0, CY = 300.0, R = 200.0;

    public bool[,] Foreground;
    public byte[,] Alpha;
    public readonly bool[,] Disc;
    public readonly Vec2d[] Landmarks;
    public readonly double CenterX;

    public DiscScene(double centerX = CX, bool soft = false, int inset = 2)
    {
        Foreground = new bool[Size, Size];
        Alpha = new byte[Size, Size];
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                double rr = CheekNumerics.Hypot(x - centerX, y - CY);
                if (soft)
                {
                    // alpha == MaskAlpha exactly at r == R: the edge to recover.
                    double ramp = MaskAlpha + 40.0 * (R - rr);
                    Alpha[y, x] = (byte)Math.Clamp(ramp, 0, 255);
                    Foreground[y, x] = Alpha[y, x] > MaskAlpha;
                }
                else
                {
                    Foreground[y, x] = rr < R;
                    Alpha[y, x] = Foreground[y, x] ? (byte)255 : (byte)0;
                }
            }
        Disc = (bool[,])Foreground.Clone();
        CenterX = centerX;
        var lm = new Vec2d[98];
        lm[60] = new(220, 200); lm[72] = new(380, 200);
        lm[96] = new(250, 200); lm[97] = new(350, 200);
        lm[55] = new(270, 320); lm[59] = new(330, 320);
        lm[16] = new(300, 530); // below the disc, whose last row is 499
        lm[90] = new(300, 430); lm[94] = new(300, 434);
        lm[54] = new(390, 330); lm[57] = new(360, 360); // h 0.606 and 0.515
        for (int k = 0; k < 17; k++)
        {
            int y = (int)Math.Round(190 + k * 21.25); // np.round(np.linspace(190, 530, 17))
            if (y < 500) lm[32 - k] = new(Edge(y) - inset, y);
            else if (y < 530) lm[32 - k] = new(320, y);
        }
        lm[16] = new(300, 530);
        for (int k = 0; k < 17; k++) lm[k] = new(Size - 1 - lm[32 - k].X, lm[32 - k].Y);
        Landmarks = lm;
    }

    // The disc's rightmost foreground column in row y.
    public int Edge(int y)
    {
        for (int x = Size - 1; x >= 0; x--)
            if (Disc[y, x]) return x;
        throw new InvalidOperationException($"row {y} has no disc");
    }

    // Foreground from edge + start to edge + stop (exclusive) on these rows.
    public void Extend(IEnumerable<int> rows, int start, int stop)
    {
        foreach (int y in rows)
        {
            int e = Edge(y);
            for (int x = e + start; x < Math.Min(e + stop, Size); x++)
            {
                Foreground[y, x] = true;
                Alpha[y, x] = 255;
            }
        }
    }

    public (float[] X, int[] Y, byte[] Src) Trace(bool farLeft = true, double tol = OutlineTol, byte[,]? gray = null)
    {
        bool[,] fg = Foreground;
        byte[,] alpha = Alpha;
        Vec2d[] lm = Landmarks;
        if (!farLeft)
        {
            (fg, alpha) = SyntheticFace.FlipHorizontally(fg, alpha);
            lm = SyntheticFace.MirrorLandmarks(lm, Size);
        }
        return TraceFarCheek(fg, alpha, lm, farLeft, tol, gray);
    }
}

// Ports dynaface-lib-python's tests/test_cheek_outline.py: the far-cheek tracer, the
// level frame and the apex and submalar search, on synthetic masks, on the synthetic-head
// renders (decoded with the console project's SkiaSharp ImageLoader) and on the shared
// fixture cheek_outlines.json, whose outlines and points the Python library wrote.
// Skipped from the Python file: the cv2 half of test_gray_image_is_the_fixed_point_luma
// (OpenCV's own conversion, which .NET doesn't have; the fixed-point rule itself is
// checked here).
public class CheekOutlineTests
{
    static double[] D(params double[] v) => v;

    // ------------------------------------------------------------ numerics

    public class Numerics
    {
        [Fact]
        public void SoftwareFmaIsCorrectlyRounded()
        {
            // The emulation must equal the hardware instruction everywhere the geometry
            // goes: products and sums of pixels, fractions and degrees.
            var rng = new Random(11);
            for (int t = 0; t < 200000; t++)
            {
                double a = (rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-6, 6));
                double b = (rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-6, 6));
                double c = t % 3 == 0 ? -a * b * (1 + (rng.NextDouble() - 0.5) * 1e-9)
                    : (rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-6, 6));
                Assert.Equal(Math.FusedMultiplyAdd(a, b, c), CheekNumerics.Fma(a, b, c));
            }
            // Halfway cases: a * b one half-ulp past a double, and its exact cancellations.
            double e = 1.0 + Math.Pow(2, -27);
            foreach (double c in D(0.0, -1.0, 1.0, Math.Pow(2, -53), -Math.Pow(2, -53), -1.0 - Math.Pow(2, -26)))
                Assert.Equal(Math.FusedMultiplyAdd(e, e, c), CheekNumerics.Fma(e, e, c));
            Assert.Equal(1.0 / 9.0 * 3.0, CheekNumerics.Fma(1.0 / 9.0, 3.0, 0.0));
        }

        [Fact]
        public void HypotIsTheReferenceFormula()
        {
            Assert.Equal(5.0, CheekNumerics.Hypot(3.0, -4.0));
            Assert.Equal(double.PositiveInfinity, CheekNumerics.Hypot(double.NaN, double.NegativeInfinity));
            Assert.True(double.IsNaN(CheekNumerics.Hypot(double.NaN, 1.0)));
            Assert.Equal(5e200, CheekNumerics.Hypot(3e200, 4e200), 1e188);
            var rng = new Random(5);
            for (int t = 0; t < 10000; t++)
            {
                double x = (rng.NextDouble() - 0.5) * 1000, y = (rng.NextDouble() - 0.5) * 1000;
                double big = Math.Max(Math.Abs(x), Math.Abs(y)), small = Math.Min(Math.Abs(x), Math.Abs(y));
                Assert.Equal(Math.Sqrt(Math.FusedMultiplyAdd(small, small, big * big)), CheekNumerics.Hypot(x, y));
            }
        }

        [Fact]
        public void ArgMinAndArgMaxFollowNumPy()
        {
            Assert.Equal(1, CheekNumerics.ArgMin(D(3, 1, 1, 2)));
            Assert.Equal(2, CheekNumerics.ArgMin(D(3, 1, double.NaN, 0)));
            Assert.Equal(0, CheekNumerics.ArgMax(D(3, 1, 3)));
            Assert.Equal(1, CheekNumerics.ArgMax(D(3, double.NaN, 4)));
        }

        [Fact]
        public void InterpFollowsNumPy()
        {
            double[] xp = D(0, 1, 3), fp = D(0, 10, 30);
            Assert.Equal(0.0, CheekNumerics.Interp(-1, xp, fp));
            Assert.Equal(30.0, CheekNumerics.Interp(5, xp, fp));
            Assert.Equal(10.0, CheekNumerics.Interp(1, xp, fp));
            Assert.Equal(30.0, CheekNumerics.Interp(3, xp, fp));
            Assert.Equal(20.0, CheekNumerics.Interp(2, xp, fp));
            Assert.True(double.IsNaN(CheekNumerics.Interp(double.NaN, xp, fp)));
        }
    }

    // ------------------------------------------------------------ TestLevels

    public class Levels
    {
        static Vec2d[] LandmarksWithVanishing(double vx = 1000.0)
        {
            var lm = SyntheticFace.BaseLandmarks();
            // The alar-base line through (350, 400) and V = (vx, 300).
            lm[55] = new(350, 400);
            lm[59] = new(480, 400 - 100 * (130 / (vx - 350)));
            return lm;
        }

        [Fact]
        public void LevelThroughKnownVanishingPoint()
        {
            var lm = LandmarksWithVanishing(1000.0);
            var frame = ObliqueFrameOf(lm);
            Assert.NotNull(frame);
            CheekFixtures.Close(new Vec2d(1000.0, 300.0), frame!.Vanishing!.Value, atol: 1e-9);
            CheekFixtures.Close(new Vec2d(0.0, 1.0), frame.Down, atol: 1e-12);
            var (point, direction) = LevelThrough(frame, lm[16]);
            Assert.Equal(lm[16], point);
            CheekFixtures.Near(1.0, Math.Sqrt(Vec2d.Dot(direction, direction)), 12);
            // The direction runs from V through the chin.
            CheekFixtures.Near(0.0, Cross(lm[16] - frame.Vanishing.Value, direction));
            // Where that level meets the vertical through x = 700: on the line V-chin.
            Vec2d met = LevelAtVertical((point, direction), new Vec2d(700.0, 100.0), frame);
            CheekFixtures.Near(700.0, met.X, 9);
            CheekFixtures.Near(600.0 - 300.0 * (300.0 / 600.0), met.Y, 9);
        }

        [Fact]
        public void SingularBranchUsesMeanDirection()
        {
            var lm = SyntheticFace.BaseLandmarks(); // parallel canthi and alar-base lines
            var frame = ObliqueFrameOf(lm)!;
            Assert.Null(frame.Vanishing);
            CheekFixtures.Close(new Vec2d(1.0, 0.0), frame.MeanDir, atol: 1e-12);
            var (point, direction) = LevelThrough(frame, lm[16]);
            CheekFixtures.Close(frame.MeanDir, direction);
            Vec2d met = LevelAtVertical((point, direction), new Vec2d(540.0, 100.0), frame);
            CheekFixtures.Close(new Vec2d(540.0, SyntheticFace.BaseChinRow), met, atol: 1e-9);
            var (h, scale) = HeightRatio(new Vec2d(540.0, 450.0), lm[16], lm[72], frame);
            CheekFixtures.Near(0.5, h, 12);
            CheekFixtures.Near(SyntheticFace.BaseH, scale, 9);
        }

        [Fact]
        public void FrameNeedsDistinctCanthi()
        {
            var lm = SyntheticFace.BaseLandmarks();
            lm[72] = lm[60];
            Assert.Null(ObliqueFrameOf(lm));
            Assert.Null(ObliqueFrameOf(lm.Take(97).ToArray()));
            Assert.Null(ObservedPoints(lm, null, null, null, true));
        }

        [Fact]
        public void DownPointsTowardChin()
        {
            var lm = SyntheticFace.BaseLandmarks().Select(p => new Vec2d(p.X, 900 - p.Y)).ToArray(); // upside down
            var frame = ObliqueFrameOf(lm)!;
            CheekFixtures.Close(new Vec2d(0.0, -1.0), frame.Down, atol: 1e-12);
            var (h, _) = HeightRatio(new Vec2d(540.0, 450.0), lm[16], lm[72], frame);
            CheekFixtures.Near(0.5, h, 12);
        }

        [Fact]
        public void HeightRatioExactOnTrapezoid()
        {
            // The chin level and the canthus level meet at V = (2000, 300): H is 318.75 on
            // the vertical x = 300 and 281.25 on x = 500.
            var lm = LandmarksWithVanishing(2000.0);
            var frame = ObliqueFrameOf(lm)!;
            CheekFixtures.Close(new Vec2d(2000.0, 300.0), frame.Vanishing!.Value, atol: 1e-6);
            foreach (var (x, expected) in new[] { (300.0, 318.75), (500.0, 281.25) })
            {
                double chinY = 600.0 - 300.0 * (x - 400.0) / 1600.0;
                foreach (var (p, want) in new[] { (new Vec2d(x, 300.0), 1.0), (new Vec2d(x, chinY), 0.0) })
                {
                    var (h, scale) = HeightRatio(p, lm[16], lm[72], frame);
                    CheekFixtures.Near(expected, scale, 9);
                    CheekFixtures.Near(want, h, 9);
                }
                var mid = new Vec2d(x, 0.5 * (300.0 + chinY));
                CheekFixtures.Near(0.5, HeightRatio(mid, lm[16], lm[72], frame).H, 12);
            }
        }

        [Fact]
        public void ContinuityAsVanishingPointRecedes()
        {
            var lm = SyntheticFace.BaseLandmarks();
            lm[55] = new(350.0, 400.0);
            var p = new Vec2d(600.0, 450.0);
            var hs = new Dictionary<double, double>();
            foreach (double eps in D(2e-6, 5e-7, 1e-7))
            {
                var c = (Vec2d[])lm.Clone();
                c[59] = new(450.0, 400.0 + 100.0 * Math.Tan(eps)); // the lines meet at eps
                var frame = ObliqueFrameOf(c)!;
                if (eps > VanishMinSin) Assert.NotNull(frame.Vanishing);
                else Assert.Null(frame.Vanishing);
                hs[eps] = HeightRatio(p, c[16], c[72], frame).H;
            }
            Assert.True(Math.Abs(hs[2e-6] - hs[5e-7]) < 1e-4);
            Assert.True(Math.Abs(hs[5e-7] - hs[1e-7]) < 1e-4);
            Assert.True(Math.Abs(hs[2e-6] - 0.5) < 1e-4);
        }

        [Fact]
        public void PerspectiveWarpExercisesVanishingPoint()
        {
            var lm = SyntheticFace.BaseLandmarks();
            var (xs, rows, src) = SyntheticFace.Outline(new[] { (0.78, 0.03, 20.0) });
            var plain = ObservedPoints(lm, xs, rows, src, true)!;
            Vec2d[] warpedLm = SyntheticFace.Warp(lm);
            var frame = ObliqueFrameOf(warpedLm)!;
            Assert.NotNull(frame.Vanishing);
            Assert.True(frame.Vanishing!.Value.IsFinite);
            // The horizontal through the chin maps onto the level through the warped chin.
            var (point, direction) = LevelThrough(frame, warpedLm[16]);
            foreach (double x in D(100.0, 300.0, 700.0))
            {
                Vec2d q = SyntheticFace.Warp(new Vec2d(x, SyntheticFace.BaseChinRow));
                Assert.True(Math.Abs(Cross(q - point, direction)) < 1e-6);
            }
            var (wXs, wRows, wSrc) = SyntheticFace.WarpOutline(xs, rows, src);
            var found = ObservedPoints(warpedLm, wXs, wRows, wSrc, true)!;
            Assert.False(found.ApexPrior);
            Assert.True(Math.Abs(found.ApexH - plain.ApexH) < 0.01);
            CheekFixtures.Close(SyntheticFace.Warp(plain.Apex), found.Apex, atol: 2.0);
        }
    }

    // ------------------------------------------------------------ TestContourSeed

    public class ContourSeedRule
    {
        static Vec2d[] P(params (double X, double Y)[] pts) => pts.Select(p => new Vec2d(p.X, p.Y)).ToArray();

        static void AllClose(double[] expected, double[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++) CheekFixtures.Close(expected[i], actual[i]);
        }

        [Fact]
        public void ReversingChinTakesOutwardCrossing()
        {
            var cont = P((300, 900), (310, 930), (330, 964), (350, 982), (380, 991), (420, 997), (440, 991));
            int[] rows = { 991, 995, 997, 980 };
            double x980 = 330 + 20 * 16 / 18.0; // on the segment from (330, 964) to (350, 982)
            double[] right = ContourSeed(cont, rows, farLeft: false); // outward is -x
            AllClose(D(380.0, 380 + 40 * 4 / 6.0, 420.0, x980), right);
            double[] left = ContourSeed(cont, rows, farLeft: true); // outward is +x
            AllClose(D(440.0, 420 + 20 * 2 / 6.0, 420.0, x980), left);
            // Without farLeft, the side is read from the polyline.
            AllClose(right, ContourSeed(cont, rows));
            var flipped = cont.Select(p => new Vec2d(1000 - p.X, p.Y)).ToArray(); // now outward is +x
            AllClose(right.Select(x => 1000 - x).ToArray(), ContourSeed(flipped, rows));
        }

        [Fact]
        public void DuplicateRowsTakeOutwardEnd()
        {
            var cont = P((300, 100), (310, 120), (320, 120), (330, 140));
            int[] rows = { 100, 110, 120, 130, 140 };
            AllClose(D(300, 305, 320, 325, 330), ContourSeed(cont, rows, farLeft: true));
            AllClose(D(300, 305, 310, 325, 330), ContourSeed(cont, rows, farLeft: false));
        }

        [Fact]
        public void RowsOutsideExtrapolateEndSegments()
        {
            var cont = P((200, 300), (210, 400), (215, 500));
            AllClose(D(198, 199, 200, 205, 215, 216),
                ContourSeed(cont, new[] { 280, 290, 300, 350, 500, 520 }, farLeft: true));
        }

        [Fact]
        public void DegenerateContours()
        {
            Assert.Empty(ContourSeed(P((1, 2)), Array.Empty<int>()));
            AllClose(D(7, 7, 7), ContourSeed(P((7, 2)), new[] { 1, 2, 3 }));
            AllClose(D(9, 9, 9), ContourSeed(P((5, 2), (9, 2), (3, 2)), new[] { 1, 2, 3 }, farLeft: true));
        }

        [Fact]
        public void ContourOutlineStartsAtCanthusAboveContourTop()
        {
            var lm = SyntheticFace.BaseLandmarks(); // the far contour's top (32) is at row 290
            lm[72] = new(500, 280); // the far canthus above it
            var (xs, rows, src) = ContourOutline(lm, true);
            Assert.Equal(280, rows[0]);
            Assert.Equal(SyntheticFace.BaseChinRow, rows[^1]);
            Assert.All(src, s => Assert.Equal(OutlineSrcContour, s));
            // Rows 280 to 289 extrapolate the first segment (32 -> 31).
            Vec2d p0 = lm[32], p1 = lm[31];
            for (int i = 0; i < 10; i++)
            {
                double want = p0.X + (rows[i] - p0.Y) * (p1.X - p0.X) / (p1.Y - p0.Y);
                CheekFixtures.Near(want, xs[i], 4);
            }
            // The right side mirrors the rule from the other contour.
            var (xsR, rowsR, _) = ContourOutline(lm, false);
            Assert.Equal(290, rowsR[0]);
            CheekFixtures.Near(260.0, xsR[0], 4);
            // Fewer than 98 landmarks: nothing.
            Assert.Empty(ContourOutline(lm.Take(50).ToArray(), true).X);
            // The float overload reads the same landmarks.
            Assert.Equal(xs, ContourOutline(lm.Select(p => p.ToVec2()).ToArray(), true).X);
        }
    }

    // ------------------------------------------------------------ TestTracer

    public class Tracer
    {
        static double[] SeedOf(Vec2d[] lm, int[] rows) =>
            ContourSeed(LeftFarContour.Select(i => lm[i]).ToArray(), rows, farLeft: true);

        [Fact]
        public void DiscHead()
        {
            var scene = new DiscScene();
            scene.Extend(CheekFixtures.Span(330, 361), 1, 6); // the nose: 5 px past the edge
            scene.Extend(CheekFixtures.Span(240, 261), 1, 61); // hair merging with the cheek
            scene.Extend(CheekFixtures.Span(400, 421), 60, 81); // a hair blob 60 px outside
            var lm = scene.Landmarks;
            foreach (int y in new[] { 424, 445 }) // contour points 4 px in the background
            {
                int k = Enumerable.Range(16, 17).First(i => lm[i].Y == y);
                lm[k] = new(scene.Edge(y) + 4, y);
            }
            var (xs, rows, src) = scene.Trace();
            Assert.Equal(190, rows[0]); // the contour top, above the canthus row
            Assert.Equal(530, rows[^1]); // the chin row
            var byRow = CheekFixtures.ByRow(rows);
            double[] seed = SeedOf(lm, rows);
            // Disc rows: the sub-pixel edge from the binary alpha, 0.8745 px past the last
            // foreground pixel.
            foreach (int y in CheekFixtures.Span(200, 240).Concat(CheekFixtures.Span(261, 330)).Concat(CheekFixtures.Span(361, 470)))
            {
                int i = byRow[y];
                Assert.Equal(OutlineSrcMask, src[i]);
                CheekFixtures.Near(scene.Edge(y) + 223 / 255.0, xs[i], 3, $"row {y}");
            }
            // The seed 4 px in the background walks inward to the same edge.
            foreach (int y in CheekFixtures.Span(422, 427).Concat(CheekFixtures.Span(443, 448)))
                Assert.True(seed[byRow[y]] > scene.Edge(y) + 2.5, $"row {y}");
            // Merged hair: the walk reaches the limit and keeps the seed.
            foreach (int y in CheekFixtures.Span(240, 261))
            {
                int i = byRow[y];
                Assert.Equal(OutlineSrcContour, src[i]);
                CheekFixtures.Near(seed[i], xs[i], 4);
            }
            // The nose rows: the seed stands in, between the tip's and the subnasale's levels only.
            foreach (int y in CheekFixtures.Span(330, 361))
            {
                int i = byRow[y];
                Assert.Equal(OutlineSrcNose, src[i]);
                CheekFixtures.Near(seed[i], xs[i], 4);
            }
            Assert.Equal(OutlineSrcMask, src[byRow[329]]);
            Assert.Equal(OutlineSrcMask, src[byRow[361]]);
            // Rows beyond the mask keep the seed.
            foreach (int y in CheekFixtures.Span(499, 531))
            {
                int i = byRow[y];
                Assert.Equal(OutlineSrcContour, src[i]);
                CheekFixtures.Near(seed[i], xs[i], 4);
            }
            Assert.Equal(OutlineSrcMask, src[byRow[498]]);
        }

        [Fact]
        public void NoseExtendsUpwardWhileDeparting()
        {
            var scene = new DiscScene();
            scene.Extend(CheekFixtures.Span(300, 361), 1, 6); // the bump reaches above the tip row
            var (_, rows, src) = scene.Trace();
            var byRow = CheekFixtures.ByRow(rows);
            foreach (int y in CheekFixtures.Span(300, 361)) Assert.Equal(OutlineSrcNose, src[byRow[y]]);
            Assert.Equal(OutlineSrcMask, src[byRow[299]]);
            // A departure above the tip that the tip's row does not share stays mask.
            scene = new DiscScene();
            scene.Extend(CheekFixtures.Span(340, 361), 1, 6);
            scene.Extend(CheekFixtures.Span(280, 301), 1, 6);
            (_, rows, src) = scene.Trace();
            byRow = CheekFixtures.ByRow(rows);
            foreach (int y in CheekFixtures.Span(340, 361)) Assert.Equal(OutlineSrcNose, src[byRow[y]]);
            foreach (int y in CheekFixtures.Span(280, 340)) Assert.Equal(OutlineSrcMask, src[byRow[y]]);
        }

        [Fact]
        public void UniformOffsetIsNotTheNose()
        {
            var scene = new DiscScene(inset: 5); // 3 px further inside the disc than usual
            var lm = scene.Landmarks;
            var frame = ObliqueFrameOf(lm)!;
            double stepPx = NoseStep * Vec2d.Dot(lm[16] - lm[72], frame.Down); // 3.3
            var (xs, rows, src) = scene.Trace();
            double[] seed = SeedOf(lm, rows);
            var depart = Enumerable.Range(0, rows.Length).Where(i => src[i] == OutlineSrcMask)
                .Select(i => xs[i] - seed[i]).ToList();
            Assert.True(CheekNumerics.Median(depart) > 1.5 * stepPx);
            Assert.True(depart.Max() > 2.5 * stepPx);
            Assert.DoesNotContain(OutlineSrcNose, src);
            Assert.Equal(309, depart.Count);
            // A nose bump of 5 px on the band rows is marked, its neighbors are not.
            scene.Extend(CheekFixtures.Span(330, 361), 1, 6);
            (xs, rows, src) = scene.Trace();
            Assert.Equal(CheekFixtures.Span(330, 361).ToHashSet(),
                rows.Where((r, i) => src[i] == OutlineSrcNose).ToHashSet());
            var byRow = CheekFixtures.ByRow(rows);
            Assert.Equal(OutlineSrcMask, src[byRow[329]]);
            Assert.Equal(OutlineSrcMask, src[byRow[361]]);
        }

        [Fact]
        public void ContourAndEdgeRowsAreNeverNoseRows()
        {
            var lm = SyntheticFace.BaseLandmarks();
            const int size = 700;
            int[] all = Enumerable.Range(0, size).ToArray();
            double[] seed = SeedOf(lm, all);
            bool InBand(int y) => y >= 394 && y <= 418; // the tip's to subnasale's
            var foreground = SyntheticFace.LeftOf(seed.Select(s => s - 5).ToArray(), size,
                (y, x) => InBand(y) && x <= 600); // no mask edge within the window there
            var alpha = CheekFixtures.AlphaOf(foreground);
            var gray = new byte[size, size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++) gray[y, x] = x < seed[y] + 8 ? (byte)180 : (byte)60;
            double stepPx = NoseStep * SyntheticFace.BaseH;
            foreach (var (image, source) in new[] { ((byte[,]?)null, OutlineSrcContour), (gray, OutlineSrcEdge) })
            {
                var (xs, rows, src) = TraceFarCheek(foreground, alpha, lm, true, gray: image);
                var depart = rows.Select((r, i) => xs[i] - seed[r]).ToArray();
                var maskDepart = depart.Where((d, i) => src[i] == OutlineSrcMask).ToList();
                Assert.True(CheekNumerics.Median(maskDepart) < -stepPx);
                for (int i = 0; i < rows.Length; i++)
                {
                    if (!InBand(rows[i])) continue;
                    Assert.Equal(source, src[i]);
                    if (source == OutlineSrcEdge) Assert.True(depart[i] > 2 * stepPx);
                    else CheekFixtures.Close(0.0, depart[i], atol: 1e-4);
                }
                Assert.DoesNotContain(OutlineSrcNose, src);
            }
        }

        [Fact]
        public void ReferenceExcludesRowsBelowStomionLevel()
        {
            const int size = 700;
            var lm = SyntheticFace.BaseLandmarks();
            lm[90] = new(400, 428); lm[94] = new(400, 432); // the stomion at row 430, h 0.567
            int[] all = Enumerable.Range(0, size).ToArray();
            double[] seed = SeedOf(lm, all);
            double stepPx = NoseStep * SyntheticFace.BaseH;
            foreach (var (bandExtra, want) in new[] { (0.0, new HashSet<int>()), (5.0, CheekFixtures.Span(394, 419).ToHashSet()) })
            {
                var edge = all.Select(y => seed[y] + ((y > 430 ? 9.0 : 1.0) + (y >= 394 && y <= 418 ? bandExtra : 0.0))).ToArray();
                var foreground = SyntheticFace.LeftOf(edge, size);
                var alpha = CheekFixtures.AlphaOf(foreground);
                var (xs, rows, src) = TraceFarCheek(foreground, alpha, lm, true);
                var depart = rows.Select((r, i) => xs[i] - seed[r]).ToArray();
                int jaw = 0, cheek = 0;
                var maskDepart = new List<double>();
                for (int i = 0; i < rows.Length; i++)
                {
                    if (src[i] != OutlineSrcMask) continue;
                    maskDepart.Add(depart[i]);
                    if (rows[i] > 430) jaw++;
                    if (rows[i] <= 430 && rows[i] >= 300) cheek++;
                }
                Assert.True(jaw > cheek);
                // The median over every mask row is the jaw's departure.
                Assert.True(CheekNumerics.Median(maskDepart) > 2 * stepPx);
                Assert.Equal(want, rows.Where((r, i) => src[i] == OutlineSrcNose).ToHashSet());
            }
        }

        [Fact]
        public void RefinedCrossingTakesNearestPair()
        {
            byte[] B(params int[] v) => v.Select(x => (byte)x).ToArray();
            // The mask's edge pair (3, 4) doesn't bracket 32; the pair one pixel inward does.
            CheekFixtures.Near(2 + 223 / 245.0, RefinedCrossing(B(255, 255, 255, 10, 5, 0), 3, 1));
            // Two pixels outward.
            var row = B(255, 255, 255, 255, 255, 0);
            CheekFixtures.Near(4 + 223 / 255.0, RefinedCrossing(row, 2, 1));
            CheekFixtures.Near(1 - 223 / 255.0, RefinedCrossing(row.Reverse().ToArray(), 3, -1));
            // Nearest first, inward before outward.
            CheekFixtures.Near(223 / 255.0, RefinedCrossing(B(255, 0, 255, 0), 1, 1));
            // No bracketing pair within AlphaSearch: the mask's pixel boundary.
            Assert.Equal(2, AlphaSearch);
            Assert.Equal(3.5, RefinedCrossing(Enumerable.Repeat((byte)255, 8).ToArray(), 3, 1));
            Assert.Equal(2.5, RefinedCrossing(new byte[8], 3, -1));
            Assert.Equal(2.5, RefinedCrossing(B(255, 255, 255, 255, 255, 255, 0), 2, 1));
            // Pairs past the border are skipped, never indexed.
            Assert.Equal(2.5, RefinedCrossing(B(255, 255, 255), 2, 1));
            Assert.Equal(-0.5, RefinedCrossing(B(255, 255, 255), 0, -1));
            CheekFixtures.Near(223 / 255.0, RefinedCrossing(B(255, 0), 0, 1));
        }

        [Fact]
        public void AlphaCrossingRefinedPastShiftedMask()
        {
            foreach (bool farLeft in new[] { true, false })
            {
                var scene = new DiscScene(soft: true);
                bool[,] fg = scene.Foreground;
                byte[,] alpha = scene.Alpha;
                Vec2d[] lm = scene.Landmarks;
                double cx = DiscScene.CX;
                if (!farLeft)
                {
                    (fg, alpha) = SyntheticFace.FlipHorizontally(fg, alpha);
                    lm = SyntheticFace.MirrorLandmarks(lm, DiscScene.Size);
                    cx = DiscScene.Size - 1 - cx;
                }
                foreach (int shift in new[] { 1, 2, 3 }) // 2 is AlphaSearch itself, still refined
                {
                    var shifted = new bool[DiscScene.Size, DiscScene.Size];
                    for (int y = 0; y < DiscScene.Size; y++)
                        for (int x = shift; x < DiscScene.Size; x++) shifted[y, x] = fg[y, x - shift];
                    var (xs, rows, src) = TraceFarCheek(shifted, alpha, lm, farLeft);
                    var sel = Enumerable.Range(0, rows.Length)
                        .Where(i => src[i] == OutlineSrcMask && rows[i] >= 200 && rows[i] <= 470).ToArray();
                    Assert.True(sel.Length > 200);
                    var radius = sel.Select(i => Math.Sqrt(Math.Pow(xs[i] - cx, 2) + Math.Pow(rows[i] - DiscScene.CY, 2))).ToArray();
                    var onBoundary = sel.Select(i => Math.Abs(xs[i] - Math.Floor(xs[i]) - 0.5) < 1e-6).ToArray();
                    if (shift <= AlphaSearch)
                    {
                        Assert.True(radius.Max(r => Math.Abs(r - DiscScene.R)) < 0.25, $"farLeft {farLeft}, shift {shift}");
                        Assert.True(onBoundary.Count(b => b) / (double)onBoundary.Length < 0.05);
                    }
                    else
                    {
                        Assert.All(onBoundary, b => Assert.True(b));
                        Assert.True(radius.Min(r => Math.Abs(r - DiscScene.R)) > 1.0);
                    }
                    // The mask's own edge pair (its binary alpha) is a pixel off.
                    var (xsB, _, srcB) = TraceFarCheek(shifted, CheekFixtures.AlphaOf(shifted), lm, farLeft);
                    var radiusB = Enumerable.Range(0, rows.Length)
                        .Where(i => srcB[i] == OutlineSrcMask && rows[i] >= 200 && rows[i] <= 470)
                        .Select(i => Math.Abs(Math.Sqrt(Math.Pow(xsB[i] - cx, 2) + Math.Pow(rows[i] - DiscScene.CY, 2)) - DiscScene.R))
                        .ToList();
                    Assert.True(CheekNumerics.Median(radiusB) > 0.5);
                }
            }
        }

        [Fact]
        public void SeedNearBorderNeverIndexesOutside()
        {
            var scene = new DiscScene(centerX: 420.0); // the disc runs past the right border
            var lm = scene.Landmarks;
            for (int k = 17; k < 33; k++)
            {
                int y = (int)lm[k].Y;
                if (y >= 213 && y <= 387) lm[k] = new(DiscScene.Size - 4, y); // 3 px from the border
            }
            var (xs, rows, src) = scene.Trace();
            var byRow = CheekFixtures.ByRow(rows);
            foreach (int y in CheekFixtures.Span(220, 380))
            {
                int i = byRow[y];
                Assert.Equal(OutlineSrcContour, src[i]);
                Assert.True(xs[i] < DiscScene.Size);
            }
            Assert.Equal(OutlineSrcMask, src[byRow[190]]);
            // Rows outside the image keep the seed too.
            lm[16] = new(300, 650);
            (xs, rows, src) = scene.Trace();
            Assert.Equal(650, rows[^1]);
            for (int i = 0; i < rows.Length; i++)
                if (rows[i] >= DiscScene.Size) Assert.Equal(OutlineSrcContour, src[i]);
        }

        [Fact]
        public void SubpixelEdgeOfSoftDisc()
        {
            var scene = new DiscScene(soft: true);
            var (xs, rows, src) = scene.Trace();
            var sel = Enumerable.Range(0, rows.Length)
                .Where(i => src[i] == OutlineSrcMask && rows[i] >= 200 && rows[i] <= 470).ToArray();
            Assert.True(sel.Length > 200);
            foreach (int i in sel)
            {
                double radius = Math.Sqrt(Math.Pow(xs[i] - DiscScene.CX, 2) + Math.Pow(rows[i] - DiscScene.CY, 2));
                Assert.True(Math.Abs(radius - DiscScene.R) < 0.25);
            }
            CheekFixtures.Near(223 / 255.0, AlphaCrossing(new byte[] { 255, 0 }, 0, 1));
            CheekFixtures.Near(0.5, AlphaCrossing(new byte[] { 10, 5 }, 1, 0));
        }

        [Fact]
        public void BothFarSidesMirror()
        {
            var scene = new DiscScene();
            scene.Extend(CheekFixtures.Span(330, 361), 1, 6);
            scene.Extend(CheekFixtures.Span(240, 261), 1, 61);
            var (xs, rows, src) = scene.Trace(farLeft: true);
            var (xsM, rowsM, srcM) = scene.Trace(farLeft: false);
            Assert.Equal(rows, rowsM);
            Assert.Equal(src, srcM);
            for (int i = 0; i < xs.Length; i++) CheekFixtures.Close(DiscScene.Size - 1 - xs[i], xsM[i], atol: 1e-3);
            Assert.Equal(CheekFixtures.Span(330, 361).ToHashSet(), rows.Where((r, i) => src[i] == OutlineSrcNose).ToHashSet());
            Assert.Equal(CheekFixtures.Span(240, 261).Concat(CheekFixtures.Span(499, 531)).ToHashSet(),
                rows.Where((r, i) => src[i] == OutlineSrcContour).ToHashSet());
        }

        [Fact]
        public void NothingWithoutLandmarksOrFrame()
        {
            var scene = new DiscScene();
            var empty = TraceFarCheek(scene.Foreground, scene.Alpha, Enumerable.Repeat(new Vec2d(1, 1), 10).ToArray(), true);
            Assert.Empty(empty.X); Assert.Empty(empty.Y); Assert.Empty(empty.Src);
            var lm = (Vec2d[])scene.Landmarks.Clone();
            lm[72] = lm[60];
            empty = TraceFarCheek(scene.Foreground, scene.Alpha, lm, true);
            Assert.Empty(empty.X); Assert.Empty(empty.Y); Assert.Empty(empty.Src);
        }

        [Fact]
        public void NonFiniteLandmarksGiveNothing()
        {
            var scene = new DiscScene();
            foreach (var (index, value) in new[] { (16, double.NaN), (72, double.PositiveInfinity) })
            {
                var lm = (Vec2d[])scene.Landmarks.Clone();
                lm[index].Y = value;
                var empty = TraceFarCheek(scene.Foreground, scene.Alpha, lm, true);
                Assert.Empty(empty.X); Assert.Empty(empty.Y); Assert.Empty(empty.Src);
                Assert.Empty(ContourOutline(lm, true).X);
                Assert.Null(ObliqueFrameOf(lm));
                Assert.Null(ObservedPoints(lm, null, null, null, true));
            }
        }

        [Fact]
        public void EdgeSourceOnGrayStep()
        {
            var lm = SyntheticFace.BaseLandmarks();
            const int size = 700;
            int[] all = Enumerable.Range(0, size).ToArray();
            double[] seed = SeedOf(lm, all);
            bool Hair(int y) => (y >= 300 && y <= 314) || (y >= 330 && y <= 390);
            var foreground = SyntheticFace.LeftOf(seed.Select(s => s + 1).ToArray(), size, (y, x) => Hair(y) && x <= 600);
            var alpha = CheekFixtures.AlphaOf(foreground);
            byte[,] Step(byte inside, byte outside)
            {
                var g = new byte[size, size];
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size; x++) g[y, x] = x < seed[y] + 8 ? inside : outside;
                return g;
            }
            var (xs, rows, src) = TraceFarCheek(foreground, alpha, lm, true, gray: Step(180, 60));
            var byRow = CheekFixtures.ByRow(rows);
            foreach (int y in CheekFixtures.Span(330, 391))
            {
                int i = byRow[y];
                Assert.Equal(OutlineSrcEdge, src[i]);
                Assert.True(Math.Abs(xs[i] - (seed[y] + 8)) < 1.0);
            }
            // Off above a plain row ratio of 0.95 (rows above 315).
            foreach (int y in CheekFixtures.Span(300, 315)) Assert.Equal(OutlineSrcContour, src[byRow[y]]);
            foreach (int y in CheekFixtures.Span(315, 330).Concat(CheekFixtures.Span(391, 600)))
                Assert.Equal(OutlineSrcMask, src[byRow[y]]);
            // Without the gray image those rows keep the seed.
            var (_, _, srcPlain) = TraceFarCheek(foreground, alpha, lm, true);
            foreach (int y in CheekFixtures.Span(330, 391)) Assert.Equal(OutlineSrcContour, srcPlain[byRow[y]]);
            // A weak step (below EdgeGradMin, 2.5 times the step) is not an edge.
            var (_, _, srcFaint) = TraceFarCheek(foreground, alpha, lm, true, gray: Step(100, 90));
            Assert.Equal(srcPlain, srcFaint);
            // ObservedPoints takes the EDGE rows' x but counts only mask rows.
            var found = ObservedPoints(lm, xs, rows, src, true)!;
            Assert.Equal(300, found.Outline[0].Y);
            Assert.Equal(472, found.Outline[^1].Y);
            CheekFixtures.Near(97 / 173.0, found.MaskFraction, 9);
            Assert.Equal(0.0, found.WindowMask);
            var edgeRows = Enumerable.Range(0, found.Outline.Length).Where(k => found.OutlineSrc[k] == OutlineSrcEdge).ToArray();
            Assert.Equal(61, edgeRows.Length);
            foreach (int k in edgeRows) Assert.True(found.Outline[k].X > seed[(int)found.Outline[k].Y] + 6);
            // No mask row in the window: the prior stays at Kaptein's height, on an edge
            // row, and no depth is read below a stand-in row.
            Assert.True(found.ApexPrior);
            Assert.Equal(362, found.Apex.Y);
            Assert.Equal(OutlineSrcEdge, found.OutlineSrc[Array.FindIndex(found.Outline, p => p.Y == 362)]);
            CheekFixtures.Within(ApexHPrior, found.ApexH, 1.0 / SyntheticFace.BaseH);
            CheekFixtures.Near(82 / 111.0, found.BelowMask, 9);
            Assert.True(double.IsNaN(found.SubmalarDepth));
            Assert.True(double.IsNaN(found.SubmalarPresent));
            Assert.True(double.IsNaN(found.Submalar.X) && double.IsNaN(found.Submalar.Y));
        }

        [Fact]
        public void GrayImageIsTheFixedPointLuma()
        {
            // The library's own conversion, so both ports read the same edge rows: OpenCV's
            // integer BT.601 weights over 2^14, rounded. (The Python test's comparison with
            // cv2.cvtColor has no .NET counterpart.)
            var rng = new Random(7);
            var rgb = new byte[37, 53, 3];
            for (int y = 0; y < 37; y++)
                for (int x = 0; x < 53; x++)
                    for (int c = 0; c < 3; c++) rgb[y, x, c] = (byte)rng.Next(256);
            byte[,] gray = GrayImage(rgb);
            Assert.Equal(37, gray.GetLength(0));
            Assert.Equal(53, gray.GetLength(1));
            for (int y = 0; y < 37; y++)
                for (int x = 0; x < 53; x++)
                {
                    long expected = ((long)rgb[y, x, 0] * 4899 + rgb[y, x, 1] * 9617 + rgb[y, x, 2] * 1868 + 8192) >> 14;
                    Assert.Equal((byte)expected, gray[y, x]);
                }
            var white = new byte[2, 3, 3];
            for (int y = 0; y < 2; y++) for (int x = 0; x < 3; x++) for (int c = 0; c < 3; c++) white[y, x, c] = 255;
            Assert.All(GrayImage(white).Cast<byte>(), v => Assert.Equal(255, v));
            Assert.All(GrayImage(new byte[2, 3, 3]).Cast<byte>(), v => Assert.Equal(0, v));
            // A FaceImage's bottom-left rows come out top-left.
            var pixels = new Rgba32[53 * 37];
            for (int y = 0; y < 37; y++)
                for (int x = 0; x < 53; x++)
                    pixels[(36 - y) * 53 + x] = new Rgba32(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2], 255);
            Assert.Equal(gray.Cast<byte>().ToArray(), GrayImage(new FaceImage(pixels, 53, 37)).Cast<byte>().ToArray());
        }

        [Fact]
        public void NoseBandBoundedByTipLevelOnFixtureLandmarks()
        {
            foreach (string name in new[] { "yaw40", "yaw42", "yaw45", "yaw48", "yaw50", "yaw-45" })
            {
                var lm = CheekFixtures.RawLandmarks(name);
                bool farLeft = CheekFixtures.Entry(name).GetProperty("cues").GetProperty("nose_offset").GetDouble() > 0;
                var frame = ObliqueFrameOf(lm)!;
                Vec2d chin = lm[16], canthus = lm[FarCanthus(farLeft)];
                double hTip = HeightRatio(lm[NoseRows.Tip], chin, canthus, frame).H;
                double hSub = HeightRatio(lm[NoseRows.Subnasale], chin, canthus, frame).H;
                double h53 = HeightRatio(lm[53], chin, canthus, frame).H;
                Assert.True(hSub < hTip, name);
                Assert.True(hTip < ApexHWindow.Low, name);
                Assert.True(h53 > 0.78, name);
            }
        }
    }

    // ------------------------------------------------------------ TestSearch

    public class Search
    {
        readonly Vec2d[] lm = SyntheticFace.BaseLandmarks();

        // ObservedPoints on the synthetic outline with these bumps, after each change
        // (first row, last row, x or null to keep it, source) is applied.
        ObliqueCheek Find((double, double, double)[]? bumps = null, Vec2d[]? landmarks = null,
            params (int First, int Last, double? X, byte Source)[] changes)
        {
            var (xs, rows, src) = SyntheticFace.Outline(bumps);
            foreach (var (first, last, x, source) in changes)
                for (int i = 0; i < rows.Length; i++)
                {
                    if (rows[i] < first || rows[i] > last) continue;
                    if (x != null) xs[i] = (float)x.Value;
                    src[i] = source;
                }
            var found = ObservedPoints(landmarks ?? lm, xs, rows, src, true);
            Assert.NotNull(found);
            return found!;
        }

        static (double, double, double)[] Bumps(params (double, double, double)[] b) => b;

        [Fact]
        public void BulgeFoundWithinTwoRows()
        {
            var found = Find(Bumps((0.78, 0.03, 20)));
            Assert.False(found.ApexPrior);
            Assert.True(Math.Abs(found.Apex.Y - SyntheticFace.RowOf(0.78)) <= 2.0);
            CheekFixtures.Within(0.78, found.ApexH, 0.01);
            Assert.True(found.ApexDev > 0.025);
            Assert.True(found.ApexDev < 0.031);
            CheekFixtures.Within(SyntheticFace.BaseX + 9.0, found.Apex.X, 0.5);
            Assert.False(found.ApexNearNose);
            Assert.False(found.NoseOn);
            CheekFixtures.Near(SyntheticFace.BaseH, found.Scale, 9);
            Assert.Equal(1.0, found.MaskFraction);
            Assert.Equal(1.0, found.WindowMask);
            Assert.Equal(1.0, found.BelowMask);
            // The band runs from the canthus level to the stomion level.
            Assert.Equal(300, found.Outline[0].Y);
            Assert.Equal(472, found.Outline[^1].Y);
            CheekFixtures.Close(new Vec2d(SyntheticFace.BaseX, 300), found.Chord.Canthus);
            CheekFixtures.Close(new Vec2d(SyntheticFace.BaseX, 472), found.Chord.Stomion);
            Assert.Equal(new HashSet<string> { "canthus", "stomion", "chin" }, found.Levels.Keys.ToHashSet());
        }

        [Fact]
        public void StraightOutlineGivesPrior()
        {
            var found = Find();
            Assert.True(found.ApexPrior);
            CheekFixtures.Within(ApexHPrior, found.ApexH, 0.5 / SyntheticFace.BaseH);
            Assert.Equal(362, found.Apex.Y);
            CheekFixtures.Near(0.0, found.ApexDev, 9);
        }

        [Fact]
        public void BulgeOutsideWindowGivesPrior()
        {
            var found = Find(Bumps((0.60, 0.03, 20)));
            Assert.True(found.ApexPrior);
            CheekFixtures.Within(ApexHPrior, found.ApexH, 0.5 / SyntheticFace.BaseH);
        }

        [Fact]
        public void SmallBulgeGivesPrior()
        {
            var found = Find(Bumps((0.78, 0.005, 20)));
            Assert.True(found.ApexPrior);
            CheekFixtures.Within(ApexHPrior, found.ApexH, 0.5 / SyntheticFace.BaseH);
        }

        [Fact]
        public void TwoBulgesPickNearerPrior()
        {
            var found = Find(Bumps((0.74, 0.03, 10), (0.81, 0.03, 10)));
            Assert.False(found.ApexPrior);
            CheekFixtures.Within(0.81, found.ApexH, 0.01);
            found = Find(Bumps((0.76, 0.03, 10), (0.83, 0.03, 10)));
            Assert.False(found.ApexPrior);
            CheekFixtures.Within(0.76, found.ApexH, 0.01);
        }

        [Fact]
        public void StepInSmoothingHaloIsNotACandidate()
        {
            // A hump whose maximum lies below the window, rising through it, with the nose
            // rows set back 12 px: the box filter's drop at the step makes the last
            // unaffected row a local maximum, which the exclusion covers.
            (int First, int Last) nose = (372, 384);
            var found = Find(Bumps((0.62, 0.04, 75)), null, (nose.First, nose.Last, SyntheticFace.BaseX - 12.0, OutlineSrcNose));
            Assert.True(found.NoseOn);
            Assert.True(found.ApexPrior);
            Assert.Equal(362, found.Apex.Y);
            // The same hump without the step has no candidate either.
            var plain = Find(Bumps((0.62, 0.04, 75)));
            Assert.True(plain.ApexPrior);
            var (xs, rows, _) = SyntheticFace.Outline(Bumps((0.62, 0.04, 75)));
            var band = new List<double>();
            var bandRows = new List<int>();
            for (int i = 0; i < rows.Length; i++)
            {
                double x = rows[i] >= nose.First && rows[i] <= nose.Last ? (float)(SyntheticFace.BaseX - 12.0) : xs[i];
                if (rows[i] < 300 || rows[i] > 472) continue;
                band.Add(x - SyntheticFace.BaseX);
                bandRows.Add(rows[i]);
            }
            double[] dev = Smooth(band.ToArray()).Select(v => v / SyntheticFace.BaseH).ToArray();
            int k = bandRows.IndexOf(nose.First - SmoothRows / 2 - 1);
            Assert.True(dev[k] > dev[k - 1]);
            Assert.True(dev[k] > dev[k + 1]);
            Assert.True(dev[k] > ApexMinDev);
        }

        [Fact]
        public void StepAtContourOrEdgeRowsIsNotACandidate()
        {
            foreach (byte source in new[] { OutlineSrcContour, OutlineSrcEdge })
            {
                var found = Find(Bumps((0.62, 0.04, 75)), null, (372, 384, SyntheticFace.BaseX - 12.0, source));
                Assert.False(found.NoseOn);
                Assert.True(found.ApexPrior);
                Assert.Equal(362, found.Apex.Y);
                Assert.True(found.WindowMask < 1.0);
            }
            // The prior never lands on such a row while the window has a mask row.
            foreach (byte source in new[] { OutlineSrcContour, OutlineSrcEdge })
            {
                var found = Find(null, null, (360, 364, SyntheticFace.BaseX - 2.0, source));
                Assert.True(found.ApexPrior);
                Assert.Equal(365, found.Apex.Y);
                CheekFixtures.Near(SyntheticFace.BaseX, found.Apex.X, 4);
                Assert.Equal((SyntheticFace.BaseChinRow - 365) / SyntheticFace.BaseH, found.ApexH);
                Assert.True(CheekNumerics.IsFinite(found.SubmalarDepth));
            }
            var nose = Find(null, null, (360, 364, SyntheticFace.BaseX - 2.0, OutlineSrcNose));
            Assert.True(nose.ApexPrior);
            Assert.True(Math.Abs(nose.Apex.Y - 362) > SmoothRows / 2 + 2);
        }

        [Fact]
        public void SubmalarIsNeverAChordEnd()
        {
            var found = Find(Bumps((0.60, 0.02, 60)));
            Assert.True(found.ApexPrior);
            Assert.Equal(362, found.Apex.Y);
            Assert.Equal(472, found.Outline[^1].Y);
            Assert.True(362 < found.Submalar.Y && found.Submalar.Y < 472, found.Submalar.ToString());
            Assert.Equal(0.0, found.SubmalarPresent);
            Assert.True(found.SubmalarDepth <= 1e-6);
            found = Find();
            Assert.True(362 < found.Submalar.Y && found.Submalar.Y < 472, found.Submalar.ToString());
        }

        [Fact]
        public void NotchPresentWithDepth()
        {
            var found = Find(Bumps((0.55, -0.01, 30)));
            Assert.True(found.ApexPrior);
            CheekFixtures.Within(0.01, found.SubmalarDepth, 0.001);
            Assert.Equal(1.0, found.SubmalarPresent);
            Assert.True(Math.Abs(found.Submalar.Y - SyntheticFace.RowOf(0.55)) <= 2.0);
            CheekFixtures.Within(0.55, found.SubmalarH, 0.01);
            Assert.Equal(found.SubmalarDepth * found.Scale, found.SubmalarDepthPx);
            CheekFixtures.Within(3.0, found.SubmalarDepthPx, 0.3);
        }

        [Fact]
        public void NoNotchGivesZeroWithPoint()
        {
            var found = Find();
            CheekFixtures.Near(0.0, found.SubmalarDepth, 9);
            CheekFixtures.Near(0.0, found.SubmalarDepthPx, 6);
            Assert.Equal(0.0, found.SubmalarPresent);
            Assert.True(found.Submalar.IsFinite);
            CheekFixtures.Near(SyntheticFace.BaseX, found.Submalar.X, 4);
            Assert.True(362 <= found.Submalar.Y && found.Submalar.Y <= 472);
        }

        [Fact]
        public void NotchInsideNoseRowsSkipped()
        {
            var found = Find(Bumps((0.643, -0.03, 6)), null, (400, 414, null, OutlineSrcNose));
            Assert.True(found.NoseOn);
            Assert.True(found.BelowMask >= OutlineMinMask);
            CheekFixtures.Near(0.0, found.SubmalarDepth, 9);
            Assert.Equal(0.0, found.SubmalarPresent);
            Assert.False(395 <= found.Submalar.Y && found.Submalar.Y <= 419);
        }

        [Fact]
        public void TooFewMaskRowsGiveNanDepth()
        {
            var found = Find(Bumps((0.55, -0.01, 30)), null, (430, 472, null, OutlineSrcContour));
            Assert.True(found.BelowMask < OutlineMinMask);
            Assert.True(double.IsNaN(found.SubmalarDepth));
            Assert.True(double.IsNaN(found.SubmalarPresent));
            Assert.True(double.IsNaN(found.Submalar.X) && double.IsNaN(found.Submalar.Y));
        }

        [Fact]
        public void ContourFallbackGivesNanDepth()
        {
            var found = ObservedPoints(lm, null, null, null, true);
            Assert.NotNull(found);
            Assert.All(found!.OutlineSrc, s => Assert.Equal(OutlineSrcContour, s));
            Assert.Equal(0.0, found.MaskFraction);
            Assert.Equal(0.0, found.WindowMask);
            Assert.Equal(0.0, found.BelowMask);
            Assert.True(double.IsNaN(found.SubmalarDepth));
            Assert.True(double.IsNaN(found.SubmalarDepthPx));
            Assert.True(double.IsNaN(found.SubmalarPresent));
            Assert.True(double.IsNaN(found.Submalar.X) && double.IsNaN(found.Submalar.Y));
            Assert.True(double.IsNaN(found.SubmalarH));
            Assert.True(found.Apex.IsFinite);
            Assert.True(found.ApexPrior);
            Assert.False(found.NoseOn);
        }

        [Fact]
        public void ShortBandGivesNone()
        {
            var (xs, rows, src) = SyntheticFace.Outline(null, 300, 301);
            Assert.Null(ObservedPoints(lm, xs, rows, src, true));
            (xs, rows, src) = SyntheticFace.Outline(null, 300, 302);
            Assert.NotNull(ObservedPoints(lm, xs, rows, src, true));
            // Ragged or empty outlines give none.
            Assert.Null(ObservedPoints(lm, xs, rows.Take(2).ToArray(), src, true));
            Assert.Null(ObservedPoints(lm, Array.Empty<float>(), Array.Empty<int>(), Array.Empty<byte>(), true));
        }

        [Fact]
        public void CommissuresMovedChangeNothing()
        {
            var plain = Find(Bumps((0.78, 0.03, 20)));
            var moved = (Vec2d[])lm.Clone();
            moved[76] += new Vec2d(0, 40);
            moved[82] += new Vec2d(0, 40);
            var found = Find(Bumps((0.78, 0.03, 20)), moved);
            Assert.Equal(plain.ApexH, found.ApexH);
            Assert.Equal(plain.Apex, found.Apex);
            Assert.Equal(plain.SubmalarDepth, found.SubmalarDepth);
        }

        [Fact]
        public void AlarBasesMovedTiltH()
        {
            var plain = Find(Bumps((0.78, 0.03, 20)));
            var tilted = (Vec2d[])lm.Clone();
            tilted[59] = new(440, 390); // V at (1160, 300): the levels fan out
            var found = Find(Bumps((0.78, 0.03, 20)), tilted);
            Assert.NotNull(ObliqueFrameOf(tilted)!.Vanishing);
            Assert.False(found.ApexPrior);
            Assert.True(Math.Abs(found.ApexH - plain.ApexH) > 0.02);
            Assert.True(found.Scale < SyntheticFace.BaseH);
        }

        [Fact]
        public void NoseTipOnOutlineSetsNoseOn()
        {
            var on = (Vec2d[])lm.Clone();
            on[54] = new(SyntheticFace.BaseX, 394);
            var found = Find(null, on);
            Assert.True(found.NoseGap <= 0.0);
            Assert.True(found.NoseOn);
            var outside = (Vec2d[])lm.Clone();
            outside[54] = new(SyntheticFace.BaseX + 5, 394);
            found = Find(null, outside);
            Assert.True(found.NoseGap < 0.0);
            Assert.True(found.NoseOn);
            var inside = (Vec2d[])lm.Clone();
            inside[54] = new(SyntheticFace.BaseX - 0.048 * SyntheticFace.BaseH, 394);
            found = Find(null, inside);
            CheekFixtures.Near(0.048, found.NoseGap, 6);
            Assert.False(found.NoseOn);
            // Nose rows set the flag whatever the gap.
            found = Find(null, inside, (400, 410, null, OutlineSrcNose));
            Assert.True(found.NoseOn);
        }

        [Fact]
        public void CandidateNearNoseTipFlagged()
        {
            var high = (Vec2d[])lm.Clone();
            high[54] = new(480, 384); // the tip at h 0.72
            var found = Find(Bumps((0.73, 0.03, 10)), high);
            Assert.False(found.ApexPrior);
            Assert.True(found.ApexNearNose);
            found = Find(Bumps((0.78, 0.03, 10)), high);
            Assert.False(found.ApexNearNose);
        }

        [Fact]
        public void FloatLandmarksReadTheSame()
        {
            var (xs, rows, src) = SyntheticFace.Outline(Bumps((0.78, 0.03, 20)));
            var a = ObservedPoints(lm, xs, rows, src, true)!;
            var b = ObservedPoints(lm.Select(p => p.ToVec2()).ToArray(), xs, rows, src, true)!;
            Assert.Equal(a.Apex, b.Apex);
            Assert.Equal(a.ApexH, b.ApexH);
            Assert.Equal(a.SubmalarDepth, b.SubmalarDepth);
        }
    }

    // ------------------------------------------------------------ TestRenders

    public class Renders
    {
        // The render's entry, its mask from the background color, its raw landmarks and farLeft.
        static (JsonElement Render, bool[,] Foreground, Vec2d[] Landmarks, bool FarLeft) Load(string name)
        {
            JsonElement render = CheekFixtures.Renders.GetProperty("renders").GetProperty(name);
            bool[,] fg = CheekFixtures.RenderForeground("cheek/" + render.GetProperty("file").GetString());
            return (render, fg, CheekFixtures.RawLandmarks(name), render.GetProperty("far_side").GetString() == "left");
        }

        static (JsonElement Render, bool[,] Foreground, bool FarLeft, (float[] X, int[] Y, byte[] Src) Outline, ObliqueCheek Found)
            Analyze(string name)
        {
            var (render, fg, lm, farLeft) = Load(name);
            var outline = TraceFarCheek(fg, CheekFixtures.AlphaOf(fg), lm, farLeft);
            var found = ObservedPoints(lm, outline.X, outline.Y, outline.Src, farLeft);
            Assert.NotNull(found);
            return (render, fg, farLeft, outline, found!);
        }

        static int Silhouette(bool[,] fg, bool farLeft, int y)
        {
            int w = fg.GetLength(1);
            if (farLeft)
            {
                for (int x = w - 1; x >= 0; x--) if (fg[y, x]) return x;
            }
            else
            {
                for (int x = 0; x < w; x++) if (fg[y, x]) return x;
            }
            throw new InvalidOperationException($"row {y} is empty");
        }

        [Fact]
        public void TracesTheRenders()
        {
            var depths = new Dictionary<string, double>();
            foreach (string name in new[] { "yaw45", "yaw-45", "yaw45_hollow3" })
            {
                var (render, fg, farLeft, (xs, rows, src), found) = Analyze(name);
                Assert.True(src.Count(s => s == OutlineSrcMask) / (double)src.Length > 0.9, name);
                for (int i = 0; i < rows.Length; i++)
                    if (src[i] == OutlineSrcMask) Assert.True(Math.Abs(xs[i] - Silhouette(fg, farLeft, rows[i])) < 2, name);
                Assert.DoesNotContain(OutlineSrcNose, src);
                Assert.False(found.NoseOn);
                Assert.True(found.NoseGap > 0.0);
                Assert.Equal(1.0, found.MaskFraction);
                JsonElement oracle = render.GetProperty("apex");
                Assert.True(Math.Abs(found.Apex.X - oracle.GetProperty("point")[0].GetDouble()) < 1.0, name);
                int apexRow = (int)found.Apex.Y;
                Assert.True(Math.Abs(found.Apex.X - Silhouette(fg, farLeft, apexRow)) < 1.0, name);
                if (found.ApexPrior)
                {
                    Assert.Equal("yaw45", name);
                    CheekFixtures.Within(ApexHPrior, found.ApexH, 1.0 / found.Scale);
                }
                else
                {
                    int first = oracle.GetProperty("plateau_rows")[0].GetInt32(), last = oracle.GetProperty("plateau_rows")[1].GetInt32();
                    Assert.True(first <= apexRow && apexRow <= last, $"{name} {apexRow}");
                }
                Assert.True(found.ApexDev > ApexMinDev);
                Assert.True(CheekNumerics.IsFinite(found.SubmalarDepth));
                depths[name] = found.SubmalarDepth;
                if (name == "yaw45_hollow3")
                {
                    Assert.Equal(1.0, found.SubmalarPresent);
                    double depthPx = render.GetProperty("submalar").GetProperty("depth_px").GetDouble();
                    CheekFixtures.Within(depthPx, found.SubmalarDepthPx, 0.3 * depthPx);
                }
                else
                {
                    Assert.Equal(0.0, found.SubmalarPresent);
                }
            }
            Assert.True(depths["yaw45_hollow3"] > depths["yaw45"]);
            Assert.True(depths["yaw45_hollow3"] > 2 * depths["yaw-45"]);
        }

        [Fact]
        public void ContourDropoutInTheWindowKeepsThePrior()
        {
            var (_, _, farLeft, (xs, rows, src), plain) = Analyze("yaw45");
            Assert.True(plain.ApexPrior);
            CheekFixtures.Within(ApexHPrior, plain.ApexH, 0.002);
            var lm = CheekFixtures.RawLandmarks("yaw45");
            var (seed, seedRows, _) = ContourOutline(lm, farLeft);
            Assert.Equal(rows, seedRows);
            double outward = farLeft ? 1.0 : -1.0;
            var sel = Enumerable.Range(0, rows.Length).Where(i => rows[i] >= 334 && rows[i] <= 338).ToArray();
            Assert.Equal(5, sel.Length);
            foreach (int i in sel)
            {
                double step = (xs[i] - seed[i]) * outward;
                Assert.True(step > 1.0 && step < 2.0, step.ToString());
            }
            int row339 = Array.IndexOf(rows, 339);
            foreach (byte source in new[] { OutlineSrcContour, OutlineSrcEdge })
            {
                var xs2 = (float[])xs.Clone();
                var src2 = (byte[])src.Clone();
                foreach (int i in sel) { xs2[i] = seed[i]; src2[i] = source; }
                var found = ObservedPoints(lm, xs2, rows, src2, farLeft)!;
                Assert.True(found.ApexPrior);
                Assert.Equal(336, plain.Apex.Y);
                Assert.Equal(339, found.Apex.Y);
                CheekFixtures.Within(0.7830, found.ApexH, 0.001);
                CheekFixtures.Within(ApexHPrior, found.ApexH, 0.02);
                Assert.Equal(OutlineSrcMask, src2[row339]);
                Assert.Equal((double)xs[row339], found.Apex.X);
                Assert.True(found.WindowMask >= OutlineMinMask);
                Assert.True(found.WindowMask < 1.0);
                Assert.False(found.NoseOn);
                Assert.True(CheekNumerics.IsFinite(found.SubmalarDepth));
                CheekFixtures.Within(plain.SubmalarDepth, found.SubmalarDepth, 0.0002);
                Assert.Equal(plain.SubmalarPresent, found.SubmalarPresent);
                // The step did make a local maximum of the smoothed deviation on the
                // shoulder above it: only the exclusion keeps it out.
                Vec2d normal = OutwardNormal(found.Chord.Canthus, found.Chord.Stomion, outward)!.Value;
                double[] dev = Smooth(found.Outline.Select(p => Vec2d.Dot(p - found.Chord.Canthus, normal)).ToArray())
                    .Select(v => v / found.Scale).ToArray();
                int k = Array.FindIndex(found.Outline, p => p.Y == 334 - SmoothRows / 2 - 1);
                Assert.True(dev[k] > dev[k - 1]);
                Assert.True(dev[k] >= dev[k + 1]);
                Assert.True(dev[k] > ApexMinDev);
            }
        }

        [Fact]
        public void StandInRowsAtThePriorNeverCarryTheApexOrTheChord()
        {
            var (_, _, farLeft, (xs, rows, src), plain) = Analyze("yaw45");
            var lm = CheekFixtures.RawLandmarks("yaw45");
            var (seed, _, _) = ContourOutline(lm, farLeft);
            double outward = farLeft ? 1.0 : -1.0;
            var sel = Enumerable.Range(0, rows.Length).Where(i => rows[i] >= 334 && rows[i] <= 338).ToArray();
            Assert.Equal(336, plain.Apex.Y);
            Assert.Equal(0.0, plain.SubmalarPresent);
            int row339 = Array.IndexOf(rows, 339);
            foreach (var (source, shift) in new[] { (OutlineSrcEdge, 20.0), (OutlineSrcEdge, 8.0), (OutlineSrcContour, -10.0) })
            {
                var xs2 = (float[])xs.Clone();
                var src2 = (byte[])src.Clone();
                foreach (int i in sel) { xs2[i] = (float)(seed[i] + shift * outward); src2[i] = source; }
                var found = ObservedPoints(lm, xs2, rows, src2, farLeft)!;
                Assert.True(found.ApexPrior);
                Assert.Equal(339, found.Apex.Y);
                Assert.Equal((double)xs[row339], found.Apex.X);
                Assert.True(Math.Abs(found.Apex.X - plain.Apex.X) < 0.5);
                Assert.True(found.WindowMask >= OutlineMinMask);
                Assert.Equal(1.0, found.BelowMask);
                CheekFixtures.Within(plain.SubmalarDepth, found.SubmalarDepth, 0.0002);
                Assert.Equal(0.0, found.SubmalarPresent);
                Assert.Equal(plain.Submalar.Y, found.Submalar.Y);
                // The stand-in rows themselves sit off the canthus-to-stomion chord by most
                // of the shift.
                Vec2d normal = OutwardNormal(found.Chord.Canthus, found.Chord.Stomion, outward)!.Value;
                var standIns = found.Outline.Where(p => p.Y >= 334 && p.Y <= 338).ToArray();
                Assert.Equal(5, standIns.Length);
                double off = standIns.Min(p => Math.Abs(Vec2d.Dot(p - found.Chord.Canthus, normal)));
                Assert.True(off > 0.5 * Math.Abs(shift));
            }
        }

        [Fact]
        public void DilatedMasksReadTheOffsetNotTheNose()
        {
            foreach (var (name, wantNose) in new[] { ("yaw45", false), ("yaw-45", false), ("yaw45_hollow3", false), ("yaw50", true) })
            {
                var (_, fg0, lm, farLeft) = Load(name);
                int outward = farLeft ? 1 : -1;
                byte[,] alpha0 = CheekFixtures.AlphaOf(fg0);
                var (xs0, rows0, src0) = TraceFarCheek(fg0, alpha0, lm, farLeft);
                Assert.Equal(wantNose, src0.Contains(OutlineSrcNose));
                int h = fg0.GetLength(0), w = fg0.GetLength(1);
                for (int k = 1; k <= 4; k++)
                {
                    var fg = (bool[,])fg0.Clone();
                    for (int j = 1; j <= k; j++) // the far edge only: np.roll along the rows
                        for (int y = 0; y < h; y++)
                            for (int x = 0; x < w; x++)
                                fg[y, x] |= fg0[y, (((x - outward * j) % w) + w) % w];
                    byte[,] alpha = CheekFixtures.AlphaOf(fg);
                    var (xs, rows, src) = TraceFarCheek(fg, alpha, lm, farLeft);
                    var found = ObservedPoints(lm, xs, rows, src, farLeft)!;
                    int nose = src.Count(s => s == OutlineSrcNose);
                    Assert.Equal(wantNose, nose > 0);
                    Assert.Equal(wantNose, found.NoseOn);
                    if (wantNose) Assert.Equal(26, nose);
                    var (xsA, rowsA, srcA) = TraceFarCheek(fg, alpha0, lm, farLeft);
                    Assert.Equal(rows0, rowsA);
                    var both = Enumerable.Range(0, rows0.Length)
                        .Where(i => srcA[i] == OutlineSrcMask && src0[i] == OutlineSrcMask).ToArray();
                    Assert.True(both.Length > 240, $"{name} {k}");
                    if (k <= AlphaSearch)
                    {
                        foreach (int i in both) CheekFixtures.Close(xs0[i], xsA[i], atol: 1e-5);
                    }
                    else
                    {
                        foreach (int i in both) Assert.True(Math.Abs(xsA[i] - Math.Floor(xsA[i]) - 0.5) < 1e-6);
                        Assert.True(both.Min(i => Math.Abs(xsA[i] - xs0[i])) > k - 1);
                    }
                }
            }
        }

        [Fact]
        public void PerspectiveWarpOnRender()
        {
            var (_, _, farLeft, (xs, rows, src), plain) = Analyze("yaw-45");
            Assert.False(farLeft);
            Assert.False(plain.ApexPrior);
            Assert.True(plain.ApexNearNose);
            var warpedLm = SyntheticFace.Warp(CheekFixtures.RawLandmarks("yaw-45"));
            var frame = ObliqueFrameOf(warpedLm)!;
            Assert.NotNull(frame.Vanishing);
            Assert.True(frame.Vanishing!.Value.IsFinite);
            var (wXs, wRows, wSrc) = SyntheticFace.WarpOutline(xs, rows, src);
            var found = ObservedPoints(warpedLm, wXs, wRows, wSrc, farLeft)!;
            Assert.False(found.ApexPrior);
            Assert.True(Math.Abs(found.ApexH - plain.ApexH) < 0.01);
            CheekFixtures.Close(SyntheticFace.Warp(plain.Apex), found.Apex, atol: 2.0);
            Assert.True(found.ApexNearNose);
            Assert.Equal(1.0, found.MaskFraction);
            Assert.True(Math.Abs(found.SubmalarH - plain.SubmalarH) < 0.01);
            Assert.True(Math.Abs(found.SubmalarDepth - plain.SubmalarDepth) < 0.001);
        }

        [Fact]
        public void MirroredTurnsAgree()
        {
            var plus = Analyze("yaw45").Found;
            var minus = Analyze("yaw-45").Found;
            Assert.True(plus.FarLeft);
            Assert.False(minus.FarLeft);
            Assert.True(Math.Abs(plus.NoseGap - minus.NoseGap) < 0.01);
            Assert.True(Math.Abs(plus.SubmalarDepth - minus.SubmalarDepth) < 0.001);
        }
    }

    // ------------------------------------------------------------ TestSharedOutlines

    // The cases in tests_data/cheek_outlines.json, which dynaface-lib-python's
    // TestSharedOutlines checks too, so both ports must trace the same outline and find
    // the same points: the tracer's outputs (the sub-pixel x, the rows and each row's
    // source), the points, the flags and the fractions must equal the Python outputs
    // frozen there, and the derived heights, deviations and levels must agree to 1e-9.
    public class SharedOutlines
    {
        static JsonElement Data => CheekFixtures.Outlines;

        public static IEnumerable<object[]> CaseNames() =>
            Data.GetProperty("cases").EnumerateArray().Select(c => new object[] { c.GetProperty("name").GetString()! });

        public static IEnumerable<object[]> FrontalNames() =>
            Data.GetProperty("frontal").EnumerateArray().Select(c => new object[] { c.GetProperty("name").GetString()! });

        static JsonElement Case(string section, string name) =>
            Data.GetProperty(section).EnumerateArray().Single(c => c.GetProperty("name").GetString() == name);

        // An image from its run-length encoded rows ([value, count, ...] each).
        static byte[,] RleImage(JsonElement rows, int height, int width)
        {
            var image = new byte[height, width];
            int y = 0;
            foreach (JsonElement runs in rows.EnumerateArray())
            {
                int x = 0;
                var flat = runs.EnumerateArray().Select(e => e.GetInt32()).ToArray();
                for (int k = 0; k < flat.Length; k += 2)
                    for (int c = 0; c < flat[k + 1]; c++) image[y, x++] = (byte)flat[k];
                Assert.Equal(width, x);
                y++;
            }
            Assert.Equal(height, y);
            return image;
        }

        // The case's alpha, gray image (or null) and foreground mask.
        static (byte[,] Alpha, byte[,]? Gray, bool[,] Foreground) Inputs(JsonElement c)
        {
            Assert.Equal(MaskAlpha, Data.GetProperty("mask").GetProperty("mask_alpha").GetInt32());
            byte[,] alpha;
            byte[,]? gray = null;
            if (c.GetProperty("kind").GetString() == "synthetic")
            {
                int h = c.GetProperty("size")[0].GetInt32(), w = c.GetProperty("size")[1].GetInt32();
                alpha = RleImage(c.GetProperty("alpha"), h, w);
                if (c.GetProperty("gray").ValueKind != JsonValueKind.Null) gray = RleImage(c.GetProperty("gray"), h, w);
            }
            else
            {
                alpha = CheekFixtures.AlphaOf(CheekFixtures.RenderForeground(c.GetProperty("file").GetString()!));
            }
            var fg = new bool[alpha.GetLength(0), alpha.GetLength(1)];
            for (int y = 0; y < fg.GetLength(0); y++)
                for (int x = 0; x < fg.GetLength(1); x++) fg[y, x] = alpha[y, x] > MaskAlpha;
            return (alpha, gray, fg);
        }

        // snake_case -> PascalCase: apex_h -> ApexH, submalar_depth_px -> SubmalarDepthPx.
        static string Pascal(string snake) =>
            string.Concat(snake.Split('_').Select(part => char.ToUpperInvariant(part[0]) + part.Substring(1)));

        [Fact]
        public void TheCasesAreTheExpectedOnes()
        {
            Assert.Equal(new[] { "soft_disc", "nose_disc", "hair_disc", "yaw45", "yaw-45", "yaw45_hollow3", "yaw50" },
                CaseNames().Select(o => (string)o[0]));
        }

        [Theory]
        [MemberData(nameof(CaseNames))]
        public void TracesAndFindsThePythonOutputs(string name)
        {
            JsonElement c = Case("cases", name);
            var (alpha, gray, foreground) = Inputs(c);
            var lm = CheekFixtures.Points(c.GetProperty("landmarks"));
            bool farLeft = c.GetProperty("far_left").GetBoolean();
            var (xs, rows, src) = TraceFarCheek(foreground, alpha, lm, farLeft, gray: gray);
            JsonElement expected = c.GetProperty("outline");
            Assert.Equal(expected.GetProperty("x").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray(), xs);
            Assert.Equal(expected.GetProperty("rows").EnumerateArray().Select(e => e.GetInt32()).ToArray(), rows);
            Assert.Equal(expected.GetProperty("src").EnumerateArray().Select(e => (byte)e.GetInt32()).ToArray(), src);

            var found = ObservedPoints(lm, xs, rows, src, farLeft);
            Assert.NotNull(found);
            JsonElement ob = c.GetProperty("observed");
            // Every field of the Python dataclass has its property, and no other.
            var properties = typeof(ObliqueCheek).GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Select(p => p.Name).ToHashSet();
            Assert.Equal(ob.EnumerateObject().Select(p => Pascal(p.Name)).ToHashSet(), properties);

            Assert.Equal(ob.GetProperty("far_left").GetBoolean(), found!.FarLeft);
            Assert.Equal(CheekFixtures.Points(ob.GetProperty("outline")), found.Outline);
            Assert.Equal(ob.GetProperty("outline_src").EnumerateArray().Select(e => (byte)e.GetInt32()).ToArray(), found.OutlineSrc);
            var chord = CheekFixtures.Points(ob.GetProperty("chord"));
            Assert.Equal(chord[0], found.Chord.Canthus);
            Assert.Equal(chord[1], found.Chord.Stomion);
            var levels = ob.GetProperty("levels");
            Assert.Equal(levels.EnumerateObject().Select(p => p.Name).ToHashSet(), found.Levels.Keys.ToHashSet());
            foreach (var (key, level) in found.Levels)
            {
                var want = CheekFixtures.Points(levels.GetProperty(key));
                CheekFixtures.Close(want[0], level.Point, atol: 1e-9, rtol: 0, what: key);
                CheekFixtures.Close(want[1], level.Direction, atol: 1e-9, rtol: 0, what: key);
            }
            Assert.Equal(CheekFixtures.Point(ob.GetProperty("apex")), found.Apex);
            Assert.Equal(CheekFixtures.Point(ob.GetProperty("submalar")), found.Submalar);
            Assert.Equal(ob.GetProperty("apex_prior").GetBoolean(), found.ApexPrior);
            Assert.Equal(ob.GetProperty("apex_near_nose").GetBoolean(), found.ApexNearNose);
            Assert.Equal(ob.GetProperty("nose_on").GetBoolean(), found.NoseOn);
            Assert.Equal(CheekFixtures.Number(ob.GetProperty("submalar_present")), found.SubmalarPresent);
            Assert.Equal(CheekFixtures.Number(ob.GetProperty("mask_fraction")), found.MaskFraction);
            Assert.Equal(CheekFixtures.Number(ob.GetProperty("window_mask")), found.WindowMask);
            Assert.Equal(CheekFixtures.Number(ob.GetProperty("below_mask")), found.BelowMask);
            foreach (string field in new[] { "apex_h", "apex_dev", "submalar_h", "submalar_depth", "submalar_depth_px", "nose_gap", "scale" })
            {
                double actual = (double)typeof(ObliqueCheek).GetProperty(Pascal(field))!.GetValue(found)!;
                CheekFixtures.Close(CheekFixtures.Number(ob.GetProperty(field)), actual, atol: 1e-9, rtol: 0, what: field);
            }
        }

        [Fact]
        public void SourcesExercised()
        {
            var sources = new Dictionary<string, HashSet<int>>();
            var sides = new HashSet<bool>();
            var present = new List<double>();
            foreach (JsonElement c in Data.GetProperty("cases").EnumerateArray())
            {
                sources[c.GetProperty("name").GetString()!] =
                    c.GetProperty("outline").GetProperty("src").EnumerateArray().Select(e => e.GetInt32()).ToHashSet();
                sides.Add(c.GetProperty("far_left").GetBoolean());
                present.Add(CheekFixtures.Number(c.GetProperty("observed").GetProperty("submalar_present")));
            }
            Assert.Equal(new HashSet<bool> { true, false }, sides);
            Assert.Equal(new HashSet<int> { OutlineSrcContour, OutlineSrcMask }, sources["soft_disc"]);
            Assert.Contains(OutlineSrcNose, sources["nose_disc"]);
            Assert.Contains(OutlineSrcEdge, sources["hair_disc"]);
            Assert.Contains(OutlineSrcNose, sources["yaw50"]);
            Assert.Contains(1.0, present);
            Assert.Contains(0.0, present);
            Assert.Contains(present, double.IsNaN);
        }

        [Theory]
        [MemberData(nameof(FrontalNames))]
        public void FrontalConstructionsMatchPython(string name)
        {
            JsonElement c = Case("frontal", name);
            var lm = CheekFixtures.Points(c.GetProperty("landmarks"));
            var fc = Cheek.FrontalConstructions(lm);
            Assert.NotNull(fc);
            JsonElement expected = c.GetProperty("constructions");
            JsonElement frame = expected.GetProperty("frame");
            CheekFixtures.Close(CheekFixtures.Point(frame.GetProperty("origin")), fc!.Frame.Origin, atol: 1e-9, rtol: 0);
            CheekFixtures.Close(CheekFixtures.Point(frame.GetProperty("across")), fc.Frame.Across, atol: 1e-9, rtol: 0);
            CheekFixtures.Close(CheekFixtures.Point(frame.GetProperty("down")), fc.Frame.Down, atol: 1e-9, rtol: 0);
            CheekFixtures.Close(frame.GetProperty("ipd").GetDouble(), fc.Frame.Ipd, atol: 1e-9, rtol: 0);
            var properties = typeof(Cheek.FrontalCheek).GetProperties().Select(p => p.Name).ToHashSet();
            Assert.Equal(expected.EnumerateObject().Select(p => Pascal(p.Name)).ToHashSet(), properties);
            foreach (JsonProperty field in expected.EnumerateObject())
            {
                if (field.Name == "frame") continue;
                object actual = typeof(Cheek.FrontalCheek).GetProperty(Pascal(field.Name))!.GetValue(fc)!;
                if (field.Name.EndsWith("_clip"))
                    Assert.Equal(field.Value.GetInt32(), (int)actual);
                else if (actual is Vec2d point)
                    CheekFixtures.Close(CheekFixtures.Point(field.Value), point, atol: 1e-9, rtol: 0, what: field.Name);
                else
                    CheekFixtures.Close(field.Value.GetDouble(), (double)actual, atol: 1e-9, rtol: 0, what: field.Name);
            }
        }
    }
}
