using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using static CheekOutline;

namespace DynafaceTests;

// A drawing call MeasureCheekMeasures made through its canvas, recorded as dynaface-lib's
// test MockFace records one: the kind ("line", "circle", "polyline" or "text"), its
// points (a text's bottom-left), color, a circle's radius, the thickness (negative for a
// filled dot) and a text's string.
internal sealed class CheekDrawCall
{
    public string Kind = "";
    public (int X, int Y)[] Points = Array.Empty<(int X, int Y)>();
    public Rgba32 Color;
    public int Radius;
    public int Thickness;
    public string? Text;

    public (int X, int Y) P0 => Points[0];
    public (int X, int Y) P1 => Points[1];

    public override string ToString() =>
        $"{Kind} [{string.Join(" ", Points.Select(p => $"({p.X},{p.Y})"))}] {Color} r={Radius} t={Thickness} \"{Text}\"";
}

// Records what the measure draws: dynaface-lib's MockFace, whose calc_text_size makes a
// label 12 px per character wide, so the label positions match its tests'.
internal sealed class RecordingCanvas : ICheekCanvas
{
    public readonly List<CheekDrawCall> Calls = new();

    public RecordingCanvas(int width) => Width = width;

    public int Width { get; }

    // Python's MockFace.calc_text_size: ((12 * len(txt), 20), 5).
    public (int Width, int Height, int Depth) TextSize(string text) => (12 * text.Length, 20, 5);

    public void Line((int X, int Y) a, (int X, int Y) b, Rgba32 color, int thickness) =>
        Calls.Add(new CheekDrawCall { Kind = "line", Points = new[] { a, b }, Color = color, Thickness = thickness });

    public void Circle((int X, int Y) center, Rgba32 color, int radius, int thickness) =>
        Calls.Add(new CheekDrawCall
        {
            Kind = "circle", Points = new[] { center }, Color = color, Radius = radius, Thickness = thickness,
        });

    public void Polyline((int X, int Y)[] points, Rgba32 color, int thickness) =>
        Calls.Add(new CheekDrawCall { Kind = "polyline", Points = points.ToArray(), Color = color, Thickness = thickness });

    public void Text((int X, int Y) origin, string text, Rgba32 color) =>
        Calls.Add(new CheekDrawCall { Kind = "text", Points = new[] { origin }, Color = color, Text = text });

    public List<CheekDrawCall> Kinds(string kind) => Calls.Where(c => c.Kind == kind).ToList();

    // The point labels, in drawing order (the text lines go to the sidebar instead).
    public List<string> Labels() => Kinds("text").Select(c => c.Text!).ToList();

    public List<string> Repr() => Calls.Select(c => c.ToString()).ToList();
}

// The fixture faces of dynaface-lib's test_measures_cheek.py as measure contexts, without
// models: a frontal face from an entry's pipeline landmarks (truncated to whole pixels, as
// the Python fixture does) with its head pose and pix2mm, and a quarter face from a
// render's raw landmarks with its outline traced from the render's own background color.
// The head pose is float, as the pipeline's is, so a JSON yaw such as 1.08 is 1.08f here.
internal static class CheekFaces
{
    public static Vec2[] Ints(IEnumerable<Vec2d> points) =>
        points.Select(p => new Vec2((float)Math.Truncate(p.X), (float)Math.Truncate(p.Y))).ToArray();

    static FaceImage Blank(JsonElement size)
    {
        int w = size[0].GetInt32(), h = size[1].GetInt32();
        return new FaceImage(new Rgba32[w * h], w, h);
    }

    static float F(JsonElement e, string name) => (float)e.GetProperty(name).GetDouble();

    public static float PipelineYaw(string name) => F(CheekFixtures.Entry(name).GetProperty("pipeline"), "yaw");
    public static float PipelinePitch(string name) => F(CheekFixtures.Entry(name).GetProperty("pipeline"), "pitch");
    public static float PipelinePix2mm(string name) => F(CheekFixtures.Entry(name).GetProperty("pipeline"), "pix2mm");
    public static float RawYaw(string name) => F(CheekFixtures.Entry(name).GetProperty("raw"), "yaw");
    public static float RawPitch(string name) => F(CheekFixtures.Entry(name).GetProperty("raw"), "pitch");

    public static Vec2[] FrontalLandmarks(string name = "img4") => Ints(CheekFixtures.PipelineLandmarks(name));

    // A frontal face (or, with lateral, a profile) from an entry's pipeline landmarks,
    // head pose and pix2mm.
    public static FaceMeasureContext Frontal(string name = "img4", float? yaw = null, float? pitch = null,
        Vec2[]? landmarks = null, bool lateral = false)
    {
        JsonElement p = CheekFixtures.Entry(name).GetProperty("pipeline");
        var headPose = new[] { yaw ?? F(p, "yaw"), pitch ?? F(p, "pitch"), F(p, "roll"), 0f, 0f, 0f };
        return FaceMeasureContext.WithFixedScale(Blank(p.GetProperty("size")), landmarks ?? FrontalLandmarks(name),
            F(p, "pix2mm"), isLateral: lateral, headPose: headPose,
            pose: lateral ? FacePose.Lateral : FacePose.Frontal);
    }

    static readonly ConcurrentDictionary<string, (Vec2[] Landmarks, bool FarLeft, (float[] X, int[] Y, byte[] Src) Outline)>
        TracedCache = new();

    // A render's raw landmarks, far side and outline, traced from the mask its background
    // color defines. Copies, so a test may change them.
    public static (Vec2[] Landmarks, bool FarLeft, (float[] X, int[] Y, byte[] Src) Outline) Traced(string name)
    {
        var (lm, farLeft, (x, y, src)) = TracedCache.GetOrAdd(name, n =>
        {
            JsonElement render = CheekFixtures.Renders.GetProperty("renders").GetProperty(n);
            bool[,] fg = CheekFixtures.RenderForeground("cheek/" + render.GetProperty("file").GetString());
            Vec2[] landmarks = Ints(CheekFixtures.RawLandmarks(n));
            bool left = render.GetProperty("far_side").GetString() == "left";
            var outline = TraceFarCheek(fg, CheekFixtures.AlphaOf(fg), landmarks.Select(v => (Vec2d)v).ToArray(), left);
            return (landmarks, left, outline);
        });
        return ((Vec2[])lm.Clone(), farLeft, ((float[])x.Clone(), (int[])y.Clone(), (byte[])src.Clone()));
    }

    // A quarter face from a render: its raw landmarks and traced outline (or the outline
    // given, (null, null, null) for the contour fallback), at the render's size.
    public static FaceMeasureContext Quarter(string name = "yaw45", float? yaw = null, float? pitch = null,
        (float[]? X, int[]? Y, byte[]? Src)? outline = null, Vec2[]? landmarks = null)
    {
        var (lm, farLeft, traced) = Traced(name);
        JsonElement entry = CheekFixtures.Entry(name), raw = entry.GetProperty("raw");
        var headPose = new[] { yaw ?? F(raw, "yaw"), pitch ?? F(raw, "pitch"), F(raw, "roll"), 0f, 0f, 0f };
        var ctx = new FaceMeasureContext(Blank(entry.GetProperty("size")), landmarks ?? lm,
            headPose: headPose, pose: FacePose.Quarter);
        var (x, y, src) = outline ?? traced;
        ctx.SetQuarterView(farLeft, x, y, src);
        return ctx;
    }

    public static Vec2[] SketchLandmarks() => Ints(CheekFixtures.RawLandmarks("quarter_sketch"));

    // The app's three-quarter sketch as a quarter face without an outline (the contour
    // fallback, unless one is given). Its nose points to the image left, so the far cheek
    // is the subject's right.
    public static FaceMeasureContext Sketch((float[]? X, int[]? Y, byte[]? Src)? outline = null)
    {
        JsonElement entry = CheekFixtures.Entry("quarter_sketch"), raw = entry.GetProperty("raw");
        var headPose = new[] { F(raw, "yaw"), F(raw, "pitch"), F(raw, "roll"), 0f, 0f, 0f };
        var ctx = new FaceMeasureContext(Blank(entry.GetProperty("size")), SketchLandmarks(),
            headPose: headPose, pose: FacePose.Quarter);
        var (x, y, src) = outline ?? (null, null, null);
        ctx.SetQuarterView(false, x, y, src);
        return ctx;
    }

    // The cheek landmarks' 20 coordinates MeasureLandmarks reports for a face.
    public static Dictionary<string, double> Coords(FaceMeasureContext ctx)
    {
        var result = new MeasureLandmarks().Calc(ctx, render: false);
        return Cheek.CheekLandmarkOutputs.ToDictionary(name => name, name => result[name]);
    }

    // A copy of an object of public read-write properties (Cheek.FrontalCheek,
    // CheekOutline.ObliqueCheek), as dataclasses.replace copies one in Python.
    public static T Copy<T>(T source) where T : class, new()
    {
        var copy = new T();
        foreach (PropertyInfo p in typeof(T).GetProperties(BindingFlags.Public | BindingFlags.Instance))
            if (p.CanRead && p.CanWrite) p.SetValue(copy, p.GetValue(source));
        return copy;
    }
}

// Ports dynaface-lib's tests/test_measures_cheek.py: MeasureCheekMeasures on fixture faces
// without models (the six items, the declared outputs, render independence, the drawing
// recorded through its canvas and the text on frontal, quarter and lateral faces, each item
// gating exactly its own drawing), and the Landmarks measure's "Cheek Landmarks" item.
//
// Not ported: the checks that each drawn point is a tuple of Python ints (the canvas
// takes (int, int)), that every value is a Python float accepted by video._number (every
// value is a double), that no numpy RuntimeWarning is raised, the text column's
// analyze_next_pt positions (the text goes to the sidebar here), and the "mock face
// without cheek_landmarks gets cheek.locate()" cases (a context always has its accessors;
// LocateAll is compared instead). Where Python records draw_cheek_landmarks() or circle()
// calls through mock.patch, these tests read the drawn pixels.
public class MeasureCheekMeasuresTests
{
    internal const string Zygion = MeasureCheekMeasures.ItemZygion;
    internal const string Reference = MeasureCheekMeasures.ItemReference;
    internal const string Outline = MeasureCheekMeasures.ItemOutline;
    internal const string Apex = MeasureCheekMeasures.ItemApex;
    internal const string Submalar = MeasureCheekMeasures.ItemSubmalar;
    internal const string HeadTurn = MeasureCheekMeasures.ItemHeadTurn;

    // The six items, in order, with the views they suit and their outputs.
    internal static readonly (string Name, FacePose[] Views, string[] Outputs)[] Items =
    {
        (Zygion, new[] { FacePose.Frontal }, new[]
        {
            "zy.r", "zy.l", "zy.w", "zy.ratio", "zy.ratio.ok", "zy.r.clip", "zy.l.clip", "zy.r.ok", "zy.l.ok", "zy.ok",
        }),
        (Reference, new[] { FacePose.Frontal }, new[] { "cr.r.d", "cr.l.d", "cr.ok", "sr.ok" }),
        (Outline, new[] { FacePose.Quarter }, new[] { "cheek.far", "cheek.mask", "cheek.nose", "cheek.nose.gap", "cheek.scale" }),
        (Apex, new[] { FacePose.Quarter }, new[]
        {
            "apex.r.h", "apex.l.h", "apex.r.dev", "apex.l.dev", "apex.r.prior", "apex.l.prior",
            "apex.r.near_nose", "apex.l.near_nose", "apex.ok",
        }),
        (Submalar, new[] { FacePose.Quarter }, new[]
        {
            "sub.r.h", "sub.l.h", "sub.r.depth", "sub.l.depth", "sub.r.depth.px", "sub.l.depth.px",
            "sub.r.present", "sub.l.present", "sub.ok",
        }),
        (HeadTurn, new[] { FacePose.Frontal, FacePose.Quarter }, new[] { "cheek.yaw", "cheek.yaw.model", "cheek.pitch", "cheek.pitch.ok" }),
    };

    internal static readonly string[] ItemNames = Items.Select(i => i.Name).ToArray();
    internal static string[] OutputsOf(params string[] items) =>
        Items.Where(i => items.Contains(i.Name)).SelectMany(i => i.Outputs).ToArray();
    // The 41 outputs, in CSV order.
    internal static readonly string[] ExpectedNames = Items.SelectMany(i => i.Outputs).ToArray();
    // The frontal items' outputs (NaN on a quarter face), the quarter items' (NaN on a
    // frontal face) and the head turn's (numeric in either view).
    internal static readonly string[] FrontalNames = OutputsOf(Zygion, Reference);
    internal static readonly string[] ObliqueNames = OutputsOf(Outline, Apex, Submalar);
    internal static readonly string[] SharedNames = OutputsOf(HeadTurn);
    internal static readonly string[] OkFlags = ExpectedNames.Where(n => n.EndsWith(".ok")).ToArray();
    internal static readonly string[] CoordNames = Cheek.CheekLandmarkOutputs.ToArray();

    static readonly Rgba32 Orange = MeasureCheekMeasures.Orange, Blue = MeasureCheekMeasures.Blue,
        Cyan = MeasureCheekMeasures.Cyan, Red = MeasureCheekMeasures.Red, Green = MeasureCheekMeasures.Green,
        Gray = MeasureCheekMeasures.Gray, Magenta = MeasureCheekMeasures.Magenta,
        Yellow = MeasureCheekMeasures.Yellow, White = MeasureCheekMeasures.White;

    // A Cheek Measures measure with only these items enabled.
    internal static MeasureCheekMeasures Only(params string[] names)
    {
        var measure = new MeasureCheekMeasures();
        foreach (var item in measure.Items) item.Enabled = names.Contains(item.Name);
        return measure;
    }

    internal static bool Same(double a, double b) => (double.IsNaN(a) && double.IsNaN(b)) || a == b;

    static List<string> Flagged(params string[] reasons) =>
        reasons.Select(r => MeasureCheekMeasures.FlagPrefix + r).ToList();

    // The measure's text lines on the context: its sidebar lines after its header, before
    // the spacer (dynaface-lib's text column).
    internal static List<string> Column(FaceMeasureContext ctx)
    {
        var lines = ctx.TextLines;
        if (lines.Count == 0) return new List<string>();
        Assert.Equal("CHEEK MEASURES", lines[0]);
        Assert.Equal("", lines[^1]);
        return lines.Skip(1).Take(lines.Count - 2).Select(line =>
        {
            Assert.StartsWith("  ", line);
            return line.Substring(2);
        }).ToList();
    }

    // One run of the measure on a context, its drawing recorded.
    internal sealed class Run
    {
        public readonly Dictionary<string, double> Result;
        public readonly RecordingCanvas Canvas;
        public readonly List<string> Column;
        public readonly FaceMeasureContext Ctx;

        public Run(FaceMeasureContext ctx, MeasureCheekMeasures? measure = null, bool render = true, int? width = null,
            Func<Cheek.FrontalCheek?>? frontal = null, Func<ObliqueCheek?>? oblique = null)
        {
            Ctx = ctx;
            ctx.TextLines.Clear();
            Canvas = new RecordingCanvas(width ?? ctx.Width);
            Result = (measure ?? new MeasureCheekMeasures()).Calc(ctx, render, Canvas, frontal, oblique);
            Column = MeasureCheekMeasuresTests.Column(ctx);
        }

        public double this[string name] => Result[name];
        public List<CheekDrawCall> Kinds(string kind) => Canvas.Kinds(kind);
        public List<string> Labels() => Canvas.Labels();
        public List<CheekDrawCall> Calls => Canvas.Calls;
    }

    static void AssertSameValues(Dictionary<string, double> expected, Dictionary<string, double> actual,
        IEnumerable<string>? names = null)
    {
        foreach (string name in names ?? ExpectedNames)
            Assert.True(Same(expected[name], actual[name]), $"{name}: {expected[name]:R} != {actual[name]:R}");
    }

    static ObliqueCheek Observed(FaceMeasureContext ctx) =>
        ObservedPoints(ctx.Landmarks, ctx.CheekOutlineX, ctx.CheekOutlineY, ctx.CheekOutlineSrc, ctx.FarLeft)!;

    // ------------------------------------------------------------ TestStructure

    public class Structure
    {
        readonly MeasureCheekMeasures _measure = new();

        [Fact]
        public void TheLabel() => Assert.Equal("CHEEK MEASURES", _measure.Label);

        [Fact]
        public void SixItemsInOrder()
        {
            Assert.Equal(new[]
            {
                "Zygion width", "Reference points", "Far cheek outline", "Malar apex", "Submalar point", "Head turn",
            }, ItemNames);
            Assert.Equal(ItemNames, _measure.Items.Select(i => i.Name));
            Assert.Equal(ItemNames, MeasureCheekMeasures.ItemNames);
            foreach (var (item, (name, views, outputs)) in _measure.Items.Zip(Items))
            {
                Assert.False(item.DisplayOnly, name);
                Assert.Equal(outputs, item.OutputNames());
                Assert.Equal(views, item.Views);
            }
            Assert.Equal(Items.Select(i => (i.Name, string.Join(",", i.Views), string.Join(",", i.Outputs))),
                MeasureCheekMeasures.ItemDefinitions.Select(d => (d.Name, string.Join(",", d.Views), string.Join(",", d.Outputs))));
        }

        [Fact]
        public void NamesInOrder()
        {
            Assert.Equal(ExpectedNames, _measure.Outputs());
            Assert.Equal(41, ExpectedNames.Length);
            Assert.Equal(new[] { 10, 4, 5, 9, 9, 4 }, Items.Select(i => i.Outputs.Length));
            Assert.Equal(ExpectedNames, MeasureCheekMeasures.AllOutputs);
            // No coordinates (MeasureLandmarks') and no drawing switches.
            foreach (string name in CoordNames) Assert.DoesNotContain(name, ExpectedNames);
            foreach (string name in ItemNames) Assert.DoesNotContain(name, ExpectedNames);
        }

        [Fact]
        public void ViewFlags()
        {
            Assert.True(_measure.Enabled);
            Assert.True(_measure.IsFrontal);
            Assert.True(_measure.IsQuarter);
            Assert.False(_measure.IsLateral);
            Assert.True(_measure.Suits(false));
            Assert.True(_measure.Suits(false, quarter: true));
            Assert.False(_measure.Suits(true));
            foreach (var (item, (name, views, _)) in _measure.Items.Zip(Items))
            {
                Assert.Equal(views.Contains(FacePose.Frontal), item.IsFrontal);
                Assert.Equal(views.Contains(FacePose.Quarter), item.IsQuarter);
                Assert.False(item.IsLateral, name);
                Assert.True(item.Enabled, name);
            }
            // UpdateForType leaves each item on only where it suits the face.
            _measure.UpdateForType(false, quarter: true);
            Assert.Equal(new[] { Outline, Apex, Submalar, HeadTurn }, _measure.Items.Where(i => i.Enabled).Select(i => i.Name));
            _measure.UpdateForType(false);
            Assert.Equal(new[] { Zygion, Reference, HeadTurn }, _measure.Items.Where(i => i.Enabled).Select(i => i.Name));
            Assert.Equal(FrontalNames.Concat(SharedNames), DynafaceMeasures.GetAllItems(new FaceMeasureBase[] { _measure }));
        }

        [Fact]
        public void RegisteredLast() => Assert.IsType<MeasureCheekMeasures>(DynafaceMeasures.AllMeasures()[^1]);

        [Fact]
        public void PtRoundsToInts()
        {
            Assert.Equal((2, 2), MeasureCheekMeasures.Pt(new Vec2d(1.5, 2.49)));
            Assert.Equal((-1, 10), MeasureCheekMeasures.Pt(new Vec2d(-0.6, 10.0)));
            Assert.Equal((2, 4), MeasureCheekMeasures.Pt(new Vec2d(2.5, 3.5))); // half to even, as round()
        }

        [Fact]
        public void OutlineColorsCoverEverySource()
        {
            Assert.Equal(new byte[] { OutlineSrcContour, OutlineSrcMask, OutlineSrcNose, OutlineSrcEdge }.OrderBy(b => b),
                MeasureCheekMeasures.OutlineColors.Keys.OrderBy(b => b));
            Assert.Equal(4, MeasureCheekMeasures.OutlineColors.Values.Distinct().Count());
        }

        // .NET-only: the numbers are written as Python's format writes them.
        [Fact]
        public void NumbersReadAsPythonWritesThem()
        {
            Assert.Equal("nan", MeasureCheekMeasures.Fixed(double.NaN, 1));
            Assert.Equal("inf", MeasureCheekMeasures.Fixed(double.PositiveInfinity, 1));
            Assert.Equal("-inf", MeasureCheekMeasures.Fixed(double.NegativeInfinity, 2));
            Assert.Equal("0.2", MeasureCheekMeasures.Fixed(0.25, 1));    // a tie, to even
            Assert.Equal("0.8", MeasureCheekMeasures.Fixed(0.75, 1));
            Assert.Equal("0.12", MeasureCheekMeasures.Fixed(0.125, 2));
            Assert.Equal("4.5", MeasureCheekMeasures.Fixed(4.45, 1));    // 4.4500000000000002
            Assert.Equal("4.4", MeasureCheekMeasures.Fixed(4.45f, 1));   // 4.4499998
            Assert.Equal("-0.0", MeasureCheekMeasures.Fixed(-0.04, 1));
            Assert.Equal("0.000", MeasureCheekMeasures.Fixed(-2.6e-05, 3, signedZero: false));
            Assert.Equal("-0.012", MeasureCheekMeasures.Fixed(-0.0123, 3, signedZero: false));
            Assert.Equal("130.6", MeasureCheekMeasures.Fixed(130.6, 1));
            Assert.Equal("40", MeasureCheekMeasures.Fixed(40.0, 0));
            Assert.Equal("12345678901234567168.0", MeasureCheekMeasures.Fixed(12345678901234567890.0, 1));
            Assert.Equal("0.0000", MeasureCheekMeasures.Fixed(double.Epsilon, 4));
        }

        // Python's util.symmetry_ratio, NaN arguments included.
        [Fact]
        public void SymmetryRatio()
        {
            Assert.Equal(1.0, MeasureCheekMeasures.SymmetryRatio(0.0, 0.0));
            Assert.Equal(0.5, MeasureCheekMeasures.SymmetryRatio(1.0, 2.0));
            Assert.Equal(0.5, MeasureCheekMeasures.SymmetryRatio(2.0, 1.0));
            Assert.True(double.IsNaN(MeasureCheekMeasures.SymmetryRatio(double.NaN, 1.0)));
            Assert.Equal(1.0, MeasureCheekMeasures.SymmetryRatio(1.0, double.NaN));
        }
    }

    // ------------------------------------------------------------ TestDeclaredOutputs

    // Every declared output comes back on a frontal and a quarter face, nothing on a
    // profile, whether or not the items are enabled and whether or not it renders, with
    // the same values either way.
    public class DeclaredOutputs
    {
        static void Check(Func<FaceMeasureContext> makeFace, string[] expected)
        {
            foreach (bool enabled in new[] { true, false })
            {
                var measure = new MeasureCheekMeasures();
                foreach (var item in measure.Items) item.Enabled = enabled;
                var results = new Dictionary<bool, Dictionary<string, double>>();
                foreach (bool render in new[] { false, true })
                {
                    var run = new Run(makeFace(), measure, render);
                    Assert.Equal(expected.OrderBy(n => n), run.Result.Keys.OrderBy(n => n));
                    results[render] = run.Result;
                    if (!render || !enabled)
                    {
                        Assert.Empty(run.Calls);
                        Assert.Empty(run.Column);
                    }
                }
                AssertSameValues(results[false], results[true], expected);
            }
        }

        [Fact]
        public void Frontal() => Check(() => CheekFaces.Frontal(), ExpectedNames);

        [Fact]
        public void Quarter() => Check(() => CheekFaces.Quarter(), ExpectedNames);

        [Fact]
        public void QuarterFallback() => Check(() => CheekFaces.Sketch(), ExpectedNames);

        [Fact]
        public void Lateral() => Check(() => CheekFaces.Frontal(lateral: true), Array.Empty<string>());

        // In the declared order, through the public Calc too.
        [Fact]
        public void InDeclaredOrder()
        {
            foreach (var ctx in new[] { CheekFaces.Frontal(), CheekFaces.Quarter(), CheekFaces.Sketch() })
                Assert.Equal(ExpectedNames, new MeasureCheekMeasures().Calc(ctx, render: false).Keys);
        }

        [Fact]
        public void FlagsAre01OrNaN()
        {
            // Only the two clip codes may read 2 (Cheek.ClipBottom); every other flag is 0.0
            // or 1.0, or NaN where the view doesn't define it.
            var flags = OkFlags.Concat(new[]
            {
                "zy.r.clip", "zy.l.clip", "apex.r.prior", "apex.l.prior", "apex.r.near_nose", "apex.l.near_nose",
                "sub.r.present", "sub.l.present", "cheek.far", "cheek.nose",
            });
            foreach (var ctx in new[] { CheekFaces.Frontal(), CheekFaces.Quarter(), CheekFaces.Sketch() })
            {
                var result = new MeasureCheekMeasures().Calc(ctx, render: false);
                foreach (string name in flags)
                {
                    double value = result[name];
                    var allowed = name.EndsWith(".clip") ? new[] { 0.0, 1.0, 2.0 } : new[] { 0.0, 1.0 };
                    Assert.True(double.IsNaN(value) || allowed.Contains(value), $"{name}: {value}");
                }
            }
        }

        [Fact]
        public void TooFewLandmarksGiveNaNNotAnException()
        {
            foreach (bool quarter in new[] { false, true })
            {
                var ctx = quarter
                    ? CheekFaces.Quarter(landmarks: CheekFaces.Traced("yaw45").Landmarks.Take(50).ToArray())
                    : CheekFaces.Frontal(landmarks: CheekFaces.FrontalLandmarks().Take(50).ToArray());
                var run = new Run(ctx);
                Assert.Equal(ExpectedNames.OrderBy(n => n), run.Result.Keys.OrderBy(n => n));
                foreach (string name in FrontalNames.Concat(ObliqueNames))
                    if (name != "cheek.far") Assert.True(double.IsNaN(run[name]), name);
                Assert.True(double.IsNaN(run["cheek.yaw"]));
                Assert.False(double.IsNaN(run["cheek.yaw.model"]));
                Assert.Equal(1.0, run["cheek.pitch.ok"]);
                // Nothing to draw but the pose line, and on a quarter face why.
                Assert.Empty(run.Calls);
                Assert.StartsWith("cheek yaw", run.Column[0]);
                Assert.Contains("(flagged)", run.Column[0]);
                Assert.Equal(quarter ? Flagged("the far cheek could not be measured") : new List<string>(),
                    run.Column.Skip(1));
            }
        }

        // A NaN chin: no pupil frame, so both modules and cheek.yaw are NaN in either view,
        // whether or not it renders, and the cheek landmarks agree (all NaN).
        [Fact]
        public void NonFiniteLandmarkGivesNaNNotAnException()
        {
            foreach (bool quarter in new[] { false, true })
                foreach (bool render in new[] { false, true })
                {
                    Vec2[] landmarks = quarter ? CheekFaces.Traced("yaw45").Landmarks : CheekFaces.FrontalLandmarks();
                    landmarks[Cheek.LmChin] = new Vec2(landmarks[Cheek.LmChin].X, float.NaN);
                    var ctx = quarter ? CheekFaces.Quarter(landmarks: landmarks) : CheekFaces.Frontal(landmarks: landmarks);
                    var run = new Run(ctx, render: render);
                    Assert.Equal(ExpectedNames.OrderBy(n => n), run.Result.Keys.OrderBy(n => n));
                    foreach (string name in FrontalNames.Concat(ObliqueNames).Append("cheek.yaw"))
                        if (name != "cheek.far") Assert.True(double.IsNaN(run[name]), name);
                    Assert.True(Same(quarter ? 1.0 : double.NaN, run["cheek.far"]));
                    Assert.False(double.IsNaN(run["cheek.yaw.model"]));
                    Assert.Equal(1.0, run["cheek.pitch.ok"]);
                    Assert.Empty(run.Calls);
                    if (render)
                    {
                        // The text alone: the pose line, on a quarter face why.
                        Assert.Contains("(flagged)", run.Column[0]);
                        Assert.Equal(quarter ? Flagged("the far cheek could not be measured") : new List<string>(),
                            run.Column.Skip(1));
                    }
                    else Assert.Empty(run.Column);
                    Vec2d[] located = Cheek.LocateAll(ctx.Landmarks, quarter ? FacePose.Quarter : FacePose.Frontal,
                        ctx.FarLeft, ctx.CheekOutlineX, ctx.CheekOutlineY, ctx.CheekOutlineSrc);
                    Assert.All(located, p => Assert.True(double.IsNaN(p.X) && double.IsNaN(p.Y)));
                }
        }
    }

    // ------------------------------------------------------------ TestFrontal

    // The frontal core on img4's pipeline landmarks.
    public class Frontal
    {
        readonly FaceMeasureContext _ctx = CheekFaces.Frontal();
        readonly Run _run;

        public Frontal() => _run = new Run(_ctx);

        [Fact]
        public void ZygionWidth()
        {
            CheekFixtures.Within(130.6, _run["zy.w"], 0.2);
            CheekFixtures.Near(130.6, _run["zy.r"] + _run["zy.l"], 0);
            CheekFixtures.Within(65.5, _run["zy.r"], 0.1);
            CheekFixtures.Within(65.1, _run["zy.l"], 0.1);
            CheekFixtures.Within(65.06 / 65.53, _run["zy.ratio"], 0.002);
            CheekFixtures.Within(28.8, _run["cr.r.d"], 0.1);
            CheekFixtures.Within(28.8, _run["cr.l.d"], 0.1);
        }

        [Fact]
        public void ValuesFollowTheConstructions()
        {
            var fc = Cheek.FrontalConstructions(_ctx.Landmarks)!;
            var c = CheekFaces.Coords(_ctx); // MeasureLandmarks' Cheek Landmarks item
            Assert.Equal(fc.ZygionR, new Vec2d(c["zy.r.x"], c["zy.r.y"]));
            Assert.Equal(fc.ZygionL, new Vec2d(c["zy.l.x"], c["zy.l.y"]));
            Assert.Equal(fc.CheekRefR, new Vec2d(c["cr.r.x"], c["cr.r.y"]));
            Assert.Equal(fc.CheekRefL, new Vec2d(c["cr.l.x"], c["cr.l.y"]));
            Assert.Equal(fc.SubmalarRefR, new Vec2d(c["sr.r.x"], c["sr.r.y"]));
            Assert.Equal(fc.SubmalarRefL, new Vec2d(c["sr.l.x"], c["sr.l.y"]));
            Assert.Equal(fc.ZygionRPx * (double)_ctx.Pix2mm, _run["zy.r"]);
            Assert.Equal(fc.CheekRefLPx * (double)_ctx.Pix2mm, _run["cr.l.d"]);
        }

        [Fact]
        public void GatesPassOnImg4()
        {
            foreach (string name in new[] { "zy.r.ok", "zy.l.ok", "zy.ok", "zy.ratio.ok", "cr.ok", "sr.ok" })
                Assert.Equal(1.0, _run[name]);
            Assert.Equal(0.0, _run["zy.r.clip"]);
            Assert.Equal(0.0, _run["zy.l.clip"]);
            Assert.Equal(1.0, _run["cheek.pitch.ok"]);
            Assert.Equal((double)0.05f, _run["cheek.pitch"]);
            CheekFixtures.Within(-0.885, _run["cheek.yaw"], 0.01);
            Assert.Equal(Cheek.CalibrateYaw(1.08f), _run["cheek.yaw.model"]);
            // A raw 1.08: below the level head's reading (1.5) by less than the table's
            // first step, so the model reads a level head.
            CheekFixtures.Within(0.0, _run["cheek.yaw.model"], 0.01);
        }

        [Fact]
        public void ObliqueOutputsNaN()
        {
            foreach (string name in ObliqueNames) Assert.True(double.IsNaN(_run[name]), name);
        }

        [Fact]
        public void DrawingCalls()
        {
            var lines = _run.Kinds("line");
            // The level line, then the two canthus verticals.
            Assert.Equal(((201, 558), (818, 558)), (lines[0].P0, lines[0].P1));
            Assert.Equal((Orange, 2), (lines[0].Color, lines[0].Thickness));
            Assert.Equal(3, lines.Count);
            foreach (var call in lines.Skip(1))
            {
                Assert.Equal((Orange, 3), (call.Color, call.Thickness));
                Assert.Equal(call.P0.X, call.P1.X);     // vertical
                Assert.True(call.P0.Y < call.P1.Y);     // canthus above the foot
            }
            var circles = _run.Kinds("circle");
            Assert.Equal(new[]
            {
                ((240, 558), White, 9, -1), ((240, 558), Blue, 7, -1), ((778, 558), White, 9, -1), ((778, 558), Blue, 7, -1),
            }, circles.Take(4).Select(c => (c.P0, c.Color, c.Radius, c.Thickness)));
            var rings = circles.Skip(4).ToList();
            Assert.Equal(4, rings.Count);
            Assert.All(rings, c => Assert.Equal((Orange, 7, 3), (c.Color, c.Radius, c.Thickness)));
            Assert.Equal(new[]
            {
                "zygion", "zygion", "cheek ref", "submalar ref", "cheek ref", "submalar ref",
            }, _run.Labels());
            var texts = _run.Kinds("text");
            Assert.Equal(new[] { Blue, Blue, Orange, Orange, Orange, Orange }, texts.Select(t => t.Color));
            // The right side's labels sit left of the point, the left side's right of it: the
            // zygion's above it, the reference points' beside their rings, centered on the
            // ring's row (the recording canvas's text is 20 px tall).
            Assert.True(texts[0].P0.X < 240);
            Assert.Equal((778 + 14, 558 - 22), texts[1].P0);
            Assert.Equal((240 - 12 * "zygion".Length - 14, 558 - 22), texts[0].P0);
            var outsideLeft = new[] { true, true, false, false };
            for (int i = 0; i < 4; i++)
            {
                var (x, y) = rings[i].P0;
                int width = 12 * texts[2 + i].Text!.Length;
                Assert.Equal((outsideLeft[i] ? x - 14 - width : x + 14, y + 10), texts[2 + i].P0);
            }
        }

        [Fact]
        public void TextColumn()
        {
            Assert.Equal(new[]
            {
                "zy.w=130.6 mm (R 65.5 / L 65.1, ratio 0.99)",
                "cheek ref depth R 28.8 / L 28.8 mm",
                "cheek yaw -0.9 (model 0.0) pitch 0.1",
            }, _run.Column);
            // Three lines at most for any one face, after the measure's header.
            Assert.True(_run.Column.Count <= 3);
            Assert.Equal(new[] { "CHEEK MEASURES" }, _ctx.TextLines.Take(1));
        }

        [Fact]
        public void ItemsOffDrawNothing()
        {
            var run = new Run(CheekFaces.Frontal(), Only());
            Assert.Empty(run.Calls);
            Assert.Empty(run.Column);
            Assert.Empty(run.Ctx.TextLines);
            AssertSameValues(_run.Result, run.Result);
        }

        [Fact]
        public void ZygionItemWithTheHeadTurn()
        {
            var run = new Run(CheekFaces.Frontal(), Only(Zygion, HeadTurn));
            Assert.Single(run.Kinds("line"));
            Assert.Equal(4, run.Kinds("circle").Count);
            Assert.Equal(new[] { "zygion", "zygion" }, run.Labels());
            Assert.Equal(new[] { "zy.w=130.6 mm (R 65.5 / L 65.1, ratio 0.99)", "cheek yaw -0.9 (model 0.0) pitch 0.1" },
                run.Column);
        }

        [Fact]
        public void ReferenceItemWithTheHeadTurn()
        {
            var run = new Run(CheekFaces.Frontal(), Only(Reference, HeadTurn));
            Assert.Equal(2, run.Kinds("line").Count);
            Assert.Equal(4, run.Kinds("circle").Count);
            Assert.Equal(4, run.Labels().Count);
            Assert.Equal(new[] { "cheek ref depth R 28.8 / L 28.8 mm", "cheek yaw -0.9 (model 0.0) pitch 0.1" }, run.Column);
        }

        // Each item alone draws its own share of the full drawing, in the same calls, and
        // writes its own line; the values never change.
        [Fact]
        public void EachItemGatesExactlyItsOwnDrawing()
        {
            var full = _run.Calls;
            var lines = full.Where(c => c.Kind == "line").ToList();
            var circles = full.Where(c => c.Kind == "circle").ToList();
            var texts = full.Where(c => c.Kind == "text").ToList();
            var expected = new Dictionary<string, (List<CheekDrawCall> Calls, List<string> Column)>
            {
                [Zygion] = (lines.Take(1).Concat(circles.Take(4)).Concat(texts.Take(2)).ToList(), new() { _run.Column[0] }),
                [Reference] = (lines.Skip(1).Concat(circles.Skip(4)).Concat(texts.Skip(2)).ToList(), new() { _run.Column[1] }),
                [Outline] = (new(), new()),
                [Apex] = (new(), new()),
                [Submalar] = (new(), new()),
                [HeadTurn] = (new(), new() { _run.Column[2] }),
            };
            foreach (string name in ItemNames)
            {
                var run = new Run(CheekFaces.Frontal(), Only(name));
                var (calls, column) = expected[name];
                Assert.Equal(calls.Select(c => c.ToString()).OrderBy(s => s, StringComparer.Ordinal),
                    run.Canvas.Repr().OrderBy(s => s, StringComparer.Ordinal));
                Assert.Equal(column, run.Column);
                Assert.Equal(ExpectedNames, run.Result.Keys);
                AssertSameValues(_run.Result, run.Result);
            }
        }

        [Fact]
        public void HeadTurnLineAlone()
        {
            // The pose line is the head turn's, whatever else is on or off.
            var run = new Run(CheekFaces.Frontal(), Only(HeadTurn));
            Assert.Equal(new[] { "cheek yaw -0.9 (model 0.0) pitch 0.1" }, run.Column);
            Assert.Empty(run.Calls);
            run = new Run(CheekFaces.Frontal(), Only(Zygion, Reference));
            Assert.Equal(_run.Column.Take(2), run.Column);
        }

        [Fact]
        public void RenderFalseDrawsNothing()
        {
            var run = new Run(CheekFaces.Frontal(), render: false);
            Assert.Empty(run.Calls);
            AssertSameValues(_run.Result, run.Result);
            // Through the public Calc, the image is left as it was; render: true draws on it.
            var ctx = CheekFaces.Frontal();
            AssertSameValues(_run.Result, new MeasureCheekMeasures().Calc(ctx, render: false));
            Assert.True(ctx.Pixels.All(p => p == default(Rgba32)));
            Assert.Empty(ctx.TextLines);
            ctx = CheekFaces.Frontal();
            AssertSameValues(_run.Result, new MeasureCheekMeasures().Calc(ctx, render: true));
            Assert.Contains(ctx.Pixels, p => p == Orange);
            Assert.Contains(ctx.Pixels, p => p == Blue);
            Assert.Equal(_run.Column, Column(ctx));
        }

        [Fact]
        public void Pitch25FailsOnlyPitchOk()
        {
            var run = new Run(CheekFaces.Frontal(pitch: 25f));
            Assert.Equal(25.0, run["cheek.pitch"]);
            Assert.Equal(0.0, run["cheek.pitch.ok"]);
            AssertSameValues(_run.Result, run.Result, ExpectedNames.Except(new[] { "cheek.pitch", "cheek.pitch.ok" }));
            Assert.Equal("cheek yaw -0.9 (model 0.0) pitch 25.0 (flagged)", run.Column[^1]);
            Assert.Equal(_run.Column[0], run.Column[0]); // zy.ok still 1
        }

        [Fact]
        public void NaNYawFailsEveryYawGate()
        {
            var run = new Run(CheekFaces.Frontal(yaw: float.NaN));
            Assert.True(double.IsNaN(run["cheek.yaw.model"]));
            Assert.False(double.IsNaN(run["cheek.yaw"])); // from the landmarks
            foreach (string name in new[] { "zy.r.ok", "zy.l.ok", "zy.ok", "zy.ratio.ok", "cr.ok", "sr.ok" })
                Assert.Equal(0.0, run[name]);
            Assert.Equal(1.0, run["cheek.pitch.ok"]); // pitch is its own flag
            CheekFixtures.Near(_run["zy.w"], run["zy.w"]); // still reported
            Assert.Contains("(flagged)", run.Column[0]);
            Assert.Contains("(flagged)", run.Column[^1]);
            // Gray dots and rings for a failed gate.
            Assert.Equal(new[] { White, Gray, White, Gray, Gray, Gray, Gray, Gray }, run.Kinds("circle").Select(c => c.Color));
            // With the pitch unknown too, every flag is 0.
            run = new Run(CheekFaces.Frontal(yaw: float.NaN, pitch: float.NaN), render: false);
            var flags = OkFlags.Select(n => run[n]).Where(v => !double.IsNaN(v)).ToList();
            Assert.Equal(7, flags.Count);
            Assert.All(flags, v => Assert.Equal(0.0, v));
        }

        [Fact]
        public void TurnedHeadIsReportedAndFlagged()
        {
            // The 10 degree render: the frontal core is computed, with the gates 0.
            Assert.Equal("frontal", CheekFixtures.Entry("yaw10").GetProperty("pipeline").GetProperty("pose").GetString());
            var run = new Run(CheekFaces.Frontal("yaw10"), render: false);
            CheekFixtures.Within(10.0, run["cheek.yaw"], 2.0);
            Assert.False(double.IsNaN(run["zy.w"]));
            foreach (string name in new[] { "zy.r.ok", "zy.l.ok", "zy.ok", "zy.ratio.ok", "cr.ok", "sr.ok" })
                Assert.Equal(0.0, run[name]);
        }

        [Fact]
        public void DisagreeingModelYawFailsTheGate()
        {
            var run = new Run(CheekFaces.Frontal(yaw: 28f), render: false); // the model says 45; the landmarks 0
            CheekFixtures.Within(44.5, run["cheek.yaw.model"], 0.5);
            foreach (string name in new[] { "zy.r.ok", "zy.l.ok", "zy.ok", "zy.ratio.ok", "cr.ok", "sr.ok" })
                Assert.Equal(0.0, run[name]);
        }

        [Fact]
        public void ClippedZygion()
        {
            // The right contour's top (0-4) pushed below the level: the level passes above
            // it, the point is landmark 0 and only that side is flagged.
            Vec2[] landmarks = CheekFaces.FrontalLandmarks()
                .Select((p, i) => i < 5 ? new Vec2(p.X, p.Y + 250) : p).ToArray();
            var ctx = CheekFaces.Frontal(landmarks: landmarks);
            var run = new Run(ctx);
            Assert.Equal((double)Cheek.ClipTop, run["zy.r.clip"]);
            Assert.Equal(0.0, run["zy.l.clip"]);
            var c = CheekFaces.Coords(ctx);
            Assert.Equal(((double)landmarks[0].X, (double)landmarks[0].Y), (c["zy.r.x"], c["zy.r.y"]));
            Assert.Equal(0.0, run["zy.r.ok"]);
            Assert.Equal(1.0, run["zy.l.ok"]);
            Assert.Equal(0.0, run["zy.ok"]);
            Assert.Equal(0.0, run["zy.ratio.ok"]);
            Assert.Equal(1.0, run["cr.ok"]);
            Assert.Equal(new[] { "zygion (clipped)", "zygion" }, run.Labels().Take(2));
            Assert.Equal(new[] { White, Gray, White, Blue }, run.Kinds("circle").Take(4).Select(k => k.Color));
            Assert.EndsWith(" (flagged)", run.Column[0]);
            Assert.EndsWith(" (flagged)", run.Column[^1]);
        }

        [Fact]
        public void FrontalCheekAccessorIsRead()
        {
            // Constructions handed over in place of the context's: here the ones of the
            // landmarks shifted 100 px right, a sentinel the landmarks would never give.
            Vec2[] shifted = CheekFaces.FrontalLandmarks().Select(p => new Vec2(p.X + 100, p.Y)).ToArray();
            var sentinel = Cheek.FrontalConstructions(shifted);
            var run = new Run(CheekFaces.Frontal(), frontal: () => sentinel);
            Assert.Equal(_run["zy.w"], run["zy.w"]);
            Assert.Equal((240 + 100, 558), run.Kinds("circle")[0].P0);
            // And an accessor with nothing gives NaN, whatever the landmarks.
            run = new Run(CheekFaces.Frontal(), frontal: () => null);
            foreach (string name in FrontalNames) Assert.True(double.IsNaN(run[name]), name);
            Assert.False(double.IsNaN(run["cheek.yaw"]));
            Assert.Equal(new[] { "cheek yaw -0.9 (model 0.0) pitch 0.1 (flagged)" }, run.Column);
        }

        [Fact]
        public void CoincidentPupilsGiveNaN()
        {
            Vec2[] landmarks = CheekFaces.FrontalLandmarks();
            landmarks[97] = landmarks[96];
            var run = new Run(CheekFaces.Frontal(landmarks: landmarks));
            Assert.Equal(ExpectedNames.OrderBy(n => n), run.Result.Keys.OrderBy(n => n));
            foreach (string name in FrontalNames.Concat(ObliqueNames).Append("cheek.yaw"))
                Assert.True(double.IsNaN(run[name]), name);
            Assert.Equal(1.0, run["cheek.pitch.ok"]);
        }

        // A NaN lateral canthus: Calc(render: true) returns wherever Calc(render: false)
        // does, with the same values, and nothing throws.
        [Fact]
        public void NaNCanthusNeverThrowsWhenRendering()
        {
            // In the landmarks: no frame at all, so nothing frontal is reported or drawn,
            // and cheek.yaw is NaN too.
            var runs = new Dictionary<bool, Run>();
            foreach (bool render in new[] { false, true })
            {
                Vec2[] landmarks = CheekFaces.FrontalLandmarks();
                landmarks[Cheek.LmCanthusL] = new Vec2(float.NaN, float.NaN);
                runs[render] = new Run(CheekFaces.Frontal(landmarks: landmarks), render: render);
            }
            Assert.Equal(ExpectedNames.OrderBy(n => n), runs[true].Result.Keys.OrderBy(n => n));
            AssertSameValues(runs[false].Result, runs[true].Result);
            foreach (string name in FrontalNames.Concat(ObliqueNames).Append("cheek.yaw"))
                Assert.True(double.IsNaN(runs[true][name]), name);
            Assert.Empty(runs[true].Calls); // the pose line only
            Assert.Equal(new[] { "cheek yaw nan (model 0.0) pitch 0.1 (flagged)" }, runs[true].Column);
            Assert.Empty(runs[false].Calls);

            // Handed over with the right canthus NaN (so its foot, reference points and
            // zygion too): those points are skipped, not rounded, the left side is drawn,
            // and the values are render: false's.
            var fc = Cheek.FrontalConstructions(CheekFaces.FrontalLandmarks())!;
            var broken = CheekFaces.Copy(fc);
            broken.CanthusR = broken.FootR = broken.CheekRefR = broken.SubmalarRefR = broken.ZygionR = Vec2d.NaN;
            broken.CheekRefRPx = broken.ZygionRPx = double.NaN;
            runs.Clear();
            foreach (bool render in new[] { false, true })
                runs[render] = new Run(CheekFaces.Frontal(), render: render, frontal: () => broken);
            AssertSameValues(runs[false].Result, runs[true].Result);
            foreach (string name in new[] { "zy.r", "zy.w", "zy.ratio", "cr.r.d" })
                Assert.True(double.IsNaN(runs[true][name]), name);
            foreach (string name in new[] { "zy.l", "cr.l.d", "cheek.yaw" })
                Assert.False(double.IsNaN(runs[true][name]), name);
            var run = runs[true];
            // No level line (its right end is NaN) and one vertical; the left zygion's rim
            // and dot and the left side's two rings; three labels.
            Assert.Single(run.Kinds("line"));
            Assert.Equal(4, run.Kinds("circle").Count);
            Assert.Equal(new[] { "zygion", "cheek ref", "submalar ref" }, run.Labels());
            Assert.Equal(3, run.Column.Count);
            Assert.Empty(runs[false].Calls);
        }
    }

    // ------------------------------------------------------------ TestQuarter

    // The oblique module on the yaw45 render's traced outline.
    public class Quarter
    {
        readonly FaceMeasureContext _ctx = CheekFaces.Quarter();
        readonly Run _run;

        public Quarter() => _run = new Run(_ctx);

        [Fact]
        public void FarSideOnly()
        {
            Assert.True(_ctx.FarLeft);
            Assert.Equal(1.0, _run["cheek.far"]);
            foreach (string name in ObliqueNames)
                Assert.Equal(name.Contains(".r."), double.IsNaN(_run[name]));
            foreach (string name in FrontalNames) Assert.True(double.IsNaN(_run[name]), name);
        }

        [Fact]
        public void ValuesFollowObservedPoints()
        {
            var ob = Observed(_ctx);
            Assert.True(ob.FarLeft);
            var c = CheekFaces.Coords(_ctx); // MeasureLandmarks' Cheek Landmarks item
            Assert.Equal(ob.Apex, new Vec2d(c["apex.l.x"], c["apex.l.y"]));
            Assert.Equal(ob.ApexH, _run["apex.l.h"]);
            Assert.Equal(ob.ApexDev, _run["apex.l.dev"]);
            Assert.Equal(ob.Submalar, new Vec2d(c["sub.l.x"], c["sub.l.y"]));
            Assert.Equal(ob.SubmalarH, _run["sub.l.h"]);
            Assert.Equal(ob.SubmalarDepth, _run["sub.l.depth"]);
            Assert.Equal(ob.SubmalarDepthPx, _run["sub.l.depth.px"]);
            Assert.Equal(ob.SubmalarPresent, _run["sub.l.present"]);
            Assert.Equal(ob.MaskFraction, _run["cheek.mask"]);
            Assert.Equal(ob.NoseGap, _run["cheek.nose.gap"]);
            Assert.Equal(ob.Scale, _run["cheek.scale"]);
        }

        [Fact]
        public void KnownFactsOfYaw45()
        {
            CheekFixtures.Within(45.0, _run["cheek.yaw"], 0.1);
            CheekFixtures.Within(45.0, _run["cheek.yaw.model"], 0.01);
            Assert.Equal((double)4.45f, _run["cheek.pitch"]);
            Assert.Equal(1.0, _run["cheek.pitch.ok"]);
            Assert.Equal(1.0, _run["apex.l.prior"]);
            CheekFixtures.Within(ApexHPrior, _run["apex.l.h"], 0.005);
            Assert.Equal(0.0, _run["apex.l.near_nose"]);
            Assert.Equal(0.0, _run["cheek.nose"]);
            Assert.True(_run["cheek.nose.gap"] > 0.04);
            Assert.Equal(1.0, _run["cheek.mask"]);
            Assert.Equal(1.0, _run["apex.ok"]);
            Assert.Equal(0.0, _run["sub.l.present"]);
            Assert.True(_run["sub.l.depth"] < SubmalarMinDepth);
            CheekFixtures.Near(_run["sub.l.depth"] * _run["cheek.scale"], _run["sub.l.depth.px"]);
            Assert.Equal(1.0, _run["sub.ok"]);
            Assert.True(_run["cheek.scale"] > 200.0);
        }

        [Fact]
        public void MirroredTurnNearNose()
        {
            // On the plain -45 render the found apex lies within 0.02 of the nose tip's h,
            // so near_nose folds into apex.ok, and the model's calibrated yaw reads -44.8.
            var ctx = CheekFaces.Quarter("yaw-45");
            Assert.False(ctx.FarLeft);
            var run = new Run(ctx);
            Assert.Equal(0.0, run["cheek.far"]);
            CheekFixtures.Within(-45.0, run["cheek.yaw"], 0.2);
            CheekFixtures.Within(-44.77, run["cheek.yaw.model"], 0.01);
            Assert.Equal(1.0, run["apex.r.near_nose"]);
            Assert.Equal(0.0, run["apex.r.prior"]);
            Assert.Equal(0.0, run["apex.ok"]);
            Assert.Equal(0.0, run["sub.ok"]);
            Assert.False(double.IsNaN(run["sub.r.depth"]));
            foreach (string name in ObliqueNames.Where(n => n.Contains(".l.")))
                Assert.True(double.IsNaN(run[name]), name);
            Assert.Equal("malar apex (nose?)", run.Labels()[0]);
            Assert.EndsWith(" (nose?)", run.Column[0]);
            Assert.EndsWith(" (absent) (flagged)", run.Column[1]);
            Assert.EndsWith(" (flagged)", run.Column[2]);
            Assert.Equal(Flagged("the apex is level with the nose tip"), run.Column.Skip(3));
            // The right cheek's labels go left of the point.
            int apexX = run.Kinds("circle")[0].P0.X;
            Assert.True(run.Kinds("text")[0].P0.X < apexX);
            // The far-right outline's level ticks point to the image right (inward).
            foreach (var call in run.Kinds("line").Take(3)) Assert.True(call.P1.X > call.P0.X);
        }

        [Fact]
        public void HollowRender()
        {
            var run = new Run(CheekFaces.Quarter("yaw45_hollow3"));
            Assert.Equal(1.0, run["sub.l.present"]);
            Assert.True(run["sub.l.depth"] >= SubmalarMinDepth);
            Assert.Equal(1.0, run["apex.ok"]);
            Assert.Equal(1.0, run["sub.ok"]);
            var circle = run.Kinds("circle")[^1];
            Assert.Equal((Green, 7, -1), (circle.Color, circle.Radius, circle.Thickness)); // a filled dot
            Assert.Equal("submalar point", run.Labels()[^1]);
            Assert.StartsWith("sub.l depth=", run.Column[1]);
            Assert.EndsWith(" (hollow)", run.Column[1]);
        }

        [Fact]
        public void DrawingCalls()
        {
            var polylines = _run.Kinds("polyline");
            Assert.Single(polylines); // every band row from the mask
            Assert.Equal((Cyan, 4), (polylines[0].Color, polylines[0].Thickness));
            var rows = polylines[0].Points.Select(p => p.Y).ToList();
            Assert.Equal(rows.OrderBy(r => r), rows);
            Assert.True(rows.Count > 100);
            var lines = _run.Kinds("line");
            Assert.Equal(4, lines.Count);
            foreach (var call in lines.Take(3)) // the canthus, stomion and chin levels, inward
            {
                Assert.Equal((Orange, 1), (call.Color, call.Thickness));
                Assert.True(call.P1.X < call.P0.X);
                double length = Math.Sqrt(Math.Pow(call.P1.X - call.P0.X, 2) + Math.Pow(call.P1.Y - call.P0.Y, 2));
                CheekFixtures.Within(0.25 * _run["cheek.scale"], length, 1.5);
            }
            Assert.Equal(polylines[0].Points[0], lines[0].P0); // from the outline
            Assert.Equal(polylines[0].Points[^1], lines[1].P0);
            Assert.Equal(((int)_ctx.Landmarks[16].X, (int)_ctx.Landmarks[16].Y), lines[2].P0); // the chin
            var chord = lines[3];
            Assert.Equal((Orange, 2), (chord.Color, chord.Thickness));
            Assert.Equal((lines[0].P0, lines[1].P0), (chord.P0, chord.P1));
            var circles = _run.Kinds("circle");
            var c = CheekFaces.Coords(_ctx);
            var apex = MeasureCheekMeasures.Pt(new Vec2d(c["apex.l.x"], c["apex.l.y"]));
            Assert.Equal((apex, White, 9, -1), (circles[0].P0, circles[0].Color, circles[0].Radius, circles[0].Thickness));
            Assert.Equal((apex, Red, 7, -1), (circles[1].P0, circles[1].Color, circles[1].Radius, circles[1].Thickness));
            // The absent submalar point is reported but not drawn: on a convex cheek it
            // sits beside a chord end, so a ring there marked nothing.
            Assert.Equal(0.0, _run["sub.l.present"]);
            Assert.False(double.IsNaN(c["sub.l.x"]));
            Assert.Equal(2, circles.Count);
            Assert.Equal(new[] { "malar apex (prior)" }, _run.Labels());
            var texts = _run.Kinds("text");
            Assert.Single(texts);
            // Away from the face, beside the dot and centered on its row.
            Assert.Equal((apex.X + 14, apex.Y + 10), texts[0].P0);
            Assert.Equal(Red, texts[0].Color);
        }

        [Fact]
        public void TextColumn()
        {
            // The pitch is 4.45f, as the pipeline's float head pose holds it, which reads
            // 4.4 (dynaface-lib's fixture keeps the JSON's double 4.45, which reads 4.5).
            Assert.Equal(new[]
            {
                "apex.l h=0.79 dev=0.016 (prior)",
                "sub.l depth=0.001 (absent)",
                "cheek yaw 45.0 (model 45.0) pitch 4.4",
            }, _run.Column);
            Assert.True(_run.Column.Count <= 5);
        }

        [Fact]
        public void ItemsOffDrawNothing()
        {
            // The frontal items stay on and draw nothing on a quarter face.
            var run = new Run(CheekFaces.Quarter(), Only(Zygion, Reference));
            Assert.Empty(run.Calls);
            Assert.Empty(run.Column);
            AssertSameValues(_run.Result, run.Result);
        }

        // Each item alone draws its own share of the full drawing, in the same calls, and
        // writes its own line; the values never change.
        [Fact]
        public void EachItemGatesExactlyItsOwnDrawing()
        {
            var full = _run.Calls;
            var outline = full.Where(c => c.Kind is "polyline" or "line").ToList();
            var apex = full.Where(c => c.Kind is "circle" or "text").ToList();
            var expected = new Dictionary<string, (List<CheekDrawCall> Calls, List<string> Column)>
            {
                [Zygion] = (new(), new()),
                [Reference] = (new(), new()),
                [Outline] = (outline, new()),
                [Apex] = (apex, new() { _run.Column[0] }),
                [Submalar] = (new(), new() { _run.Column[1] }), // absent: no dot, a line
                [HeadTurn] = (new(), new() { _run.Column[2] }),
            };
            foreach (string name in ItemNames)
            {
                var run = new Run(CheekFaces.Quarter(), Only(name));
                var (calls, column) = expected[name];
                Assert.Equal(calls.Select(c => c.ToString()).OrderBy(s => s, StringComparer.Ordinal),
                    run.Canvas.Repr().OrderBy(s => s, StringComparer.Ordinal));
                Assert.Equal(column, run.Column);
                Assert.Equal(ExpectedNames, run.Result.Keys);
                AssertSameValues(_run.Result, run.Result);
            }
            // A present hollow: the submalar item alone draws its dot and label.
            var hollow = new Run(CheekFaces.Quarter("yaw45_hollow3"), Only(Submalar));
            Assert.Equal(new[] { "circle", "text" }, hollow.Calls.Select(c => c.Kind));
            Assert.Equal((Green, 7, -1), (hollow.Calls[0].Color, hollow.Calls[0].Radius, hollow.Calls[0].Thickness));
            Assert.Equal(new[] { "submalar point" }, hollow.Labels());
            Assert.Single(hollow.Column);
            Assert.EndsWith(" (hollow)", hollow.Column[0]);
        }

        [Fact]
        public void RenderFalseDrawsNothing()
        {
            var run = new Run(CheekFaces.Quarter(), render: false);
            Assert.Empty(run.Calls);
            Assert.Empty(run.Column);
            AssertSameValues(_run.Result, run.Result);
            // Through the public Calc: nothing drawn without render, the outline with it.
            var ctx = CheekFaces.Quarter();
            new MeasureCheekMeasures().Calc(ctx, render: false);
            Assert.True(ctx.Pixels.All(p => p == default(Rgba32)));
            ctx = CheekFaces.Quarter();
            AssertSameValues(_run.Result, new MeasureCheekMeasures().Calc(ctx, render: true));
            Assert.Contains(ctx.Pixels, p => p == Cyan);
            Assert.Contains(ctx.Pixels, p => p == Red);
            Assert.Equal(_run.Column, Column(ctx));
        }

        [Fact]
        public void OutlineColorsBySource()
        {
            // Rows of every source: contour, nose and edge rows recolor their runs.
            var (_, _, (xs, rows, src)) = CheekFaces.Traced("yaw45");
            float canthusRow = _ctx.Landmarks[72].Y;
            int first = Array.FindIndex(rows, r => r >= canthusRow + 10);
            for (int i = first; i < first + 10; i++) src[i] = OutlineSrcContour;
            for (int i = first + 20; i < first + 30; i++) src[i] = OutlineSrcEdge;
            for (int i = first + 40; i < first + 45; i++) src[i] = OutlineSrcNose;
            var run = new Run(CheekFaces.Quarter(outline: (xs, rows, src)));
            var polylines = run.Kinds("polyline");
            Assert.Equal(new[] { Cyan, Gray, Cyan, Yellow, Cyan, Magenta, Cyan }, polylines.Select(p => p.Color));
            Assert.All(polylines, p => Assert.Equal(4, p.Thickness));
            // The runs join: each starts where the previous one ended.
            for (int i = 1; i < polylines.Count; i++)
                Assert.Equal(polylines[i - 1].Points[^1], polylines[i].Points[0]);
            int bandRows = polylines.Sum(p => p.Points.Length - 1) + 1;
            Assert.True(bandRows > 100);
            // Nose rows set cheek.nose; the apex is then gray and flagged.
            Assert.Equal(1.0, run["cheek.nose"]);
            Assert.Equal(0.0, run["apex.ok"]);
            Assert.Equal(Gray, run.Kinds("circle")[1].Color);
            Assert.True(run["cheek.mask"] < 1.0);
        }

        [Fact]
        public void ContourFallback()
        {
            // The app's sketch, no outline arrays: the contour stands in, so the band is all
            // gray, no row came from the mask, the hollow is unknown, and the apex is the
            // prior.
            var ctx = CheekFaces.Sketch();
            var run = new Run(ctx);
            Assert.Equal(ExpectedNames.OrderBy(n => n), run.Result.Keys.OrderBy(n => n));
            Assert.Equal(0.0, run["cheek.far"]);
            Assert.Equal(0.0, run["cheek.mask"]);
            foreach (string name in new[] { "apex.r.h", "apex.r.dev", "cheek.scale" })
                Assert.False(double.IsNaN(run[name]), name);
            var c = CheekFaces.Coords(ctx);
            Assert.False(double.IsNaN(c["apex.r.x"]) || double.IsNaN(c["apex.r.y"]));
            Assert.True(double.IsNaN(c["sub.r.x"]) && double.IsNaN(c["sub.r.y"]));
            foreach (string name in ObliqueNames.Where(n => n.Contains(".l.")))
                Assert.True(double.IsNaN(run[name]), name);
            foreach (string name in new[] { "sub.r.h", "sub.r.depth", "sub.r.depth.px", "sub.r.present" })
                Assert.True(double.IsNaN(run[name]), name);
            Assert.Equal(0.0, run["apex.ok"]);
            Assert.Equal(0.0, run["sub.ok"]);
            Assert.Equal(0.0, run["cheek.nose"]);
            CheekFixtures.Within(-31.9, run["cheek.yaw"], 0.1);
            CheekFixtures.Within(-32.9, run["cheek.yaw.model"], 0.1);
            var polylines = run.Kinds("polyline");
            Assert.Single(polylines);
            Assert.Equal(Gray, polylines[0].Color);
            var circles = run.Kinds("circle");
            Assert.Equal(2, circles.Count); // the apex only: no submalar drawing
            Assert.Equal(Gray, circles[1].Color);
            Assert.Equal(1.0, run["apex.r.prior"]);
            CheekFixtures.Within(ApexHPrior, run["apex.r.h"], 0.005);
            Assert.Equal(new[] { "malar apex (prior)" }, run.Labels());
            Assert.StartsWith("apex.r h=0.79 dev=", run.Column[0]);
            Assert.EndsWith(" (prior)", run.Column[0]);
            Assert.EndsWith(" (flagged)", run.Column[1]);
            Assert.Equal(Flagged(
                "not turned enough (31.9 degrees, needs 40 to 50)",
                "the cheek's edge is unclear (hair or background)",
                "no submalar depth (the edge is unclear below the apex)"), run.Column.Skip(2));
            // The same through explicit arrays from ContourOutline.
            var explicitOutline = ContourOutline(CheekFaces.SketchLandmarks(), false);
            var run2 = new Run(CheekFaces.Sketch(outline: explicitOutline), render: false);
            AssertSameValues(run.Result, run2.Result);
        }

        // An intensity-edge run 20 px outside the seed on the five rows at Kaptein's height
        // of the yaw45 outline (a hair strand within the walk's reach): the apex is the mask
        // row beside it with the mask's own x, apex.ok stays 1, and the submalar point stays
        // absent with the plain outline's depth.
        [Fact]
        public void StandInRowsAtThePriorNeverMakeAHollow()
        {
            var (landmarks, farLeft, (xs, rows, src)) = CheekFaces.Traced("yaw45");
            Assert.True(farLeft);
            var plainCtx = CheekFaces.Quarter();
            var plain = new MeasureCheekMeasures().Calc(plainCtx, render: false);
            var (seedX, seedY, _) = ContourOutline(landmarks, true);
            var seedAt = CheekFixtures.ByRow(seedY);
            for (int i = 0; i < rows.Length; i++)
            {
                if (rows[i] < 334 || rows[i] > 338) continue;
                xs[i] = (float)(seedX[seedAt[rows[i]]] + 20.0);
                src[i] = OutlineSrcEdge;
            }
            var ctx = CheekFaces.Quarter(outline: (xs, rows, src));
            var result = new MeasureCheekMeasures().Calc(ctx, render: false);
            var plainAt = CheekFaces.Coords(plainCtx);
            var at = CheekFaces.Coords(ctx);
            Assert.Equal(336.0, plainAt["apex.l.y"]);
            Assert.Equal(339.0, at["apex.l.y"]);
            Assert.Equal(1.0, result["apex.l.prior"]);
            Assert.Equal(1.0, result["apex.ok"]);
            Assert.True(Math.Abs(at["apex.l.x"] - plainAt["apex.l.x"]) < 0.5);
            CheekFixtures.Within(0.9713, result["cheek.mask"], 0.001);
            Assert.Equal(0.0, result["sub.l.present"]);
            Assert.Equal(1.0, result["sub.ok"]);
            CheekFixtures.Within(plain["sub.l.depth"], result["sub.l.depth"], 2e-4);
            Assert.Equal(plainAt["sub.l.y"], at["sub.l.y"]);
        }

        [Fact]
        public void ObliqueCheekAccessorIsRead()
        {
            // Observed points handed over in place of the context's: the yaw45 points with a
            // sentinel height, on a face that carries no outline at all.
            var ob = Observed(CheekFaces.Quarter());
            var sentinel = CheekFaces.Copy(ob);
            sentinel.ApexH = 0.5;
            sentinel.Apex = new Vec2d(600.0, 300.0);
            var run = new Run(CheekFaces.Quarter(outline: (null, null, null)), oblique: () => sentinel);
            Assert.Equal(0.5, run["apex.l.h"]);
            Assert.Equal(1.0, run["apex.ok"]);
            Assert.Equal((600, 300), run.Kinds("circle")[0].P0);
            Assert.Equal("apex.l h=0.50 dev=0.016 (prior)", run.Column[0]);
            // And an accessor with nothing gives NaN, whatever the outline.
            run = new Run(CheekFaces.Quarter(), oblique: () => null);
            foreach (string name in ObliqueNames)
            {
                if (name == "cheek.far") Assert.Equal(1.0, run[name]);
                else Assert.True(double.IsNaN(run[name]), name);
            }
            Assert.Empty(run.Calls);
            Assert.Equal(new[] { "cheek yaw 45.0 (model 45.0) pitch 4.4 (flagged)" }
                .Concat(Flagged("the far cheek could not be measured")), run.Column);
        }

        // Records the warnings written to Trace about the reported side.
        sealed class SideWarnings : TraceListener
        {
            public readonly List<string> Messages = new();

            void Add(string? message)
            {
                if (message != null && message.Contains("observed cheek points"))
                    lock (Messages) Messages.Add(message);
            }

            public override void Write(string? message) => Add(message);
            public override void WriteLine(string? message) => Add(message);

            public override void TraceEvent(TraceEventCache? eventCache, string source, TraceEventType eventType,
                int id, string? format, params object?[]? args) =>
                Add(args == null || format == null ? format : string.Format(format, args));

            public override void TraceEvent(TraceEventCache? eventCache, string source, TraceEventType eventType,
                int id, string? message) => Add(message);
        }

        // Observed points that name the other side are reported on the face's, with a
        // warning, so cheek.far and the reported side never disagree.
        [Fact]
        public void FarSideFollowsTheFace()
        {
            var ob = Observed(CheekFaces.Quarter());
            var other = CheekFaces.Copy(ob);
            other.FarLeft = false;
            var listener = new SideWarnings();
            Trace.Listeners.Add(listener);
            try
            {
                var run = new Run(CheekFaces.Quarter(), oblique: () => other);
                Assert.Single(listener.Messages);
                Assert.Contains("subject's right, the face says the left", listener.Messages[0]);
                Assert.Equal(1.0, run["cheek.far"]);
                AssertSameValues(_run.Result, run.Result);
                Assert.Equal(_run.Column, run.Column);
                Assert.Equal(_run.Canvas.Repr(), run.Canvas.Repr());
                // Agreeing sides log nothing.
                listener.Messages.Clear();
                run = new Run(CheekFaces.Quarter(), oblique: () => ob);
                Assert.Empty(listener.Messages);
                AssertSameValues(_run.Result, run.Result);
                Assert.Equal(_run.Canvas.Repr(), run.Canvas.Repr());
            }
            finally
            {
                Trace.Listeners.Remove(listener);
            }
        }

        [Fact]
        public void Pitch25FailsOnlyPitchOk()
        {
            var run = new Run(CheekFaces.Quarter(pitch: 25f));
            Assert.Equal(0.0, run["cheek.pitch.ok"]);
            AssertSameValues(_run.Result, run.Result, ExpectedNames.Except(new[] { "cheek.pitch", "cheek.pitch.ok" }));
            Assert.Equal(1.0, run["apex.ok"]);
            Assert.Equal(new[] { "cheek yaw 45.0 (model 45.0) pitch 25.0 (flagged)" }
                .Concat(Flagged("head tipped up or down (pitch 25.0, needs within 20)")), run.Column.TakeLast(2));
        }

        [Fact]
        public void NaNYawFailsEveryGate()
        {
            var run = new Run(CheekFaces.Quarter(yaw: float.NaN, pitch: float.NaN));
            Assert.True(double.IsNaN(run["cheek.yaw.model"]));
            CheekFixtures.Within(45.0, run["cheek.yaw"], 0.1);
            var flags = OkFlags.Select(n => run[n]).Where(v => !double.IsNaN(v)).ToList();
            Assert.Equal(3, flags.Count); // apex.ok, sub.ok, cheek.pitch.ok
            Assert.All(flags, v => Assert.Equal(0.0, v));
            Assert.False(double.IsNaN(run["apex.l.h"])); // still reported
            Assert.Equal(Gray, run.Kinds("circle")[1].Color);
        }

        [Fact]
        public void YawGate()
        {
            // The landmark yaw is what the gate reads: a model yaw 10 degrees off fails the
            // agreement, one 3 degrees off passes.
            var table = Cheek.YawCalibration;
            var run = new Run(CheekFaces.Quarter(yaw: (float)table[table.Count - 3].Raw), render: false); // 55
            CheekFixtures.Within(55.0, run["cheek.yaw.model"], 0.01);
            Assert.Equal(0.0, run["apex.ok"]);
            Assert.Equal(0.0, run["sub.ok"]);
            run = new Run(CheekFaces.Quarter(yaw: (float)table[table.Count - 5].Raw), render: false); // 48
            CheekFixtures.Within(48.0, run["cheek.yaw.model"], 0.01);
            Assert.Equal(1.0, run["apex.ok"]);
        }

        [Fact]
        public void SubmalarLineNeverReadsNegativeZero()
        {
            // A convex cheek's depth just below zero reads 0.000.
            var ok = new Dictionary<string, double> { ["sub.ok"] = 1.0 };
            var ob = new ObliqueCheek { SubmalarDepth = -2.6e-05, SubmalarPresent = 0.0 };
            Assert.Equal(new[] { "sub.l depth=0.000 (absent)" }, MeasureCheekMeasures.ColumnSubmalar(ob, true, ok));
            ob = new ObliqueCheek { SubmalarDepth = -0.0123, SubmalarPresent = 0.0 };
            Assert.Equal(new[] { "sub.r depth=-0.012 (absent)" }, MeasureCheekMeasures.ColumnSubmalar(ob, false, ok));
            // A flagged one says so.
            Assert.Equal(new[] { "sub.r depth=-0.012 (absent) (flagged)" },
                MeasureCheekMeasures.ColumnSubmalar(ob, false, new Dictionary<string, double> { ["sub.ok"] = 0.0 }));
        }
    }

    // ------------------------------------------------------------ TestFlagReasons

    // A flagged three-quarter result says why, one "flagged:" line per reason after the
    // item lines, under the items whose lines carry the flag; an unflagged result and a
    // frontal face write none.
    public class FlagReasons
    {
        static List<string> Reasons(Run run) =>
            run.Column.Where(line => line.StartsWith(MeasureCheekMeasures.FlagPrefix)).ToList();

        [Fact]
        public void UnflaggedResultsWriteNone()
        {
            foreach (var ctx in new[] { CheekFaces.Quarter(), CheekFaces.Frontal() })
                Assert.Empty(Reasons(new Run(ctx)));
            Assert.Empty(new Run(CheekFaces.Quarter("yaw50"), render: false).Column); // nothing without rendering
        }

        [Fact]
        public void TurnedTooFarWithTheNoseOnTheOutline()
        {
            // The 50 degree render, like a photo turned past the window: the turn just past
            // 50 and the nose on the far outline, whose rows also leave no submalar depth.
            // The nose is named, not an unclear edge.
            var run = new Run(CheekFaces.Quarter("yaw50"));
            Assert.Equal(0.0, run["apex.ok"]);
            Assert.Equal(1.0, run["cheek.nose"]);
            Assert.True(Math.Abs(run["cheek.yaw"]) > 50.0);
            Assert.Equal(Flagged(
                "turned too far (just over 50 degrees, needs 40 to 50)",
                "the nose hides the far cheek",
                "no submalar depth (the nose covers the cheek below the apex)"), Reasons(run));
            Assert.Equal(Reasons(run), run.Column.TakeLast(3)); // after the others
        }

        [Fact]
        public void TheTurnEstimatesDiffer()
        {
            var table = Cheek.YawCalibration;
            var run = new Run(CheekFaces.Quarter(yaw: (float)table[table.Count - 3].Raw)); // reads 55
            Assert.Equal(0.0, run["apex.ok"]);
            Assert.Equal(Flagged("the turn estimates differ (45.0 and 55.0 degrees)"), Reasons(run));
        }

        [Fact]
        public void ReasonsFollowTheItems()
        {
            var expected = new Dictionary<string, List<string>>
            {
                [Outline] = new(),
                [Apex] = Flagged("turned too far (just over 50 degrees, needs 40 to 50)", "the nose hides the far cheek"),
                [Submalar] = Flagged("no submalar depth (the nose covers the cheek below the apex)"),
                [HeadTurn] = Flagged("turned too far (just over 50 degrees, needs 40 to 50)", "the nose hides the far cheek"),
            };
            foreach (var (name, reasons) in expected)
            {
                var measure = new MeasureCheekMeasures();
                foreach (string item in MeasureCheekMeasures.ItemNames) measure.SetItemEnabled(item, item == name);
                Assert.Equal(reasons, Reasons(new Run(CheekFaces.Quarter("yaw50"), measure)));
            }
        }

        [Fact]
        public void EveryFailedGateIsNamed()
        {
            // Each apex gate alone, on the observed points of the plain 45 render.
            var ob = Observed(CheekFaces.Quarter());
            var cases = new (string Reason, Action<ObliqueCheek> Change)[]
            {
                ("the nose hides the far cheek", o => o.NoseOn = true),
                ("the apex is level with the nose tip", o => o.ApexNearNose = true),
                ("the cheek's edge is unclear (hair or background)", o => o.WindowMask = 0.5),
            };
            foreach (var (reason, change) in cases)
            {
                var changed = CheekFaces.Copy(ob);
                change(changed);
                Assert.Equal(new[] { reason }, MeasureCheekMeasures.ApexReasons(changed, 45.0, 45.0, true));
            }
            Assert.Empty(MeasureCheekMeasures.ApexReasons(ob, 45.0, 45.0, true));
            Assert.Equal(new[] { "not turned enough (38.2 degrees, needs 40 to 50)" },
                MeasureCheekMeasures.ApexReasons(ob, -38.2, -38.0, true));
            Assert.Equal(new[] { "the turn is unknown" }, MeasureCheekMeasures.ApexReasons(ob, double.NaN, 45.0, false));
            Assert.Equal(new[] { "the model's turn is unknown" }, MeasureCheekMeasures.ApexReasons(ob, 45.0, double.NaN, false));
            // The nose rows explain the window, so no unclear edge beside the nose.
            var both = CheekFaces.Copy(ob);
            both.NoseOn = true;
            both.WindowMask = 0.5;
            Assert.Equal(new[] { "the nose hides the far cheek" }, MeasureCheekMeasures.ApexReasons(both, 45.0, 45.0, true));
        }

        [Fact]
        public void ValuesOnTheLimitReadPastIt()
        {
            Assert.Equal("50.2", MeasureCheekMeasures.Beyond(50.2, 50.0));
            Assert.Equal("just over 50", MeasureCheekMeasures.Beyond(50.04, 50.0));
            Assert.Equal("just under 40", MeasureCheekMeasures.Beyond(39.96, 40.0));
            Assert.Equal("just under -20", MeasureCheekMeasures.Beyond(-20.03, 20.0));
            Assert.Equal("25.0", MeasureCheekMeasures.Beyond(25.0, 20.0));
            Assert.Equal("nan", MeasureCheekMeasures.Beyond(double.NaN, 20.0)); // .NET-only
        }
    }

    // ------------------------------------------------------------ TestLabelsAtTheEdge

    // A label stays on its point's outside, never crossing the face: one that would cross
    // the image's edge there is clamped to it and raised above its point.
    public class LabelsAtTheEdge
    {
        // Each point label as (x, y, width, text), in drawing order.
        static List<(int X, int Y, int Width, string Text)> Labels(Run run) =>
            run.Kinds("text").Select(c => (c.P0.X, c.P0.Y, 12 * c.Text!.Length, c.Text)).ToList();

        [Fact]
        public void FarLeftApexNearTheRightEdge()
        {
            // The yaw45 render's far cheek is the subject's left, so its labels go right,
            // beside the apex; with the image cut to 20 px past the apex the label is
            // clamped to the right edge and raised above the apex, still off the face
            // rather than across it.
            var wideCtx = CheekFaces.Quarter();
            var wide = new Run(wideCtx);
            var result = CheekFaces.Coords(wideCtx);
            var (apexX, apexY) = MeasureCheekMeasures.Pt(new Vec2d(result["apex.l.x"], result["apex.l.y"]));
            Assert.True(wideCtx.FarLeft);
            Assert.Equal((apexX + 14, apexY + 10), (Labels(wide)[0].X, Labels(wide)[0].Y));
            var ctx = CheekFaces.Quarter();
            var run = new Run(ctx, width: apexX + 20);
            Assert.Equal(result["apex.l.x"], CheekFaces.Coords(ctx)["apex.l.x"]);
            var labels = Labels(run);
            Assert.Equal(labels.Select(l => l.Text), run.Labels());
            Assert.Single(labels); // the apex alone: no hollow, no label
            foreach (var (x, _, width, text) in labels)
            {
                Assert.True(x >= 0, text);
                Assert.True(x + width <= apexX + 20, text);
            }
            Assert.Equal((apexX + 20 - 12 * "malar apex (prior)".Length, apexY - 22), (labels[0].X, labels[0].Y));
        }

        [Fact]
        public void FarRightApexNearTheLeftEdge()
        {
            // The yaw-45 render's far cheek is the subject's right, so its labels go left;
            // with the landmarks and outline shifted so the apex sits 20 px from the left
            // edge the label is clamped to x = 0 and raised above the apex.
            var wideCtx = CheekFaces.Quarter("yaw-45");
            var wide = new Run(wideCtx);
            var result = CheekFaces.Coords(wideCtx);
            Assert.False(wideCtx.FarLeft);
            int apexX = MeasureCheekMeasures.Pt(new Vec2d(result["apex.r.x"], 0)).X;
            Assert.True(Labels(wide)[0].X < apexX);
            int shift = apexX - 20;
            var (landmarks, _, (xs, rows, src)) = CheekFaces.Traced("yaw-45");
            var ctx = CheekFaces.Quarter("yaw-45",
                outline: (xs.Select(x => x - shift).ToArray(), rows, src),
                landmarks: landmarks.Select(p => new Vec2(p.X - shift, p.Y)).ToArray());
            var run = new Run(ctx);
            var shifted = CheekFaces.Coords(ctx);
            CheekFixtures.Near(result["apex.r.x"] - shift, shifted["apex.r.x"]);
            Assert.Equal(20, MeasureCheekMeasures.Pt(new Vec2d(shifted["apex.r.x"], 0)).X);
            var labels = Labels(run);
            Assert.Single(labels);
            foreach (var (x, _, width, text) in labels)
            {
                Assert.True(x >= 0, text);
                Assert.True(x + width <= ctx.Width, text);
            }
            int shiftedApexY = MeasureCheekMeasures.Pt(new Vec2d(0, shifted["apex.r.y"])).Y;
            Assert.Equal((0, shiftedApexY - 22), (labels[0].X, labels[0].Y));
        }

        [Fact]
        public void FrontalLabelsFitBesideOrAbove()
        {
            // The frontal labels fit on their outside at the image's width: the zygion's
            // above the dot, the reference points' beside the ring; none comes within
            // LabelGap of another or of a dot.
            var ctx = CheekFaces.Frontal();
            var run = new Run(ctx);
            var labels = Labels(run);
            Assert.Equal((240 - 12 * "zygion".Length - 14, 558 - 22), (labels[0].X, labels[0].Y));
            Assert.Equal((778 + 14, 558 - 22), (labels[1].X, labels[1].Y));
            var boxes = labels.Select(l => new MeasureCheekMeasures.Box(l.X, l.Y - 20, l.X + l.Width, l.Y + 5))
                .Concat(run.Kinds("circle").Select(c =>
                    MeasureCheekMeasures.DotBox(new Vec2d(c.P0.X, c.P0.Y), c.Radius)))
                .ToList();
            foreach (var (x, _, width, text) in labels)
            {
                Assert.True(x >= 0, text);
                Assert.True(x + width <= ctx.Width, text);
            }
            for (int i = 0; i < labels.Count; i++)
                for (int j = i + 1; j < boxes.Count; j++)
                    Assert.False(boxes[i].Near(boxes[j]), $"{labels[i].Text} near box {j}");
        }

        [Fact]
        public void ClampedOnItsOutside()
        {
            // On an image narrower than the label plus its offset, the label is clamped to
            // the image's edge on its outside and raised above its point.
            var canvas = new RecordingCanvas(100);
            MeasureCheekMeasures.WriteLabel(canvas, new Vec2d(50, 40), "malar apex", Red, false);
            MeasureCheekMeasures.WriteLabel(canvas, new Vec2d(50, 40), "malar apex", Red, true);
            Assert.Equal(new[] { ((0, 18), "malar apex", Red), ((0, 18), "malar apex", Red) },
                canvas.Kinds("text").Select(c => (c.P0, c.Text!, c.Color)));
            // Room on the right but not the left: a label that belongs left stays left,
            // clamped to x = 0, rather than crossing to the right (over the face).
            canvas = new RecordingCanvas(200);
            MeasureCheekMeasures.WriteLabel(canvas, new Vec2d(50, 40), "malar apex", Red, true);
            Assert.Equal((0, 18), canvas.Kinds("text")[0].P0);
            // With room, beside the point, centered on its row.
            canvas = new RecordingCanvas(200);
            MeasureCheekMeasures.WriteLabel(canvas, new Vec2d(50, 40), "malar apex", Red, false);
            Assert.Equal((64, 50), canvas.Kinds("text")[0].P0);
        }
    }

    // ------------------------------------------------------------ TestPlaceLabel

    // PlaceLabel: beside or above the point, and off what is drawn.
    public class PlaceLabel
    {
        static readonly (int, int, int) Size = (120, 20, 5); // width, height above the baseline, depth below

        static MeasureCheekMeasures.Box B(int l, int t, int r, int b) => new(l, t, r, b);

        [Fact]
        public void BesideAndAbove()
        {
            Assert.Equal((314, 210), MeasureCheekMeasures.PlaceLabel((300, 200), Size, false, 1024, new()));
            Assert.Equal((166, 210), MeasureCheekMeasures.PlaceLabel((300, 200), Size, true, 1024, new()));
            Assert.Equal((314, 178), MeasureCheekMeasures.PlaceLabel((300, 200), Size, false, 1024, new(), beside: false));
        }

        [Fact]
        public void TakenRowsMoveTheLabel()
        {
            // Its own row taken (a label already in its box): one line down (20 + 5 +
            // LabelGap = 29 px); that taken too: one line up; and the box is added to taken.
            var own = B(314, 190, 434, 215);
            var taken = new List<MeasureCheekMeasures.Box> { own };
            Assert.Equal((314, 239), MeasureCheekMeasures.PlaceLabel((300, 200), Size, false, 1024, taken));
            var last = taken[^1];
            Assert.Equal((314, 219, 434, 244), (last.Left, last.Top, last.Right, last.Bottom));
            taken = new() { own, B(314, 219, 434, 244) };
            Assert.Equal((314, 181), MeasureCheekMeasures.PlaceLabel((300, 200), Size, false, 1024, taken));
            // Every row taken: it stays on its own row.
            taken = new() { B(0, 0, 1024, 1024) };
            Assert.Equal((314, 210), MeasureCheekMeasures.PlaceLabel((300, 200), Size, false, 1024, taken));
        }

        [Fact]
        public void ADotBesideItsLabelIsClear()
        {
            // A label beside its own dot (rim 9 px) keeps LabelGap from it.
            var taken = new List<MeasureCheekMeasures.Box> { MeasureCheekMeasures.DotBox(new Vec2d(300, 200), MeasureCheekMeasures.DotRim) };
            Assert.Equal((314, 210), MeasureCheekMeasures.PlaceLabel((300, 200), Size, false, 1024, taken));
            Assert.Equal((166, 210), MeasureCheekMeasures.PlaceLabel((300, 200), Size, true, 1024, taken));
        }
    }

    // ------------------------------------------------------------ TestLateral

    public class Lateral
    {
        [Fact]
        public void NoOutputsAndNoDrawing()
        {
            var measure = new MeasureCheekMeasures();
            foreach (bool render in new[] { false, true })
            {
                var run = new Run(CheekFaces.Frontal(lateral: true), measure, render);
                Assert.Empty(run.Result);
                Assert.Empty(run.Calls);
                Assert.Empty(run.Column);
                Assert.Empty(new MeasureCheekMeasures().Calc(CheekFaces.Frontal(lateral: true), render));
            }
        }
    }

    // ------------------------------------------------------------ TestDrawCheekLandmarks

    // FaceMeasureContext.DrawCheekLandmarks, read from the drawn pixels: a dot (and with
    // names, a code) for every cheek landmark that exists in the view, nothing for the
    // rest, nor for a submalar point the outline shows no hollow at. (dynaface-lib records
    // its circle and write_text calls instead.)
    public class DrawCheekLandmarks
    {
        static readonly Rgba32 CheekOrange = new(255, 128, 0, 255);

        // The pixel at a top-left point of the context's image.
        internal static Rgba32 At(FaceMeasureContext ctx, (int X, int Y) p) =>
            ctx.Pixels[(ctx.Height - 1 - p.Y) * ctx.Width + p.X];

        internal static (int X, int Y) Rounded(Vec2d p) => MeasureCheekMeasures.Pt(p);

        static int Count(FaceMeasureContext ctx, Rgba32 color) => ctx.Pixels.Count(p => p == color);

        [Fact]
        public void FrontalFaceDrawsItsSixConstructions()
        {
            var ctx = CheekFaces.Frontal();
            Vec2d[] points = ctx.CheekLandmarks;
            Assert.All(points.Take(6), p => Assert.True(p.IsFinite));
            ctx.DrawCheekLandmarks(size: 0.5f, names: false);
            foreach (var p in points.Take(6)) Assert.Equal(CheekOrange, At(ctx, Rounded(p)));
            // The same dot size rule as the landmarks': radius max(1, round(3 * 0.5)) = 2,
            // so 6 dots of 13 pixels each.
            Assert.Equal(6 * 13, Count(ctx, CheekOrange));
            // The codes sit just past the dots, in the same color.
            var named = CheekFaces.Frontal();
            named.DrawCheekLandmarks(size: 0.5f);
            Assert.True(Count(named, CheekOrange) > 6 * 13);
        }

        [Fact]
        public void Options()
        {
            var ctx = CheekFaces.Frontal();
            var color = new Rgba32(1, 2, 3, 255);
            ctx.DrawCheekLandmarks(size: 1.0f, color: color, names: false);
            Assert.Equal(6 * 29, Count(ctx, color)); // radius 3
        }

        [Fact]
        public void QuarterFaceDrawsTheFarCheeksPoints()
        {
            // yaw45_hollow3 has a hollow under the apex: both points are drawn.
            var ctx = CheekFaces.Quarter("yaw45_hollow3");
            Assert.True(ctx.IsQuarter && ctx.FarLeft);
            Assert.Equal(1.0, ctx.ObliqueCheek()!.SubmalarPresent);
            ctx.DrawCheekLandmarks(names: false);
            Vec2d[] points = ctx.CheekLandmarks;
            Assert.Equal(CheekOrange, At(ctx, Rounded(points[Cheek.CheekLmApexL])));
            Assert.Equal(CheekOrange, At(ctx, Rounded(points[Cheek.CheekLmSubmalarL])));
        }

        [Fact]
        public void AbsentSubmalarPointIsNotDrawn()
        {
            // yaw45 shows no hollow: its submalar point is still a cheek landmark, and
            // exported, but only the apex is drawn, as the cheek measure does.
            var ctx = CheekFaces.Quarter();
            Assert.Equal(0.0, ctx.ObliqueCheek()!.SubmalarPresent);
            Vec2d[] points = ctx.CheekLandmarks;
            Assert.True(points[Cheek.CheekLmSubmalarL].IsFinite);
            ctx.DrawCheekLandmarks(size: 0.5f, names: false);
            Assert.Equal(CheekOrange, At(ctx, Rounded(points[Cheek.CheekLmApexL])));
            Assert.NotEqual(CheekOrange, At(ctx, Rounded(points[Cheek.CheekLmSubmalarL])));
            Assert.Equal(13, Count(ctx, CheekOrange));
        }

        [Fact]
        public void NothingWithoutAPoint()
        {
            var lateral = CheekFaces.Frontal(lateral: true);
            Assert.All(lateral.CheekLandmarks, p => Assert.True(double.IsNaN(p.X) && double.IsNaN(p.Y)));
            lateral.DrawCheekLandmarks();
            Assert.True(lateral.Pixels.All(p => p == default(Rgba32)));
            var noFace = new FaceMeasureContext(new FaceImage(new Rgba32[16 * 16], 16, 16), Array.Empty<Vec2>());
            noFace.DrawCheekLandmarks();
            Assert.True(noFace.Pixels.All(p => p == default(Rgba32)));
        }
    }

    // ------------------------------------------------------------ TestCheekLandmarksItem

    // MeasureLandmarks' "Cheek Landmarks" item on fixture faces: its 20 outputs are
    // ctx.CheekLandmarks, NaN where absent, and it draws every cheek landmark that exists in
    // the view, but a submalar point without a hollow, while it is enabled.
    public class CheekLandmarksItem
    {
        static readonly Rgba32 CheekOrange = MeasureLandmarks.CheekColor;

        static Dictionary<string, double> CheckValues(FaceMeasureContext ctx)
        {
            var result = new MeasureLandmarks().Calc(ctx, render: false);
            double[] flat = ctx.CheekLandmarks.SelectMany(p => new[] { p.X, p.Y }).ToArray();
            for (int i = 0; i < CoordNames.Length; i++)
                Assert.True(Same(flat[i], result[CoordNames[i]]), CoordNames[i]);
            // The rule every caller locates by.
            Vec2d[] located = Cheek.LocateAll(ctx.Landmarks, ctx.Pose, ctx.FarLeft,
                ctx.CheekOutlineX, ctx.CheekOutlineY, ctx.CheekOutlineSrc);
            double[] expected = located.SelectMany(p => new[] { p.X, p.Y }).ToArray();
            for (int i = 0; i < CoordNames.Length; i++)
                Assert.True(Same(expected[i], result[CoordNames[i]]), CoordNames[i]);
            return CoordNames.ToDictionary(n => n, n => result[n]);
        }

        static List<string> Finite(Dictionary<string, double> values) =>
            CoordNames.Where(n => !double.IsNaN(values[n])).ToList();

        [Fact]
        public void FrontalFace()
        {
            var values = CheckValues(CheekFaces.Frontal());
            Assert.Equal(CoordNames.Take(12), Finite(values));
            // The constructions' pinned places, as the cheek measure drew them.
            Assert.Equal((240, 558), MeasureCheekMeasures.Pt(new Vec2d(values["zy.r.x"], values["zy.r.y"])));
            Assert.Equal((778, 558), MeasureCheekMeasures.Pt(new Vec2d(values["zy.l.x"], values["zy.l.y"])));
        }

        [Fact]
        public void QuarterFace()
        {
            var values = CheckValues(CheekFaces.Quarter());
            Assert.Equal(new[] { "apex.l.x", "apex.l.y", "sub.l.x", "sub.l.y" }, Finite(values));
            Assert.Equal(336.0, values["apex.l.y"]);
        }

        [Fact]
        public void AbsentPointsAreNaN()
        {
            var sketch = CheekFaces.Sketch();
            Assert.True(sketch.IsQuarter);
            Assert.Equal(new[] { "apex.r.x", "apex.r.y" }, Finite(CheckValues(sketch)));
            Assert.Empty(Finite(CheckValues(CheekFaces.Frontal(lateral: true))));
        }

        [Fact]
        public void Drawing()
        {
            var ctx = CheekFaces.Frontal();
            new MeasureLandmarks().Calc(ctx, render: true);
            // A black dot of radius 3 under an orange one of radius 2, for each of the six,
            // over the 98 landmarks' black and white dots.
            Vec2d[] points = ctx.CheekLandmarks;
            foreach (var p in points.Take(6))
            {
                var (x, y) = DrawCheekLandmarks.Rounded(p);
                Assert.Equal(CheekOrange, DrawCheekLandmarks.At(ctx, (x, y)));
                Assert.Equal(new Rgba32(0, 0, 0, 255), DrawCheekLandmarks.At(ctx, (x + 3, y)));
            }
            Assert.Equal(6 * 13, ctx.Pixels.Count(p => p == CheekOrange));
            Assert.Contains(ctx.Pixels, p => p == new Rgba32(255, 255, 255, 255)); // the 98
        }

        [Fact]
        public void QuarterDrawingSkipsAnAbsentSubmalarPoint()
        {
            // Absent (yaw45): the apex alone; present (yaw45_hollow3): both. The values are
            // the same either way, the submalar point's included.
            foreach (var (name, rows) in new[]
            {
                ("yaw45", new[] { Cheek.CheekLmApexL }),
                ("yaw45_hollow3", new[] { Cheek.CheekLmApexL, Cheek.CheekLmSubmalarL }),
            })
            {
                var ctx = CheekFaces.Quarter(name);
                Assert.Equal(new[] { "apex.l.x", "apex.l.y", "sub.l.x", "sub.l.y" }, Finite(CheckValues(ctx)));
                new MeasureLandmarks().Calc(ctx, render: true);
                Assert.Equal(13 * rows.Length, ctx.Pixels.Count(p => p == CheekOrange));
                foreach (int row in rows)
                    Assert.Equal(CheekOrange, DrawCheekLandmarks.At(ctx, DrawCheekLandmarks.Rounded(ctx.CheekLandmarks[row])));
            }
        }

        [Fact]
        public void DrawingGatedByTheItem()
        {
            var measure = new MeasureLandmarks();
            measure.SetItemEnabled(MeasureLandmarks.CheekItem, false);
            var ctx = CheekFaces.Frontal();
            var result = measure.Calc(ctx, render: true);
            Assert.DoesNotContain(ctx.Pixels, p => p == CheekOrange);
            Assert.Contains(ctx.Pixels, p => p == new Rgba32(255, 255, 255, 255)); // the 98 still drawn
            ctx = CheekFaces.Frontal();
            new MeasureLandmarks().Calc(ctx, render: false);
            Assert.True(ctx.Pixels.All(p => p == default(Rgba32)));
            // Off, the values are still returned, and GetAllItems drops them.
            Assert.True(CoordNames.All(result.ContainsKey));
            var listed = DynafaceMeasures.GetAllItems(new FaceMeasureBase[] { measure });
            Assert.Empty(CoordNames.Intersect(listed));
            Assert.Equal(196, listed.Count);
        }

        [Fact]
        public void NothingDrawnForNaNRows()
        {
            var sketch = CheekFaces.Sketch();
            new MeasureLandmarks().Calc(sketch, render: true);
            Assert.Equal(13, sketch.Pixels.Count(p => p == CheekOrange));
            Assert.Equal(CheekOrange,
                DrawCheekLandmarks.At(sketch, DrawCheekLandmarks.Rounded(sketch.CheekLandmarks[Cheek.CheekLmApexR])));
            var lateral = CheekFaces.Frontal(lateral: true);
            new MeasureLandmarks().Calc(lateral, render: true);
            Assert.DoesNotContain(lateral.Pixels, p => p == CheekOrange);
        }
    }
}
