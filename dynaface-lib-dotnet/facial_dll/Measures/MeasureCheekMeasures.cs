#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;

// Cheek Measures: on a frontal face, the zygion width and the cheek and submalar reference
// points constructed from the landmarks; on a three-quarter (quarter) face, the far
// cheek's outline and the malar apex and submalar point observed on it; on both, the head
// turn the gates read. A port of dynaface-lib's measures_cheek.AnalyzeCheekMeasures, the
// reference implementation, line for line in double. The cheek landmarks' coordinates
// themselves are MeasureLandmarks' "Cheek Landmarks" item (Cheek.CheekLandmarkOutputs),
// not outputs of this measure. Frontal distances are millimeters (ctx.Pix2mm, measured on
// the landmarks being measured); oblique heights and depths are fractions of H, the
// chin-to-canthus distance (cheek.scale), because the pupil scale over-reads a turned head
// by about 1.4.
//
// Six items, in this order, each a group (MeasureItemInfo's outputs): a switch named for
// the one thing it draws, standing for the numbers about it, and limited to the views
// where it means something (MeasureItemInfo's views), so on a quarter face the frontal
// items are off after UpdateForType or MeasureSelection.Apply and the other way round.
// The values never depend on render or on which items are enabled, and every one of the
// 41 outputs is returned either way, in the items' order (AllOutputs). For a profile this
// measure returns no outputs (an empty dictionary), as dynaface-lib's does.
//
//  1. "Zygion width" (frontal), NaN on a quarter face: zy.r, zy.l (each zygion's distance
//     from the midline, mm), zy.w (their sum), zy.ratio (SymmetryRatio(zy.r, zy.l)),
//     zy.ratio.ok (zy.ok and |cheek.yaw| <= 2), zy.r.clip, zy.l.clip (Cheek.ClipNone,
//     ClipTop or ClipBottom), zy.r.ok, zy.l.ok (that side unclipped and the yaw gate
//     passes: |cheek.yaw| <= 5 with the agreement), zy.ok (both). Draws the orange zygion
//     level line, a blue zygion dot with a white rim at each end (gray when that side's
//     ok flag is 0), each labeled "zygion" (" (clipped)" appended), and the zygion line
//     of the text.
//  2. "Reference points" (frontal), NaN on a quarter face: cr.r.d, cr.l.d (each cheek
//     reference point's depth below the lateral canthus, mm), cr.ok, sr.ok (the yaw
//     gate). Draws an orange vertical from each lateral canthus to its foot and an open
//     orange ring at each reference point (gray when cr.ok is 0), labeled "cheek
//     reference point" and "submalar reference point", and the reference line.
//  3. "Far cheek outline" (quarter), NaN on a frontal face: cheek.far (0 the subject's
//     right, 1 the left, on every quarter face), cheek.mask, cheek.nose, cheek.nose.gap,
//     cheek.scale (H, px). Draws the far outline as a 4 px polyline colored by each row's
//     source (OutlineColors), the canthus, stomion and chin levels as 1 px orange ticks
//     inward by 0.25 H, and the canthus-to-stomion chord as a 2 px orange line.
//  4. "Malar apex" (quarter), NaN on a frontal face and for the near side: apex.r.h,
//     apex.l.h, apex.r.dev, apex.l.dev, apex.r.prior, apex.l.prior, apex.r.near_nose,
//     apex.l.near_nose, apex.ok (|(|cheek.yaw| - 45)| <= 5 with the agreement, no nose on
//     the outline, not near the nose, and at least CheekOutline.OutlineMinMask of the
//     search window's rows from the mask). Draws a red dot with a white rim (gray when
//     apex.ok is 0), labeled "malar apex" plus " (prior)" or " (nose?)", and the apex line.
//  5. "Submalar point" (quarter), NaN on a frontal face and for the near side: sub.r.h,
//     sub.l.h, sub.r.depth, sub.l.depth (fraction of H, positive inward), sub.r.depth.px,
//     sub.l.depth.px, sub.r.present, sub.l.present, sub.ok (apex.ok and a known depth).
//     Draws a green dot labeled "submalar point" only when a hollow is present (an absent
//     point sits wherever the outline comes nearest the chord, not a landmark), and the
//     submalar line (nothing when its depth is NaN).
//  6. "Head turn" (frontal and quarter): cheek.yaw (Cheek.LandmarkYaw, NaN without a pupil
//     frame), cheek.yaw.model (Cheek.CalibrateYaw of SPIGA's yaw), cheek.pitch (SPIGA's raw
//     pitch), cheek.pitch.ok (|cheek.pitch| <= 20). Writes the pose line in either view.
//
// Every value is a double; flags are 0.0 or 1.0; NaN means not applicable in this view or
// not determinable: no pupil frame (fewer than 98 landmarks, one that isn't finite, or
// coincident pupils) blanks every frontal and oblique output and cheek.yaw, except
// cheek.far, which follows the face's FarLeft on every quarter face; the head pose's own
// values always come back (NaN when the context has no head pose: .NET-only, since
// dynaface-lib's faces always carry one). "Right" and "left" are the subject's, and the
// far (measured) cheek of a quarter face is the context's FarLeft, which names the reported
// side; observed points that say otherwise are reported on the face's side, with a
// Trace warning, as dynaface-lib logs one.
//
// Drawing: labels are written at the standard measure text size outside their point,
// away from the face, or on its other side when the outside would cross the image's edge
// (WriteLabel). The text, written to the sidebar like every .NET measure's (a header, one line
// each, a spacer), holds Python's text-column lines: at most three headline lines for any
// one face, each written by its item ("zy.w=130.6 mm (R 65.5 / L 65.1, ratio 0.99)",
// "cheek ref depth R 28.8 / L 28.8 mm", "apex.l h=0.79 dev=0.012 (prior)",
// "sub.l depth=0.006 (hollow)" or "(absent)", "cheek yaw 44.6 (model 42.3) pitch 4.7"),
// then on a flagged quarter face one line per reason, each starting with FlagPrefix
// ("flagged: "). Numbers are written as Python's format writes them (Fixed: rounded half
// to even on the exact value, "nan" for NaN), so the two ports' lines read the same.
public class MeasureCheekMeasures : FaceMeasureBase
{
    // The six items' names: each is a switch named for what it draws, standing for the
    // outputs about it.
    public const string ItemZygion = "Zygion width";
    public const string ItemReference = "Reference points";
    public const string ItemOutline = "Far cheek outline";
    public const string ItemApex = "Malar apex";
    public const string ItemSubmalar = "Submalar point";
    public const string ItemHeadTurn = "Head turn";

    // The items, in order: (name, the views it suits, its outputs in CSV order). The
    // frontal items' outputs are NaN on a quarter face and the quarter items' on a frontal
    // face (and always for the near side); the head turn's are numeric in either view.
    public static readonly IReadOnlyList<(string Name, IReadOnlyList<FacePose> Views, IReadOnlyList<string> Outputs)>
        ItemDefinitions = Array.AsReadOnly(new (string, IReadOnlyList<FacePose>, IReadOnlyList<string>)[]
        {
            (ItemZygion, Views(FacePose.Frontal), Names(
                "zy.r", "zy.l", "zy.w", "zy.ratio", "zy.ratio.ok",
                "zy.r.clip", "zy.l.clip", "zy.r.ok", "zy.l.ok", "zy.ok")),
            (ItemReference, Views(FacePose.Frontal), Names("cr.r.d", "cr.l.d", "cr.ok", "sr.ok")),
            (ItemOutline, Views(FacePose.Quarter), Names(
                "cheek.far", "cheek.mask", "cheek.nose", "cheek.nose.gap", "cheek.scale")),
            (ItemApex, Views(FacePose.Quarter), Names(
                "apex.r.h", "apex.l.h", "apex.r.dev", "apex.l.dev", "apex.r.prior", "apex.l.prior",
                "apex.r.near_nose", "apex.l.near_nose", "apex.ok")),
            (ItemSubmalar, Views(FacePose.Quarter), Names(
                "sub.r.h", "sub.l.h", "sub.r.depth", "sub.l.depth", "sub.r.depth.px", "sub.l.depth.px",
                "sub.r.present", "sub.l.present", "sub.ok")),
            (ItemHeadTurn, Views(FacePose.Frontal, FacePose.Quarter), Names(
                "cheek.yaw", "cheek.yaw.model", "cheek.pitch", "cheek.pitch.ok")),
        });

    // The six items' names, in order.
    public static readonly IReadOnlyList<string> ItemNames =
        Array.AsReadOnly(ItemDefinitions.Select(d => d.Name).ToArray());

    // Every output, in CSV order: the items' outputs, item by item (41).
    public static readonly IReadOnlyList<string> AllOutputs =
        Array.AsReadOnly(ItemDefinitions.SelectMany(d => d.Outputs).ToArray());

    // The drawing's colors (dynaface-lib's, RGB).
    public static readonly Rgba32 Orange = new Rgba32(255, 140, 0, 255);   // levels, verticals, rings, chord
    public static readonly Rgba32 Blue = new Rgba32(31, 119, 180, 255);    // the zygion dots and labels
    public static readonly Rgba32 Cyan = new Rgba32(0, 200, 255, 255);     // outline rows from the mask
    public static readonly Rgba32 Red = new Rgba32(214, 39, 40, 255);      // the malar apex
    public static readonly Rgba32 Green = new Rgba32(44, 160, 44, 255);    // the submalar point
    public static readonly Rgba32 Gray = new Rgba32(110, 110, 110, 255);   // a flagged point; contour rows
    public static readonly Rgba32 Magenta = new Rgba32(255, 0, 255, 255);  // outline rows the nose joins
    public static readonly Rgba32 Yellow = new Rgba32(255, 230, 0, 255);   // rows an intensity edge stood in for
    public static readonly Rgba32 White = new Rgba32(255, 255, 255, 255);  // the rim of a filled dot

    // The far outline's color by row source (CheekOutline.OutlineSrc*).
    public static readonly IReadOnlyDictionary<byte, Rgba32> OutlineColors = new Dictionary<byte, Rgba32>
    {
        [CheekOutline.OutlineSrcContour] = Gray,
        [CheekOutline.OutlineSrcMask] = Cyan,
        [CheekOutline.OutlineSrcNose] = Magenta,
        [CheekOutline.OutlineSrcEdge] = Yellow,
    };

    // The dots: a white rim of DotRim px under a filled dot of DotRadius px; a reference
    // point is an open ring of DotRadius px and RingThickness px.
    public const int DotRim = 9, DotRadius = 7, RingThickness = 3;
    // A label sits LabelDx px outside its point, its baseline LabelDy px above it.
    public const int LabelDx = 14, LabelDy = 22;
    // The zygion level line runs this far past each zygion, in IPD units.
    public const double LevelOverhangIpd = 0.15;
    // Each oblique level is drawn from the outline inward by this fraction of H.
    public const double ObliqueLevelLength = 0.25;
    // The far outline's polyline, px; the level ticks are 1 px, the chord 2 px.
    public const int OutlineThickness = 4;

    // Each text line that says why a three-quarter result is flagged starts with this.
    public const string FlagPrefix = "flagged: ";

    public override string Label => "CHEEK MEASURES";

    public MeasureCheekMeasures()
    {
        foreach (var (name, views, outputs) in ItemDefinitions)
            Items.Add(new MeasureItemInfo(name, outputs: outputs, views: views));
        IsFrontal = true;
        IsQuarter = true;
        IsLateral = false;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true) =>
        Calc(ctx, render, new ContextCanvas(ctx));

    // Calc drawing on a canvas: the context's own (ContextCanvas) or a test's. The
    // frontal constructions and the observed points come from ctx.FrontalCheek() and
    // ctx.ObliqueCheek() (the context's last FindCheekLandmarks, which Analyze runs
    // first), unless frontalCheek or obliqueCheek stands in for that accessor, as a face
    // may hand over its own in dynaface-lib.
    internal Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render, ICheekCanvas canvas,
        Func<Cheek.FrontalCheek?>? frontalCheek = null, Func<CheekOutline.ObliqueCheek?>? obliqueCheek = null)
    {
        var result = new Dictionary<string, double>();
        if (ctx.IsLateral) return result;
        var data = new Dictionary<string, double>();
        foreach (string name in Outputs()) data[name] = double.NaN;
        Vec2[] landmarks = ctx.Landmarks;
        Cheek.Frame? frame = Cheek.PupilFrame(landmarks);
        double yaw = frame != null ? Cheek.LandmarkYaw(landmarks, frame) : double.NaN;
        double yawModel = Cheek.CalibrateYaw(HeadPoseAt(ctx, 0));
        double pitch = HeadPoseAt(ctx, 1);
        data["cheek.yaw"] = yaw;
        data["cheek.yaw.model"] = yawModel;
        data["cheek.pitch"] = pitch;
        // Reported, and folded into no other flag: pitch is uncalibrated.
        data["cheek.pitch.ok"] = Flag(Math.Abs(pitch) <= Cheek.PitchGrossRaw);
        // False through NaN, so an unknown yaw fails every yaw gate.
        bool agree = Math.Abs(yaw - yawModel) <= Cheek.YawAgreeDeg;
        var column = new List<string>();
        var reasons = new List<string>();
        double viewOk;
        if (ctx.IsQuarter)
        {
            bool farLeft = ctx.FarLeft;
            CheekOutline.ObliqueCheek? ob = obliqueCheek != null ? obliqueCheek() : ctx.ObliqueCheek();
            data["cheek.far"] = Flag(farLeft);
            if (ob != null) FillOblique(data, ob, farLeft, yaw, agree);
            if (render && ob != null)
            {
                bool outsideLeft = !farLeft;
                if (IsEnabled(ItemOutline)) DrawOutline(canvas, ob, farLeft);
                if (IsEnabled(ItemApex))
                {
                    DrawApex(canvas, ob, outsideLeft, data);
                    column.Add(ColumnApex(ob, farLeft));
                }
                if (IsEnabled(ItemSubmalar))
                {
                    DrawSubmalar(canvas, ob, outsideLeft);
                    column.AddRange(ColumnSubmalar(ob, farLeft, data));
                }
            }
            if (render) reasons = QuarterReasons(ob, data, yaw, yawModel, agree);
            viewOk = data["apex.ok"];
        }
        else
        {
            Cheek.FrontalCheek? fc = frontalCheek != null ? frontalCheek() : ctx.FrontalCheek();
            if (fc != null) FillFrontal(data, fc, ctx.Pix2mm, yaw, agree);
            if (render && fc != null)
            {
                if (IsEnabled(ItemZygion))
                {
                    DrawZygion(canvas, fc, data);
                    column.Add(ColumnZygion(data));
                }
                if (IsEnabled(ItemReference))
                {
                    DrawReference(canvas, fc, data);
                    column.Add(ColumnReference(data));
                }
            }
            viewOk = data["zy.ok"];
        }
        if (render && IsEnabled(ItemHeadTurn)) column.Add(ColumnYaw(data, viewOk));
        foreach (string reason in reasons) column.Add(FlagPrefix + reason);
        if (column.Count > 0)
        {
            ctx.AddHeader(Label);
            foreach (string txt in column) ctx.AddValue(txt);
            ctx.AddSpacer();
        }
        // Every declared output, in the items' order (dynaface-lib's select_outputs).
        foreach (string name in Outputs()) result[name] = data[name];
        return result;
    }

    // SPIGA's head pose value at index i (0 yaw, 1 pitch), NaN when the context has none
    // (dynaface-lib's getattr(face, "yaw", math.nan)).
    static double HeadPoseAt(FaceMeasureContext ctx, int i) =>
        ctx.HeadPose != null && ctx.HeadPose.Length > i ? ctx.HeadPose[i] : double.NaN;

    // The frontal items' values and gates from the constructions (their coordinates are
    // MeasureLandmarks').
    static void FillFrontal(Dictionary<string, double> data, Cheek.FrontalCheek fc, double pix2mm,
        double yaw, bool agree)
    {
        double zyR = fc.ZygionRPx * pix2mm;
        double zyL = fc.ZygionLPx * pix2mm;
        data["zy.r"] = zyR;
        data["zy.l"] = zyL;
        data["zy.w"] = zyR + zyL;
        data["zy.ratio"] = SymmetryRatio(zyR, zyL);
        data["zy.r.clip"] = fc.ZygionRClip;
        data["zy.l.clip"] = fc.ZygionLClip;
        bool poseOk = Math.Abs(yaw) <= Cheek.FrontalYawGate && agree;
        data["zy.r.ok"] = Flag(poseOk && fc.ZygionRClip == Cheek.ClipNone);
        data["zy.l.ok"] = Flag(poseOk && fc.ZygionLClip == Cheek.ClipNone);
        data["zy.ok"] = Flag(data["zy.r.ok"] == 1.0 && data["zy.l.ok"] == 1.0);
        data["zy.ratio.ok"] = Flag(data["zy.ok"] == 1.0 && Math.Abs(yaw) <= Cheek.ZygionRatioYawGate);
        data["cr.r.d"] = fc.CheekRefRPx * pix2mm;
        data["cr.l.d"] = fc.CheekRefLPx * pix2mm;
        data["cr.ok"] = data["sr.ok"] = Flag(poseOk);
    }

    // The quarter items' values and gates from the observed points (their coordinates
    // are MeasureLandmarks'), for the far side only: the face's, which also set cheek.far.
    // Observed points that name the other side are reported on the face's, with a
    // warning, so cheek.far and the reported side never disagree.
    static void FillOblique(Dictionary<string, double> data, CheekOutline.ObliqueCheek ob, bool farLeft,
        double yaw, bool agree)
    {
        if (ob.FarLeft != farLeft)
            System.Diagnostics.Trace.TraceWarning(
                "The observed cheek points say the far cheek is the subject's {0}, "
                + "the face says the {1}: reported on the face's side.",
                ob.FarLeft ? "left" : "right", farLeft ? "left" : "right");
        string s = farLeft ? "l" : "r";
        data[$"apex.{s}.h"] = ob.ApexH;
        data[$"apex.{s}.dev"] = ob.ApexDev;
        data[$"apex.{s}.prior"] = Flag(ob.ApexPrior);
        data[$"apex.{s}.near_nose"] = Flag(ob.ApexNearNose);
        data[$"sub.{s}.h"] = ob.SubmalarH;
        data[$"sub.{s}.depth"] = ob.SubmalarDepth;
        data[$"sub.{s}.depth.px"] = ob.SubmalarDepthPx;
        data[$"sub.{s}.present"] = ob.SubmalarPresent;
        data["cheek.mask"] = ob.MaskFraction;
        data["cheek.nose"] = Flag(ob.NoseOn);
        data["cheek.nose.gap"] = ob.NoseGap;
        data["cheek.scale"] = ob.Scale;
        data["apex.ok"] = Flag(
            Math.Abs(Math.Abs(yaw) - Cheek.ObliqueYaw) <= Cheek.ObliqueYawGate
            && agree
            && !ob.NoseOn
            && !ob.ApexNearNose
            && ob.WindowMask >= CheekOutline.OutlineMinMask);
        data["sub.ok"] = Flag(data["apex.ok"] == 1.0 && !double.IsNaN(ob.SubmalarDepth));
    }

    // ------------------------------------------------------------------ drawing

    // Writes a point's label outside it at the standard measure text size: its baseline
    // LabelDy px above the point, and LabelDx px to its left (the text's right end there)
    // or to its right. A label that would cross the image's edge on its outside (start
    // before x = 0 or end past the canvas width) is written on the point's other side
    // instead; if it crosses on both sides, its x is clamped so the text stays inside.
    // outsideLeft: whether the label goes to the point's left (the subject's right side,
    // whose outside is the image left).
    internal static void WriteLabel(ICheekCanvas canvas, Vec2d point, string text, Rgba32 color, bool outsideLeft)
    {
        var (x, y) = Pt(point);
        int width = canvas.TextWidth(text);
        int left = x - width - LabelDx, right = x + LabelDx;
        int first = outsideLeft ? left : right, second = outsideLeft ? right : left;
        int textX;
        if (first >= 0 && first + width <= canvas.Width) textX = first;
        else if (second >= 0 && second + width <= canvas.Width) textX = second;
        else textX = Math.Max(0, Math.Min(first, canvas.Width - width));
        canvas.Text((textX, y - LabelDy), text, color);
    }

    // The zygion level line and the two zygion dots with their labels.
    static void DrawZygion(ICheekCanvas canvas, Cheek.FrontalCheek fc, Dictionary<string, double> data)
    {
        Cheek.Frame frame = fc.Frame;
        double overhang = LevelOverhangIpd * frame.Ipd;
        Vec2d levelR = frame.Point(-fc.ZygionRPx - overhang, fc.ZygionLevel);
        Vec2d levelL = frame.Point(fc.ZygionLPx + overhang, fc.ZygionLevel);
        if (Finite(levelR, levelL)) canvas.Line(Pt(levelR), Pt(levelL), Orange, 2);
        foreach (var (point, clip, ok, outsideLeft) in new[]
        {
            (fc.ZygionR, fc.ZygionRClip, data["zy.r.ok"], true),
            (fc.ZygionL, fc.ZygionLClip, data["zy.l.ok"], false),
        })
        {
            if (!Finite(point)) continue;
            canvas.Circle(Pt(point), White, DotRim, -1);
            canvas.Circle(Pt(point), ok == 1.0 ? Blue : Gray, DotRadius, -1);
            string label = "zygion" + (clip != Cheek.ClipNone ? " (clipped)" : "");
            WriteLabel(canvas, point, label, Blue, outsideLeft);
        }
    }

    // The canthus verticals and the open rings of the reference points with their labels.
    static void DrawReference(ICheekCanvas canvas, Cheek.FrontalCheek fc, Dictionary<string, double> data)
    {
        Rgba32 color = data["cr.ok"] == 1.0 ? Orange : Gray;
        foreach (var (canthus, foot) in new[] { (fc.CanthusR, fc.FootR), (fc.CanthusL, fc.FootL) })
            if (Finite(canthus, foot)) canvas.Line(Pt(canthus), Pt(foot), Orange, 3);
        foreach (var (point, text, outsideLeft) in new[]
        {
            (fc.CheekRefR, "cheek reference point", true),
            (fc.SubmalarRefR, "submalar reference point", true),
            (fc.CheekRefL, "cheek reference point", false),
            (fc.SubmalarRefL, "submalar reference point", false),
        })
        {
            if (!Finite(point)) continue;
            canvas.Circle(Pt(point), color, DotRadius, RingThickness);
            WriteLabel(canvas, point, text, Orange, outsideLeft);
        }
    }

    // The far outline colored by source, the three level ticks and the chord. farLeft:
    // the face's far side, which decides which way is inward.
    static void DrawOutline(ICheekCanvas canvas, CheekOutline.ObliqueCheek ob, bool farLeft)
    {
        Vec2d[] points = ob.Outline;
        byte[] src = ob.OutlineSrc;
        int n = Math.Min(points.Length, src.Length);
        // Each run of rows with one source is one polyline, through the first row of the
        // next run too, so the segment between two rows takes the upper row's color and
        // no segment is skipped.
        int start = 0;
        while (start < n - 1)
        {
            int end = start;
            while (end < n - 1 && src[end] == src[start]) end++;
            var run = new Vec2d[end - start + 1];
            Array.Copy(points, start, run, 0, run.Length);
            if (Finite(run))
                canvas.Polyline(run.Select(Pt).ToArray(),
                    OutlineColors.TryGetValue(src[start], out Rgba32 c) ? c : Gray, OutlineThickness);
            start = end;
        }
        var (a, b) = ob.Chord;
        double outward = farLeft ? 1.0 : -1.0;
        double length = ObliqueLevelLength * ob.Scale;
        if (ob.Levels.TryGetValue("chin", out var chin))
        {
            foreach (var (name, tickStart) in new[] { ("canthus", a), ("stomion", b), ("chin", chin.Point) })
            {
                if (!ob.Levels.TryGetValue(name, out var level)) continue;
                Vec2d direction = level.Direction;
                Vec2d inward = direction.X * outward < 0.0 ? direction : -direction;
                Vec2d tickEnd = tickStart + inward * length;
                if (Finite(tickStart, tickEnd)) canvas.Line(Pt(tickStart), Pt(tickEnd), Orange, 1);
            }
        }
        if (Finite(a, b)) canvas.Line(Pt(a), Pt(b), Orange, 2);
    }

    // The malar apex's dot and label. outsideLeft: whether the label goes to the point's
    // left (the near side of a far-right cheek: not farLeft).
    static void DrawApex(ICheekCanvas canvas, CheekOutline.ObliqueCheek ob, bool outsideLeft,
        Dictionary<string, double> data)
    {
        if (!Finite(ob.Apex)) return;
        canvas.Circle(Pt(ob.Apex), White, DotRim, -1);
        canvas.Circle(Pt(ob.Apex), data["apex.ok"] == 1.0 ? Red : Gray, DotRadius, -1);
        WriteLabel(canvas, ob.Apex, "malar apex" + ApexNote(ob), Red, outsideLeft);
    }

    // The submalar point's dot and label, when a hollow is present. Only a present hollow
    // is drawn: an absent point is wherever the outline comes nearest the chord, on a
    // convex cheek a row beside one of the chord's ends, so a ring there marked nothing
    // and overprinted the apex label.
    static void DrawSubmalar(ICheekCanvas canvas, CheekOutline.ObliqueCheek ob, bool outsideLeft)
    {
        if (ob.SubmalarPresent != 1.0 || !Finite(ob.Submalar)) return;
        canvas.Circle(Pt(ob.Submalar), Green, DotRadius, -1);
        WriteLabel(canvas, ob.Submalar, "submalar point", Green, outsideLeft);
    }

    // ------------------------------------------------------------------ the text

    // Why a quarter face's result is flagged, for the enabled items' lines: the far cheek
    // not measured at all (no observed points), under "Malar apex", "Submalar point" or
    // "Head turn"; the apex's failed gates (ApexReasons) when apex.ok is 0, under "Malar
    // apex" or "Head turn", whose lines carry that flag; a submalar depth that couldn't be
    // read, under "Submalar point"; and a gross pitch, under "Head turn".
    List<string> QuarterReasons(CheekOutline.ObliqueCheek? ob, Dictionary<string, double> data,
        double yaw, double yawModel, bool agree)
    {
        bool apex = IsEnabled(ItemApex), submalar = IsEnabled(ItemSubmalar);
        bool headTurn = IsEnabled(ItemHeadTurn);
        var reasons = new List<string>();
        if (ob == null)
        {
            if (apex || submalar || headTurn) reasons.Add("the far cheek could not be measured");
        }
        else
        {
            if (data["apex.ok"] != 1.0 && (apex || headTurn)) reasons.AddRange(ApexReasons(ob, yaw, yawModel, agree));
            if (submalar && double.IsNaN(ob.SubmalarDepth)) reasons.Add(SubmalarReason(ob));
        }
        if (headTurn && data["cheek.pitch.ok"] != 1.0)
        {
            string pitch = Beyond(data["cheek.pitch"], Cheek.PitchGrossRaw);
            reasons.Add($"head tipped up or down (pitch {pitch}, needs within {Fixed(Cheek.PitchGrossRaw, 0)})");
        }
        return reasons;
    }

    // Why apex.ok is 0, as short phrases, in the order its gates are checked: the turn
    // outside ObliqueYaw +/- ObliqueYawGate, the two yaws more than YawAgreeDeg apart, the
    // nose on the far outline, the apex within ApexNearNoseH of the nose tip's height, and
    // fewer than OutlineMinMask of the search window's rows from the mask. Every gate that
    // failed has its phrase, except that the window's rows are not blamed on an unclear
    // edge when the nose is on the outline: its rows are the ones that didn't come from
    // the mask.
    internal static List<string> ApexReasons(CheekOutline.ObliqueCheek ob, double yaw, double yawModel, bool agree)
    {
        var reasons = new List<string>();
        double low = Cheek.ObliqueYaw - Cheek.ObliqueYawGate;
        double high = Cheek.ObliqueYaw + Cheek.ObliqueYawGate;
        string window = $"needs {Fixed(low, 0)} to {Fixed(high, 0)}";
        double turn = Math.Abs(yaw);
        if (double.IsNaN(turn)) reasons.Add("the turn is unknown");
        else if (turn > high) reasons.Add($"turned too far ({Beyond(turn, high)} degrees, {window})");
        else if (turn < low) reasons.Add($"not turned enough ({Beyond(turn, low)} degrees, {window})");
        if (!agree && !double.IsNaN(turn))
        {
            if (double.IsNaN(yawModel)) reasons.Add("the model's turn is unknown");
            else reasons.Add($"the turn estimates differ ({Fixed(yaw, 1)} and {Fixed(yawModel, 1)} degrees)");
        }
        if (ob.NoseOn) reasons.Add("the nose hides the far cheek");
        if (ob.ApexNearNose) reasons.Add("the apex is level with the nose tip");
        if (ob.WindowMask < CheekOutline.OutlineMinMask && !ob.NoseOn)
            reasons.Add("the cheek's edge is unclear (hair or background)");
        return reasons;
    }

    // Why the submalar depth couldn't be read: too few of the rows from the apex down came
    // from the mask. When nose rows make up at least half of the rows that didn't, the
    // nose is named instead of an unclear edge.
    internal static string SubmalarReason(CheekOutline.ObliqueCheek ob)
    {
        Vec2d[] rows = ob.Outline;
        byte[] src = ob.OutlineSrc;
        double apexY = ob.Apex.Y;
        if (rows.Length > 0 && src.Length == rows.Length && !double.IsNaN(apexY) && !double.IsInfinity(apexY))
        {
            int others = 0, nose = 0;
            for (int i = 0; i < rows.Length; i++)
            {
                if (!(rows[i].Y >= apexY)) continue;
                if (src[i] != CheekOutline.OutlineSrcMask) others++;
                if (src[i] == CheekOutline.OutlineSrcNose) nose++;
            }
            if (nose > 0 && 2 * nose >= others)
                return "no submalar depth (the nose covers the cheek below the apex)";
        }
        return "no submalar depth (the edge is unclear below the apex)";
    }

    // A value outside a limit, to one decimal place, or "just over 50" (or "just under
    // 40", or "just under -20" for a negative value past -20) where one place would round
    // it onto the limit, so the text never reads "50.0 degrees, needs 40 to 50".
    internal static string Beyond(double value, double bound)
    {
        string text = Fixed(value, 1);
        if (double.IsNaN(value) || double.IsInfinity(value)
            || Math.Abs(double.Parse(text, CultureInfo.InvariantCulture)) != bound)
            return text;
        double limit = value < 0.0 ? -bound : bound;
        string side = value > limit ? "over" : "under";
        return $"just {side} {limit.ToString("G6", CultureInfo.InvariantCulture)}";
    }

    // The apex label's suffix: " (prior)" when the prior stood in for a local maximum,
    // " (nose?)" when its height is within ApexNearNoseH of the nose tip's, " (prior,
    // nose?)" for both, "" otherwise.
    internal static string ApexNote(CheekOutline.ObliqueCheek ob)
    {
        var notes = new List<string>();
        if (ob.ApexPrior) notes.Add("prior");
        if (ob.ApexNearNose) notes.Add("nose?");
        return notes.Count > 0 ? $" ({string.Join(", ", notes)})" : "";
    }

    // The zygion headline: the width, the two midline distances, the ratio.
    internal static string ColumnZygion(IReadOnlyDictionary<string, double> data)
    {
        string txt = $"zy.w={Fixed(data["zy.w"], 1)} mm (R {Fixed(data["zy.r"], 1)} / L {Fixed(data["zy.l"], 1)},"
            + $" ratio {Fixed(data["zy.ratio"], 2)})";
        return txt + (data["zy.ok"] == 1.0 ? "" : " (flagged)");
    }

    // The reference headline: the cheek reference points' depths.
    internal static string ColumnReference(IReadOnlyDictionary<string, double> data) =>
        $"cheek ref depth R {Fixed(data["cr.r.d"], 1)} / L {Fixed(data["cr.l.d"], 1)} mm";

    // The apex headline, named for the face's far side.
    internal static string ColumnApex(CheekOutline.ObliqueCheek ob, bool farLeft)
    {
        string s = farLeft ? "l" : "r";
        return $"apex.{s} h={Fixed(ob.ApexH, 2)} dev={Fixed(ob.ApexDev, 3)}{ApexNote(ob)}";
    }

    // The submalar headline when its depth is known (nothing otherwise), named for the
    // face's far side, flagged when sub.ok is 0. A depth just below zero reads 0.000, not
    // -0.000 (dynaface-lib rounds it and adds 0.0).
    internal static List<string> ColumnSubmalar(CheekOutline.ObliqueCheek ob, bool farLeft,
        IReadOnlyDictionary<string, double> data)
    {
        var lines = new List<string>();
        if (double.IsNaN(ob.SubmalarDepth)) return lines;
        string s = farLeft ? "l" : "r";
        string state = ob.SubmalarPresent == 1.0 ? "hollow" : "absent";
        string flagged = data["sub.ok"] == 1.0 ? "" : " (flagged)";
        lines.Add($"sub.{s} depth={Fixed(ob.SubmalarDepth, 3, signedZero: false)} ({state}){flagged}");
        return lines;
    }

    // The pose headline, flagged when the view's gate or the pitch failed.
    internal static string ColumnYaw(IReadOnlyDictionary<string, double> data, double viewOk)
    {
        string txt = $"cheek yaw {Fixed(data["cheek.yaw"], 1)} (model {Fixed(data["cheek.yaw.model"], 1)})"
            + $" pitch {Fixed(data["cheek.pitch"], 1)}";
        bool flagged = viewOk != 1.0 || data["cheek.pitch.ok"] != 1.0;
        return txt + (flagged ? " (flagged)" : "");
    }

    // ------------------------------------------------------------------ helpers

    // A point as whole pixels, each coordinate rounded half to even (Python's round()).
    internal static (int X, int Y) Pt(Vec2d p) =>
        ((int)Math.Round(p.X, MidpointRounding.ToEven), (int)Math.Round(p.Y, MidpointRounding.ToEven));

    // A flag as 1.0 or 0.0.
    static double Flag(bool value) => value ? 1.0 : 0.0;

    // Whether every point can be drawn: all of their coordinates are finite. A point
    // that isn't is skipped rather than drawn, so Calc(render: true) returns wherever
    // Calc(render: false) does.
    static bool Finite(params Vec2d[] points)
    {
        foreach (Vec2d p in points)
            if (!p.IsFinite) return false;
        return true;
    }

    // dynaface-lib's util.symmetry_ratio, with Python's min and max (each keeps its first
    // argument unless the second compares smaller or larger, so a NaN first argument
    // gives NaN): 1.0 when both are zero, else min / max.
    internal static double SymmetryRatio(double a, double b)
    {
        if (a == 0.0 && b == 0.0) return 1.0;
        double min = b < a ? b : a;
        double max = b > a ? b : a;
        return min / max;
    }

    // A number as Python's f"{value:.{places}f}" writes it: the exact binary value rounded
    // half to even at that many places, "nan", "inf" or "-inf" when not finite, with the
    // sign of a negative value that rounds to zero ("-0.0") unless signedZero is false,
    // as dynaface-lib's round(value, places) + 0.0 drops it. Never culture-dependent, and
    // the same on every runtime.
    internal static string Fixed(double value, int places, bool signedZero = true)
    {
        if (double.IsNaN(value)) return "nan";
        if (double.IsInfinity(value)) return value > 0 ? "inf" : "-inf";
        long bits = BitConverter.DoubleToInt64Bits(value);
        bool negative = bits < 0;
        int exponent = (int)((bits >> 52) & 0x7FF);
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0) exponent = 1; else mantissa |= 1L << 52;
        exponent -= 1075;
        // |value| * 10^places = numerator / denominator, exactly.
        BigInteger numerator = new BigInteger(mantissa) * BigInteger.Pow(10, places);
        BigInteger denominator = BigInteger.One;
        if (exponent > 0) numerator <<= exponent; else denominator <<= -exponent;
        BigInteger quotient = BigInteger.DivRem(numerator, denominator, out BigInteger remainder);
        int half = (remainder * 2).CompareTo(denominator);
        if (half > 0 || (half == 0 && !quotient.IsEven)) quotient += 1;
        string digits = quotient.ToString(CultureInfo.InvariantCulture).PadLeft(places + 1, '0');
        string text = places == 0
            ? digits
            : digits.Substring(0, digits.Length - places) + "." + digits.Substring(digits.Length - places);
        bool sign = negative && (signedZero || !quotient.IsZero);
        return sign ? "-" + text : text;
    }

    static IReadOnlyList<FacePose> Views(params FacePose[] views) => Array.AsReadOnly(views);

    static IReadOnlyList<string> Names(params string[] names) => Array.AsReadOnly(names);

    // Draws on the context's image: the drawing calls in top-left whole pixels, the same
    // calls dynaface-lib makes on its face (line, circle, polylines, write_text).
    sealed class ContextCanvas : ICheekCanvas
    {
        readonly FaceMeasureContext _ctx;

        public ContextCanvas(FaceMeasureContext ctx) => _ctx = ctx;

        public int Width => _ctx.Width;

        public int TextWidth(string text) => FaceRenderer.GetTextSize(text, FaceRenderer.TEXT_SIZE_MEASURE).X;

        public void Line((int X, int Y) a, (int X, int Y) b, Rgba32 color, int thickness) =>
            _ctx.DrawLine(new Vec2(a.X, a.Y), new Vec2(b.X, b.Y), color, thickness);

        public void Circle((int X, int Y) center, Rgba32 color, int radius, int thickness)
        {
            if (thickness < 0)
                _ctx.DrawCircleAt(new Vec2(center.X, center.Y), color, radius);
            else
                FaceRenderer.DrawRing(_ctx.Pixels, _ctx.Width, _ctx.Height,
                    center.X, _ctx.Height - 1 - center.Y, color, radius, thickness);
        }

        public void Polyline((int X, int Y)[] points, Rgba32 color, int thickness)
        {
            for (int i = 1; i < points.Length; i++) Line(points[i - 1], points[i], color, thickness);
        }

        // origin: the text's bottom-left, as OpenCV's putText takes it; DrawImageText takes
        // its top-left.
        public void Text((int X, int Y) origin, string text, Rgba32 color)
        {
            int height = FaceRenderer.GetTextSize(text, FaceRenderer.TEXT_SIZE_MEASURE).Y;
            _ctx.DrawImageText(new Vec2(origin.X, origin.Y - height), text, color);
        }
    }
}

// What MeasureCheekMeasures draws, in top-left whole pixels of the image: dynaface-lib's
// face.line, face.circle (a filled dot for a negative thickness, else an open ring),
// face.polyline and face.write_text with face.calc_text_size, so a test can record the
// drawing as dynaface-lib's tests record a mock face's. The measure draws on its context
// through a private implementation.
internal interface ICheekCanvas
{
    // The image's width, px, which the labels keep inside.
    int Width { get; }

    // A label's width at the standard measure text size, px.
    int TextWidth(string text);

    void Line((int X, int Y) a, (int X, int Y) b, Rgba32 color, int thickness);

    void Circle((int X, int Y) center, Rgba32 color, int radius, int thickness);

    void Polyline((int X, int Y)[] points, Rgba32 color, int thickness);

    // origin: the text's bottom-left (OpenCV's putText origin).
    void Text((int X, int Y) origin, string text, Rgba32 color);
}
