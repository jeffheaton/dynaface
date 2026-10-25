using System;
using System.Collections.Generic;

// Dental Display: area of the inner lip opening, split at the facial midline (the tilted
// perpendicular bisector of the pupils, ctx.CalcBisect). WFLW: landmarks 88-95 = inner lip
// contour. Sides are the person's, like eye.right: dental_right is the image-left half.
// Mirrors dynaface-lib's AnalyzeDentalArea: each half is the outline clipped to one side of
// the midline, so a closed mouth (an outline with no area) measures ~0 instead of failing,
// and the zero fallback only covers real failures.
public class MeasureDentalArea : FaceMeasureBase
{
    // Floating-point dust from a closed mouth counts as zero, so two empty halves keep a
    // symmetry ratio of 1 (dynaface-lib's DENTAL_MIN_AREA_MM2).
    const double MinAreaMm2 = 1e-6;

    public override string Label => "DENTAL";

    static readonly Rgba32 RightColor = new Rgba32(255, 60, 60, 255);
    static readonly Rgba32 LeftColor = new Rgba32(60, 60, 255, 255);

    public MeasureDentalArea()
    {
        Items.Add(new MeasureItemInfo("dental_area"));
        Items.Add(new MeasureItemInfo("dental_left"));
        Items.Add(new MeasureItemInfo("dental_right"));
        Items.Add(new MeasureItemInfo("dental_ratio"));
        Items.Add(new MeasureItemInfo("dental_diff"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool renderArea = IsEnabled("dental_area");
        bool renderLeft = IsEnabled("dental_left");
        bool renderRight = IsEnabled("dental_right");
        bool renderRatio = IsEnabled("dental_ratio");
        bool renderDiff = IsEnabled("dental_diff");

        var poly = Slice(ctx.Landmarks, 88, 95);
        var (p1, p2) = ctx.CalcBisect();

        float rightArea, leftArea;
        try
        {
            rightArea = HalfArea(ctx, poly, p1, p2, imageLeft: true, render && renderRight, RightColor);
            leftArea = HalfArea(ctx, poly, p1, p2, imageLeft: false, render && renderLeft, LeftColor);
        }
        catch (Exception)
        {
            // A real failure, not a closed mouth: same zero fallback as dynaface-lib.
            rightArea = leftArea = 0f;
        }

        float total = rightArea + leftArea;
        float ratio = FaceMeasureContext.SymmetryRatio(leftArea, rightArea);
        float diff = MathHelpers.Abs(leftArea - rightArea);

        if (render)
        {
            ctx.AddHeader(Label);
            if (renderArea) ctx.AddValue($"total: {total:F1} mm²");
            if (renderLeft) ctx.AddValue($"left:  {leftArea:F1} mm²");
            if (renderRight) ctx.AddValue($"right: {rightArea:F1} mm²");
            if (renderRatio) ctx.AddValue($"ratio: {ratio:F2}");
            if (renderDiff) ctx.AddValue($"diff:  {diff:F2} mm²");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double>
        {
            ["dental_area"] = total,
            ["dental_left"] = leftArea,
            ["dental_right"] = rightArea,
            ["dental_ratio"] = ratio,
            ["dental_diff"] = diff,
        };
    }

    static float HalfArea(FaceMeasureContext ctx, Vec2[] poly, Vec2 p1, Vec2 p2, bool imageLeft,
        bool render, Rgba32 color)
    {
        var half = FaceMeasureContext.ClipPolygonToSide(poly, p1, p2, imageLeft);
        if (half.Length < 3) return 0f;
        if (render) ctx.MeasurePolygon(half, render: true, color: color);
        double area = FaceMeasureContext.PolygonAreaPx(half) * ctx.Pix2mm * ctx.Pix2mm;
        return area >= MinAreaMm2 ? (float)area : 0f;
    }

    static Vec2[] Slice(Vec2[] lm, int first, int last)
    {
        var result = new Vec2[last - first + 1];
        for (int i = 0; i < result.Length; i++) result[i] = lm[first + i];
        return result;
    }
}
