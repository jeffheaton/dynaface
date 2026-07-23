using System.Collections.Generic;

// Dental Display: area of the inner lip opening, split left/right by the face midline.
// WFLW: landmarks 88-95 = inner lip contour.
public class MeasureDentalArea : FaceMeasureBase
{
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

        float midX = ctx.Landmarks.Length > 97
            ? (ctx.Landmarks[96].X + ctx.Landmarks[97].X) * 0.5f
            : ctx.Width * 0.5f;

        float rightArea = 0f, leftArea = 0f;
        try
        {
            var (leftPoly, rightPoly) = FaceMeasureContext.SplitPolygonByX(poly, midX);
            rightArea = leftPoly.Length >= 3 ? ctx.MeasurePolygon(leftPoly, render && renderRight, RightColor) : 0f;
            leftArea = rightPoly.Length >= 3 ? ctx.MeasurePolygon(rightPoly, render && renderLeft, LeftColor) : 0f;
        }
        catch { /* malformed polygon — skip fill */ }

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

    static Vec2[] Slice(Vec2[] lm, int first, int last)
    {
        var result = new Vec2[last - first + 1];
        for (int i = 0; i < result.Length; i++) result[i] = lm[first + i];
        return result;
    }
}
