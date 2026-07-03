using System.Collections.Generic;

// Eye Area: polygon area of each eye opening and their asymmetry.
// WFLW: landmarks 60-67 = right eye contour, 68-75 = left eye contour.
public class MeasureEyeArea : FaceMeasureBase
{
    public override string Label => "EYE AREA";

    static readonly Rgba32 RightColor = new Rgba32(255,  60,  60, 255);
    static readonly Rgba32 LeftColor  = new Rgba32( 60,  60, 255, 255);

    public MeasureEyeArea()
    {
        Items.Add(new MeasureItemInfo("eye.left"));
        Items.Add(new MeasureItemInfo("eye.right"));
        Items.Add(new MeasureItemInfo("eye.diff"));
        Items.Add(new MeasureItemInfo("eye.ratio"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool renderEyeL    = IsEnabled("eye.left");
        bool renderEyeR    = IsEnabled("eye.right");
        bool renderEyeDiff = IsEnabled("eye.diff");
        bool renderEyeRatio = IsEnabled("eye.ratio");

        var rightContour = Slice(ctx.Landmarks, 60, 67);
        var leftContour  = Slice(ctx.Landmarks, 68, 75);

        float rightArea = ctx.MeasurePolygon(rightContour, render: render && renderEyeR, color: RightColor);
        float leftArea  = ctx.MeasurePolygon(leftContour,  render: render && renderEyeL, color: LeftColor);
        float diff  = MathHelpers.Abs(rightArea - leftArea);
        float ratio = FaceMeasureContext.SymmetryRatio(rightArea, leftArea);

        if (render && renderEyeR)
        {
            var rLabel = ctx.Landmarks[66];
            FaceRenderer.DrawText(ctx.Pixels, ctx.Width, ctx.Height,
                MathHelpers.RoundToInt(rLabel.X - 120),
                MathHelpers.RoundToInt(ctx.Height - 1 - (rLabel.Y + 24)),
                $"R {rightArea:F1}", RightColor, scale: FaceRenderer.TEXT_SIZE_MEASURE);
        }
        if (render && renderEyeL)
        {
            var lLabel = ctx.Landmarks[74];
            FaceRenderer.DrawText(ctx.Pixels, ctx.Width, ctx.Height,
                MathHelpers.RoundToInt(lLabel.X - 30),
                MathHelpers.RoundToInt(ctx.Height - 1 - (lLabel.Y + 24)),
                $"L {leftArea:F1}", LeftColor, scale: FaceRenderer.TEXT_SIZE_MEASURE);
        }

        if (render)
        {
            ctx.AddHeader(Label);
            if (renderEyeR)     ctx.AddValue($"eye.r: {rightArea:F1} mm²");
            if (renderEyeL)     ctx.AddValue($"eye.l: {leftArea:F1} mm²");
            if (renderEyeDiff)  ctx.AddValue($"diff:  {diff:F2} mm²");
            if (renderEyeRatio) ctx.AddValue($"ratio: {ratio:F2}");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double>
        {
            ["eye.left"]  = leftArea,
            ["eye.right"] = rightArea,
            ["eye.diff"]  = diff,
            ["eye.ratio"] = ratio,
        };
    }

    static Vec2[] Slice(Vec2[] lm, int first, int last)
    {
        var result = new Vec2[last - first + 1];
        for (int i = 0; i < result.Length; i++) result[i] = lm[first + i];
        return result;
    }
}
