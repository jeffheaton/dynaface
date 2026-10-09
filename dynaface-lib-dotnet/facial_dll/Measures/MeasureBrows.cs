using System.Collections.Generic;

// Brow Asymmetry: vertical gap between two parallel lines drawn through the brow
// landmarks at the head-tilt angle, which cancels head tilt (a symmetric face scores
// ~0 at any tilt). WFLW: 35=image-left brow, 36=image-left brow ref, 44=image-right brow.
//
// dynaface-lib's AnalyzeBrows extends both lines with util.line_to_edge, which lands
// both on the RIGHT image edge, and compares their y values there — i.e. at the same
// x. Because the lines are parallel that gap is the same at every x, so it's computed
// here directly: |(y44 - y35) - tan(tilt) * (x44 - x35)|. Python truncates the two
// edge points to ints first, so the two ports can differ by up to 1px.
//
// ctx.LineToEdge (a directional ray-cast, not a port of util.line_to_edge) is used only
// to draw the arrows, which point outward to opposite edges. Their tips therefore sit
// at different x and their y difference is NOT brow.d; comparing y values at opposite
// edges was a past bug that added ~1023*|tan(tilt)| px for a tilted face.
public class MeasureBrows : FaceMeasureBase
{
    public override string Label => "BROW";

    static readonly Rgba32 ArrowColor = new Rgba32(255, 165, 0, 255); // orange

    public MeasureBrows()
    {
        Items.Add(new MeasureItemInfo("brow.d"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render2 = IsEnabled("brow.d");
        float tilt = ctx.CalcFaceRotationDeg();

        Vec2 rightBrow = ctx.Landmarks[35];
        Vec2 leftBrow = ctx.Landmarks[44];
        float slope = MathHelpers.Tan(tilt * MathHelpers.Deg2Rad);
        float diffPx = MathHelpers.Abs((leftBrow.Y - rightBrow.Y) - slope * (leftBrow.X - rightBrow.X));
        float diff = diffPx * ctx.Pix2mm;

        if (render && render2)
        {
            Vec2? rightEdge = ctx.LineToEdge(rightBrow, 180f + tilt);
            Vec2? leftEdge = ctx.LineToEdge(leftBrow, tilt);
            if (rightEdge != null)
                ctx.DrawArrow(ctx.Landmarks[36], rightEdge.Value, ArrowColor, arrowAtEnd: true);
            if (leftEdge != null)
                ctx.DrawArrow(leftBrow, leftEdge.Value, ArrowColor, arrowAtEnd: true);

            ctx.AddHeader(Label);
            ctx.AddValue($"d.brow: {diff:F2} mm");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["brow.d"] = diff };
    }
}
