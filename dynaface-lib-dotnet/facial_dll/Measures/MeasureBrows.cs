using System.Collections.Generic;

// Brow Asymmetry: vertical difference between the heights of brow inner landmarks
// after projecting horizontally (accounting for face tilt) to the image edges.
// WFLW: 35=right brow medial, 36=right brow ref, 44=left brow medial.
//
// NOTE: this ray-casting uses ctx.LineToEdge, a directional-ray boundary intersection
// (cos/sin against the 4 edges, nearest hit wins) — a different algorithm in kind
// from dynaface-lib's util.line_to_edge (an undirected slope test against all 4
// edges in a fixed right/left/top/bottom priority order). The two aren't a clean
// 1:1 match; this was already the case before this port and is left as-is rather
// than risk a mismatched "fix" with no reference run to verify against.
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

        Vec2? rightEdge = ctx.LineToEdge(ctx.Landmarks[35], 180f + tilt);
        Vec2? leftEdge = ctx.LineToEdge(ctx.Landmarks[44], tilt);

        if (rightEdge == null || leftEdge == null)
            return new Dictionary<string, double>();

        float diff = MathHelpers.Abs(rightEdge.Value.Y - leftEdge.Value.Y) * ctx.Pix2mm;

        if (render && render2)
        {
            ctx.DrawArrow(ctx.Landmarks[36], rightEdge.Value, ArrowColor, arrowAtEnd: true);
            ctx.DrawArrow(ctx.Landmarks[44], leftEdge.Value, ArrowColor, arrowAtEnd: true);

            ctx.AddHeader(Label);
            ctx.AddValue($"d.brow: {diff:F2} mm");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["brow.d"] = diff };
    }
}
