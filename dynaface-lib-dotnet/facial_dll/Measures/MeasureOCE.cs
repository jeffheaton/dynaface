using System.Collections.Generic;

// Oral Commissure Excursion (OCE): distance from each mouth corner to upper lip center.
// WFLW: 76=right commissure, 82=left commissure, 85=upper lip center region.
public class MeasureOCE : FaceMeasureBase
{
    public override string Label => "ORAL CE";

    public MeasureOCE()
    {
        Items.Add(new MeasureItemInfo("oce.l"));
        Items.Add(new MeasureItemInfo("oce.r"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render2L = IsEnabled("oce.l");
        bool render2R = IsEnabled("oce.r");
        float oceR = ctx.Measure(ctx.Landmarks[76], ctx.Landmarks[85], render: render && render2R, dir: "l");
        float oceL = ctx.Measure(ctx.Landmarks[82], ctx.Landmarks[85], render: render && render2L, dir: "r");

        if (render && (render2R || render2L))
        {
            ctx.AddHeader(Label);
            if (render2R) ctx.AddValue($"oce.r: {oceR:F1} mm");
            if (render2L) ctx.AddValue($"oce.l: {oceL:F1} mm");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["oce.l"] = oceL, ["oce.r"] = oceR };
    }
}
