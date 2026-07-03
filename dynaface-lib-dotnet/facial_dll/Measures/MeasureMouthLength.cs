using System.Collections.Generic;

// Mouth Length (ML): horizontal distance between mouth corners.
// WFLW: landmark 88 = right commissure, 92 = left commissure (inner lip).
public class MeasureMouthLength : FaceMeasureBase
{
    public override string Label => "MOUTH LENGTH";

    public MeasureMouthLength()
    {
        Items.Add(new MeasureItemInfo("ml"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render1 = IsEnabled("ml");
        float mm = ctx.Measure(ctx.Landmarks[88], ctx.Landmarks[92], render: render && render1, dir: "r");

        if (render && render1)
        {
            ctx.AddHeader(Label);
            ctx.AddValue($"ml: {mm:F1} mm");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["ml"] = mm };
    }
}
