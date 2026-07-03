using System.Collections.Generic;

// Outer Eye Corners (OE): distance between lateral canthi.
// WFLW: landmark 60 = right lateral canthus, 72 = left lateral canthus.
public class MeasureOuterEyeCorners : FaceMeasureBase
{
    public override string Label => "OUTER EYE";

    public MeasureOuterEyeCorners()
    {
        Items.Add(new MeasureItemInfo("oe"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render1 = IsEnabled("oe");
        float mm = ctx.Measure(ctx.Landmarks[60], ctx.Landmarks[72], render: render && render1, dir: "r");

        if (render && render1)
        {
            ctx.AddHeader(Label);
            ctx.AddValue($"oe: {mm:F1} mm");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["oe"] = mm };
    }
}
