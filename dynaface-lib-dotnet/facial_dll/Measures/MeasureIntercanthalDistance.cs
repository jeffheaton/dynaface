using System.Collections.Generic;

// Intercanthal Distance (ID): distance between inner corners of the eyes.
// WFLW: landmark 64 = right inner canthus, 68 = left inner canthus.
public class MeasureIntercanthalDistance : FaceMeasureBase
{
    public override string Label => "INTERCANTHAL";

    public MeasureIntercanthalDistance()
    {
        Items.Add(new MeasureItemInfo("id"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render1 = IsEnabled("id");
        float mm = ctx.Measure(ctx.Landmarks[64], ctx.Landmarks[68], render: render && render1, dir: "r");

        if (render && render1)
        {
            ctx.AddHeader(Label);
            ctx.AddValue($"id: {mm:F1} mm");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["id"] = mm };
    }
}
