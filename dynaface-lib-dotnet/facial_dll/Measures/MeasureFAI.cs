using System.Collections.Generic;

// Facial Angle of Incisor (FAI): asymmetry between left and right mouth-to-eye distances.
// WFLW: 64=right inner canthus, 68=left inner canthus, 76=right commissure, 82=left commissure.
public class MeasureFAI : FaceMeasureBase
{
    public override string Label => "FAI";

    public MeasureFAI()
    {
        Items.Add(new MeasureItemInfo("fai"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render2 = IsEnabled("fai");
        float d1 = ctx.Measure(ctx.Landmarks[64], ctx.Landmarks[76], render: render && render2, dir: "l");
        float d2 = ctx.Measure(ctx.Landmarks[68], ctx.Landmarks[82], render: render && render2, dir: "r");
        float fai = MathHelpers.Abs(d1 - d2);

        if (render && render2)
        {
            ctx.WriteImageLabel($"FAI={fai:F2}");
            ctx.AddHeader(Label);
            ctx.AddValue($"fai: {fai:F2}");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["fai"] = fai };
    }
}
