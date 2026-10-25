using System.Collections.Generic;

// Nose Frontal: two widths, in millimeters, between landmarks on the nose's lower outline.
// Mirrors dynaface-lib's AnalyzeNoseFrontal.
//   nose.alar:    alar width, between the outer edges of the nostrils (WFLW 55 and 59).
//   nose.nostril: between the points midway along each nostril (WFLW 56 and 58).
// These replace four earlier columns that held only these two widths: nostril was the
// alar span padded by 20%, dorsal.base the alar span, and nose.tip and dorsal.bridge both
// the 56-58 span. WFLW has no landmarks on the sides of the nose higher up, so widths
// along the dorsum can't be measured from landmarks.
public class MeasureNoseFrontal : FaceMeasureBase
{
    public override string Label => "NOSE FRONTAL";

    public MeasureNoseFrontal()
    {
        Items.Add(new MeasureItemInfo("nose.alar"));
        Items.Add(new MeasureItemInfo("nose.nostril"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool renderAlar = IsEnabled("nose.alar");
        bool renderNostril = IsEnabled("nose.nostril");

        var lm = ctx.Landmarks;
        float alar = ctx.Measure(lm[55], lm[59], render: render && renderAlar, dir: "s");
        float nostril = ctx.Measure(lm[56], lm[58], render: render && renderNostril, dir: "s");

        if (render && (renderAlar || renderNostril))
        {
            ctx.AddHeader(Label);
            if (renderAlar) ctx.AddValue($"alar:    {alar:F1} mm");
            if (renderNostril) ctx.AddValue($"nostril: {nostril:F1} mm");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double>
        {
            ["nose.alar"] = alar,
            ["nose.nostril"] = nostril,
        };
    }
}
