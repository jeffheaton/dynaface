using System.Collections.Generic;

// Nose Frontal: four horizontal measurements at increasing heights across the nose.
// WFLW nose region: 51-59.
//   52 = dorsal bridge,  53 = dorsal base,  54 = nasal tip apex
//   55 = right ala,      56 = right nostril, 57 = columella
//   58 = left nostril,   59 = left ala
public class MeasureNoseFrontal : FaceMeasureBase
{
    public override string Label => "NOSE FRONTAL";

    public MeasureNoseFrontal()
    {
        Items.Add(new MeasureItemInfo("nostril"));
        Items.Add(new MeasureItemInfo("nose.tip"));
        Items.Add(new MeasureItemInfo("dorsal.base"));
        Items.Add(new MeasureItemInfo("dorsal.bridge"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render1 = IsEnabled("nostril");
        bool render2 = IsEnabled("nose.tip");
        bool render3 = IsEnabled("dorsal.base");
        bool render4 = IsEnabled("dorsal.bridge");

        var lm = ctx.Landmarks;
        float x1 = lm[55].X, x2 = lm[59].X;
        float w  = 0.1f * MathHelpers.Abs(x2 - x1);

        float nostril = ctx.Measure(
            new Vec2(x1 - w, lm[55].Y), new Vec2(x2 + w, lm[59].Y),
            render: render && render1, dir: "s");

        float noseTip = ctx.Measure(
            new Vec2(lm[56].X, lm[54].Y), new Vec2(lm[58].X, lm[54].Y),
            render: render && render2, dir: "s");

        float dorsalBase = ctx.Measure(
            new Vec2(x1, lm[53].Y), new Vec2(x2, lm[53].Y),
            render: render && render3, dir: "s");

        float dorsalBridge = ctx.Measure(
            new Vec2(lm[56].X, lm[52].Y), new Vec2(lm[58].X, lm[52].Y),
            render: render && render4, dir: "s");

        if (render && (render1 || render2 || render3 || render4))
        {
            ctx.AddHeader(Label);
            if (render1) ctx.AddValue($"nostril:     {nostril:F1} mm");
            if (render2) ctx.AddValue($"tip:         {noseTip:F1} mm");
            if (render3) ctx.AddValue($"dorsal.base: {dorsalBase:F1} mm");
            if (render4) ctx.AddValue($"bridge:      {dorsalBridge:F1} mm");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double>
        {
            ["nostril"]      = nostril,
            ["nose.tip"]     = noseTip,
            ["dorsal.base"]  = dorsalBase,
            ["dorsal.bridge"] = dorsalBridge,
        };
    }
}
