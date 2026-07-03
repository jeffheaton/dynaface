using System.Collections.Generic;

// Lateral-view-only measurements computed from LateralAnalyzer's 6 landmarks:
//   NFA (Nasofrontal Angle) at Nasion, between Glabella and Nasal Tip.
//   NLA (Nasolabial Angle) at Subnasal, between Nasal Tip and Pogonion.
//   Tip projection (AT, NT, AT/NT) — a right triangle from Nasal Tip and Nasion.
//
// AT/NT/AT_NT-ratio now have their own items and are returned (dynaface-lib used
// to declare only a "tip_proj" item that matched none of its computed dict keys,
// so these were drawn but never actually returned — fixed upstream; "tip_proj"
// is kept as the shared render-gate for the tip-projection drawing). Note AT/NT
// are in raw pixels, not mm — dynaface-lib never multiplies them by pix2mm.
public class MeasureLateral : FaceMeasureBase
{
    public override string Label => "LATERAL";

    static readonly Rgba32 ArrowColor = new Rgba32(0, 200, 255, 255);

    public MeasureLateral()
    {
        Items.Add(new MeasureItemInfo("nfa"));
        Items.Add(new MeasureItemInfo("nla"));
        Items.Add(new MeasureItemInfo("tip_proj"));
        Items.Add(new MeasureItemInfo("at"));
        Items.Add(new MeasureItemInfo("nt"));
        Items.Add(new MeasureItemInfo("at_nt"));
        IsFrontal = false;
        IsLateral = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool renderNfa = IsEnabled("nfa");
        bool renderNla = IsEnabled("nla");

        if (!ctx.IsLateral || ctx.LateralLandmarks == null)
            return new Dictionary<string, double>();

        var lm = ctx.LateralLandmarks;
        Vec2 glabella    = lm[LateralLandmarkFinder.Glabella];
        Vec2 nasion      = lm[LateralLandmarkFinder.Nasion];
        Vec2 nasalTip    = lm[LateralLandmarkFinder.NasalTip];
        Vec2 subnasal    = lm[LateralLandmarkFinder.Subnasal];
        Vec2 pogonion    = lm[LateralLandmarkFinder.Pogonion];

        float nfa = AngleAt(nasion, glabella, nasalTip);
        float nla = AngleAt(subnasal, nasalTip, pogonion);

        if (render && renderNfa)
        {
            ctx.DrawArrow(nasion, glabella, ArrowColor, thickness: 2, arrowAtStart: false, arrowAtEnd: true);
            ctx.DrawArrow(nasion, nasalTip, ArrowColor, thickness: 2, arrowAtStart: false, arrowAtEnd: true);
            ctx.DrawImageText(new Vec2(nasion.X + 10, nasion.Y), $"NFA={nfa:F2}", new Rgba32(255, 255, 255, 255));
        }
        if (render && renderNla)
        {
            ctx.DrawArrow(subnasal, nasalTip, ArrowColor, thickness: 2, arrowAtStart: false, arrowAtEnd: true);
            ctx.DrawArrow(subnasal, pogonion, ArrowColor, thickness: 2, arrowAtStart: false, arrowAtEnd: true);
            ctx.DrawImageText(new Vec2(subnasal.X + 10, subnasal.Y), $"NLA={nla:F2}", new Rgba32(255, 255, 255, 255));
        }

        Vec2 ptA = nasalTip, ptN = nasion;
        var ptT = new Vec2(ptN.X, ptA.Y);
        float at = Vec2.Distance(ptA, ptT);
        float nt = Vec2.Distance(ptN, ptT);
        float atNtRatio = nt != 0f ? at / nt : float.NaN;

        if (render && IsEnabled("tip_proj"))
        {
            ctx.DrawArrow(ptA, ptT, ArrowColor, thickness: 2, arrowAtStart: false, arrowAtEnd: true);
            ctx.DrawArrow(ptN, ptT, ArrowColor, thickness: 2, arrowAtStart: false, arrowAtEnd: true);
            ctx.DrawImageText(ptA, "A", new Rgba32(255, 255, 255, 255));
            ctx.DrawImageText(ptN, "N", new Rgba32(255, 255, 255, 255));
            ctx.DrawImageText(ptT, "T", new Rgba32(255, 255, 255, 255));
            ctx.DrawImageText(new Vec2((ptA.X + ptT.X) / 2 + 5, ptA.Y + 25), $"AT={at:F2}", new Rgba32(255, 255, 255, 255));
            ctx.DrawImageText(new Vec2((ptN.X + ptT.X) / 2 + 5, (ptN.Y + ptT.Y) / 2), $"NT={nt:F2}", new Rgba32(255, 255, 255, 255));
            ctx.WriteImageLabel($"AT/NT={atNtRatio:F2}");
        }

        if (render && (renderNfa || renderNla))
        {
            ctx.AddHeader(Label);
            if (renderNfa) ctx.AddValue($"nfa: {nfa:F2}°");
            if (renderNla) ctx.AddValue($"nla: {nla:F2}°");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double>
        {
            ["nfa"]   = nfa,
            ["nla"]   = nla,
            ["at"]    = at,
            ["nt"]    = nt,
            ["at_nt"] = atNtRatio,
        };
    }

    static float AngleAt(Vec2 center, Vec2 p1, Vec2 p2)
    {
        Vec2 v1 = p1 - center, v2 = p2 - center;
        float mag1 = v1.Magnitude, mag2 = v2.Magnitude;
        if (mag1 <= 0f || mag2 <= 0f) return 0f;
        float cosAngle = MathHelpers.Clamp(Vec2.Dot(v1, v2) / (mag1 * mag2), -1f, 1f);
        return MathHelpers.Acos(cosAngle) * MathHelpers.Rad2Deg;
    }
}
