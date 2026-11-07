using System.Collections.Generic;

// Dumps all facial landmark (x,y) coordinates as named fields, and draws a plain
// black-outline/white-fill dot at every landmark (distinct from FaceRenderer's
// DrawLandmarksOnto, which numbers + region-colors dots for pipeline debugging).
//
// NUM_LANDMARKS=98 matches dynaface-lib's AnalyzeLandmarks (fixed upstream — it
// used to be 97, silently dropping WFLW index 97/the left pupil while keeping
// index 96/the right pupil; that was an off-by-one, not intentional).
//
// The last item, "Cheek Landmarks" (CheekItem), is a group of 20 outputs
// (Cheek.CheekLandmarkOutputs: zy.r.x, zy.r.y, ... sub.l.y): ctx.CheekLandmarks, the six
// frontal constructions on a frontal face and the far cheek's malar apex and submalar point
// on a quarter face (the submalar point only where its depth is known), NaN where a
// landmark doesn't exist in the view. While the item is enabled, every cheek landmark that
// exists in the view is drawn the way the 98 are, a black dot under one in orange
// (CheekColor), without labels (ctx.DrawCheekLandmarks), except a quarter face's submalar
// point where the outline shows no hollow, which is reported but not drawn, as
// MeasureCheekMeasures leaves it undrawn. Mirrors dynaface-lib's AnalyzeLandmarks.
//
// Output goes to ctx.Values (structured numeric), not the TextLines sidebar —
// 216 flat coordinate fields would make for an unusable console sidebar, and
// dynaface-lib's own calc() -> Dict[str,Any] contract is a flat data dict, not
// display text.
//
// Suits frontal and quarter faces (a landmark's pixel position needs no millimeter), and
// so does the Cheek Landmarks item.
public class MeasureLandmarks : FaceMeasureBase
{
    public const int NumLandmarksDumped = 98;

    // The item that stands for the cheek landmarks' coordinates.
    public const string CheekItem = "Cheek Landmarks";

    // The cheek landmarks' dot color (dynaface-lib's CHEEK_COLOR).
    public static readonly Rgba32 CheekColor = new Rgba32(255, 128, 0, 255);

    public override string Label => "LANDMARKS";

    public MeasureLandmarks()
    {
        for (int i = 1; i <= NumLandmarksDumped; i++)
        {
            Items.Add(new MeasureItemInfo($"landmark-{i}-x"));
            Items.Add(new MeasureItemInfo($"landmark-{i}-y"));
        }
        Items.Add(new MeasureItemInfo(CheekItem, outputs: Cheek.CheekLandmarkOutputs));
        IsFrontal = true;
        IsQuarter = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        if (render)
        {
            for (int i = 0; i < ctx.Landmarks.Length; i++)
                ctx.DrawCircleAt(ctx.Landmarks[i], new Rgba32(0, 0, 0, 255), radius: 3);
            for (int i = 0; i < ctx.Landmarks.Length; i++)
                ctx.DrawCircleAt(ctx.Landmarks[i], new Rgba32(255, 255, 255, 255), radius: 2);
            if (IsEnabled(CheekItem))
            {
                // dynaface-lib's draw_cheek_landmarks at size 1.0 (radius 3), then 0.5 (radius 2).
                ctx.DrawCheekLandmarks(size: 1.0f, color: new Rgba32(0, 0, 0, 255), names: false);
                ctx.DrawCheekLandmarks(size: 0.5f, color: CheekColor, names: false);
            }
        }

        var data = new Dictionary<string, double>();
        int count = MathHelpers.Min(ctx.Landmarks.Length, NumLandmarksDumped);
        for (int i = 0; i < count; i++)
        {
            int n = i + 1;
            data[$"landmark-{n}-x"] = ctx.Landmarks[i].X;
            data[$"landmark-{n}-y"] = ctx.Landmarks[i].Y;
            ctx.AddValue($"landmark-{n}-x", ctx.Landmarks[i].X);
            ctx.AddValue($"landmark-{n}-y", ctx.Landmarks[i].Y);
        }
        Vec2d[] points = ctx.CheekLandmarks;
        for (int i = 0; i < Cheek.CheekLandmarkCodes.Count; i++)
        {
            string code = Cheek.CheekLandmarkCodes[i];
            data[$"{code}.x"] = points[i].X;
            data[$"{code}.y"] = points[i].Y;
            ctx.AddValue($"{code}.x", points[i].X);
            ctx.AddValue($"{code}.y", points[i].Y);
        }
        return data;
    }
}
