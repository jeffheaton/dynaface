using System.Collections.Generic;

// Dumps all facial landmark (x,y) coordinates as named fields, and draws a plain
// black-outline/white-fill dot at every landmark (distinct from FaceRenderer's
// DrawLandmarksOnto, which numbers + region-colors dots for pipeline debugging).
//
// NUM_LANDMARKS=98 matches dynaface-lib's AnalyzeLandmarks (fixed upstream — it
// used to be 97, silently dropping WFLW index 97/the left pupil while keeping
// index 96/the right pupil; that was an off-by-one, not intentional).
//
// Output goes to ctx.Values (structured numeric), not the TextLines sidebar —
// 196 flat coordinate fields would make for an unusable console sidebar, and
// dynaface-lib's own calc() -> Dict[str,Any] contract is a flat data dict, not
// display text.
public class MeasureLandmarks : FaceMeasureBase
{
    public const int NumLandmarksDumped = 98;

    public override string Label => "LANDMARKS";

    public MeasureLandmarks()
    {
        for (int i = 1; i <= NumLandmarksDumped; i++)
        {
            Items.Add(new MeasureItemInfo($"landmark-{i}-x"));
            Items.Add(new MeasureItemInfo($"landmark-{i}-y"));
        }
        IsFrontal = true;
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
        return data;
    }
}
