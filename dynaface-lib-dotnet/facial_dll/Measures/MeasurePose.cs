using System.Collections.Generic;

// Pose: reports SPIGA's raw headpose output (pitch/roll/yaw) for the aligned crop.
// Pitch is nodding ("yes"), yaw is turning side to side ("no"), roll is tilting the
// head toward a shoulder.
public class MeasurePose : FaceMeasureBase
{
    public override string Label => "POSE";

    public MeasurePose()
    {
        Items.Add(new MeasureItemInfo("pitch"));
        Items.Add(new MeasureItemInfo("roll"));
        Items.Add(new MeasureItemInfo("yaw"));
        IsFrontal = true;
        IsLateral = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render2Pitch = IsEnabled("pitch");
        bool render2Roll = IsEnabled("roll");
        bool render2Yaw = IsEnabled("yaw");

        float yaw = ctx.HeadPose != null && ctx.HeadPose.Length > 0 ? ctx.HeadPose[0] : 0f;
        float pitch = ctx.HeadPose != null && ctx.HeadPose.Length > 1 ? ctx.HeadPose[1] : 0f;
        float roll = ctx.HeadPose != null && ctx.HeadPose.Length > 2 ? ctx.HeadPose[2] : 0f;

        if (render && (render2Pitch || render2Roll || render2Yaw))
        {
            ctx.AddHeader(Label);
            if (render2Pitch) ctx.AddValue($"pitch: {pitch:F2}°");
            if (render2Roll) ctx.AddValue($"roll:  {roll:F2}°");
            if (render2Yaw) ctx.AddValue($"yaw:   {yaw:F2}°");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["pitch"] = pitch, ["roll"] = roll, ["yaw"] = yaw };
    }
}
