using System.Collections.Generic;

// Position: face tilt, pixel-to-mm conversion, and pupillary distance.
// Also draws the vertical face midline (bisect) on the image.
public class MeasurePosition : FaceMeasureBase
{
    public override string Label => "POSITION";

    static readonly Rgba32 BisectColor = new Rgba32(200, 200, 200, 160);

    public MeasurePosition()
    {
        Items.Add(new MeasureItemInfo("tilt"));
        Items.Add(new MeasureItemInfo("px2mm"));
        Items.Add(new MeasureItemInfo("pd"));
        IsFrontal = true;
        IsLateral = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render2Tilt  = IsEnabled("tilt");
        bool render2Px2mm = IsEnabled("px2mm");
        bool render2Pd    = IsEnabled("pd");

        // Matches dynaface-lib: tilt is only actually computed when both render and
        // its own item are enabled (otherwise it stays 0) — a quirk of the Python
        // original, replicated here rather than silently "fixed".
        float tilt = (render && render2Tilt) ? ctx.CalcFaceRotationDeg() : 0f;

        // dynaface-lib's AnalyzePosition recomputes pd/pix2mm FRESH from whatever
        // landmarks are current when this measure runs (i.e. the post-crop ones) —
        // a deliberately separate calculation from ctx.Pix2mm (which reflects the
        // pre-crop pupil distance used for every OTHER measure's mm conversions;
        // see FacePipeline.RunFrontal). In lateral mode only pd gets recomputed;
        // pix2mm keeps the fixed lateral value.
        float pix2mm = DynafaceConstants.LateralPix2mm;
        float pd = 260f;
        bool havePupils = ctx.Landmarks != null && ctx.Landmarks.Length > DynafaceConstants.LmLeftPupil;
        if (havePupils)
        {
            pd = Vec2.Distance(
                ctx.Landmarks[DynafaceConstants.LmRightPupil], ctx.Landmarks[DynafaceConstants.LmLeftPupil]);
            if (!ctx.IsLateral)
                pix2mm = DynafaceConfig.PupilDistMm / pd;
        }

        if (render && render2Tilt)
        {
            var (top, bottom) = ctx.CalcBisect();
            ctx.DrawLine(top, bottom, BisectColor, thickness: 1);
        }

        if (render && (render2Tilt || render2Pd || render2Px2mm))
        {
            ctx.AddHeader(Label);
            if (render2Tilt)  ctx.AddValue($"tilt:   {tilt:F2}°");
            if (render2Pd)    ctx.AddValue($"pd:     {pd:F0} px");
            if (render2Px2mm) ctx.AddValue($"px2mm:  {pix2mm:F3}");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["tilt"] = tilt, ["px2mm"] = pix2mm, ["pd"] = pd };
    }
}
