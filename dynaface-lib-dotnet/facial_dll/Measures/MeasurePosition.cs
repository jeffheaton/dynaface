using System.Collections.Generic;

// Position: face tilt, pixel-to-mm conversion, and pupillary distance.
// Also draws the vertical face midline (bisect) on the image.
//
// px2mm is NaN on a quarter (three-quarter) face (ctx.IsQuarter), where no millimeter is
// defined: its pupil scale is foreshortened by the turn, and a scale column would put
// that number back into the CSV. Its line isn't drawn there; tilt and the pixel pd are
// reported and drawn as on a frontal face. Suits frontal, quarter and lateral faces.
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
        IsQuarter = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        bool render2Tilt = IsEnabled("tilt");
        bool render2Px2mm = IsEnabled("px2mm");
        bool render2Pd = IsEnabled("pd");

        // Always computed, whether or not it's drawn (like dynaface-lib's AnalyzePosition).
        float tilt = ctx.CalcFaceRotationDeg();

        // dynaface-lib's AnalyzePosition recomputes pd/pix2mm FRESH from whatever
        // landmarks are current when this measure runs, with the face's own pupillary
        // distance, rather than reading ctx.Pix2mm. The context measures its scale the
        // same way, so the two agree. In lateral mode only pd gets recomputed; pix2mm
        // keeps the fixed lateral value.
        float pix2mm = DynafaceConstants.LateralPix2mm;
        float pd = 260f;
        bool havePupils = ctx.Landmarks != null && ctx.Landmarks.Length > DynafaceConstants.LmLeftPupil;
        if (havePupils)
        {
            var (measuredPd, measuredPix2mm) = FaceMeasureContext.CalcPd(ctx.Landmarks, ctx.PupilDistMm);
            pd = measuredPd;
            if (!ctx.IsLateral)
                pix2mm = measuredPix2mm;
            // No millimeter is defined on a turned head: the pupil scale is
            // foreshortened, so report no scale (tilt and the pixel pd stay).
            if (ctx.IsQuarter)
                pix2mm = float.NaN;
        }

        if (render && render2Tilt)
        {
            var (top, bottom) = ctx.CalcBisect();
            ctx.DrawLine(top, bottom, BisectColor, thickness: 1);
        }

        // Like dynaface-lib, a scale that isn't a number (a quarter face's) isn't drawn.
        bool showPx2mm = render2Px2mm && !float.IsNaN(pix2mm);
        if (render && (render2Tilt || render2Pd || showPx2mm))
        {
            ctx.AddHeader(Label);
            if (render2Tilt) ctx.AddValue($"tilt:   {tilt:F2}°");
            if (render2Pd) ctx.AddValue($"pd:     {pd:F0} px");
            if (showPx2mm) ctx.AddValue($"px2mm:  {pix2mm:F3}");
            ctx.AddSpacer();
        }

        return new Dictionary<string, double> { ["tilt"] = tilt, ["px2mm"] = pix2mm, ["pd"] = pd };
    }
}
