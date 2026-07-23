using System.Collections.Generic;

// Skin Tone: samples 2 cheek rectangles + draws a corner swatch showing the
// average (mean, not mode — see below) sampled color as HUE/SAT/BRT text.
// WFLW: landmarks 2,3,30,55,59 define the cheek sample rectangles.
//
// dynaface-lib's calc() uses mean averaging (a since-removed, never-read
// `USE_MODE = True` class attribute used to suggest mode-averaging, but calc()
// always read the module-level `USE_MODE = False` instead — that dead attribute
// was deleted upstream; no scipy.stats.mode equivalent is needed here).
//
// Items/return keys are hue/sat/brightness (dynaface-lib used to declare a
// single "hsv" item that didn't match any of its computed dict keys, so the
// computed values were drawn but never actually returned — fixed upstream).
public class MeasureSkinTone : FaceMeasureBase
{
    public override string Label => "SKIN TONE";

    static readonly int[] RequiredLandmarks = { 2, 3, 30, 55, 59 };

    public MeasureSkinTone()
    {
        Items.Add(new MeasureItemInfo("hue"));
        Items.Add(new MeasureItemInfo("sat"));
        Items.Add(new MeasureItemInfo("brightness"));
        IsFrontal = true;
        SyncItems();
    }

    public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
    {
        var results = new Dictionary<string, double>();
        bool anyEnabled = IsEnabled("hue") || IsEnabled("sat") || IsEnabled("brightness");
        if (!(render && anyEnabled)) return results;

        var lm = ctx.Landmarks;
        foreach (int idx in RequiredLandmarks)
            if (idx >= lm.Length) return results;

        float squareArea = ctx.Width * ctx.Height * 0.025f;
        int squareSize = (int)MathHelpers.Sqrt(squareArea);
        int offsetX = (int)(ctx.Width * 0.02f);
        int offsetY = (int)(ctx.Height * 0.02f);

        var swatchTopLeft = new Vec2(ctx.Width - offsetX - squareSize, offsetY);
        var swatchBottomRight = new Vec2(ctx.Width - offsetX, offsetY + squareSize);

        var cheek1TopLeft = new Vec2(lm[3].X, lm[2].Y);
        var cheek1BottomRight = new Vec2(lm[55].X, lm[55].Y);
        var cheek2TopLeft = new Vec2(lm[30].X, lm[59].Y);
        var cheek2BottomRight = new Vec2(lm[59].X, lm[30].Y);

        Rgba32[] sample1 = ctx.SampleRectangleTopLeft(cheek1TopLeft, cheek1BottomRight);
        Rgba32[] sample2 = ctx.SampleRectangleTopLeft(cheek2TopLeft, cheek2BottomRight);
        if (sample1.Length == 0 || sample2.Length == 0) return results;

        int n = sample1.Length + sample2.Length;
        long sumR = 0, sumG = 0, sumB = 0;
        foreach (var p in sample1) { sumR += p.R; sumG += p.G; sumB += p.B; }
        foreach (var p in sample2) { sumR += p.R; sumG += p.G; sumB += p.B; }

        byte cr = (byte)(sumR / n), cg = (byte)(sumG / n), cb = (byte)(sumB / n);
        var colorRgb = new Rgba32(cr, cg, cb, 255);

        DrawRectTopLeft(ctx, swatchTopLeft, swatchBottomRight, colorRgb, filled: true);
        DrawRectTopLeft(ctx, swatchTopLeft, swatchBottomRight, new Rgba32(0, 0, 0, 255), filled: false);

        var (h, s, v) = ColorUtils.RgbToHsv(cr / 255f, cg / 255f, cb / 255f);
        int hueDeg = (int)(h * 360f);
        int satPct = (int)(s * 100f);
        int briPct = (int)(v * 100f);

        results["hue"] = hueDeg;
        results["sat"] = satPct;
        results["brightness"] = briPct;

        string[] lines = { $"HUE: {hueDeg}", $"SAT: {satPct}", $"BRT: {briPct}" };
        var sizes = new Vec2i[lines.Length];
        for (int i = 0; i < lines.Length; i++)
            sizes[i] = FaceRenderer.GetTextSize(lines[i], FaceRenderer.TEXT_SIZE_MEASURE);

        int totalTextHeight = 5 * (lines.Length - 1);
        foreach (var sz in sizes) totalTextHeight += sz.Y;

        int currentY = (int)swatchTopLeft.Y + (squareSize - totalTextHeight) / 2;
        for (int i = 0; i < lines.Length; i++)
        {
            int textX = (int)swatchTopLeft.X + (squareSize - sizes[i].X) / 2;
            ctx.DrawImageText(new Vec2(textX, currentY), lines[i], new Rgba32(255, 255, 255, 255));
            currentY += sizes[i].Y + 5;
        }

        return results;
    }

    static void DrawRectTopLeft(FaceMeasureContext ctx, Vec2 topLeft, Vec2 bottomRight, Rgba32 color, bool filled)
    {
        int left = MathHelpers.RoundToInt(topLeft.X);
        int right = MathHelpers.RoundToInt(bottomRight.X);
        int bottomBL = ctx.Height - 1 - MathHelpers.RoundToInt(bottomRight.Y);
        int topBL = ctx.Height - 1 - MathHelpers.RoundToInt(topLeft.Y);
        FaceRenderer.DrawRect(ctx.Pixels, ctx.Width, ctx.Height, left, bottomBL, right, topBL, color, filled: filled);
    }
}
