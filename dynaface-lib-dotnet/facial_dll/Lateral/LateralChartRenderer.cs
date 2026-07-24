using System;

// Draws the sagittal-profile chart directly onto the final lateral crop image,
// replacing dynaface-lib's matplotlib debug chart (lateral.py's analyze_lateral +
// AnalyzeFace._overlay_lateral_analysis) with a pure pixel-buffer equivalent built
// on FaceRenderer's existing rasterizer (no drawing-library dependency here, same
// as the rest of facial_dll).
//
// Python builds this as a *separate* matplotlib figure (aspect='equal', y inverted,
// xlim=[-25,512]) then rescales it to height 1024 and pastes it flush-right over
// render_img. Here sagittal_y is already in the same 0..1024 pixel space as the
// crop's own height (SagittalProfile extracts it row-by-row from that same crop),
// so no figure/rescale step is needed — a 1:1 pixel line draw lands in the same
// place a height-1024-normalized matplotlib render would.
public static class LateralChartRenderer
{
    static readonly string[] LandmarkNames =
    {
        "Soft Tissue Glabella",
        "Soft Tissue Nasion",
        "Nasal Tip",
        "Subnasal Point",
        "Mento Labial Point",
        "Soft Tissue Pogonion",
    };

    static readonly Rgba32 White = new Rgba32(255, 255, 255, 255);
    static readonly Rgba32 Black = new Rgba32(0, 0, 0, 255);
    static readonly Rgba32 LandmarkDot = new Rgba32(20, 160, 20, 255);

    const int LineThickness = 3;
    const int DotRadius = 6;
    const int LabelTextScale = 2;
    const int LabelGap = 12;
    const int PanelLeftPad = 25; // matches Python's xlim(-25, ...) left margin
    const int PanelRightPad = 220; // room for the widest landmark label text
    const int PanelMaxDataWidth = 512; // matches Python's xlim(..., 512) right clip

    // Draws the sagittal profile line and the 6 labeled lateral landmarks onto
    // `image` (mutated in place), right-aligned exactly as Python's
    // _overlay_lateral_analysis pastes its chart flush against the right edge.
    public static void Overlay(FaceImage image, LateralAnalyzer.Result result)
    {
        if (!image.IsValid || result.SagittalX == null || result.SagittalX.Length == 0) return;

        int n = result.SagittalX.Length;
        float minX = result.SagittalX[0];
        for (int i = 1; i < n; i++)
            if (result.SagittalX[i] < minX) minX = result.SagittalX[i];

        float maxShiftedX = 0f;
        for (int i = 0; i < n; i++)
        {
            float shifted = result.SagittalX[i] - minX;
            if (shifted > maxShiftedX) maxShiftedX = shifted;
        }

        // Deep profiles (neck/shoulder rows) would otherwise grow the panel far
        // past what Python's fixed xlim window ever shows, eating into the face.
        int panelWidth = PanelLeftPad + MathHelpers.RoundToInt(maxShiftedX) + PanelRightPad;
        panelWidth = MathHelpers.Min(panelWidth, PanelLeftPad + PanelMaxDataWidth);
        panelWidth = MathHelpers.Clamp(panelWidth, 1, image.Width);
        int xOffset = image.Width - panelWidth;

        FaceRenderer.DrawRect(
            image.Pixels, image.Width, image.Height,
            left: xOffset, bottom: 0, right: image.Width - 1, top: image.Height - 1,
            color: White, filled: true, alpha: 1f);

        int prevX = int.MinValue, prevY = int.MinValue;
        for (int i = 0; i < n; i++)
        {
            int x = xOffset + PanelLeftPad + MathHelpers.RoundToInt(result.SagittalX[i] - minX);
            int yTop = MathHelpers.RoundToInt(result.SagittalY[i]);
            int y = ToBottomLeft(yTop, image.Height);

            if (prevX != int.MinValue)
                FaceRenderer.DrawLine(image.Pixels, image.Width, image.Height, prevX, prevY, x, y, Black, LineThickness);
            prevX = x;
            prevY = y;
        }

        for (int i = 0; i < result.LateralLandmarks.Length && i < LandmarkNames.Length; i++)
        {
            Vec2 lm = result.LateralLandmarks[i];
            if (lm.X < 0 || lm.Y < 0) continue;

            int x = xOffset + PanelLeftPad + MathHelpers.RoundToInt(lm.X - minX);
            int y = ToBottomLeft(MathHelpers.RoundToInt(lm.Y), image.Height);

            FaceRenderer.DrawCircle(image.Pixels, image.Width, image.Height, x, y, LandmarkDot, DotRadius);
            FaceRenderer.DrawText(
                image.Pixels, image.Width, image.Height,
                x + DotRadius + LabelGap, y - FaceRenderer.GlyphHeight * LabelTextScale / 2,
                LandmarkNames[i], Black, LabelTextScale, outlined: false);
        }

        DrawLegend(image, xOffset);
    }

    // Python's chart carries a framed upper-left legend whose only labeled artist
    // is the profile line itself (ax2.legend(frameon=True, loc="upper left")).
    static void DrawLegend(FaceImage image, int panelXOffset)
    {
        const string entry = "Sagittal Profile";
        const int margin = 10; // inset from the panel's top-left corner
        const int pad = 8;  // inner box padding
        const int swatchW = 24; // line swatch width
        const int gap = 8;  // swatch-to-text gap

        var textSize = FaceRenderer.GetTextSize(entry, LabelTextScale);
        int boxW = pad + swatchW + gap + textSize.X + pad;
        int boxH = pad + textSize.Y + pad;

        int left = panelXOffset + margin;
        int top = image.Height - 1 - margin;         // bottom-left coords
        int bottom = top - boxH;
        int right = left + boxW;

        FaceRenderer.DrawRect(image.Pixels, image.Width, image.Height,
            left, bottom, right, top, White, filled: true, alpha: 0.8f);
        FaceRenderer.DrawRect(image.Pixels, image.Width, image.Height,
            left, bottom, right, top, Black, filled: false, alpha: 1f);

        int midY = bottom + boxH / 2;
        FaceRenderer.DrawLine(image.Pixels, image.Width, image.Height,
            left + pad, midY, left + pad + swatchW, midY, Black, LineThickness);
        FaceRenderer.DrawText(image.Pixels, image.Width, image.Height,
            left + pad + swatchW + gap, midY - textSize.Y / 2,
            entry, Black, LabelTextScale, outlined: false);
    }

    static int ToBottomLeft(int topLeftY, int height) => MathHelpers.Clamp(height - 1 - topLeftY, 0, height - 1);
}
