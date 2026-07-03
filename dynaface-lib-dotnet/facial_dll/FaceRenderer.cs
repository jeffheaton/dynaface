using System;
using System.Collections.Generic;

// PIPELINE STAGE 4 of 4 — Face Rendering
//
// Receives the 1024×1024 upright pixel buffer from BlazeFaceDetector and the
// 98 WFLW landmark positions from FaceLandmarkDetector, and draws annotated
// dots and numbered labels in-place.
//
// All pixel-level drawing lives here. FaceLandmarkDetector knows nothing about
// rendering; CameraSource knows nothing about pixels.
//
// All public drawing methods use Unity bottom-left pixel coordinates
// (x=0 left, y=0 bottom) — the same order as FaceImage.Pixels.
//
// WFLW region → color mapping:
//   0-32  face contour   → green
//   33-50 eyebrows       → blue
//   51-59 nose           → magenta
//   60-75 eyes           → red
//   76-87 outer lip      → cyan
//   88-95 inner lip      → yellow
//   96-97 pupil centres  → red
public static class FaceRenderer
{
    public enum LandmarkDisplayMode { Off, Lm }
    public static LandmarkDisplayMode DisplayMode = LandmarkDisplayMode.Off;

    public const int TEXT_SIZE_MEASURE = 3;
    public const int TEXT_SIZE_STATS   = 32;
    public const int ARROW_THICKNESS   = 3;

    const int DotRadius = 4;
    const int NumberScale    = 2;
    const int NumberDistance = 15;
    const int GlyphWidth     = 5;
    public const int GlyphHeight  = 7;
    const int GlyphSpacing   = 1;

    static readonly Rgba32 ContourColor       = new Rgba32( 25, 235,  50, 255);
    static readonly Rgba32 EyebrowColor       = new Rgba32( 35,  70, 255, 255);
    static readonly Rgba32 NoseColor          = new Rgba32(245,  25, 230, 255);
    static readonly Rgba32 EyeColor           = new Rgba32(255,  35,  35, 255);
    static readonly Rgba32 OuterLipColor      = new Rgba32( 20, 225, 225, 255);
    static readonly Rgba32 InnerLipColor      = new Rgba32(235, 225,  25, 255);
    static readonly Rgba32 NumberColor        = new Rgba32(255, 255, 255, 255);
    static readonly Rgba32 NumberOutlineColor = new Rgba32( 20,  20,  20, 255);

    static readonly byte[,] DigitRows =
    {
        { 0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110 }, // 0
        { 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 }, // 1
        { 0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111 }, // 2
        { 0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110 }, // 3
        { 0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010 }, // 4
        { 0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110 }, // 5
        { 0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110 }, // 6
        { 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000 }, // 7
        { 0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110 }, // 8
        { 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110 }  // 9
    };

    static readonly Dictionary<char, byte[]> GlyphMap = BuildGlyphMap();

    static Dictionary<char, byte[]> BuildGlyphMap()
    {
        var m = new Dictionary<char, byte[]>
        {
            ['0'] = new byte[] { 0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110 },
            ['1'] = new byte[] { 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 },
            ['2'] = new byte[] { 0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111 },
            ['3'] = new byte[] { 0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110 },
            ['4'] = new byte[] { 0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010 },
            ['5'] = new byte[] { 0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110 },
            ['6'] = new byte[] { 0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110 },
            ['7'] = new byte[] { 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000 },
            ['8'] = new byte[] { 0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110 },
            ['9'] = new byte[] { 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110 },

            ['A'] = new byte[] { 0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
            ['B'] = new byte[] { 0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110 },
            ['C'] = new byte[] { 0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110 },
            ['D'] = new byte[] { 0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110 },
            ['E'] = new byte[] { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111 },
            ['F'] = new byte[] { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000 },
            ['G'] = new byte[] { 0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111 },
            ['H'] = new byte[] { 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
            ['I'] = new byte[] { 0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 },
            ['J'] = new byte[] { 0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100 },
            ['K'] = new byte[] { 0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001 },
            ['L'] = new byte[] { 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111 },
            ['M'] = new byte[] { 0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001 },
            ['N'] = new byte[] { 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001 },
            ['O'] = new byte[] { 0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 },
            ['P'] = new byte[] { 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000 },
            ['Q'] = new byte[] { 0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101 },
            ['R'] = new byte[] { 0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001 },
            ['S'] = new byte[] { 0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110 },
            ['T'] = new byte[] { 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 },
            ['U'] = new byte[] { 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 },
            ['V'] = new byte[] { 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100 },
            ['W'] = new byte[] { 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001 },
            ['X'] = new byte[] { 0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001 },
            ['Y'] = new byte[] { 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100 },
            ['Z'] = new byte[] { 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111 },

            [' '] = new byte[] { 0, 0, 0, 0, 0, 0, 0 },
            ['.'] = new byte[] { 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100 },
            [':'] = new byte[] { 0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000 },
            ['-'] = new byte[] { 0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000 },
            ['%'] = new byte[] { 0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011 },
        };
        return m;
    }

    // Draws landmarks onto an existing pixel buffer in-place according to DisplayMode.
    // landmarks must be in top-left pixel coordinates (x=0 left, y=0 top).
    public static void DrawLandmarksOnto(Rgba32[] pixels, int width, int height, Vec2[] landmarks)
    {
        if (DisplayMode == LandmarkDisplayMode.Off) return;
        if (pixels == null || landmarks == null || landmarks.Length < SpigaLandmarkDetector.NumWflwLandmarks)
            return;

        float maxPixelY = MathHelpers.Max(0f, height - 1f);
        var bottomLeft = new Vec2[SpigaLandmarkDetector.NumWflwLandmarks];
        for (int i = 0; i < SpigaLandmarkDetector.NumWflwLandmarks; i++)
            bottomLeft[i] = new Vec2(landmarks[i].X, maxPixelY - landmarks[i].Y);

        for (int i = 0; i < SpigaLandmarkDetector.NumWflwLandmarks; i++)
            DrawLandmarkNumber(pixels, width, height, bottomLeft, i);

        for (int i = 0; i < SpigaLandmarkDetector.NumWflwLandmarks; i++)
            DrawDot(pixels, width, height,
                MathHelpers.RoundToInt(bottomLeft[i].X),
                MathHelpers.RoundToInt(bottomLeft[i].Y),
                GetLandmarkColor(i));
    }

    // -------------------------------------------------------------------------
    // Public drawing API — all coordinates in bottom-left texture space
    // -------------------------------------------------------------------------

    public static void DrawLine(
        Rgba32[] pixels, int width, int height,
        int x0, int y0, int x1, int y1,
        Rgba32 color, int thickness = 1)
    {
        int r  = (thickness - 1) / 2;
        int dx = MathHelpers.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = MathHelpers.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;
        int x = x0, y = y0;
        while (true)
        {
            if (r <= 0) SetPixelSafe(pixels, width, height, x, y, color);
            else        DrawCircle(pixels, width, height, x, y, color, r);
            if (x == x1 && y == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x += sx; }
            if (e2 <  dx) { err += dx; y += sy; }
        }
    }

    public static void DrawArrowHead(
        Rgba32[] pixels, int width, int height,
        int tipX, int tipY, int tailX, int tailY,
        Rgba32 color, int par = 15, int thickness = 2)
    {
        float angle = MathHelpers.Atan2(tipY - tailY, tipX - tailX);
        float cosA  = MathHelpers.Cos(angle);
        float sinA  = MathHelpers.Sin(angle);

        int w1x = tipX + MathHelpers.RoundToInt(-par * cosA - (par / 2f) * sinA);
        int w1y = tipY + MathHelpers.RoundToInt(-par * sinA + (par / 2f) * cosA);
        int w2x = tipX + MathHelpers.RoundToInt(-par * cosA + (par / 2f) * sinA);
        int w2y = tipY - MathHelpers.RoundToInt((par / 2f) * cosA + par * sinA);

        DrawLine(pixels, width, height, tipX, tipY, w1x, w1y, color, thickness);
        DrawLine(pixels, width, height, tipX, tipY, w2x, w2y, color, thickness);
    }

    public static void DrawArrow(
        Rgba32[] pixels, int width, int height,
        int x0, int y0, int x1, int y1,
        Rgba32 color, int thickness = 2,
        bool arrowAtStart = true, bool arrowAtEnd = true)
    {
        DrawLine(pixels, width, height, x0, y0, x1, y1, color, thickness);
        if (arrowAtStart) DrawArrowHead(pixels, width, height, x0, y0, x1, y1, color, 15, thickness);
        if (arrowAtEnd)   DrawArrowHead(pixels, width, height, x1, y1, x0, y0, color, 15, thickness);
    }

    public static void DrawRect(
        Rgba32[] pixels, int width, int height,
        int left, int bottom, int right, int top,
        Rgba32 color, int thickness = 2, bool filled = false, float alpha = 1f)
    {
        if (filled)
        {
            int x0 = MathHelpers.Clamp(left,   0, width  - 1);
            int x1 = MathHelpers.Clamp(right,  0, width  - 1);
            int y0 = MathHelpers.Clamp(bottom, 0, height - 1);
            int y1 = MathHelpers.Clamp(top,    0, height - 1);
            float inv = 1f - alpha;
            for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
            {
                int idx = y * width + x;
                Rgba32 src = pixels[idx];
                pixels[idx] = new Rgba32(
                    (byte)(color.R * alpha + src.R * inv),
                    (byte)(color.G * alpha + src.G * inv),
                    (byte)(color.B * alpha + src.B * inv),
                    255);
            }
        }
        else
        {
            DrawLine(pixels, width, height, left,  bottom, right, bottom, color, thickness);
            DrawLine(pixels, width, height, right, bottom, right, top,    color, thickness);
            DrawLine(pixels, width, height, right, top,    left,  top,    color, thickness);
            DrawLine(pixels, width, height, left,  top,    left,  bottom, color, thickness);
        }
    }

    public static void DrawCircle(
        Rgba32[] pixels, int width, int height,
        int cx, int cy, Rgba32 color, int radius = 4)
    {
        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            if (dx * dx + dy * dy <= radius * radius)
                SetPixelSafe(pixels, width, height, cx + dx, cy + dy, color);
        }
    }

    public static void DrawText(
        Rgba32[] pixels, int width, int height,
        int originX, int originY,
        string text, Rgba32 color,
        int scale = 1, bool outlined = true)
    {
        if (string.IsNullOrEmpty(text)) return;
        string upper = text.ToUpperInvariant();
        if (outlined)
        {
            var shadow = new Rgba32(20, 20, 20, 255);
            for (int oy = -1; oy <= 1; oy++)
            for (int ox = -1; ox <= 1; ox++)
            {
                if (ox == 0 && oy == 0) continue;
                DrawTextRaw(pixels, width, height, originX + ox, originY + oy, upper, scale, shadow);
            }
        }
        DrawTextRaw(pixels, width, height, originX, originY, upper, scale, color);
    }

    public static Vec2i GetTextSize(string text, int scale = 1)
    {
        if (string.IsNullOrEmpty(text)) return Vec2i.Zero;
        int w = text.Length * GlyphWidth * scale + (text.Length - 1) * GlyphSpacing * scale;
        return new Vec2i(w, GlyphHeight * scale);
    }

    public static float MeasurePolygon(
        Rgba32[] pixels, int width, int height,
        Vec2i[] contour,
        bool render = true,
        Rgba32 color = default,
        float alpha = 0.4f)
    {
        if (contour == null || contour.Length < 3) return 0f;
        if (render) FillPolygon(pixels, width, height, contour, color, alpha);

        double area = 0;
        int n = contour.Length;
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            area += (double)contour[i].X * contour[j].Y;
            area -= (double)contour[j].X * contour[i].Y;
        }
        return (float)(Math.Abs(area) * 0.5);
    }

    public static Rgba32[] ExtractHorizRow(
        Rgba32[] pixels, int width, int height,
        int y, int x1 = 0, int x2 = -1)
    {
        if (x2 < 0) x2 = width;
        x1 = MathHelpers.Clamp(x1, 0, width);
        x2 = MathHelpers.Clamp(x2, 0, width);
        y  = MathHelpers.Clamp(y,  0, height - 1);
        int count = MathHelpers.Max(0, x2 - x1);
        var result = new Rgba32[count];
        Array.Copy(pixels, y * width + x1, result, 0, count);
        return result;
    }

    public static Rgba32[] SampleRectangle(
        Rgba32[] pixels, int width, int height,
        int left, int bottom, int right, int top)
    {
        left   = MathHelpers.Clamp(left,   0, width);
        right  = MathHelpers.Clamp(right,  0, width);
        bottom = MathHelpers.Clamp(bottom, 0, height);
        top    = MathHelpers.Clamp(top,    0, height);

        int w = MathHelpers.Max(0, right  - left);
        int h = MathHelpers.Max(0, top    - bottom);
        var result = new Rgba32[w * h];
        for (int row = 0; row < h; row++)
            Array.Copy(pixels, (bottom + row) * width + left, result, row * w, w);
        return result;
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    static void DrawLandmarkNumber(
        Rgba32[] pixels, int width, int height,
        Vec2[] points, int landmarkIndex)
    {
        Vec2 point        = points[landmarkIndex];
        Vec2 regionCenter = GetLabelRegionCenter(points, landmarkIndex);
        Vec2 direction    = point - regionCenter;

        if (direction.SqrMagnitude < 1f)
            direction = GetFallbackLabelDirection(landmarkIndex);
        else
            direction = direction.Normalized();

        string text      = landmarkIndex.ToString();
        int    textWidth  = GetTextWidth(text, NumberScale);
        int    textHeight = GlyphHeight * NumberScale;

        float   labelRadius = NumberDistance + MathHelpers.Max(textWidth, textHeight) * 0.5f;
        Vec2    labelCenter = point + direction * labelRadius;

        int originX = MathHelpers.Clamp(
            MathHelpers.RoundToInt(labelCenter.X - textWidth  * 0.5f),
            1, MathHelpers.Max(1, width  - textWidth  - 2));
        int originY = MathHelpers.Clamp(
            MathHelpers.RoundToInt(labelCenter.Y - textHeight * 0.5f),
            1, MathHelpers.Max(1, height - textHeight - 2));

        DrawOutlinedNumber(pixels, width, height, originX, originY, text, NumberScale);
    }

    static Vec2 GetLabelRegionCenter(Vec2[] points, int index)
    {
        if (index <= 32) return AverageRange(points,  0, 32);
        if (index <= 41) return AverageRange(points, 33, 41);
        if (index <= 50) return AverageRange(points, 42, 50);
        if (index <= 59) return AverageRange(points, 51, 59);
        if (index <= 67) return AverageRange(points, 60, 67);
        if (index <= 75) return AverageRange(points, 68, 75);
        if (index <= 95) return AverageRange(points, 76, 95);
        return index == 96
            ? AverageRange(points, 60, 67)
            : AverageRange(points, 68, 75);
    }

    static Vec2 GetFallbackLabelDirection(int index)
    {
        if (index >= 51 && index <= 54)
            return index % 2 == 0 ? Vec2.Right : Vec2.Left;
        if (index == 57) return Vec2.Down;
        if (index == 96 || index == 97)
            return new Vec2(0.75f, 0.65f).Normalized();
        return Vec2.Up;
    }

    static Vec2 AverageRange(Vec2[] points, int firstInclusive, int lastInclusive)
    {
        Vec2 sum   = Vec2.Zero;
        int  count = 0;
        for (int i = firstInclusive; i <= lastInclusive; i++) { sum += points[i]; count++; }
        return count > 0 ? sum / count : Vec2.Zero;
    }

    static Rgba32 GetLandmarkColor(int index)
    {
        if (index <= 32) return ContourColor;
        if (index <= 50) return EyebrowColor;
        if (index <= 59) return NoseColor;
        if (index <= 75) return EyeColor;
        if (index <= 87) return OuterLipColor;
        if (index <= 95) return InnerLipColor;
        return EyeColor;
    }

    static void DrawDot(Rgba32[] pixels, int width, int height, int cx, int cy, Rgba32 color)
        => DrawCircle(pixels, width, height, cx, cy, color, DotRadius);

    static int GetTextWidth(string text, int scale)
    {
        if (string.IsNullOrEmpty(text)) return 0;
        return text.Length * GlyphWidth * scale + (text.Length - 1) * GlyphSpacing * scale;
    }

    static void DrawOutlinedNumber(
        Rgba32[] pixels, int width, int height,
        int originX, int originY, string text, int scale)
    {
        for (int oy = -1; oy <= 1; oy++)
        for (int ox = -1; ox <= 1; ox++)
        {
            if (ox == 0 && oy == 0) continue;
            DrawNumberRaw(pixels, width, height, originX + ox, originY + oy, text, scale, NumberOutlineColor);
        }
        DrawNumberRaw(pixels, width, height, originX, originY, text, scale, NumberColor);
    }

    static void DrawNumberRaw(
        Rgba32[] pixels, int width, int height,
        int originX, int originY, string text, int scale, Rgba32 color)
    {
        int cursorX = originX;
        for (int ci = 0; ci < text.Length; ci++)
        {
            int digit = text[ci] - '0';
            if ((uint)digit <= 9u)
                DrawDigit(pixels, width, height, cursorX, originY, digit, scale, color);
            cursorX += (GlyphWidth + GlyphSpacing) * scale;
        }
    }

    static void DrawDigit(
        Rgba32[] pixels, int width, int height,
        int originX, int originY, int digit, int scale, Rgba32 color)
    {
        for (int row = 0; row < GlyphHeight; row++)
        {
            byte rowBits = DigitRows[digit, row];
            for (int col = 0; col < GlyphWidth; col++)
            {
                if ((rowBits & (1 << (GlyphWidth - 1 - col))) == 0) continue;
                int blockX = originX + col * scale;
                int blockY = originY + (GlyphHeight - 1 - row) * scale;
                for (int py = 0; py < scale; py++)
                for (int px = 0; px < scale; px++)
                    SetPixelSafe(pixels, width, height, blockX + px, blockY + py, color);
            }
        }
    }

    static void DrawTextRaw(
        Rgba32[] pixels, int width, int height,
        int originX, int originY, string text, int scale, Rgba32 color)
    {
        int cursorX = originX;
        foreach (char ch in text)
        {
            if (GlyphMap.TryGetValue(ch, out byte[] rows))
                DrawGlyph(pixels, width, height, cursorX, originY, rows, scale, color);
            cursorX += (GlyphWidth + GlyphSpacing) * scale;
        }
    }

    static void DrawGlyph(
        Rgba32[] pixels, int width, int height,
        int originX, int originY, byte[] rows, int scale, Rgba32 color)
    {
        for (int row = 0; row < GlyphHeight; row++)
        {
            byte rowBits = rows[row];
            for (int col = 0; col < GlyphWidth; col++)
            {
                if ((rowBits & (1 << (GlyphWidth - 1 - col))) == 0) continue;
                int blockX = originX + col * scale;
                int blockY = originY + (GlyphHeight - 1 - row) * scale;
                for (int py = 0; py < scale; py++)
                for (int px = 0; px < scale; px++)
                    SetPixelSafe(pixels, width, height, blockX + px, blockY + py, color);
            }
        }
    }

    static void FillPolygon(
        Rgba32[] pixels, int width, int height,
        Vec2i[] contour, Rgba32 color, float alpha)
    {
        int n = contour.Length;
        int yMin = int.MaxValue, yMax = int.MinValue;
        foreach (var p in contour)
        {
            if (p.Y < yMin) yMin = p.Y;
            if (p.Y > yMax) yMax = p.Y;
        }
        yMin = MathHelpers.Clamp(yMin, 0, height - 1);
        yMax = MathHelpers.Clamp(yMax, 0, height - 1);
        float inv = 1f - alpha;

        for (int y = yMin; y <= yMax; y++)
        {
            var xs = new System.Collections.Generic.List<int>();
            for (int i = 0; i < n; i++)
            {
                Vec2i a = contour[i], b = contour[(i + 1) % n];
                if ((a.Y <= y && b.Y > y) || (b.Y <= y && a.Y > y))
                {
                    float t = (float)(y - a.Y) / (b.Y - a.Y);
                    xs.Add(MathHelpers.RoundToInt(a.X + t * (b.X - a.X)));
                }
            }
            xs.Sort();
            for (int i = 0; i + 1 < xs.Count; i += 2)
            {
                int xLeft  = MathHelpers.Clamp(xs[i],     0, width - 1);
                int xRight = MathHelpers.Clamp(xs[i + 1], 0, width - 1);
                int rowBase = y * width;
                for (int x = xLeft; x <= xRight; x++)
                {
                    Rgba32 src = pixels[rowBase + x];
                    pixels[rowBase + x] = new Rgba32(
                        (byte)(color.R * alpha + src.R * inv),
                        (byte)(color.G * alpha + src.G * inv),
                        (byte)(color.B * alpha + src.B * inv),
                        255);
                }
            }
        }
    }

    static void SetPixelSafe(Rgba32[] pixels, int width, int height, int x, int y, Rgba32 color)
    {
        if ((uint)x < (uint)width && (uint)y < (uint)height)
            pixels[y * width + x] = color;
    }
}
