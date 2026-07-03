// Pure C# replacement for Python's colorsys.rgb_to_hsv (no System.Drawing dependency).
internal static class ColorUtils
{
    // r,g,b in [0,1]. Returns (h,s,v) each in [0,1], matching colorsys.rgb_to_hsv exactly.
    internal static (float h, float s, float v) RgbToHsv(float r, float g, float b)
    {
        float max = MathHelpers.Max(r, MathHelpers.Max(g, b));
        float min = MathHelpers.Min(r, MathHelpers.Min(g, b));
        float v = max;

        if (max == min) return (0f, 0f, v);

        float delta = max - min;
        float s = delta / max;

        float h;
        if (max == r)      h = (g - b) / delta;
        else if (max == g) h = 2f + (b - r) / delta;
        else               h = 4f + (r - g) / delta;

        h = (h / 6f) % 1f;
        if (h < 0f) h += 1f;

        return (h, s, v);
    }
}
