// Pure C# replacements for UnityEngine.Mathf.
// All methods are inlineable thin wrappers over System.MathF / System.Math.
internal static class MathHelpers
{
    public const float Epsilon = 1e-6f;
    public const float Deg2Rad = System.MathF.PI / 180f;
    public const float Rad2Deg = 180f / System.MathF.PI;

    public static int   RoundToInt(float f) => (int)System.MathF.Round(f, System.MidpointRounding.AwayFromZero);
    public static int   FloorToInt(float f) => (int)System.MathF.Floor(f);
    public static int   CeilToInt(float f)  => (int)System.MathF.Ceiling(f);

    public static float Clamp(float v, float min, float max) => v < min ? min : v > max ? max : v;
    public static int   Clamp(int v, int min, int max)       => v < min ? min : v > max ? max : v;
    public static float Clamp01(float v)                     => v < 0f  ? 0f  : v > 1f  ? 1f  : v;

    public static float Abs(float f) => System.MathF.Abs(f);
    public static int   Abs(int i)   => System.Math.Abs(i);

    public static float Max(float a, float b) => a > b ? a : b;
    public static int   Max(int a, int b)     => a > b ? a : b;
    public static float Min(float a, float b) => a < b ? a : b;
    public static int   Min(int a, int b)     => a < b ? a : b;

    public static float Lerp(float a, float b, float t) => a + (b - a) * t;

    public static float Atan2(float y, float x) => System.MathF.Atan2(y, x);
    public static float Cos(float f)             => System.MathF.Cos(f);
    public static float Sin(float f)             => System.MathF.Sin(f);
    public static float Acos(float f)            => System.MathF.Acos(System.MathF.Max(-1f, System.MathF.Min(1f, f)));
    public static float Sqrt(float f)            => System.MathF.Sqrt(f);
    public static float Exp(float f)             => System.MathF.Exp(f);
}
