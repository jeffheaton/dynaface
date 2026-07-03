// Pure C# replacement for UnityEngine.Vector2.
// Used throughout facial_dll so the assembly has no UnityEngine dependency.
public struct Vec2
{
    public float X, Y;

    public Vec2(float x, float y) { X = x; Y = y; }

    public static Vec2 operator +(Vec2 a, Vec2 b) => new Vec2(a.X + b.X, a.Y + b.Y);
    public static Vec2 operator -(Vec2 a, Vec2 b) => new Vec2(a.X - b.X, a.Y - b.Y);
    public static Vec2 operator -(Vec2 v)          => new Vec2(-v.X, -v.Y);
    public static Vec2 operator *(Vec2 v, float s) => new Vec2(v.X * s, v.Y * s);
    public static Vec2 operator *(float s, Vec2 v) => new Vec2(v.X * s, v.Y * s);
    public static Vec2 operator /(Vec2 v, float s) => new Vec2(v.X / s, v.Y / s);

    public float SqrMagnitude => X * X + Y * Y;
    public float Magnitude    => System.MathF.Sqrt(X * X + Y * Y);

    public Vec2 Normalized()
    {
        float m = Magnitude;
        return m > 0f ? this / m : Zero;
    }

    public static float Distance(Vec2 a, Vec2 b) => (b - a).Magnitude;

    public static float Dot(Vec2 a, Vec2 b) => a.X * b.X + a.Y * b.Y;

    public static Vec2 Lerp(Vec2 a, Vec2 b, float t) =>
        new Vec2(a.X + (b.X - a.X) * t, a.Y + (b.Y - a.Y) * t);

    public static readonly Vec2 Zero  = new Vec2(0f,  0f);
    public static readonly Vec2 One   = new Vec2(1f,  1f);
    public static readonly Vec2 Up    = new Vec2(0f,  1f);
    public static readonly Vec2 Down  = new Vec2(0f, -1f);
    public static readonly Vec2 Right = new Vec2(1f,  0f);
    public static readonly Vec2 Left  = new Vec2(-1f, 0f);

    public override string ToString() => $"({X:F3},{Y:F3})";
}
