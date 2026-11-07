using System;

/// <summary>
/// A 2D point or vector in double precision: the cheek geometry's (<see cref="Cheek"/>,
/// <see cref="CheekOutline"/>) counterpart of a float64 NumPy (x, y) pair, so its
/// numbers can match dynaface-lib's to the bit. <see cref="Vec2"/> (float) widens to it
/// exactly. Each operator is the elementwise operation NumPy does, one rounding per
/// component, so a transcribed formula keeps NumPy's results when it keeps its order.
/// Equality treats NaN as equal to NaN (like <c>double.Equals</c> and NumPy's
/// <c>assert_array_equal</c>), so a missing point (<see cref="NaN"/>) equals another.
/// </summary>
public struct Vec2d : IEquatable<Vec2d>
{
    public double X, Y;

    public Vec2d(double x, double y) { X = x; Y = y; }

    public static Vec2d operator +(Vec2d a, Vec2d b) => new Vec2d(a.X + b.X, a.Y + b.Y);
    public static Vec2d operator -(Vec2d a, Vec2d b) => new Vec2d(a.X - b.X, a.Y - b.Y);
    public static Vec2d operator -(Vec2d v) => new Vec2d(-v.X, -v.Y);
    public static Vec2d operator *(Vec2d v, double s) => new Vec2d(v.X * s, v.Y * s);
    public static Vec2d operator *(double s, Vec2d v) => new Vec2d(s * v.X, s * v.Y);
    public static Vec2d operator /(Vec2d v, double s) => new Vec2d(v.X / s, v.Y / s);

    /// <summary>A Vec2's exact double value.</summary>
    public static implicit operator Vec2d(Vec2 v) => new Vec2d(v.X, v.Y);

    /// <summary>a · b as NumPy's <c>a @ b</c> computes it for two pairs: a.X * b.X + a.Y * b.Y.</summary>
    public static double Dot(Vec2d a, Vec2d b) => a.X * b.X + a.Y * b.Y;

    /// <summary>The z component of a × b: a.X * b.Y - a.Y * b.X.</summary>
    public static double Cross(Vec2d a, Vec2d b) => a.X * b.Y - a.Y * b.X;

    /// <summary>The point that isn't there: (NaN, NaN).</summary>
    public static Vec2d NaN => new Vec2d(double.NaN, double.NaN);

    /// <summary>Whether both coordinates are finite (neither NaN nor infinite).</summary>
    public bool IsFinite => !double.IsNaN(X) && !double.IsInfinity(X) && !double.IsNaN(Y) && !double.IsInfinity(Y);

    /// <summary>The nearest float point, for drawing and the float APIs.</summary>
    public Vec2 ToVec2() => new Vec2((float)X, (float)Y);

    public bool Equals(Vec2d other) => X.Equals(other.X) && Y.Equals(other.Y);
    public override bool Equals(object obj) => obj is Vec2d v && Equals(v);
    public override int GetHashCode() => (X, Y).GetHashCode();
    public override string ToString() => $"({X:R},{Y:R})";
}
