// Pure C# replacement for UnityEngine.Color32.
// Used throughout facial_dll so the assembly has no UnityEngine dependency.
public struct Rgba32
{
    public byte R, G, B, A;

    public Rgba32(byte r, byte g, byte b, byte a) { R = r; G = g; B = b; A = a; }

    public static bool operator ==(Rgba32 a, Rgba32 b) =>
        a.R == b.R && a.G == b.G && a.B == b.B && a.A == b.A;
    public static bool operator !=(Rgba32 a, Rgba32 b) => !(a == b);
    public override bool Equals(object obj) => obj is Rgba32 c && this == c;
    public override int GetHashCode() => (R, G, B, A).GetHashCode();
    public override string ToString() => $"({R},{G},{B},{A})";
}
