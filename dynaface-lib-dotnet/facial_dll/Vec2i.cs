// Pure C# replacement for UnityEngine.Vector2Int.
// Used in FaceRenderer polygon fill / text size APIs.
public struct Vec2i
{
    public int X, Y;

    public Vec2i(int x, int y) { X = x; Y = y; }

    public static readonly Vec2i Zero = new Vec2i(0, 0);

    public override string ToString() => $"({X},{Y})";
}
