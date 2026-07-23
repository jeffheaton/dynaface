// Pure C# replacement for UnityEngine.Texture2D at the pipeline boundary.
// Carries a flat row-major Rgba32 pixel array (Unity bottom-left order: y=0 at bottom)
// plus image dimensions.  The array is owned by the caller and not copied.
public struct FaceImage
{
    public Rgba32[] Pixels;
    public int Width;
    public int Height;

    public FaceImage(Rgba32[] pixels, int width, int height)
    {
        Pixels = pixels;
        Width = width;
        Height = height;
    }

    public bool IsValid => Pixels != null && Width > 0 && Height > 0;
}
