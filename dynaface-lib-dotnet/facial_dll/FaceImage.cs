using System;
using System.Runtime.InteropServices;

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

    // An image from 8-bit RGBA bytes, 4 to a pixel (R, G, B, A), in rows from the bottom,
    // as FaceImage keeps them and as Unity's Texture2D.GetRawTextureData gives an RGBA32
    // texture's. Pass topDown: true for rows from the top, as most image decoders give
    // them. A Color32[] buffer works too, through MemoryMarshal.AsBytes. Copies the bytes.
    public static FaceImage FromRgba(ReadOnlySpan<byte> rgba, int width, int height, bool topDown = false)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(width <= 0 ? nameof(width) : nameof(height), "The size must be above 0.");
        if (rgba.Length != width * height * 4)
            throw new ArgumentException($"A {width}×{height} image needs {width * height * 4} bytes, not {rgba.Length}.", nameof(rgba));

        var pixels = new Rgba32[width * height];
        ReadOnlySpan<Rgba32> source = MemoryMarshal.Cast<byte, Rgba32>(rgba);
        if (topDown)
            for (int y = 0; y < height; y++)
                source.Slice((height - 1 - y) * width, width).CopyTo(pixels.AsSpan(y * width, width));
        else
            source.CopyTo(pixels);
        return new FaceImage(pixels, width, height);
    }

    // The pixels as 8-bit RGBA bytes, 4 to a pixel, in rows from the bottom, or from the
    // top with topDown: true.
    public byte[] ToRgba(bool topDown = false)
    {
        var rgba = new byte[Width * Height * 4];
        ToRgba(rgba, topDown);
        return rgba;
    }

    // Writes the pixels as RGBA bytes into destination, which must hold exactly
    // Width × Height × 4 of them, so a buffer can be used again frame after frame.
    public void ToRgba(Span<byte> destination, bool topDown = false)
    {
        RequirePixels();
        if (destination.Length != Width * Height * 4)
            throw new ArgumentException(
                $"A {Width}×{Height} image needs {Width * Height * 4} bytes, not {destination.Length}.", nameof(destination));

        Span<Rgba32> target = MemoryMarshal.Cast<byte, Rgba32>(destination);
        ReadOnlySpan<Rgba32> pixels = Pixels.AsSpan(0, Width * Height);
        if (topDown)
            for (int y = 0; y < Height; y++)
                pixels.Slice((Height - 1 - y) * Width, Width).CopyTo(target.Slice(y * Width, Width));
        else
            pixels.CopyTo(target);
    }

    // A copy turned clockwise, as the image is seen, by quarterTurns quarter turns: 1
    // turns its top edge to the right, 2 turns it upside down, and -1 (or 3) turns it
    // counterclockwise. An odd number swaps the width and the height.
    public FaceImage Rotate(int quarterTurns)
    {
        RequirePixels();
        int turns = ((quarterTurns % 4) + 4) % 4;
        int w = Width, h = Height;
        int outWidth = turns % 2 == 1 ? h : w;
        int outHeight = turns % 2 == 1 ? w : h;
        var source = Pixels;
        var rotated = new Rgba32[w * h];

        // Rows from the bottom, so a clockwise turn sends (x, y) to (y, w - 1 - x).
        for (int y = 0; y < outHeight; y++)
            for (int x = 0; x < outWidth; x++)
            {
                int sx, sy;
                switch (turns)
                {
                    case 1: sx = w - 1 - y; sy = x; break;
                    case 2: sx = w - 1 - x; sy = h - 1 - y; break;
                    case 3: sx = y; sy = h - 1 - x; break;
                    default: sx = x; sy = y; break;
                }
                rotated[y * outWidth + x] = source[sy * w + sx];
            }
        return new FaceImage(rotated, outWidth, outHeight);
    }

    // A copy mirrored left to right, as the pipeline mirrors a profile facing right.
    public FaceImage FlipHorizontal()
    {
        RequirePixels();
        var flipped = (Rgba32[])Pixels.Clone();
        ImageUtils.FlipHorizontal(flipped, Width, Height);
        return new FaceImage(flipped, Width, Height);
    }

    // A copy turned upside down by mirroring top to bottom.
    public FaceImage FlipVertical()
    {
        RequirePixels();
        return new FaceImage(ImageUtils.FlipVertical(Pixels, Width, Height), Width, Height);
    }

    void RequirePixels()
    {
        if (!IsValid || Pixels.Length < Width * Height)
            throw new InvalidOperationException($"The image has no pixels for its {Width}×{Height} size.");
    }
}
