namespace DynafaceTests;

// FaceImage's helpers: RGBA bytes in and out, quarter turns and flips. FaceImage keeps
// its rows from the bottom; these tests describe images as they are seen, from the top.
public class FaceImageTests
{
    static readonly Rgba32 Red = new(255, 0, 0, 255);
    static readonly Rgba32 Blue = new(0, 0, 255, 255);
    static readonly Rgba32 Black = new(0, 0, 0, 255);

    // A 3×2 image, seen as rows from the top, with each pixel numbered in its red.
    //   0 1 2
    //   3 4 5
    static FaceImage Numbered() => FaceImage.FromRgba(
        Enumerable.Range(0, 6).SelectMany(i => new byte[] { (byte)i, 10, 20, 255 }).ToArray(), 3, 2, topDown: true);

    // The pixels' numbers, as rows from the top.
    static int[][] Seen(FaceImage image)
    {
        byte[] rgba = image.ToRgba(topDown: true);
        return Enumerable.Range(0, image.Height)
            .Select(y => Enumerable.Range(0, image.Width).Select(x => (int)rgba[(y * image.Width + x) * 4]).ToArray())
            .ToArray();
    }

    [Fact]
    public void RgbaBytesInAndOut()
    {
        byte[] bytes = Enumerable.Range(0, 3 * 2 * 4).Select(i => (byte)i).ToArray();
        var image = FaceImage.FromRgba(bytes, 3, 2);
        Assert.Equal(new Rgba32(0, 1, 2, 3), image.Pixels[0]);   // R, G, B, A
        Assert.Equal(new Rgba32(20, 21, 22, 23), image.Pixels[5]);
        Assert.Equal(bytes, image.ToRgba());

        // Rows from the top are the same rows in the other order.
        var topDown = FaceImage.FromRgba(bytes, 3, 2, topDown: true);
        Assert.Equal(image.Pixels[3..].Concat(image.Pixels[..3]), topDown.Pixels);
        Assert.Equal(bytes, topDown.ToRgba(topDown: true));

        var buffer = new byte[bytes.Length];
        image.ToRgba(buffer);
        Assert.Equal(bytes, buffer);
    }

    [Fact]
    public void RgbaSizesMustMatch()
    {
        Assert.Throws<ArgumentException>(() => FaceImage.FromRgba(new byte[23], 3, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => FaceImage.FromRgba(Array.Empty<byte>(), 0, 2));
        Assert.Throws<ArgumentException>(() => Numbered().ToRgba(new byte[23]));
        Assert.Throws<InvalidOperationException>(() => default(FaceImage).ToRgba());
    }

    [Fact]
    public void QuarterTurns()
    {
        Assert.Equal(new[] { new[] { 0, 1, 2 }, new[] { 3, 4, 5 } }, Seen(Numbered().Rotate(0)));
        // Clockwise: the top edge turns to the right.
        Assert.Equal(new[] { new[] { 3, 0 }, new[] { 4, 1 }, new[] { 5, 2 } }, Seen(Numbered().Rotate(1)));
        Assert.Equal(new[] { new[] { 5, 4, 3 }, new[] { 2, 1, 0 } }, Seen(Numbered().Rotate(2)));
        Assert.Equal(new[] { new[] { 2, 5 }, new[] { 1, 4 }, new[] { 0, 3 } }, Seen(Numbered().Rotate(3)));
        Assert.Equal(Seen(Numbered().Rotate(3)), Seen(Numbered().Rotate(-1)));
        Assert.Equal(Seen(Numbered()), Seen(Numbered().Rotate(1).Rotate(1).Rotate(1).Rotate(1)));
        Assert.Equal((2, 3), (Numbered().Rotate(1).Width, Numbered().Rotate(1).Height));
    }

    // The Unity app's own clockwise turn (FaceImageBridge.RotateCw90, checked on a 3×2
    // image), which Rotate(1) replaces, on bottom-up buffers.
    [Fact]
    public void ClockwiseMatchesTheUnityApp()
    {
        var image = Numbered();
        int w = image.Width, h = image.Height;
        var expected = new Rgba32[w * h];
        for (int dy = 0; dy < w; dy++)
            for (int dx = 0; dx < h; dx++)
                expected[dy * h + dx] = image.Pixels[dx * w + (w - 1 - dy)];
        Assert.Equal(expected, image.Rotate(1).Pixels);
    }

    [Fact]
    public void Flips()
    {
        Assert.Equal(new[] { new[] { 2, 1, 0 }, new[] { 5, 4, 3 } }, Seen(Numbered().FlipHorizontal()));
        Assert.Equal(new[] { new[] { 3, 4, 5 }, new[] { 0, 1, 2 } }, Seen(Numbered().FlipVertical()));
        Assert.Equal(Seen(Numbered().Rotate(2)), Seen(Numbered().FlipHorizontal().FlipVertical()));
    }

    [Fact]
    public void HelpersCopy()
    {
        var image = FaceImage.FromRgba(new byte[] { 255, 0, 0, 255, 0, 0, 255, 255 }, 2, 1);
        foreach (var made in new[] { image.Rotate(0), image.Rotate(1), image.FlipHorizontal(), image.FlipVertical() })
            made.Pixels[0] = Black;
        Assert.Equal(new[] { Red, Blue }, image.Pixels);
    }
}
