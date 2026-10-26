namespace DynafaceTests;

// FaceRenderer's landmark overlay, the numbered dots behind the Unity app's "LM"
// toggle, and its small pixel helpers, drawn on blank buffers. FaceRenderer's
// public methods use bottom-left pixel coordinates; DrawLandmarksOnto takes
// landmarks in top-left ones and flips them. No other test uses the overlay's
// static settings, and these tests restore them, so they can't leak into others.
public class FaceRendererTests
{
    const int Size = 600;
    static readonly Rgba32 White = new(255, 255, 255, 255);
    static readonly Rgba32 LabelOutline = new(20, 20, 20, 255);

    // The 98 landmarks on a grid, 40 pixels apart, in top-left coordinates.
    static Vec2[] Grid() =>
        Enumerable.Range(0, 98).Select(i => new Vec2(100 + i % 10 * 40, 100 + i / 10 * 40)).ToArray();

    static Vec2[] AllAt(float x, float y) => Enumerable.Repeat(new Vec2(x, y), 98).ToArray();

    static void WithOverlay(Action draw, int numberScale = 2)
    {
        var mode = FaceRenderer.DisplayMode;
        int scale = FaceRenderer.NumberScale;
        try
        {
            FaceRenderer.DisplayMode = FaceRenderer.LandmarkDisplayMode.Lm;
            FaceRenderer.NumberScale = numberScale;
            draw();
        }
        finally
        {
            FaceRenderer.DisplayMode = mode;
            FaceRenderer.NumberScale = scale;
        }
    }

    static Rgba32[] DrawLandmarks(Vec2[] landmarks, int numberScale = 2)
    {
        var pixels = new Rgba32[Size * Size];
        WithOverlay(() => FaceRenderer.DrawLandmarksOnto(pixels, Size, Size, landmarks), numberScale);
        return pixels;
    }

    // The pixel at a top-left position.
    static Rgba32 At(Rgba32[] pixels, Vec2 p) => pixels[(Size - 1 - (int)p.Y) * Size + (int)p.X];

    static IEnumerable<(int x, int y)> Where(Rgba32[] pixels, Rgba32 color) =>
        Enumerable.Range(0, pixels.Length).Where(i => pixels[i] == color).Select(i => (i % Size, i / Size));

    [Fact]
    public void NothingIsDrawnWhenOff()
    {
        var pixels = new Rgba32[Size * Size];
        FaceRenderer.DrawLandmarksOnto(pixels, Size, Size, Grid()); // the default is off
        Assert.All(pixels, p => Assert.Equal(default, p));
    }

    [Fact]
    public void NothingIsDrawnWithoutAll98Landmarks()
    {
        Assert.All(DrawLandmarks(Grid().Take(97).ToArray()), p => Assert.Equal(default, p));
        WithOverlay(() =>
        {
            FaceRenderer.DrawLandmarksOnto(null!, Size, Size, Grid());
            FaceRenderer.DrawLandmarksOnto(new Rgba32[Size * Size], Size, Size, null!);
        });
    }

    [Fact]
    public void EachDotHasItsRegionsColor()
    {
        Vec2[] grid = Grid();
        Rgba32[] pixels = DrawLandmarks(grid);
        for (int i = 0; i < 98; i++)
        {
            Rgba32 expected = i switch
            {
                <= 32 => new(25, 235, 50, 255), // face outline: green
                <= 50 => new(35, 70, 255, 255), // brows: blue
                <= 59 => new(245, 25, 230, 255), // nose: magenta
                <= 75 => new(255, 35, 35, 255), // eyes: red
                <= 87 => new(20, 225, 225, 255), // outer lip: cyan
                <= 95 => new(235, 225, 25, 255), // inner lip: yellow
                _ => new(255, 35, 35, 255), // pupils: red
            };
            Assert.Equal(expected, At(pixels, grid[i]));
        }
    }

    [Fact]
    public void LabelsAreWhiteWithADarkOutline()
    {
        Rgba32[] pixels = DrawLandmarks(Grid());
        Assert.NotEmpty(Where(pixels, White));
        Assert.NotEmpty(Where(pixels, LabelOutline));
    }

    [Fact]
    public void LabelsOnARegionsCenterTakeTheirFallbackDirection()
    {
        // With every landmark at one point, each sits on its region's center, so
        // labels go up, except 51 and 53 (left), 52 and 54 (right), 57 (down) and
        // the pupils, 96 and 97 (up and to the right).
        Rgba32[] pixels = DrawLandmarks(AllAt(300, 300));
        int cx = 300, cy = Size - 1 - 300;
        var labels = Where(pixels, White).ToList();
        Assert.Contains(labels, p => p.y > cy + 15); // up
        Assert.Contains(labels, p => p.y < cy - 15); // down
        Assert.Contains(labels, p => p.x < cx - 15); // left
        Assert.Contains(labels, p => p.x > cx + 15); // right
    }

    [Fact]
    public void LabelsStayInsideTheImage()
    {
        // Landmarks in a corner: their dots are cut off, but their labels are moved
        // inside the image.
        Rgba32[] pixels = DrawLandmarks(AllAt(0, 0));
        Assert.NotEmpty(Where(pixels, White));
    }

    [Fact]
    public void NumberScaleSizesTheLabels()
    {
        int small = Where(DrawLandmarks(Grid(), numberScale: 2), White).Count();
        int large = Where(DrawLandmarks(Grid(), numberScale: 4), White).Count();
        Assert.True(large > small, $"{large} white pixels at scale 4, {small} at scale 2");
    }

    [Fact]
    public void ExtractHorizRowClamps()
    {
        const int w = 4, h = 3;
        var pixels = Enumerable.Range(0, w * h).Select(i => new Rgba32((byte)i, 0, 0, 255)).ToArray();
        byte[] Row(int y, int x1 = 0, int x2 = -1) =>
            FaceRenderer.ExtractHorizRow(pixels, w, h, y, x1, x2).Select(p => p.R).ToArray();

        Assert.Equal(new byte[] { 4, 5, 6, 7 }, Row(1)); // the whole row by default
        Assert.Equal(new byte[] { 9, 10 }, Row(2, 1, 3));
        Assert.Equal(new byte[] { 8, 9, 10, 11 }, Row(9, -5, 99)); // clamped to the image
        Assert.Empty(Row(0, 3, 1));
    }

    [Fact]
    public void EmptyInputsMeasureNothing()
    {
        Assert.Equal(Vec2i.Zero, FaceRenderer.GetTextSize(""));
        var pixels = new Rgba32[Size * Size];
        Assert.Equal(0f, FaceRenderer.MeasurePolygon(pixels, Size, Size, null!));
        Assert.Equal(0f, FaceRenderer.MeasurePolygon(pixels, Size, Size, new[] { new Vec2i(1, 1), new Vec2i(5, 5) }));
    }
}
