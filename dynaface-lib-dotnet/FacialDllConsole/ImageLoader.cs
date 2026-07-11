using System;
using System.IO;
using System.Net.Http;
using SkiaSharp;

// Converts between disk images (PNG/JPEG/etc. via SkiaSharp) and FaceImage.
// FaceImage stores pixels in Unity bottom-left order (y=0 at bottom);
// standard image files are top-left (y=0 at top), so Load/Save both flip Y.
public static class ImageLoader
{
    // Accepts a local path or an http(s) URL, mirroring dynaface-lib's
    // load_face_image() (which downloads URL inputs before decoding).
    public static FaceImage Load(string path)
    {
        using var bitmap = IsUrl(path) ? DecodeFromUrl(path) : SKBitmap.Decode(path);
        int w = bitmap.Width, h = bitmap.Height;
        var pixels   = new Rgba32[w * h];
        var skPixels = bitmap.Pixels; // top-left row-major SKColor[]

        for (int y = 0; y < h; y++)
        {
            int dRow = h - 1 - y; // flip to bottom-left
            for (int x = 0; x < w; x++)
            {
                var c = skPixels[y * w + x];
                pixels[dRow * w + x] = new Rgba32(c.Red, c.Green, c.Blue, c.Alpha);
            }
        }

        return new FaceImage(pixels, w, h);
    }

    public static bool IsUrl(string path) =>
        path.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
        path.StartsWith("https://", StringComparison.OrdinalIgnoreCase);

    static SKBitmap DecodeFromUrl(string url)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
        byte[] bytes = http.GetByteArrayAsync(url).GetAwaiter().GetResult();
        return SKBitmap.Decode(bytes);
    }

    public static void Save(FaceImage image, string path)
    {
        int w = image.Width, h = image.Height;
        var skPixels = new SKColor[w * h];

        for (int y = 0; y < h; y++)
        {
            int sRow = h - 1 - y; // flip from bottom-left
            for (int x = 0; x < w; x++)
            {
                var p = image.Pixels[sRow * w + x];
                skPixels[y * w + x] = new SKColor(p.R, p.G, p.B, p.A);
            }
        }

        using var bitmap = new SKBitmap(w, h);
        bitmap.Pixels = skPixels;

        using var stream = File.OpenWrite(path);
        bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
    }
}
