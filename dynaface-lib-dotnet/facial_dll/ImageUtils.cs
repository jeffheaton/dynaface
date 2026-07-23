using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("DynafaceTests")]

// Pure static image utilities with no MonoBehaviour dependency.
// Methods here are internal so the Tests assembly can reach them without
// exposing them in the public API.
internal static class ImageUtils
{
    // Flips a row-major Rgba32 pixel array horizontally in-place, for any width/height
    // (not just square images — needed both for the square aligned-crop case and for
    // the raw, generally non-square camera frame/original photo).
    // This is the mirror correction applied to iOS WebCamTexture crops and dynaface-lib's
    // facing-right-in-lateral-view flip; see the MIRROR CONTRACT before changing it.
    internal static void FlipHorizontal(Rgba32[] px, int width, int height)
    {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width / 2; x++)
            {
                int a = y * width + x, b = y * width + (width - 1 - x);
                (px[a], px[b]) = (px[b], px[a]);
            }
    }

    // Reverses row order (flips the image vertically). Used at the boundary between
    // Unity bottom-left storage and the top-left-semantic math used by the crop/pose
    // pipeline (StyleGanCropper/LateralCropper/PoseClassifier): flip once on the way
    // in, run the (Python-equivalent) top-left math, flip once on the way out.
    internal static Rgba32[] FlipVertical(Rgba32[] src, int width, int height)
    {
        var dst = new Rgba32[width * height];
        for (int y = 0; y < height; y++)
            System.Array.Copy(src, (height - 1 - y) * width, dst, y * width, width);
        return dst;
    }

    // Creates a new image where one half is original and the other half is a
    // horizontal mirror of the kept half. Used by MirroredLandmarkDetector.
    //   mirrorLeft = true  → right half replaced with mirror of left half
    //   mirrorLeft = false → left half replaced with mirror of right half
    internal static FaceImage CreateHalfMirror(FaceImage src, bool mirrorLeft)
    {
        int w = src.Width, h = src.Height, mid = w / 2;
        var dst = new Rgba32[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int srcX = mirrorLeft
                    ? (x < mid ? x : w - 1 - x)   // right mirrors left
                    : (x >= mid ? x : w - 1 - x);  // left mirrors right
                dst[y * w + x] = src.Pixels[y * w + srcX];
            }
        return new FaceImage(dst, w, h);
    }

    // Bilinear resize of a row-major Rgba32 pixel array.
    // Pixels are in Unity bottom-left order (y=0 at bottom).
    // Uses center-pixel mapping: each destination pixel center maps to the
    // proportionally equivalent position in the source image.
    internal static Rgba32[] ResizeBilinear(
        Rgba32[] src, int srcWidth, int srcHeight,
        int dstWidth, int dstHeight)
        => ResizeBilinear(src, srcWidth, srcHeight, 0, 0, srcWidth, srcHeight, dstWidth, dstHeight);

    // Bilinear resize of a sub-region (ROI) of a row-major Rgba32 pixel array.
    // roiX/roiY: top-left corner of the source region in pixel coords.
    // roiW/roiH: width and height of the source region.
    // Output is roiW×roiH resampled to dstWidth×dstHeight.
    internal static Rgba32[] ResizeBilinear(
        Rgba32[] src, int srcWidth, int srcHeight,
        int roiX, int roiY, int roiW, int roiH,
        int dstWidth, int dstHeight)
    {
        var dst = new Rgba32[dstWidth * dstHeight];
        float xScale = (float)roiW / dstWidth;
        float yScale = (float)roiH / dstHeight;
        float xMax = roiX + roiW - 1f;
        float yMax = roiY + roiH - 1f;

        for (int dy = 0; dy < dstHeight; dy++)
            for (int dx = 0; dx < dstWidth; dx++)
            {
                float sx = Clamp(roiX + (dx + 0.5f) * xScale - 0.5f, roiX, xMax);
                float sy = Clamp(roiY + (dy + 0.5f) * yScale - 0.5f, roiY, yMax);

                int x0 = (int)sx;
                int y0 = (int)sy;
                int x1 = x0 + 1 < srcWidth ? x0 + 1 : x0;
                int y1 = y0 + 1 < srcHeight ? y0 + 1 : y0;

                float tx = sx - x0;
                float ty = sy - y0;

                Rgba32 c00 = src[y0 * srcWidth + x0];
                Rgba32 c10 = src[y0 * srcWidth + x1];
                Rgba32 c01 = src[y1 * srcWidth + x0];
                Rgba32 c11 = src[y1 * srcWidth + x1];

                dst[dy * dstWidth + dx] =
                    LerpColor(LerpColor(c00, c10, tx), LerpColor(c01, c11, tx), ty);
            }
        return dst;
    }

    static Rgba32 LerpColor(Rgba32 a, Rgba32 b, float t) => new Rgba32(
        (byte)(a.R + (b.R - a.R) * t),
        (byte)(a.G + (b.G - a.G) * t),
        (byte)(a.B + (b.B - a.B) * t),
        (byte)(a.A + (b.A - a.A) * t));

    static float Clamp(float v, float min, float max) =>
        v < min ? min : v > max ? max : v;

    // General 2x3 affine warp, matching cv2.warpAffine's convention: `m` is the
    // FORWARD src->dst transform (same row layout cv2 expects), and is inverted
    // internally to do the actual dst->src sampling (cv2 never honors WARP_INVERSE_MAP
    // here, so callers should never pre-invert `m`):
    //   [ m00 m01 m02 ]   [x]
    //   [ m10 m11 m12 ] * [y]
    //                     [1]
    // Unlike ResizeBilinear (which mirrors cv2.resize's center-pixel convention),
    // this addresses pixel (x,y) directly with no +-0.5 offset, matching
    // cv2.warpAffine's own pixel-addressing convention.
    internal static Rgba32[] WarpAffine(
        Rgba32[] src, int srcWidth, int srcHeight,
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        int dstWidth, int dstHeight,
        bool bilinear, Rgba32 fillColor)
    {
        var dst = new Rgba32[dstWidth * dstHeight];

        float det = m00 * m11 - m01 * m10;
        if (System.MathF.Abs(det) < 1e-12f)
        {
            for (int i = 0; i < dst.Length; i++) dst[i] = fillColor;
            return dst;
        }

        var (i00, i01, i02, i10, i11, i12) = InvertAffine(m00, m01, m02, m10, m11, m12);

        for (int dy = 0; dy < dstHeight; dy++)
            for (int dx = 0; dx < dstWidth; dx++)
            {
                float sx = i00 * dx + i01 * dy + i02;
                float sy = i10 * dx + i11 * dy + i12;

                dst[dy * dstWidth + dx] = bilinear
                    ? SampleBilinearOrFill(src, srcWidth, srcHeight, sx, sy, fillColor)
                    : SampleNearestOrFill(src, srcWidth, srcHeight, sx, sy, fillColor);
            }
        return dst;
    }

    // Inverts a forward 2x3 affine (same row layout as cv2.warpAffine's M).
    // Returns the identity transform if the matrix is (near-)singular — callers
    // that care about the singular case (e.g. WarpAffine) should check the
    // determinant themselves first, since "identity" is rarely the right fallback.
    internal static (float i00, float i01, float i02, float i10, float i11, float i12) InvertAffine(
        float m00, float m01, float m02, float m10, float m11, float m12)
    {
        float det = m00 * m11 - m01 * m10;
        if (System.MathF.Abs(det) < 1e-12f) return (1f, 0f, 0f, 0f, 1f, 0f);

        float invDet = 1f / det;
        float i00 = m11 * invDet, i01 = -m01 * invDet;
        float i10 = -m10 * invDet, i11 = m00 * invDet;
        float i02 = -(i00 * m02 + i01 * m12);
        float i12 = -(i10 * m02 + i11 * m12);
        return (i00, i01, i02, i10, i11, i12);
    }

    // Direct port of dynaface-lib's util.safe_clip: clips a `width`x`height` region
    // starting at (x,y) out of `src`, padding any out-of-bounds area with `background`.
    // Returns the new image plus how much the requested origin was clamped inward
    // (both 0 unless x<0 or y<0). Row-major convention is whatever `src` itself uses
    // (top-left or bottom-left) — the caller just needs to stay consistent.
    internal static (Rgba32[] pixels, int offsetX, int offsetY) SafeClip(
        Rgba32[] src, int srcWidth, int srcHeight,
        int x, int y, int width, int height, Rgba32 background)
    {
        int xStart = MathHelpers.Max(x, 0);
        int yStart = MathHelpers.Max(y, 0);
        int xEnd = MathHelpers.Min(x + width, srcWidth);
        int yEnd = MathHelpers.Min(y + height, srcHeight);
        int clippedWidth = xEnd - xStart;
        int clippedHeight = yEnd - yStart;

        var dst = new Rgba32[width * height];
        for (int i = 0; i < dst.Length; i++) dst[i] = background;

        int newXStart = MathHelpers.Max(0, -x);
        int newYStart = MathHelpers.Max(0, -y);

        if (clippedWidth > 0 && clippedHeight > 0)
        {
            for (int row = 0; row < clippedHeight; row++)
            {
                int srcRow = (yStart + row) * srcWidth + xStart;
                int dstRow = (newYStart + row) * width + newXStart;
                System.Array.Copy(src, srcRow, dst, dstRow, clippedWidth);
            }
        }
        return (dst, newXStart, newYStart);
    }

    // Direct port of dynaface-lib's util.scale_crop_points. Truncates toward zero
    // (matches Python's int(...), not a round).
    internal static Vec2[] ScaleCropPoints(Vec2[] pts, int cropX, int cropY, float scale)
    {
        var result = new Vec2[pts.Length];
        for (int i = 0; i < pts.Length; i++)
            result[i] = new Vec2((int)(pts[i].X * scale - cropX), (int)(pts[i].Y * scale - cropY));
        return result;
    }

    // Direct port of dynaface-lib's util.rotate_crop_points: rotates each point about
    // `center` by -angleDegrees (note the sign flip, matching Python exactly).
    internal static Vec2[] RotateCropPoints(Vec2[] points, Vec2 center, float angleDegrees)
    {
        float angleRadians = -angleDegrees * MathHelpers.Deg2Rad;
        float cosTheta = MathHelpers.Cos(angleRadians);
        float sinTheta = MathHelpers.Sin(angleRadians);

        var result = new Vec2[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            float vx = points[i].X - center.X;
            float vy = points[i].Y - center.Y;
            float rx = cosTheta * vx - sinTheta * vy;
            float ry = sinTheta * vx + cosTheta * vy;
            result[i] = new Vec2(MathHelpers.RoundToInt(rx + center.X), MathHelpers.RoundToInt(ry + center.Y));
        }
        return result;
    }

    // Direct port of dynaface-lib's util.straighten. Rotates about the image center
    // via cv2.getRotationMatrix2D's documented formula (scale=1.0), filling anything
    // exposed by the rotation with the image's own average color — matching the
    // fix upstream (util.straighten used to build a separate average-color canvas
    // that got fully overwritten by the rotated image at an always-zero offset, so
    // the fill never actually showed; the fix passes the average color directly as
    // cv2.warpAffine's borderValue instead).
    internal static Rgba32[] Straighten(Rgba32[] src, int width, int height, float angleRadians)
    {
        float angleDegrees = angleRadians * (180f / System.MathF.PI);
        if (angleDegrees > 45f) angleDegrees -= 180f;
        else if (angleDegrees < -45f) angleDegrees += 180f;

        int cx = width / 2, cy = height / 2;
        float rad = angleDegrees * MathHelpers.Deg2Rad;
        float alpha = MathHelpers.Cos(rad);
        float beta = MathHelpers.Sin(rad);

        float m02 = (1f - alpha) * cx - beta * cy;
        float m12 = beta * cx + (1f - alpha) * cy;

        return WarpAffine(src, width, height,
            alpha, beta, m02, -beta, alpha, m12,
            width, height, bilinear: true, fillColor: AverageColor(src));
    }

    // Matches dynaface-lib's util.calculate_average_rgb: mean R/G/B over the whole image.
    internal static Rgba32 AverageColor(Rgba32[] src)
    {
        if (src.Length == 0) return new Rgba32(0, 0, 0, 255);

        long sumR = 0, sumG = 0, sumB = 0;
        foreach (var p in src) { sumR += p.R; sumG += p.G; sumB += p.B; }

        return new Rgba32(
            (byte)(sumR / src.Length), (byte)(sumG / src.Length), (byte)(sumB / src.Length), 255);
    }

    static Rgba32 TapOrFill(Rgba32[] src, int width, int height, int x, int y, Rgba32 fillColor) =>
        (uint)x < (uint)width && (uint)y < (uint)height ? src[y * width + x] : fillColor;

    static Rgba32 SampleNearestOrFill(Rgba32[] src, int width, int height, float x, float y, Rgba32 fillColor)
    {
        int ix = MathHelpers.RoundToInt(x);
        int iy = MathHelpers.RoundToInt(y);
        return TapOrFill(src, width, height, ix, iy, fillColor);
    }

    static Rgba32 SampleBilinearOrFill(Rgba32[] src, int width, int height, float x, float y, Rgba32 fillColor)
    {
        int x0 = MathHelpers.FloorToInt(x), y0 = MathHelpers.FloorToInt(y);
        int x1 = x0 + 1, y1 = y0 + 1;
        float tx = x - x0, ty = y - y0;

        Rgba32 c00 = TapOrFill(src, width, height, x0, y0, fillColor);
        Rgba32 c10 = TapOrFill(src, width, height, x1, y0, fillColor);
        Rgba32 c01 = TapOrFill(src, width, height, x0, y1, fillColor);
        Rgba32 c11 = TapOrFill(src, width, height, x1, y1, fillColor);

        float r0 = MathHelpers.Lerp(c00.R, c10.R, tx), r1 = MathHelpers.Lerp(c01.R, c11.R, tx);
        float g0 = MathHelpers.Lerp(c00.G, c10.G, tx), g1 = MathHelpers.Lerp(c01.G, c11.G, tx);
        float b0 = MathHelpers.Lerp(c00.B, c10.B, tx), b1 = MathHelpers.Lerp(c01.B, c11.B, tx);
        float a0 = MathHelpers.Lerp(c00.A, c10.A, tx), a1 = MathHelpers.Lerp(c01.A, c11.A, tx);

        return new Rgba32(
            (byte)MathHelpers.Clamp(MathHelpers.RoundToInt(MathHelpers.Lerp(r0, r1, ty)), 0, 255),
            (byte)MathHelpers.Clamp(MathHelpers.RoundToInt(MathHelpers.Lerp(g0, g1, ty)), 0, 255),
            (byte)MathHelpers.Clamp(MathHelpers.RoundToInt(MathHelpers.Lerp(b0, b1, ty)), 0, 255),
            (byte)MathHelpers.Clamp(MathHelpers.RoundToInt(MathHelpers.Lerp(a0, a1, ty)), 0, 255));
    }
}
