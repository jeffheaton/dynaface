using System.Collections.Generic;

// Direct port of dynaface-lib's lateral.py _process_image / _extract_sagittal_profile /
// _shift_sagittal_profile. Runs U^2-Net background removal (the 3rd, independent
// network), thresholds+morph-closes+inverts the resulting mask into a binary
// silhouette, then extracts the sagittal (profile) line: for each image row, the x of
// the first foreground pixel.
//
// Coordinate note: works in TOP-LEFT row order throughout (row 0 = visual top),
// matching dynaface-lib's own numpy/cv2 convention and the top-left landmarks this
// profile gets compared against later (LateralLandmarkFinder).
public static class SagittalProfile
{
    const int U2NetSize = 320;
    const int MorphKernelSize = 10;
    const int BinaryThreshold = 32;

    static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    // photo: the lateral-cropped image (bottom-left FaceImage storage).
    // Returns the inverted, morph-closed binary silhouette (255=background,
    // 0=foreground) in top-left row order, width*height flat array.
    public static byte[] ProcessImage(IDynafaceInference inference, FaceImage photo)
    {
        int width = photo.Width, height = photo.Height;
        Rgba32[] topLeftPixels = ImageUtils.FlipVertical(photo.Pixels, width, height);

        float[] mask320 = RunU2Net(inference, topLeftPixels, width, height);
        float[] maskFull = ResizeMaskBilinear(mask320, U2NetSize, U2NetSize, width, height);

        var binary = new byte[width * height];
        for (int i = 0; i < binary.Length; i++)
            binary[i] = MathHelpers.Clamp(maskFull[i], 0f, 1f) * 255f > BinaryThreshold ? (byte)255 : (byte)0;

        binary = MorphClose(binary, width, height, MorphKernelSize);

        for (int i = 0; i < binary.Length; i++)
            binary[i] = (byte)(255 - binary[i]);

        return binary;
    }

    // For each row, the x of the first foreground (0-valued) pixel. Rows with no
    // foreground pixel are skipped entirely (not padded), matching dynaface-lib —
    // the returned arrays may be shorter than `height`.
    public static (int[] sagittalX, int[] sagittalY) ExtractSagittalProfile(byte[] binaryTopLeft, int width, int height)
    {
        var xs = new List<int>();
        var ys = new List<int>();
        for (int y = 0; y < height; y++)
        {
            int rowBase = y * width;
            for (int x = 0; x < width; x++)
            {
                if (binaryTopLeft[rowBase + x] == 0)
                {
                    xs.Add(x);
                    ys.Add(y);
                    break;
                }
            }
        }
        return (xs.ToArray(), ys.ToArray());
    }

    // Shifts so the minimum x becomes 0; returns the shift amount too (added back
    // to the final lateral landmark x-coordinates by LateralLandmarkFinder).
    public static (float[] shiftedX, float shiftX) ShiftSagittalProfile(int[] sagittalX)
    {
        int minX = sagittalX.Length > 0 ? sagittalX[0] : 0;
        foreach (int x in sagittalX)
            if (x < minX) minX = x;

        var shifted = new float[sagittalX.Length];
        for (int i = 0; i < sagittalX.Length; i++) shifted[i] = sagittalX[i] - minX;
        return (shifted, minX);
    }

    static float[] RunU2Net(IDynafaceInference inference, Rgba32[] topLeftPixels, int width, int height)
    {
        Rgba32[] resized = ImageUtils.ResizeBilinear(topLeftPixels, width, height, U2NetSize, U2NetSize);

        int plane = U2NetSize * U2NetSize;
        float[] tensor = new float[3 * plane];
        for (int i = 0; i < plane; i++)
        {
            Rgba32 c = resized[i];
            tensor[0 * plane + i] = (c.R / 255f - ImageNetMean[0]) / ImageNetStd[0];
            tensor[1 * plane + i] = (c.G / 255f - ImageNetMean[1]) / ImageNetStd[1];
            tensor[2 * plane + i] = (c.B / 255f - ImageNetMean[2]) / ImageNetStd[2];
        }

        return inference.RunU2Net(tensor); // flat [320*320] raw sigmoid saliency map
    }

    static float[] ResizeMaskBilinear(float[] src, int srcW, int srcH, int dstW, int dstH)
    {
        var dst = new float[dstW * dstH];
        float xScale = (float)srcW / dstW;
        float yScale = (float)srcH / dstH;
        float xMax = srcW - 1f, yMax = srcH - 1f;

        for (int dy = 0; dy < dstH; dy++)
            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = MathHelpers.Clamp((dx + 0.5f) * xScale - 0.5f, 0f, xMax);
                float sy = MathHelpers.Clamp((dy + 0.5f) * yScale - 0.5f, 0f, yMax);

                int x0 = (int)sx, y0 = (int)sy;
                int x1 = MathHelpers.Min(x0 + 1, srcW - 1);
                int y1 = MathHelpers.Min(y0 + 1, srcH - 1);
                float tx = sx - x0, ty = sy - y0;

                float v00 = src[y0 * srcW + x0], v10 = src[y0 * srcW + x1];
                float v01 = src[y1 * srcW + x0], v11 = src[y1 * srcW + x1];
                float v0 = v00 + (v10 - v00) * tx;
                float v1 = v01 + (v11 - v01) * tx;
                dst[dy * dstW + dx] = v0 + (v1 - v0) * ty;
            }
        return dst;
    }

    // Dilate-then-erode with a flat kernelSize x kernelSize structuring element,
    // using OpenCV's default anchor for an even kernel (ksize/2 via integer
    // division — an asymmetric [-anchor, ksize-1-anchor] window, not a symmetric +-k/2).
    static byte[] MorphClose(byte[] binary, int width, int height, int kernelSize)
    {
        int anchor = kernelSize / 2;
        int lo = -anchor;
        int hi = kernelSize - 1 - anchor;

        byte[] dilated = MorphPass(binary, width, height, lo, hi, dilate: true);
        return MorphPass(dilated, width, height, lo, hi, dilate: false);
    }

    static byte[] MorphPass(byte[] src, int width, int height, int lo, int hi, bool dilate)
    {
        var dst = new byte[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                bool result = !dilate; // dilate: OR, starts false; erode: AND, starts true
                for (int dy = lo; dy <= hi; dy++)
                {
                    int sy = y + dy;
                    if ((uint)sy >= (uint)height) continue;
                    int rowBase = sy * width;
                    for (int dx = lo; dx <= hi; dx++)
                    {
                        int sx = x + dx;
                        if ((uint)sx >= (uint)width) continue;
                        bool v = src[rowBase + sx] != 0;
                        if (dilate) result |= v;
                        else result &= v;
                    }
                }
                dst[y * width + x] = result ? (byte)255 : (byte)0;
            }
        return dst;
    }
}
