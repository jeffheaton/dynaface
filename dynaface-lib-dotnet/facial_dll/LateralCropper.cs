// Direct port of dynaface-lib's AnalyzeFace.crop_lateral, for a profile and,
// centered, for a three-quarter (quarter) face.
//
// Fits the full landmark vertical band into the 1024 target with configurable
// top/bottom padding. A profile never scales up past filling the frame, so its
// padding can grow; a quarter face (center) always fills the crop with its band and
// pads, whatever the frame's size, as the frontal crop fixes the pupils' spacing, so
// the cheek landmarks' pixel-sized rules follow the face, not the frame it was
// photographed in. Horizontally, the profile's leading edge (leftmost landmark; the
// face always faces left here) is placed a fixed margin from the left, so the face
// uses the space at the left and the sagittal chart overlaid at the right cuts off
// near the front of the ear; a quarter face's landmarks are centered instead, so the
// two mirrored turns of one face crop alike — a different scale/crop strategy than
// StyleGanCropper's pupil-distance targeting.
//
// Coordinate note: same top-left-internally, bottom-left-at-the-boundary convention
// as StyleGanCropper (see its header comment).
public static class LateralCropper
{
    // photo: full image (a profile: post-flip-to-facing-left, post 1.5x-width-pad; a
    // quarter face: the image as given, never flipped or padded).
    // landmarks: ALL landmarks (not just pupils), in top-left pixel space, already
    // shifted for any padding already applied by the caller.
    // center: place the landmarks' horizontal middle at the middle of the crop instead
    // of the leading edge at LateralLeftMarginRatio, and scale the band to fill the
    // crop, for a quarter face, which is never flipped and gets no chart, so the two
    // mirrored turns of one face crop alike. Mirrors crop_lateral(center=True).
    public static (FaceImage crop, Vec2[] landmarks) Crop(FaceImage photo, Vec2[] landmarks, bool center = false)
    {
        const int target = DynafaceConstants.StyleganWidth;
        int width = photo.Width, height = photo.Height;

        int minX, minY, maxY;
        if (landmarks != null && landmarks.Length > 0)
        {
            minX = (int)landmarks[0].X;
            minY = maxY = (int)landmarks[0].Y;
            foreach (var p in landmarks)
            {
                int x = (int)p.X;
                int y = (int)p.Y;
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
        else
        {
            minX = 0;
            minY = 0;
            maxY = height - 1;
        }

        int bandH = MathHelpers.Max(1, maxY - minY);

        int padTopT = MathHelpers.RoundToInt(target * DynafaceConstants.LateralPadTopRatio);
        int padBotT = MathHelpers.RoundToInt(target * DynafaceConstants.LateralPadBottomRatio);
        int padTotalT = MathHelpers.Min(padTopT + padBotT, target - 1);

        float sVert = MathHelpers.Max(1e-6f, (target - padTotalT) / (float)bandH);
        float sFill = MathHelpers.Max(target / (float)width, target / (float)height);
        // A profile takes the smaller so the pads fit without scaling up past the
        // frame; a quarter face takes the band's own scale, whatever the frame, with
        // the sides padded by SafeClip when the scaled frame is narrower.
        float scale = center ? sVert : MathHelpers.Min(sFill, sVert);

        int newWidth = MathHelpers.Max(1, MathHelpers.RoundToInt(width * scale));
        int newHeight = MathHelpers.Max(1, MathHelpers.RoundToInt(height * scale));

        Rgba32[] topLeftPixels = ImageUtils.FlipVertical(photo.Pixels, width, height);
        Rgba32[] resized = ImageUtils.ResizeBilinear(topLeftPixels, width, height, newWidth, newHeight);

        // Horizontal crop: leading edge (nose tip) at a fixed left margin, or the
        // landmarks' middle at the crop's middle.
        int cropX;
        if (landmarks == null || landmarks.Length == 0)
        {
            cropX = (newWidth - target) / 2;
        }
        else if (center)
        {
            // The landmarks' middle at the crop's middle, both as pixel centers (the
            // + 1.0 and the target / 2.0), so the two mirrored turns of one face, whose
            // mirror maps column x to width - 1 - x, crop as mirror images of each
            // other at any scale. Rounded half to even, as Python's round().
            float minXs = landmarks[0].X, maxXs = landmarks[0].X;
            foreach (var p in landmarks)
            {
                if (p.X < minXs) minXs = p.X;
                if (p.X > maxXs) maxXs = p.X;
            }
            double middle = (minXs + (double)maxXs + 1.0) / 2.0;
            cropX = (int)System.Math.Round(middle * scale - target / 2.0, System.MidpointRounding.ToEven);
        }
        else
        {
            int minXScaled = MathHelpers.RoundToInt(minX * scale);
            cropX = minXScaled - MathHelpers.RoundToInt(target * DynafaceConstants.LateralLeftMarginRatio);
        }
        if (newWidth >= target)
            cropX = MathHelpers.Clamp(cropX, 0, newWidth - target);

        int minYScaled = MathHelpers.RoundToInt(minY * scale);
        int maxYScaled = MathHelpers.RoundToInt(maxY * scale);
        int cropY = minYScaled - padTopT;

        if (newHeight >= target)
        {
            cropY = MathHelpers.Clamp(cropY, 0, newHeight - target);
            int needDown = (maxYScaled + padBotT) - (cropY + target);
            if (needDown > 0)
                cropY = MathHelpers.Min(cropY + needDown, newHeight - target);
        }

        var (clipped, _, _) = ImageUtils.SafeClip(
            resized, newWidth, newHeight,
            cropX, cropY, target, target, DynafaceConstants.FillColorWhite);

        Vec2[] finalLandmarks = landmarks != null
            ? ImageUtils.ScaleCropPoints(landmarks, cropX, cropY, scale)
            : System.Array.Empty<Vec2>();

        Rgba32[] bottomLeftPixels = ImageUtils.FlipVertical(clipped, target, target);
        return (new FaceImage(bottomLeftPixels, target, target), finalLandmarks);
    }
}
