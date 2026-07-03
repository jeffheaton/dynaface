// Direct port of dynaface-lib's AnalyzeFace.crop_lateral.
//
// Fits the full landmark vertical band into the 1024 target with configurable
// top/bottom padding, horizontally anchored to the right-pupil landmark — a
// different scale/crop strategy than StyleGanCropper's pupil-distance targeting.
//
// Coordinate note: same top-left-internally, bottom-left-at-the-boundary convention
// as StyleGanCropper (see its header comment).
public static class LateralCropper
{
    // photo: full image (post-flip-to-facing-left, post 1.5x-width-pad).
    // landmarks: ALL landmarks (not just pupils), in top-left pixel space, already
    // shifted for any padding already applied by the caller.
    public static (FaceImage crop, Vec2[] landmarks) Crop(FaceImage photo, Vec2[] landmarks)
    {
        const int target = DynafaceConstants.StyleganWidth;
        int width = photo.Width, height = photo.Height;

        int minY, maxY;
        if (landmarks != null && landmarks.Length > 0)
        {
            minY = maxY = (int)landmarks[0].Y;
            foreach (var p in landmarks)
            {
                int y = (int)p.Y;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
        else
        {
            minY = 0;
            maxY = height - 1;
        }

        int bandH = MathHelpers.Max(1, maxY - minY);

        int padTopT   = MathHelpers.RoundToInt(target * DynafaceConstants.LateralPadTopRatio);
        int padBotT   = MathHelpers.RoundToInt(target * DynafaceConstants.LateralPadBottomRatio);
        int padTotalT = MathHelpers.Min(padTopT + padBotT, target - 1);

        float sVert = MathHelpers.Max(1e-6f, (target - padTotalT) / (float)bandH);
        float sFill = MathHelpers.Max(target / (float)width, target / (float)height);
        float scale = MathHelpers.Min(sFill, sVert);

        int newWidth  = MathHelpers.Max(1, MathHelpers.RoundToInt(width * scale));
        int newHeight = MathHelpers.Max(1, MathHelpers.RoundToInt(height * scale));

        Rgba32[] topLeftPixels = ImageUtils.FlipVertical(photo.Pixels, width, height);
        Rgba32[] resized = ImageUtils.ResizeBilinear(topLeftPixels, width, height, newWidth, newHeight);

        int cropX;
        if (landmarks == null || landmarks.Length <= DynafaceConstants.LmRightPupil)
        {
            cropX = (newWidth - target) / 2;
        }
        else
        {
            int rpXScaled = MathHelpers.RoundToInt(landmarks[DynafaceConstants.LmRightPupil].X * scale);
            cropX = rpXScaled - MathHelpers.RoundToInt(DynafaceConstants.StyleganRightPupilX);
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
