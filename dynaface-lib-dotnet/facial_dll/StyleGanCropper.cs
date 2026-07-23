using System;

// Direct port of dynaface-lib's AnalyzeFace.crop_stylegan.
//
// Coordinate note: works entirely in top-left-oriented pixel buffers internally
// (matching cv2's convention, and dynaface-lib's formulas verbatim) — flips `photo`
// to top-left once at entry, flips the finished crop back to FaceImage's bottom-left
// storage convention once at exit. Landmark VALUES are top-left throughout (both in
// and out), matching the contract the rest of the pipeline expects.
public static class StyleGanCropper
{
    // photo: full image (pre-crop). landmarks: 98 WFLW points in top-left pixel space.
    // yaw: headpose[0] (degrees), used for foreshortening correction when > 5.
    // tiltThreshold: DynafaceConstants.DefaultTiltThreshold (i.e. off) unless overridden.
    // pupilLeftOverride/pupilRightOverride: the .NET-idiomatic replacement for
    // dynaface-lib's crop_stylegan(pupils=...) zero-tuple-sentinel override — when
    // omitted, pupils are auto-detected from landmarks[97]/[96] exactly as Python does.
    public static (FaceImage crop, Vec2[] landmarks, float? faceRotationRad) Crop(
        FaceImage photo, Vec2[] landmarks, float yaw,
        float tiltThreshold = DynafaceConstants.DefaultTiltThreshold,
        Vec2? pupilLeftOverride = null, Vec2? pupilRightOverride = null)
    {
        Vec2 pupilLeft = pupilLeftOverride ?? landmarks[DynafaceConstants.LmLeftPupil];
        Vec2 pupilRight = pupilRightOverride ?? landmarks[DynafaceConstants.LmRightPupil];

        var workingLandmarks = (Vec2[])landmarks.Clone();

        // calculate_face_rotation(pupils) with pupils=(left,right) — that exact
        // argument order is what determines tilt's sign; matches Python precisely.
        float rotRad = MathHelpers.Atan2(pupilRight.Y - pupilLeft.Y, pupilRight.X - pupilLeft.X);
        float tiltDeg = ToDegrees(rotRad);

        float? faceRotationRad = null;
        if (tiltThreshold >= 0f && MathHelpers.Abs(tiltDeg) > tiltThreshold)
        {
            faceRotationRad = rotRad;
            var center = new Vec2(photo.Width / 2, photo.Height / 2);
            workingLandmarks = ImageUtils.RotateCropPoints(workingLandmarks, center, tiltDeg);
        }

        // Pupil distance is rotation-invariant, so using the pre-rotation pupils here
        // (rather than re-reading them from workingLandmarks post-rotation) matches
        // Python's actual numeric output either way — matches the order dynaface-lib
        // itself uses (calc_pd happens before crop_stylegan's own rotation step).
        float d = Vec2.Distance(pupilLeft, pupilRight);
        if (d == 0f)
            throw new InvalidOperationException("Can't process face: pupils must be in different locations");

        if (yaw > 5f)
            d = CorrectDistance2DForYaw(d, yaw) * 1.3f;

        int width = photo.Width, height = photo.Height;
        Rgba32[] topLeftPixels = ImageUtils.FlipVertical(photo.Pixels, width, height);

        if (faceRotationRad.HasValue)
            topLeftPixels = ImageUtils.Straighten(topLeftPixels, width, height, faceRotationRad.Value);

        float ar = (float)width / height;
        int newWidth = (int)(width * (DynafaceConstants.StyleganPupilDist / d));
        int newHeight = (int)(newWidth / ar);
        float scale = newWidth / (float)width;

        int cropX = (int)(workingLandmarks[DynafaceConstants.LmRightPupil].X * scale - DynafaceConstants.StyleganRightPupilX);
        int cropY = (int)(workingLandmarks[DynafaceConstants.LmRightPupil].Y * scale - DynafaceConstants.StyleganRightPupilY);

        Rgba32[] resized = ImageUtils.ResizeBilinear(topLeftPixels, width, height, newWidth, newHeight);
        var (clipped, _, _) = ImageUtils.SafeClip(
            resized, newWidth, newHeight,
            cropX, cropY, DynafaceConstants.StyleganWidth, DynafaceConstants.StyleganWidth,
            DynafaceConstants.FillColorWhite);

        Vec2[] finalLandmarks = ImageUtils.ScaleCropPoints(workingLandmarks, cropX, cropY, scale);

        Rgba32[] bottomLeftPixels = ImageUtils.FlipVertical(
            clipped, DynafaceConstants.StyleganWidth, DynafaceConstants.StyleganWidth);

        var cropImage = new FaceImage(bottomLeftPixels, DynafaceConstants.StyleganWidth, DynafaceConstants.StyleganWidth);
        return (cropImage, finalLandmarks, faceRotationRad);
    }

    static float ToDegrees(float radians)
    {
        float tilt = radians * (180f / MathF.PI);
        if (tilt > 90f) tilt -= 180f;
        else if (tilt < -90f) tilt += 180f;
        return tilt;
    }

    static float CorrectDistance2DForYaw(float measuredDistance, float yawDegrees)
    {
        float yawRadians = yawDegrees * MathHelpers.Deg2Rad;
        float correctionFactor = MathHelpers.Max(MathHelpers.Cos(yawRadians), 1e-6f);
        return measuredDistance / correctionFactor;
    }
}
