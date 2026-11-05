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
        int width = photo.Width, height = photo.Height;
        CropPlan plan = Plan(width, height, landmarks, yaw, tiltThreshold, pupilLeftOverride, pupilRightOverride);

        Rgba32[] topLeftPixels = ImageUtils.FlipVertical(photo.Pixels, width, height);

        if (plan.FaceRotationRad.HasValue)
            topLeftPixels = ImageUtils.Straighten(topLeftPixels, width, height, plan.FaceRotationRad.Value);

        Rgba32[] resized = ImageUtils.ResizeBilinear(
            topLeftPixels, width, height, plan.ResizedWidth, plan.ResizedHeight);
        var (clipped, _, _) = ImageUtils.SafeClip(
            resized, plan.ResizedWidth, plan.ResizedHeight,
            plan.CropX, plan.CropY, DynafaceConstants.StyleganWidth, DynafaceConstants.StyleganWidth,
            DynafaceConstants.FillColorWhite);

        Rgba32[] bottomLeftPixels = ImageUtils.FlipVertical(
            clipped, DynafaceConstants.StyleganWidth, DynafaceConstants.StyleganWidth);

        var cropImage = new FaceImage(bottomLeftPixels, DynafaceConstants.StyleganWidth, DynafaceConstants.StyleganWidth);
        return (cropImage, plan.Landmarks, plan.FaceRotationRad);
    }

    // The landmarks Crop would return, without building the image. For the same
    // landmarks, yaw, options and image size the result equals Crop(...).landmarks
    // exactly, integer truncation included, because both use the same layout math
    // (Plan). It suits a consumer that stores a frame's uncropped landmarks (e.g. from
    // FacePipeline.Run(crop: false)) and needs the cropped ones without decoding the
    // frame, like a video's per-frame measures. imageWidth/imageHeight: the size of the
    // image the landmarks were found in. Throws like Crop when the pupils coincide.
    //
    // A crop: false run's Pix2mm was measured on the uncropped pupils, so it doesn't
    // fit these landmarks. Build the FaceMeasureContext from them with the run's
    // PupilDistMm, and it measures the matching scale itself.
    //
    // .NET-only: dynaface-lib's crop_stylegan always builds the image.
    public static Vec2[] CropLandmarks(
        int imageWidth, int imageHeight, Vec2[] landmarks, float yaw,
        float tiltThreshold = DynafaceConstants.DefaultTiltThreshold,
        Vec2? pupilLeftOverride = null, Vec2? pupilRightOverride = null)
    {
        if (imageWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageWidth), imageWidth, "The image width must be positive.");
        if (imageHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageHeight), imageHeight, "The image height must be positive.");

        return Plan(imageWidth, imageHeight, landmarks, yaw, tiltThreshold, pupilLeftOverride, pupilRightOverride)
            .Landmarks;
    }

    // Everything the crop works out from the landmarks before it touches a pixel.
    struct CropPlan
    {
        public Vec2[] Landmarks;        // the final landmarks, in the crop's top-left pixels
        public float? FaceRotationRad;  // set only when the tilt is straightened
        public int ResizedWidth;
        public int ResizedHeight;
        public int CropX;               // the crop window's top-left corner, in the
        public int CropY;               // resized image's top-left pixels
    }

    static CropPlan Plan(
        int width, int height, Vec2[] landmarks, float yaw, float tiltThreshold,
        Vec2? pupilLeftOverride, Vec2? pupilRightOverride)
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
            var center = new Vec2(width / 2, height / 2);
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

        float ar = (float)width / height;
        int newWidth = (int)(width * (DynafaceConstants.StyleganPupilDist / d));
        int newHeight = (int)(newWidth / ar);
        float scale = newWidth / (float)width;

        int cropX = (int)(workingLandmarks[DynafaceConstants.LmRightPupil].X * scale - DynafaceConstants.StyleganRightPupilX);
        int cropY = (int)(workingLandmarks[DynafaceConstants.LmRightPupil].Y * scale - DynafaceConstants.StyleganRightPupilY);

        return new CropPlan
        {
            Landmarks = ImageUtils.ScaleCropPoints(workingLandmarks, cropX, cropY, scale),
            FaceRotationRad = faceRotationRad,
            ResizedWidth = newWidth,
            ResizedHeight = newHeight,
            CropX = cropX,
            CropY = cropY,
        };
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
