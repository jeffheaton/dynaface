// Stateless face-processing service — orchestrates all 3 independent inference calls
// (BlazeFace, SPIGA, U^2-Net) plus pose classification, cropping, and (when lateral)
// the lateral numeric analysis, mirroring dynaface-lib's AnalyzeFace.load_image order:
//
//   BlazeFace (bbox only) -> SPIGA (landmarks+pose in original-image space)
//     -> pose classification -> frontal: StyleGAN crop / lateral: flip+pad+lateral crop
//        -> lateral only: LateralAnalyzer.Analyze (sagittal profile -> 6 landmarks)
//
// Initialize once at startup with Initialize(inference) before calling Run. Like
// dynaface-lib's load_image(), the "run lateral analysis if lateral" decision lives
// INSIDE the library, not in the caller — Run() returns a fully-populated result,
// including LateralAnalysis when applicable, with no further pose-gated calls needed.
public static class FacePipeline
{
    static BlazeFaceDetector     _blazeFace;
    static SpigaLandmarkDetector _spiga;
    static IDynafaceInference   _inference;

    public static bool IsReady => _blazeFace?.IsReady == true && _spiga?.IsReady == true;

    // True when the last detection's decoded eye keypoints passed sanity checks.
    // False means the crop was aligned from an estimated bbox — likely a bad orientation.
    public static bool LastDetectionEyesOk { get; private set; }

    public static void Initialize(IDynafaceInference inference)
    {
        _inference = inference;
        _blazeFace = new BlazeFaceDetector(inference);
        _spiga     = new SpigaLandmarkDetector(inference);
    }

    // Run the full face-detection, pose-classification, and crop pipeline.
    // flipHorizontal — true for camera captures on a real iOS device (MIRROR CONTRACT §2).
    // forcePose — FacePose.Detect auto-classifies frontal/lateral (dynaface-lib's default);
    //             Frontal/Quarter force a frontal-style crop; Lateral forces lateral
    //             classification (still auto-resolving left/right facing from yaw).
    // tiltThreshold — off (-1) by default, matching dynaface-lib; only frontal crops use it.
    // crop — false skips the frontal StyleGAN crop, returning the working image and
    //        original-space landmarks (pix2mm still computed). Mirrors dynaface-lib's
    //        load_image(crop=False), including its quirk that the LATERAL branch
    //        always crops regardless of this flag.
    // Returns null if no face is detected or a model isn't ready.
    public static FacePipelineResult? Run(
        FaceImage photo, int rotationAngle, bool flipHorizontal,
        FacePose forcePose = FacePose.Detect,
        float tiltThreshold = DynafaceConstants.DefaultTiltThreshold,
        bool crop = true)
    {
        LastDetectionEyesOk = false;
        if (!IsReady || !photo.IsValid) return null;

        FaceImage working = photo;
        if (flipHorizontal)
        {
            var flippedPixels = (Rgba32[])photo.Pixels.Clone();
            ImageUtils.FlipHorizontal(flippedPixels, photo.Width, photo.Height);
            working = new FaceImage(flippedPixels, photo.Width, photo.Height);
        }

        var initial = DetectBboxAndLandmarks(working, rotationAngle);
        if (initial == null) return null;
        var (landmarks, headPose) = initial.Value;

        bool isLateral, facingLeft;
        FacePose resolvedPose;
        if (forcePose == FacePose.Detect)
        {
            (isLateral, facingLeft) = PoseClassifier.IsLateral(headPose, landmarks, working.Width);
            resolvedPose = isLateral ? FacePose.Lateral : FacePose.Frontal;
        }
        else if (forcePose == FacePose.Frontal || forcePose == FacePose.Quarter)
        {
            isLateral = false;
            facingLeft = false;
            resolvedPose = forcePose;
        }
        else
        {
            (isLateral, facingLeft) = PoseClassifier.ForceLateral(headPose, landmarks, working.Width);
            resolvedPose = FacePose.Lateral;
        }

        return isLateral
            ? RunLateral(working, landmarks, headPose, facingLeft, resolvedPose)
            : RunFrontal(working, landmarks, headPose, resolvedPose, tiltThreshold, crop);
    }

    static FacePipelineResult? RunFrontal(
        FaceImage working, Vec2[] landmarks, float[] headPose, FacePose resolvedPose,
        float tiltThreshold, bool crop)
    {
        // Matches dynaface-lib's call order exactly: calc_pd() (from the ORIGINAL,
        // pre-crop landmarks) happens before crop_stylegan() rescales everything —
        // Pix2mm is NOT recomputed from the post-crop landmarks, even though their
        // pupil distance ends up close to (but not exactly) the 260px crop target.
        float pupillaryDistance = ComputePupillaryDistance(landmarks);
        float pix2mm = pupillaryDistance > 0f
            ? DynafaceConfig.PupilDistMm / pupillaryDistance
            : DynafaceConfig.PupilDistMm / 256f;

        FaceImage outImage      = working;
        Vec2[]    outLandmarks  = landmarks;
        float?    faceRotationRad = null;
        if (crop)
        {
            float yaw = headPose != null && headPose.Length > 0 ? headPose[0] : 0f;
            (outImage, outLandmarks, faceRotationRad) =
                StyleGanCropper.Crop(working, landmarks, yaw, tiltThreshold);
        }

        return new FacePipelineResult
        {
            AlignedCrop        = outImage,
            Wflw98             = outLandmarks,
            HeadPose           = headPose,
            Pose               = resolvedPose,
            IsLateral          = false,
            Flipped            = false,
            Pix2mm             = pix2mm,
            PupillaryDistance  = pupillaryDistance,
            FaceRotationRad    = faceRotationRad,
        };
    }

    static FacePipelineResult? RunLateral(
        FaceImage working, Vec2[] landmarks, float[] headPose, bool facingLeft, FacePose resolvedPose)
    {
        bool flipped = !facingLeft;
        FaceImage source     = working;
        Vec2[]    srcLandmarks = landmarks;
        float[]   srcHeadPose  = headPose;

        if (flipped)
        {
            var mirroredPixels = (Rgba32[])working.Pixels.Clone();
            ImageUtils.FlipHorizontal(mirroredPixels, working.Width, working.Height);
            source = new FaceImage(mirroredPixels, working.Width, working.Height);

            // dynaface-lib re-runs full detection (bbox + landmarks) on the flipped
            // image rather than just remapping coordinates — do the same.
            var redetected = DetectBboxAndLandmarks(source, 0);
            if (redetected == null) return null;
            (srcLandmarks, srcHeadPose) = redetected.Value;
        }

        var (padded, paddedLandmarks) = PadCanvasWidth(source, srcLandmarks, 1.5f);
        var (crop, cropLandmarks) = LateralCropper.Crop(padded, paddedLandmarks);

        // Matches dynaface-lib's load_image(), which calls analyze_lateral() itself
        // once lateral_pos is known — the caller never has to branch on IsLateral.
        LateralAnalyzer.Result? lateralAnalysis = LateralAnalyzer.Analyze(_inference, crop, cropLandmarks);

        // Also matches load_image(): the sagittal chart is composited onto the
        // image inside the library (_overlay_lateral_analysis) before any measures
        // draw, so the returned crop already includes it.
        if (lateralAnalysis != null)
            LateralChartRenderer.Overlay(crop, lateralAnalysis.Value);

        return new FacePipelineResult
        {
            AlignedCrop       = crop,
            Wflw98            = cropLandmarks,
            HeadPose          = srcHeadPose,
            Pose              = resolvedPose,
            IsLateral         = true,
            Flipped           = flipped,
            Pix2mm            = DynafaceConstants.LateralPix2mm,
            PupillaryDistance = 0f, // dynaface-lib never calls calc_pd() in the lateral branch
            FaceRotationRad   = null,
            LateralAnalysis   = lateralAnalysis,
        };
    }

    // Runs a probe detection to determine whether a landscape image needs 90° rotation.
    // Returns 0 if the image orientation appears correct, 90 otherwise.
    public static int SuggestRotation(FaceImage image)
    {
        if (!IsReady) return 0;
        _blazeFace.TryDetectBbox(image, 0, out _, out _);
        return _blazeFace.LastDetectionEyesOk ? 0 : 90;
    }

    static (Vec2[] landmarks, float[] pose)? DetectBboxAndLandmarks(FaceImage image, int rotationAngle)
    {
        if (!_blazeFace.TryDetectBbox(image, rotationAngle, out var bbox, out _))
        {
            LastDetectionEyesOk = _blazeFace.LastDetectionEyesOk;
            return null;
        }
        LastDetectionEyesOk = _blazeFace.LastDetectionEyesOk;
        return _spiga.Detect(image, bbox);
    }

    // Adds columns on the left/right (black-filled) so the canvas is 1.5x its original
    // width, shifting landmark X accordingly. Orientation-agnostic (only inserts
    // columns, doesn't touch row order), so no top-left/bottom-left flip is needed here.
    static (FaceImage image, Vec2[] landmarks) PadCanvasWidth(FaceImage photo, Vec2[] landmarks, float widthMultiplier)
    {
        int w = photo.Width, h = photo.Height;
        int newW = MathHelpers.RoundToInt(w * widthMultiplier);
        int totalPad = MathHelpers.Max(newW - w, 0);
        if (totalPad <= 0) return (photo, landmarks);

        int padLeft = totalPad / 2;

        var padded = new Rgba32[newW * h];
        var black = DynafaceConstants.FillColorBlack;
        for (int i = 0; i < padded.Length; i++) padded[i] = black;

        for (int y = 0; y < h; y++)
            System.Array.Copy(photo.Pixels, y * w, padded, y * newW + padLeft, w);

        var newLandmarks = new Vec2[landmarks.Length];
        for (int i = 0; i < landmarks.Length; i++)
            newLandmarks[i] = new Vec2(landmarks[i].X + padLeft, landmarks[i].Y);

        return (new FaceImage(padded, newW, h), newLandmarks);
    }

    // Pixel distance between the two pupil landmarks (dynaface-lib's util_calc_pd's
    // first return value). 0 if landmarks are missing/too short (shouldn't happen in
    // practice — SPIGA always returns 98 points).
    static float ComputePupillaryDistance(Vec2[] landmarks)
    {
        if (landmarks != null && landmarks.Length > DynafaceConstants.LmLeftPupil)
            return Vec2.Distance(
                landmarks[DynafaceConstants.LmRightPupil], landmarks[DynafaceConstants.LmLeftPupil]);
        return 0f;
    }
}

public struct FacePipelineResult
{
    public FaceImage AlignedCrop;       // final 1024x1024 crop (StyleGAN or lateral)
    public Vec2[]    Wflw98;            // 98 landmarks, remapped into AlignedCrop's pixel space
    public float[]   HeadPose;          // raw 6-value [yaw,pitch,roll,tx,ty,tz] passthrough
    public FacePose  Pose;
    public bool      IsLateral;
    public bool      Flipped;           // true if lateral and originally facing right
    public float     Pix2mm;
    public float     PupillaryDistance; // pre-crop pupil pixel distance; 0 in lateral mode
    public float?    FaceRotationRad;   // null unless tilt correction triggered (frontal only)
    public LateralAnalyzer.Result? LateralAnalysis; // lateral only; null if IsLateral is false
                                                      // or if analysis failed (no sagittal profile)
}
