// Stateless face-processing service — orchestrates all 3 independent inference calls
// (BlazeFace, SPIGA, U^2-Net) plus pose classification, cropping, (when lateral) the
// lateral numeric analysis and (when three-quarter) the far cheek's outline, mirroring
// dynaface-lib's AnalyzeFace.load_image order:
//
//   BlazeFace (bbox only) -> SPIGA (landmarks+pose in original-image space)
//     -> pose classification -> frontal: StyleGAN crop / quarter: centered band crop /
//        lateral: flip+pad+lateral crop
//        -> lateral only: LateralAnalyzer.Analyze (sagittal profile -> 6 landmarks)
//        -> quarter only: U^2-Net mask -> CheekOutline.TraceFarCheek (far cheek's outline)
//
// Initialize once at startup with Initialize(inference) before calling Run. Like
// dynaface-lib's load_image(), the "run lateral analysis if lateral" and "trace the far
// cheek if quarter" decisions live INSIDE the library, not in the caller — Run() returns
// a fully-populated result, including LateralAnalysis or the cheek outline when
// applicable, with no further pose-gated calls needed.
public static class FacePipeline
{
    static BlazeFaceDetector _blazeFace;
    static SpigaLandmarkDetector _spiga;
    static IDynafaceInference _inference;

    public static bool IsReady => _blazeFace?.IsReady == true && _spiga?.IsReady == true;

    // True when the last detection's decoded eye keypoints passed sanity checks.
    // False means the crop was aligned from an estimated bbox — likely a bad orientation.
    public static bool LastDetectionEyesOk { get; private set; }

    public static void Initialize(IDynafaceInference inference)
    {
        _inference = inference;
        _blazeFace = new BlazeFaceDetector(inference);
        _spiga = new SpigaLandmarkDetector(inference);
    }

    // Run the full face-detection, pose-classification, and crop pipeline.
    // flipHorizontal — true for camera captures on a real iOS device (MIRROR CONTRACT §2).
    // forcePose — FacePose.Detect auto-classifies frontal/quarter/lateral (dynaface-lib's
    //             default; quarter only with DynafaceConfig.AutoQuarter on, beside
    //             AutoLateral: see Cheek.ClassifyQuarter); Frontal forces the frontal
    //             crop; Quarter takes the quarter path, with the far cheek read from the
    //             landmarks; Lateral forces lateral classification (still auto-resolving
    //             left/right facing from yaw).
    // tiltThreshold — off (-1) by default, matching dynaface-lib; only frontal crops use it.
    // crop — false skips the frontal StyleGAN crop, returning the working image and
    //        original-space landmarks (pix2mm still computed). Mirrors dynaface-lib's
    //        load_image(crop=False), including its quirk that the LATERAL and QUARTER
    //        branches always crop regardless of this flag.
    // previousPose — the pose of a video's previous kept frame, or null for a still
    //        image or the first frame; VideoAnalyzer passes it. The quarter classification
    //        holds a frame in that view a little past its yaw window
    //        (Cheek.QuarterYawHold), so a turn at the window's edge keeps one view
    //        instead of alternating crops from frame to frame. Mirrors load_image's
    //        previous_pose.
    // Returns null if no face is detected or a model isn't ready.
    public static FacePipelineResult? Run(
        FaceImage photo, int rotationAngle, bool flipHorizontal,
        FacePose forcePose = FacePose.Detect,
        float tiltThreshold = DynafaceConstants.DefaultTiltThreshold,
        bool crop = true,
        FacePose? previousPose = null)
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

        var view = ResolvePose(forcePose, landmarks, headPose, working.Width, previousPose);
        if (view.IsLateral)
            return RunLateral(working, landmarks, headPose, view.FacingLeft, view.Pose);
        if (view.IsQuarter)
            return RunQuarter(working, landmarks, headPose, view.FarLeft, view.Pose,
                image => SagittalProfile.ForegroundMask(_inference, image));
        return RunFrontal(working, landmarks, headPose, view.Pose, tiltThreshold, crop);
    }

    // The view a face's raw landmarks and head pose (before any crop or flip) give:
    // its pose, whether it's a profile and which way it faces, and whether it's a
    // quarter face and which cheek is the far one. Mirrors load_image()'s pose block.
    // Detecting, the quarter classification comes first: a 35 to 55 degree turn is a
    // quarter face, and only a nose past the far contour is left to the profile test.
    // When that declines the face too, the turn is still a quarter face (its cheek.nose
    // flags it), never a frontal one, whose millimeter measures would run on a turned
    // head. A forced Quarter reads the far side from the landmarks; a forced Frontal is
    // frontal; any other forced pose is a profile.
    internal static (FacePose Pose, bool IsLateral, bool FacingLeft, bool IsQuarter, bool FarLeft) ResolvePose(
        FacePose forcePose, Vec2[] landmarks, float[] headPose, int imageWidth, FacePose? previousPose)
    {
        bool previousQuarter = previousPose == FacePose.Quarter;
        if (forcePose == FacePose.Detect)
        {
            var (isQuarter, farLeft) = PoseClassifier.IsQuarter(headPose, landmarks, previousQuarter);
            bool isLateral = false, facingLeft = false;
            if (!isQuarter)
            {
                (isLateral, facingLeft) = PoseClassifier.IsLateral(headPose, landmarks, imageWidth);
                if (!isLateral)
                    (isQuarter, farLeft) = PoseClassifier.IsQuarter(
                        headPose, landmarks, previousQuarter, requireMargin: false);
            }
            if (isQuarter) return (FacePose.Quarter, false, false, true, farLeft);
            if (isLateral) return (FacePose.Lateral, true, facingLeft, false, false);
            return (FacePose.Frontal, false, false, false, false);
        }
        if (forcePose == FacePose.Quarter)
        {
            var (isQuarter, farLeft) = PoseClassifier.ForceQuarter(headPose, landmarks);
            return (FacePose.Quarter, false, false, isQuarter, farLeft);
        }
        if (forcePose == FacePose.Frontal)
            return (FacePose.Frontal, false, false, false, false);
        var (lateral, left) = PoseClassifier.ForceLateral(headPose, landmarks, imageWidth);
        return (FacePose.Lateral, lateral, left, false, false);
    }

    internal static FacePipelineResult? RunFrontal(
        FaceImage working, Vec2[] landmarks, float[] headPose, FacePose resolvedPose,
        float tiltThreshold, bool crop)
    {
        FaceImage outImage = working;
        Vec2[] outLandmarks = landmarks;
        float? faceRotationRad = null;
        if (crop)
        {
            float yaw = headPose != null && headPose.Length > 0 ? headPose[0] : 0f;
            (outImage, outLandmarks, faceRotationRad) =
                StyleGanCropper.Crop(working, landmarks, yaw, tiltThreshold);
        }

        // Matches dynaface-lib's call order: calc_pd() runs after crop_stylegan(), so
        // Pix2mm is measured in the same (post-crop) pixel space as the landmarks
        // every measure multiplies it by. Coincident pupils give no scale: NaN.
        float pupilDistMm = DynafaceConfig.PupilDistMm;
        var (pupillaryDistance, pix2mm) = FaceMeasureContext.CalcPd(outLandmarks, pupilDistMm);

        return new FacePipelineResult
        {
            AlignedCrop = outImage,
            Wflw98 = outLandmarks,
            HeadPose = headPose,
            Pose = resolvedPose,
            IsLateral = false,
            Flipped = false,
            Pix2mm = pix2mm,
            PupillaryDistance = pupillaryDistance,
            PupilDistMm = pupilDistMm,
            FaceRotationRad = faceRotationRad,
        };
    }

    static FacePipelineResult? RunLateral(
        FaceImage working, Vec2[] landmarks, float[] headPose, bool facingLeft, FacePose resolvedPose)
    {
        bool flipped = !facingLeft;
        FaceImage source = working;
        Vec2[] srcLandmarks = landmarks;

        if (flipped)
        {
            var mirroredPixels = (Rgba32[])working.Pixels.Clone();
            ImageUtils.FlipHorizontal(mirroredPixels, working.Width, working.Height);
            source = new FaceImage(mirroredPixels, working.Width, working.Height);

            // dynaface-lib re-runs full detection (bbox + landmarks) on the flipped
            // image rather than just remapping coordinates — do the same. Its head pose
            // is the mirror image's, so it's dropped: like load_image(), the result
            // reports the pose found in the photo as given (yaw > 0, facing right).
            var redetected = DetectBboxAndLandmarks(source, 0);
            if (redetected == null) return null;
            srcLandmarks = redetected.Value.landmarks;
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
            AlignedCrop = crop,
            Wflw98 = cropLandmarks,
            HeadPose = headPose,
            Pose = resolvedPose,
            IsLateral = true,
            Flipped = flipped,
            Pix2mm = DynafaceConstants.LateralPix2mm,
            PupillaryDistance = 0f, // dynaface-lib never calls calc_pd() in the lateral branch
            PupilDistMm = DynafaceConfig.PupilDistMm,
            FaceRotationRad = null,
            LateralAnalysis = lateralAnalysis,
        };
    }

    // The quarter branch of load_image(): always cropped, like a profile, with the
    // centered band crop (LateralCropper.Crop(center: true)) that keeps the chin in frame
    // and crops the two mirrored turns of one face alike; never flipped or straightened
    // (roll is handled inside the cheek constructions), and without the profile's
    // side-bar widening, which exists for the sagittal chart. The scale is the pupil
    // scale measured on the cropped landmarks, as for a frontal face, foreshortened by
    // the turn: no millimeter is defined on a turned head, so the millimeter measures
    // don't suit the view. Then the far cheek's outline is traced on the crop from the
    // background model's mask (foregroundMask; SagittalProfile.ForegroundMask in Run),
    // which the result keeps for FacePipelineResult.RetraceFarCheek.
    internal static FacePipelineResult RunQuarter(
        FaceImage working, Vec2[] landmarks, float[] headPose, bool farLeft, FacePose resolvedPose,
        System.Func<FaceImage, (bool[,] Foreground, byte[,] Alpha)> foregroundMask)
    {
        var (crop, cropLandmarks) = LateralCropper.Crop(working, landmarks, center: true);

        // After the crop, so Pix2mm matches the (rescaled) landmarks, as for a frontal face.
        float pupilDistMm = DynafaceConfig.PupilDistMm;
        var (pupillaryDistance, pix2mm) = FaceMeasureContext.CalcPd(cropLandmarks, pupilDistMm);

        // The far cheek's outline, the one image-dependent step of the cheek landmarks:
        // traced on the cropped image, as a profile's outline is.
        var tracer = FarCheekTracer.FromImage(crop, foregroundMask);
        var (outlineX, outlineY, outlineSrc) = tracer.Trace(cropLandmarks, farLeft);

        return new FacePipelineResult
        {
            AlignedCrop = crop,
            Wflw98 = cropLandmarks,
            HeadPose = headPose,
            Pose = resolvedPose,
            IsLateral = false,
            Flipped = false,
            Pix2mm = pix2mm,
            PupillaryDistance = pupillaryDistance,
            PupilDistMm = pupilDistMm,
            FaceRotationRad = null, // only the frontal crop straightens
            IsQuarter = true,
            FarLeft = farLeft,
            CheekOutlineX = outlineX,
            CheekOutlineY = outlineY,
            CheekOutlineSrc = outlineSrc,
            FarCheekMask = tracer,
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
}

public struct FacePipelineResult
{
    public FaceImage AlignedCrop;       // final 1024x1024 crop (StyleGAN, quarter band or lateral)
    public Vec2[] Wflw98;            // 98 landmarks, remapped into AlignedCrop's pixel space
    public float[] HeadPose;          // raw 6-value [yaw,pitch,roll,tx,ty,tz] passthrough
    public FacePose Pose;
    public bool IsLateral;
    public bool Flipped;           // true if lateral and originally facing right (never for a quarter face)
    public float Pix2mm;            // measured on Wflw98; build a FaceMeasureContext from Wflw98 and
                                    // PupilDistMm rather than reusing it with other landmarks
    public float PupillaryDistance; // pupil pixel distance in AlignedCrop's space; 0 in lateral mode
    public float PupilDistMm;       // the pupillary distance (mm) the run assumed: DynafaceConfig.PupilDistMm
                                    // at the time. Save it with the landmarks to measure them the same way later.
    public float? FaceRotationRad;   // null unless tilt correction triggered (frontal only)
    public LateralAnalyzer.Result? LateralAnalysis; // lateral only; null if IsLateral is false
                                                    // or if analysis failed (no sagittal profile)

    // A three-quarter (quarter) face, the counterpart of IsLateral: Pose is
    // FacePose.Quarter, cropped with the centered band crop, never flipped. Mirrors
    // dynaface-lib's AnalyzeFace.quarter. Detected only with DynafaceConfig.AutoQuarter
    // on (beside AutoLateral), or forced with FacePose.Quarter.
    public bool IsQuarter;
    // Quarter only: whether the far (measured) cheek is the subject's left, the photo
    // taken from the subject's right. Mirrors AnalyzeFace.far_left; false otherwise.
    public bool FarLeft;
    // Quarter only: the far cheek's traced outline (CheekOutline.TraceFarCheek), its
    // sub-pixel x in each row of CheekOutlineY, from the far outer eye corner's row or
    // the far contour's top row, whichever is higher, to the chin row, in AlignedCrop's
    // top-left pixels, and each row's source (CheekOutline.OutlineSrc*). Mirrors
    // AnalyzeFace.cheek_outline_x/_y/_src; null for any other face. Pass them to a
    // context with FaceMeasureContext.SetQuarterView (FaceMeasureContext.FromResult and
    // FaceAnalysisPersistence do).
    public float[] CheekOutlineX;
    public int[] CheekOutlineY;
    public byte[] CheekOutlineSrc;

    // What that outline was traced from (the mask, its alpha and the gray image of
    // AlignedCrop), for RetraceFarCheek; null for any other face. Never saved.
    internal FarCheekTracer FarCheekMask;

    // Traces the far cheek's outline again on other landmarks of this quarter face (in
    // AlignedCrop's pixels, such as a video's smoothed ones), from the mask this run's
    // background model gave, so its rows, seed and nose rows belong to the landmarks it
    // will be measured with, at the cost of the trace alone, never a second model pass.
    // Returns the outline (x, rows, sources), or null when there's nothing to trace from
    // (a frontal face or a profile). Leaves this result as it is. Mirrors dynaface-lib's
    // AnalyzeFace.retrace_far_cheek(), which VideoAnalyzer calls after smoothing.
    public (float[] X, int[] Y, byte[] Src)? RetraceFarCheek(Vec2[] landmarks)
    {
        if (FarCheekMask == null || !IsQuarter) return null;
        return FarCheekMask.Trace(landmarks, FarLeft);
    }
}
