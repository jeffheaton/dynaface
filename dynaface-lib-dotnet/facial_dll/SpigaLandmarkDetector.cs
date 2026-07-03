// PIPELINE STAGE 2 — SPIGA Face Landmark + Pose Detection
//
// Given the FULL original image and a bbox from BlazeFaceDetector, builds SPIGA's own
// bbox-derived crop (pad bbox to a square of side max(w,h)*1.6, centered on the bbox,
// scaled to 256x256 — matches dynaface_onnx.py's _crop_affine/_preprocess_landmarks
// exactly), runs SPIGA, and inverse-maps the resulting 98-point landmarks back into
// the ORIGINAL image's own pixel space — matching dynaface-lib's find_landmarks()
// contract (landmarks relative to the original image, not a pre-aligned crop).
//
// Coordinate note: internally this works in `photo`'s own raw pixel-index space
// (bottom-left, same as FaceImage.Pixels) throughout the crop/affine math — the ONLY
// row-flip is the well-isolated one needed to feed SPIGA a standard y=0-at-top tensor
// (same idiom used here as before). The returned landmarks are converted to top-left
// semantic space (y=0 = visual top of `photo`) as the very last step, matching the
// coordinate contract the rest of the pipeline (pose classification, StyleGAN/lateral
// cropping, FaceMeasureContext) expects.
public class SpigaLandmarkDetector
{
    public const int NumWflwLandmarks = 98;

    const float SpigaTargetDist = 1.6f;
    const int   SpigaCropSize   = 256;

    readonly IDynafaceInference _runner;

    public bool IsReady => _runner?.IsReady == true;

    public SpigaLandmarkDetector(IDynafaceInference runner)
    {
        _runner = runner;
    }

    // bbox: (x,y,w,h) in `photo`'s own raw pixel-index space, e.g. from
    // BlazeFaceDetector.TryDetectBbox.
    // Returns (landmarks in photo's top-left pixel space, pose[6] raw
    // [yaw,pitch,roll,tx,ty,tz] passthrough), or null on failure.
    public (Vec2[] landmarks, float[] pose)? Detect(FaceImage photo, (int x, int y, int w, int h) bbox)
    {
        if (!IsReady || !photo.IsValid) return null;

        int width  = photo.Width;
        int height = photo.Height;

        // Forward affine: photo's raw pixel-index space -> 256x256 crop, itself still
        // in raw/array-index orientation (no top/bottom semantics attached here).
        float side   = MathHelpers.Max(bbox.w, bbox.h) * SpigaTargetDist;
        float cx     = bbox.x + bbox.w / 2f;
        float cy     = bbox.y + bbox.h / 2f;
        float scale  = SpigaCropSize / side;
        float center = SpigaCropSize / 2f;

        float m00 = scale, m01 = 0f,   m02 = center - scale * cx;
        float m10 = 0f,    m11 = scale, m12 = center - scale * cy;

        // Nearest-neighbor + zero-fill, matching cv2.warpAffine(..., flags=INTER_NEAREST,
        // borderMode=BORDER_CONSTANT, borderValue=0) — distinct from the bilinear/white-fill
        // used by the StyleGAN/lateral croppers later in the pipeline.
        Rgba32[] cropRaw = ImageUtils.WarpAffine(
            photo.Pixels, width, height,
            m00, m01, m02, m10, m11, m12,
            SpigaCropSize, SpigaCropSize,
            bilinear: false, fillColor: default);

        // Build NCHW tensor [3,256,256]. cropRaw is still in photo's own raw/array
        // orientation; SPIGA expects y=0 at top, so flip Y here (X needs no flip).
        int n = SpigaCropSize;
        float[] tensor = new float[3 * n * n];
        for (int tensorY = 0; tensorY < n; tensorY++)
        for (int tensorX = 0; tensorX < n; tensorX++)
        {
            int readY    = n - 1 - tensorY;
            Rgba32 color = cropRaw[readY * n + tensorX];
            int idx      = tensorY * n + tensorX;
            tensor[0 * n * n + idx] = color.R / 255f;
            tensor[1 * n * n + idx] = color.G / 255f;
            tensor[2 * n * n + idx] = color.B / 255f;
        }

        var result = _runner.RunSpiga(tensor);
        if (result == null) return null;

        var (landmarksNorm, pose) = result.Value;
        if (landmarksNorm == null || landmarksNorm.Length < NumWflwLandmarks * 2) return null;

        var (i00, i01, i02, i10, i11, i12) = ImageUtils.InvertAffine(m00, m01, m02, m10, m11, m12);

        var landmarks = new Vec2[NumWflwLandmarks];
        for (int i = 0; i < NumWflwLandmarks; i++)
        {
            float modelX = landmarksNorm[i * 2]     * SpigaCropSize;
            float modelY = landmarksNorm[i * 2 + 1] * SpigaCropSize;

            // Undo the tensor's Y-flip to get back into cropRaw's own raw/array
            // orientation before inverse-mapping through the crop affine.
            float cropArrayY = (SpigaCropSize - 1) - modelY;

            float rawX = i00 * modelX + i01 * cropArrayY + i02;
            float rawY = i10 * modelX + i11 * cropArrayY + i12;

            // photo's raw/array space (bottom-left) -> top-left semantic space,
            // matching the coordinate contract the rest of the pipeline expects.
            float topLeftY = (height - 1) - rawY;
            landmarks[i] = new Vec2(rawX, topLeftY);
        }

        return (landmarks, pose);
    }
}
