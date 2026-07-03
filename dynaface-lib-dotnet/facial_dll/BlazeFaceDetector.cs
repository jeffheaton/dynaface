using System.Collections.Generic;

// PIPELINE STAGE 1 — Face bbox detection.
//
// Accepts any FaceImage + rotation angle, runs the BlazeFace short-range detector via
// IDynafaceInference, and returns a face bounding box in `photo`'s own raw pixel-index
// space (same indexing as photo.Pixels, i.e. bottom-left/Unity order) — matching
// dynaface-lib's dynaface_onnx.py `detect_face()` contract, plus rotation-retry and an
// eye-keypoint sanity heuristic that Python has no equivalent of (both .NET-only
// value-adds, kept from the previous version of this class).
//
// This class no longer produces an aligned crop itself — that's now
// StyleGanCropper/LateralCropper's job (see FacePipeline), after SPIGA landmarks are
// available, matching dynaface-lib's own pipeline order.
public class BlazeFaceDetector
{
    const int   InputSize      = 128;
    const int   AnchorCount    = 896;
    const float ScoreThreshold = 0.5f;
    const float NmsIouThresh   = 0.3f;

    readonly IDynafaceInference _runner;
    float[,] _anchors;

    public bool IsReady => _runner?.IsReady == true && _anchors != null;

    // True when the winning detection's decoded eye keypoints passed sanity checks.
    // False means the keypoints look implausible (e.g. a sideways/upside-down image) —
    // used by FacePipeline.SuggestRotation and the console's rotation-retry loop.
    public bool LastDetectionEyesOk { get; private set; }

    public BlazeFaceDetector(IDynafaceInference runner)
    {
        _runner = runner;
        GenerateAnchors();
    }

    void GenerateAnchors()
    {
        _anchors = new float[AnchorCount, 4];
        int index = 0;

        AddAnchorGrid(ref index, gridSize: 16, anchorsPerCell: 2);
        AddAnchorGrid(ref index, gridSize: 8,  anchorsPerCell: 6);
    }

    void AddAnchorGrid(ref int index, int gridSize, int anchorsPerCell)
    {
        for (int row = 0; row < gridSize; row++)
        for (int col = 0; col < gridSize; col++)
        for (int anchor = 0; anchor < anchorsPerCell; anchor++)
        {
            _anchors[index, 0] = (col + 0.5f) / gridSize;
            _anchors[index, 1] = (row + 0.5f) / gridSize;
            _anchors[index, 2] = 1f;
            _anchors[index, 3] = 1f;
            index++;
        }
    }

    // Detects the highest-confidence face and returns its bbox (x,y,w,h) in `photo`'s
    // own raw pixel-index space. Returns false if no face scores above threshold.
    public bool TryDetectBbox(FaceImage photo, int videoRotationAngle,
        out (int x, int y, int w, int h) bbox, out float score)
    {
        bbox = default;
        score = 0f;
        LastDetectionEyesOk = false;
        if (!IsReady || !photo.IsValid) return false;

        int rotation = NormalizeRotation(videoRotationAngle);
        int N = InputSize;
        int W = photo.Width;
        int H = photo.Height;

        bool  sideways = rotation == 90 || rotation == 270;
        float uprightW = sideways ? H : W;
        float uprightH = sideways ? W : H;

        int squarePx = MathHelpers.Min(W, H);
        float rawCropOffX   = (W - squarePx) * 0.5f / W;
        float rawCropOffY   = (H - squarePx) * 0.5f / H;
        float rawCropScaleX = (float)squarePx / W;
        float rawCropScaleY = (float)squarePx / H;

        float uprightCropOffX   = sideways ? rawCropOffY   : rawCropOffX;
        float uprightCropOffY   = sideways ? rawCropOffX   : rawCropOffY;
        float uprightCropScaleX = sideways ? rawCropScaleY : rawCropScaleX;
        float uprightCropScaleY = sideways ? rawCropScaleX : rawCropScaleY;

        int roiX = (int)((W - squarePx) * 0.5f);
        int roiY = (int)((H - squarePx) * 0.5f);
        Rgba32[] modelPixels = ImageUtils.ResizeBilinear(
            photo.Pixels, W, H,
            roiX, roiY, squarePx, squarePx,
            N, N);

        float[] inputData = new float[N * N * 3];
        for (int tensorY = 0; tensorY < N; tensorY++)
        for (int tensorX = 0; tensorX < N; tensorX++)
        {
            int sourceY, sourceX;
            switch (rotation)
            {
                case 90:  sourceY = N - 1 - tensorX; sourceX = tensorY;         break;
                case 180: sourceY = tensorY;          sourceX = N - 1 - tensorX; break;
                case 270: sourceY = tensorX;          sourceX = N - 1 - tensorY; break;
                default:  sourceY = N - 1 - tensorY; sourceX = tensorX;          break;
            }

            Rgba32 color = modelPixels[sourceY * N + sourceX];
            int dest = (tensorY * N + tensorX) * 3;
            inputData[dest]     = color.R / 255f * 2f - 1f;
            inputData[dest + 1] = color.G / 255f * 2f - 1f;
            inputData[dest + 2] = color.B / 255f * 2f - 1f;
        }

        var runResult = _runner.RunBlazeFace(inputData);
        if (runResult == null) return false;

        var (regsArr, scoresArr) = runResult.Value;
        int totalRegs   = regsArr.Length;
        int totalScores = scoresArr.Length;
        if (totalRegs % AnchorCount != 0 || totalScores < AnchorCount) return false;

        int regStride = totalRegs / AnchorCount;
        if (regStride < 4) return false;

        // Decode every anchor scoring above threshold (needed for NMS — the single
        // best-scoring anchor doesn't always survive NMS as the best surviving box
        // on images with more than one face-like region).
        var anchorIdxOf = new List<int>();
        var boxX0 = new List<float>();
        var boxY0 = new List<float>();
        var boxX1 = new List<float>();
        var boxY1 = new List<float>();
        var boxScore = new List<float>();

        for (int i = 0; i < AnchorCount; i++)
        {
            float s = Sigmoid(scoresArr[i]);
            if (s < ScoreThreshold) continue;

            int baseOffset = i * regStride;
            float anchorX128 = _anchors[i, 0] * N;
            float anchorY128 = _anchors[i, 1] * N;

            float centerX128  = regsArr[baseOffset + 0] + anchorX128;
            float centerY128  = regsArr[baseOffset + 1] + anchorY128;
            float boxWidth128 = regsArr[baseOffset + 2];
            float boxHeight128 = regsArr[baseOffset + 3];

            if (!IsFinite(centerX128) || !IsFinite(centerY128) ||
                !IsFinite(boxWidth128) || !IsFinite(boxHeight128) ||
                boxWidth128 <= 0f || boxHeight128 <= 0f)
                continue;

            float halfW = boxWidth128 * 0.5f, halfH = boxHeight128 * 0.5f;

            anchorIdxOf.Add(i);
            boxX0.Add((centerX128 - halfW) / N);
            boxY0.Add((centerY128 - halfH) / N);
            boxX1.Add((centerX128 + halfW) / N);
            boxY1.Add((centerY128 + halfH) / N);
            boxScore.Add(s);
        }
        if (anchorIdxOf.Count == 0) return false;

        var keep = GreedyNms(boxX0, boxY0, boxX1, boxY1, boxScore, NmsIouThresh);
        if (keep.Count == 0) return false;

        int best = keep[0]; // highest score surviving NMS
        int bestAnchorIdx = anchorIdxOf[best];

        float bx1 = uprightCropOffX + boxX0[best] * uprightCropScaleX;
        float by1 = uprightCropOffY + boxY0[best] * uprightCropScaleY;
        float bx2 = uprightCropOffX + boxX1[best] * uprightCropScaleX;
        float by2 = uprightCropOffY + boxY1[best] * uprightCropScaleY;

        LastDetectionEyesOk = CheckEyeKeypoints(
            regsArr, bestAnchorIdx, regStride, N,
            uprightCropOffX, uprightCropOffY, uprightCropScaleX, uprightCropScaleY,
            uprightW, uprightH, bx1, by1, bx2, by2);

        // Map the (upright, normalized) box corners back to photo's own raw
        // pixel-index space via the existing rotation-aware inverse mapping.
        Vec2 c0 = UprightNormalizedToRawPixel(bx1, by1, rotation, W, H);
        Vec2 c1 = UprightNormalizedToRawPixel(bx2, by2, rotation, W, H);

        float xMin = MathHelpers.Min(c0.X, c1.X), xMax = MathHelpers.Max(c0.X, c1.X);
        float yMin = MathHelpers.Min(c0.Y, c1.Y), yMax = MathHelpers.Max(c0.Y, c1.Y);

        int ix0 = MathHelpers.Clamp(MathHelpers.RoundToInt(xMin), 0, W);
        int iy0 = MathHelpers.Clamp(MathHelpers.RoundToInt(yMin), 0, H);
        int ix1 = MathHelpers.Clamp(MathHelpers.RoundToInt(xMax), 0, W);
        int iy1 = MathHelpers.Clamp(MathHelpers.RoundToInt(yMax), 0, H);
        if (ix1 <= ix0 || iy1 <= iy0) return false;

        bbox  = (ix0, iy0, ix1 - ix0, iy1 - iy0);
        score = boxScore[best];
        return true;
    }

    // Eye-keypoint sanity heuristic — evaluated against the winning anchor's own
    // 6-keypoint regressor output, used only as a signal (LastDetectionEyesOk),
    // never to build a crop. Unchanged logic from the previous version of this class.
    bool CheckEyeKeypoints(
        float[] regsArr, int bestAnchorIdx, int regStride, int N,
        float uprightCropOffX, float uprightCropOffY, float uprightCropScaleX, float uprightCropScaleY,
        float uprightW, float uprightH, float bx1, float by1, float bx2, float by2)
    {
        if (regStride < 16) return false;

        int baseOffset = bestAnchorIdx * regStride;
        float anchorX128 = _anchors[bestAnchorIdx, 0] * N;
        float anchorY128 = _anchors[bestAnchorIdx, 1] * N;

        float eye0X128 = regsArr[baseOffset + 4] + anchorX128;
        float eye0Y128 = regsArr[baseOffset + 5] + anchorY128;
        float eye1X128 = regsArr[baseOffset + 6] + anchorX128;
        float eye1Y128 = regsArr[baseOffset + 7] + anchorY128;

        Vec2 eye0 = ModelPointToUprightNormalized(eye0X128, eye0Y128, N,
            uprightCropOffX, uprightCropOffY, uprightCropScaleX, uprightCropScaleY);
        Vec2 eye1 = ModelPointToUprightNormalized(eye1X128, eye1Y128, N,
            uprightCropOffX, uprightCropOffY, uprightCropScaleX, uprightCropScaleY);

        Vec2 rightEye, leftEye;
        if (eye0.X <= eye1.X) { rightEye = eye0; leftEye = eye1; }
        else                  { rightEye = eye1; leftEye = eye0; }

        float faceW = bx2 - bx1, faceH = by2 - by1;
        Vec2  rightEyePx    = new Vec2(rightEye.X * uprightW, rightEye.Y * uprightH);
        Vec2  leftEyePx     = new Vec2(leftEye.X  * uprightW, leftEye.Y  * uprightH);
        float eyeDistancePx = Vec2.Distance(rightEyePx, leftEyePx);

        float marginX   = faceW * 0.35f;
        float marginY   = faceH * 0.35f;
        bool  withinBox =
            rightEye.X > bx1 - marginX && rightEye.X < bx2 + marginX &&
            leftEye.X  > bx1 - marginX && leftEye.X  < bx2 + marginX &&
            rightEye.Y > by1 - marginY && rightEye.Y < by2 + marginY &&
            leftEye.Y  > by1 - marginY && leftEye.Y  < by2 + marginY;

        float eyeBoxTopLimit  = by1 + faceH * 0.70f;
        bool  eyesInUpperFace = rightEye.Y < eyeBoxTopLimit && leftEye.Y < eyeBoxTopLimit;
        float boxWidthPx      = MathHelpers.Max(1f, faceW * uprightW);
        bool  eyeDistanceOk   = eyeDistancePx >= MathHelpers.Max(6f, boxWidthPx * 0.12f) &&
                                eyeDistancePx <= boxWidthPx * 0.90f;

        return withinBox && eyesInUpperFace && eyeDistanceOk;
    }

    static Vec2 ModelPointToUprightNormalized(
        float x128, float y128, int modelSize,
        float cropOffX, float cropOffY, float cropScaleX, float cropScaleY)
    {
        return new Vec2(
            cropOffX + (x128 / modelSize) * cropScaleX,
            cropOffY + (y128 / modelSize) * cropScaleY);
    }

    static Vec2 UprightNormalizedToRawPixel(
        float uprightX, float uprightY,
        int rotation, int width, int height)
    {
        float maxX = MathHelpers.Max(0, width  - 1);
        float maxY = MathHelpers.Max(0, height - 1);
        switch (rotation)
        {
            case 90:  return new Vec2(uprightY * maxX,        (1f - uprightX) * maxY);
            case 180: return new Vec2((1f - uprightX) * maxX, uprightY * maxY);
            case 270: return new Vec2((1f - uprightY) * maxX, uprightX * maxY);
            default:  return new Vec2(uprightX * maxX,        (1f - uprightY) * maxY);
        }
    }

    // Greedy non-max suppression; returns indices (into the input lists) to keep,
    // highest score first. Mirrors dynaface_onnx.py's _nms.
    // internal (not private) so DynafaceTests can exercise it directly.
    internal static List<int> GreedyNms(
        List<float> x0, List<float> y0, List<float> x1, List<float> y1,
        List<float> scores, float iouThresh)
    {
        int n = scores.Count;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        System.Array.Sort(order, (a, b) => scores[b].CompareTo(scores[a]));

        var areas = new float[n];
        for (int i = 0; i < n; i++)
            areas[i] = MathHelpers.Max(0f, x1[i] - x0[i]) * MathHelpers.Max(0f, y1[i] - y0[i]);

        var suppressed = new bool[n];
        var keep = new List<int>();
        for (int oi = 0; oi < n; oi++)
        {
            int i = order[oi];
            if (suppressed[i]) continue;
            keep.Add(i);

            for (int oj = oi + 1; oj < n; oj++)
            {
                int j = order[oj];
                if (suppressed[j]) continue;

                float yy1 = MathHelpers.Max(y0[i], y0[j]);
                float xx1 = MathHelpers.Max(x0[i], x0[j]);
                float yy2 = MathHelpers.Min(y1[i], y1[j]);
                float xx2 = MathHelpers.Min(x1[i], x1[j]);

                float inter = MathHelpers.Max(0f, xx2 - xx1) * MathHelpers.Max(0f, yy2 - yy1);
                float iou   = inter / (areas[i] + areas[j] - inter + 1e-9f);
                if (iou > iouThresh) suppressed[j] = true;
            }
        }
        return keep;
    }

    static float Sigmoid(float value)
    {
        value = MathHelpers.Clamp(value, -100f, 100f);
        return 1f / (1f + MathHelpers.Exp(-value));
    }

    static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    static int NormalizeRotation(int angle)
    {
        int normalized = ((angle % 360) + 360) % 360;
        if (normalized == 0 || normalized == 90 || normalized == 180 || normalized == 270)
            return normalized;
        return (MathHelpers.RoundToInt(normalized / 90f) * 90) % 360;
    }
}
