using System.Collections.Generic;

// Transient context passed to each FaceMeasureBase.Calc() call.
// All coordinates are in top-left pixel space (x=0 left, y=0 top)
// to match what FaceLandmarkDetector returns. Conversion to Unity
// bottom-left happens internally before calling FaceRenderer helpers.
public class FaceMeasureContext
{
    public Rgba32[] Pixels    { get; }
    public int      Width     { get; }
    public int      Height    { get; }

    // WFLW landmarks in top-left pixel coordinates.
    public Vec2[] Landmarks { get; }

    // Millimetres per pixel. For frontal faces this must be computed from the
    // ORIGINAL (pre-StyleGAN-crop) pupil distance — see FacePipeline.RunFrontal —
    // not re-derived from these (post-crop) landmarks, so it's always supplied
    // explicitly rather than computed internally from Landmarks.
    public float Pix2mm { get; }

    // True for a lateral-view analysis. When true, LateralLandmarks holds the 6
    // points from LateralAnalyzer (Glabella/Nasion/NasalTip/Subnasal/MentoLabial/
    // Pogonion — see LateralLandmarkFinder's index constants), otherwise null.
    public bool   IsLateral        { get; }
    public Vec2[] LateralLandmarks { get; }

    // Raw SPIGA headpose passthrough [yaw, pitch, roll, tx, ty, tz] from
    // FacePipelineResult.HeadPose (degrees for yaw/pitch/roll). May be null/short
    // for contexts built without a full pipeline run (e.g. simple unit tests) —
    // consumers (MeasurePose) fall back to 0 in that case.
    public float[] HeadPose { get; }

    public readonly List<string> TextLines = new();

    // Structured numeric output (mirrors dynaface-lib's calc() -> Dict[str, Any]
    // contract) — used by measures whose result doesn't fit the TextLines sidebar
    // model, e.g. MeasureLandmarks' 194 flat landmark-coordinate fields.
    public readonly Dictionary<string, double> Values = new();

    static readonly Rgba32 MeasureColor = new Rgba32(255, 215,   0, 255); // gold arrows
    static readonly Rgba32 LineColor    = new Rgba32(200, 200, 200, 180); // faint white

    const int ImageLabelX = 20;
    int _imageLabelY = 20;

    public FaceMeasureContext(
        FaceImage photo, Vec2[] landmarks, float pix2mm,
        bool isLateral = false, Vec2[] lateralLandmarks = null, float[] headPose = null)
    {
        Width            = photo.Width;
        Height           = photo.Height;
        Pixels           = photo.Pixels; // caller must ensure this array is not shared
        Landmarks        = landmarks;
        Pix2mm           = pix2mm;
        IsLateral        = isLateral;
        LateralLandmarks = lateralLandmarks;
        HeadPose         = headPose;
    }

    // Convenience overload for simple/test scenarios that don't have a
    // FacePipelineResult on hand: auto-computes Pix2mm from the landmarks' own
    // pupil distance (frontal-only assumption), matching FacePipeline.ComputePix2mm.
    public FaceMeasureContext(FaceImage photo, Vec2[] landmarks)
        : this(photo, landmarks, ComputePix2mmFromLandmarks(landmarks))
    {
    }

    static float ComputePix2mmFromLandmarks(Vec2[] lm)
    {
        if (lm != null && lm.Length > DynafaceConstants.LmLeftPupil)
        {
            float dist = Vec2.Distance(lm[DynafaceConstants.LmRightPupil], lm[DynafaceConstants.LmLeftPupil]);
            if (dist > 1f) return DynafaceConstants.StdPupilDistMm / dist;
        }
        return DynafaceConstants.StdPupilDistMm / 256f;
    }

    // -------------------------------------------------------------------------
    // Coordinate helpers
    // -------------------------------------------------------------------------

    int BLY(float topLeftY) => MathHelpers.RoundToInt(Height - 1f - topLeftY);

    // -------------------------------------------------------------------------
    // Geometric helpers
    // -------------------------------------------------------------------------

    public (Vec2 top, Vec2 bottom) CalcBisect()
    {
        float midX = Landmarks.Length > 97
            ? (Landmarks[96].X + Landmarks[97].X) * 0.5f
            : Width * 0.5f;
        return (new Vec2(midX, 0), new Vec2(midX, Height - 1));
    }

    public float CalcFaceRotationDeg()
    {
        if (Landmarks == null || Landmarks.Length <= 97) return 0f;
        Vec2 r = Landmarks[96], l = Landmarks[97];
        return MathHelpers.Atan2(l.Y - r.Y, l.X - r.X) * MathHelpers.Rad2Deg;
    }

    // Extend a ray from start at angleDeg (0°=right, 90°=down in top-left space)
    // to the image boundary. Returns null if the start is already at the edge.
    public Vec2? LineToEdge(Vec2 start, float angleDeg)
    {
        float rad = angleDeg * MathHelpers.Deg2Rad;
        float dx  = MathHelpers.Cos(rad);
        float dy  = MathHelpers.Sin(rad);

        float tMax = float.MaxValue;
        if (dx >  1e-6f) tMax = MathHelpers.Min(tMax, (Width  - 1 - start.X) / dx);
        if (dx < -1e-6f) tMax = MathHelpers.Min(tMax, -start.X / dx);
        if (dy >  1e-6f) tMax = MathHelpers.Min(tMax, (Height - 1 - start.Y) / dy);
        if (dy < -1e-6f) tMax = MathHelpers.Min(tMax, -start.Y / dy);

        if (tMax <= 0f || tMax >= float.MaxValue) return null;
        return new Vec2(start.X + dx * tMax, start.Y + dy * tMax);
    }

    // Split a polygon by a vertical line at midX.
    // Returns (leftPoly, rightPoly) in top-left coords.
    public static (Vec2[] left, Vec2[] right) SplitPolygonByX(Vec2[] poly, float midX)
    {
        var leftPts  = new List<Vec2>();
        var rightPts = new List<Vec2>();
        int n = poly.Length;

        for (int i = 0; i < n; i++)
        {
            Vec2 a = poly[i], b = poly[(i + 1) % n];

            if (a.X <= midX) leftPts.Add(a); else rightPts.Add(a);

            bool aCross = a.X <= midX, bCross = b.X <= midX;
            if (aCross != bCross)
            {
                float t     = (midX - a.X) / (b.X - a.X);
                var   cross = new Vec2(midX, a.Y + t * (b.Y - a.Y));
                leftPts.Add(cross);
                rightPts.Add(cross);
            }
        }

        return (leftPts.ToArray(), rightPts.ToArray());
    }

    // min/max symmetry ratio: 1.0 = perfect symmetry, 0 = fully asymmetric.
    public static float SymmetryRatio(float a, float b)
    {
        float max = MathHelpers.Max(a, b), min = MathHelpers.Min(a, b);
        return max > 0f ? min / max : 1f;
    }

    // -------------------------------------------------------------------------
    // Drawing wrappers (all accept top-left coords)
    // -------------------------------------------------------------------------

    // Draws a bidirectional measurement arrow and returns the distance in mm.
    // dir controls an optional inline label drawn offset from the arrow midpoint:
    //   ""  = no label   "r" = right of midpoint   "l" = right of pt1
    //   "s" = right of pt2   "a" = left edge, midpoint shifted up
    public float Measure(Vec2 pt1, Vec2 pt2, bool render = true, Rgba32 color = default, string dir = "")
    {
        float dist = Vec2.Distance(pt1, pt2) * Pix2mm;
        if (render)
        {
            Rgba32 c = color.A == 0 ? MeasureColor : color;
            FaceRenderer.DrawArrow(Pixels, Width, Height,
                MathHelpers.RoundToInt(pt1.X), BLY(pt1.Y),
                MathHelpers.RoundToInt(pt2.X), BLY(pt2.Y),
                c, thickness: FaceRenderer.ARROW_THICKNESS);

            if (!string.IsNullOrEmpty(dir))
            {
                string txt = $"{dist:F2}mm";
                DrawImageText(CalcMeasureLabelPos(pt1, pt2, dir), txt, new Rgba32(255, 255, 255, 255));
            }
        }
        return dist;
    }

    Vec2 CalcMeasureLabelPos(Vec2 pt1, Vec2 pt2, string dir)
    {
        float midX = (pt1.X + pt2.X) * 0.5f;
        float midY = (pt1.Y + pt2.Y) * 0.5f;
        const int offset = 15;
        return dir switch
        {
            "r" => new Vec2(midX + offset, midY),
            "s" => new Vec2(pt2.X  + offset, pt2.Y),
            "a" => new Vec2(MathHelpers.Min(pt1.X, pt2.X) + offset, midY - 20),
            _   => new Vec2(pt1.X  + offset, midY),
        };
    }

    public void DrawImageText(Vec2 topLeftPos, string text, Rgba32 color)
    {
        var size = FaceRenderer.GetTextSize(text, FaceRenderer.TEXT_SIZE_MEASURE);
        int blY  = BLY(MathHelpers.RoundToInt(topLeftPos.Y) + size.Y);
        FaceRenderer.DrawText(Pixels, Width, Height,
            MathHelpers.RoundToInt(topLeftPos.X), blY,
            text, color, scale: FaceRenderer.TEXT_SIZE_MEASURE);
    }

    public void WriteImageLabel(string text, Rgba32 color = default)
    {
        Rgba32 c = color.A == 0 ? new Rgba32(255, 255, 255, 255) : color;
        DrawImageText(new Vec2(ImageLabelX, _imageLabelY), text, c);
        var size = FaceRenderer.GetTextSize(text, FaceRenderer.TEXT_SIZE_MEASURE);
        _imageLabelY += size.Y * 2;
    }

    public float MeasurePolygon(Vec2[] contour, bool render = true,
        Rgba32 color = default, float alpha = 0.4f)
    {
        var blContour = new Vec2i[contour.Length];
        for (int i = 0; i < contour.Length; i++)
            blContour[i] = new Vec2i(MathHelpers.RoundToInt(contour[i].X), BLY(contour[i].Y));

        Rgba32 c = color.A == 0 ? new Rgba32(0, 100, 255, 255) : color;
        float areaPx = FaceRenderer.MeasurePolygon(Pixels, Width, Height, blContour, render, c, alpha);
        return areaPx * Pix2mm * Pix2mm;
    }

    public void DrawLine(Vec2 pt1, Vec2 pt2, Rgba32 color, int thickness = 2)
    {
        FaceRenderer.DrawLine(Pixels, Width, Height,
            MathHelpers.RoundToInt(pt1.X), BLY(pt1.Y),
            MathHelpers.RoundToInt(pt2.X), BLY(pt2.Y),
            color, thickness);
    }

    public void DrawArrow(Vec2 pt1, Vec2 pt2, Rgba32 color, int thickness = 2,
        bool arrowAtStart = false, bool arrowAtEnd = true)
    {
        FaceRenderer.DrawArrow(Pixels, Width, Height,
            MathHelpers.RoundToInt(pt1.X), BLY(pt1.Y),
            MathHelpers.RoundToInt(pt2.X), BLY(pt2.Y),
            color, thickness, arrowAtStart, arrowAtEnd);
    }

    public void DrawCircleAt(Vec2 pt, Rgba32 color, int radius = 4)
    {
        FaceRenderer.DrawCircle(Pixels, Width, Height,
            MathHelpers.RoundToInt(pt.X), BLY(pt.Y), color, radius);
    }

    // Samples a rectangle given in top-left coordinates, returning RGB pixel values
    // (n,3 conceptually — flattened as an Rgba32[] here). Used by MeasureSkinTone.
    public Rgba32[] SampleRectangleTopLeft(Vec2 topLeft, Vec2 bottomRight)
    {
        int left   = MathHelpers.RoundToInt(topLeft.X);
        int right  = MathHelpers.RoundToInt(bottomRight.X);
        int top    = BLY(topLeft.Y);
        int bottom = BLY(bottomRight.Y);
        if (bottom > top) (bottom, top) = (top, bottom); // BLY flips vertical order
        return FaceRenderer.SampleRectangle(Pixels, Width, Height, left, bottom, right, top + 1);
    }

    // -------------------------------------------------------------------------
    // Sidebar text / structured output
    // -------------------------------------------------------------------------

    public void AddHeader(string label) { TextLines.Add(label); }
    public void AddValue(string text)   { TextLines.Add("  " + text); }
    public void AddSpacer()             { TextLines.Add(""); }

    public void AddValue(string key, double value) { Values[key] = value; }

    // -------------------------------------------------------------------------
    // Output
    // -------------------------------------------------------------------------

    public FaceImage ToImage() => new FaceImage(Pixels, Width, Height);
}
