#nullable enable
using System;
using System.Collections.Generic;
using static CheekNumerics;

/// <summary>
/// Cheek landmarks: the frame, the frontal constructions, the yaw calibration and the
/// quarter-face classification. A transcription of dynaface-lib's <c>dynaface/cheek.py</c>,
/// the reference implementation, name for name in PascalCase, in <c>double</c> with the
/// same operation order.
///
/// <para>On a frontal face, the zygion, the cheek reference point and the submalar
/// reference point of each side are constructed from the 98 WFLW landmarks in a frame on
/// the pupil axis (<see cref="Frame"/>), so head roll never enters a number. On a
/// three-quarter (quarter) face, the far cheek's malar apex and submalar point are
/// observed on its outline by <see cref="CheekOutline"/> (<see cref="ObserveFarCheek(Vec2d[], float[], int[], byte[], bool)"/>);
/// <see cref="Locate(Vec2d[], bool, bool, bool, float[], int[], byte[])"/> is the one rule
/// that turns a face's landmarks and view into the two results,
/// <see cref="LandmarkArray"/> fills the ten cheek landmarks from them, NaN where a point
/// doesn't exist in the view, and <see cref="LocateAll(Vec2d[], FacePose, bool, float[], int[], byte[])"/>
/// does both from a pose. <see cref="CalibrateYaw"/>, <see cref="NoseOffset(Vec2d[], Frame)"/>,
/// <see cref="LandmarkYaw(Vec2d[], Frame)"/> and <see cref="NoseMargin(Vec2d[], Frame)"/>
/// are the pose cues, and <see cref="ClassifyQuarter(Vec2d[], double, bool, bool)"/> is the
/// rule that calls a face a quarter face.</para>
///
/// <para>Every number here is a top-left pixel of the image the landmarks came from, a
/// degree, or a ratio; millimeters are applied by the measure (* pix2mm), never here.
/// Each function taking landmarks has a <see cref="Vec2"/>[] overload (the pipeline's
/// landmarks, widened exactly) and a <see cref="Vec2d"/>[] one.</para>
/// </summary>
public static class Cheek
{
    /// <summary>
    /// The ten cheek landmarks, in the order of AnalyzeFace.cheek_landmarks' rows. "Right"
    /// and "left" are the subject's. Rows 0 to 5 exist on a frontal face, rows 6 to 9 on a
    /// quarter face, for its far cheek only; only the observed oblique point is called an
    /// apex.
    /// </summary>
    public static readonly IReadOnlyList<string> CheekLandmarkNames = Array.AsReadOnly(new[]
    {
        "Zygion (right)",                   // 0, frontal
        "Zygion (left)",                    // 1, frontal
        "Cheek reference point (right)",    // 2, frontal
        "Cheek reference point (left)",     // 3, frontal
        "Submalar reference point (right)", // 4, frontal
        "Submalar reference point (left)",  // 5, frontal
        "Malar apex (right)",               // 6, quarter, far cheek
        "Malar apex (left)",                // 7, quarter, far cheek
        "Submalar point (right)",           // 8, quarter, far cheek
        "Submalar point (left)",            // 9, quarter, far cheek
    });

    /// <summary>
    /// Each cheek landmark's code, in <see cref="CheekLandmarkNames"/> order: the prefix of
    /// its outputs (apex.l.x is "apex.l"'s x), and the label drawn beside it.
    /// </summary>
    public static readonly IReadOnlyList<string> CheekLandmarkCodes = Array.AsReadOnly(new[]
    {
        "zy.r", "zy.l", "cr.r", "cr.l", "sr.r", "sr.l", "apex.r", "apex.l", "sub.r", "sub.l",
    });

    /// <summary>
    /// The cheek landmarks' coordinates as outputs, px of the standard image: each code's x
    /// then y, in <see cref="CheekLandmarkNames"/> order (zy.r.x, zy.r.y, zy.l.x, ... sub.l.y),
    /// NaN where the landmark doesn't exist in the view.
    /// </summary>
    public static readonly IReadOnlyList<string> CheekLandmarkOutputs = Outputs();

    static IReadOnlyList<string> Outputs()
    {
        var list = new List<string>();
        foreach (string code in CheekLandmarkCodes)
        {
            list.Add(code + ".x");
            list.Add(code + ".y");
        }
        return list.AsReadOnly();
    }

    public const int CheekLmZygionR = 0, CheekLmZygionL = 1;
    public const int CheekLmCheekRefR = 2, CheekLmCheekRefL = 3;
    public const int CheekLmSubmalarRefR = 4, CheekLmSubmalarRefL = 5;
    public const int CheekLmApexR = 6, CheekLmApexL = 7;
    public const int CheekLmSubmalarR = 8, CheekLmSubmalarL = 9;

    /// <summary>The zygion level below the interpupillary line, in IPD units (an orbitale proxy).</summary>
    public const double ZygionLevelIpd = 0.3;
    /// <summary>Wilkinson: the cheek reference point is this far down the canthus vertical to the contour.</summary>
    public const double CheekRefFraction = 1.0 / 3.0;
    /// <summary>The submalar reference point is this far from the cheek reference point down to the stomion level.</summary>
    public const double SubmalarRefFraction = 0.5;
    /// <summary>The polyline the right zygion level crosses (0-4).</summary>
    public static readonly IReadOnlyList<int> RightContourTop = Array.AsReadOnly(new[] { 0, 1, 2, 3, 4 });
    /// <summary>The polyline the left zygion level crosses (32-28).</summary>
    public static readonly IReadOnlyList<int> LeftContourTop = Array.AsReadOnly(new[] { 32, 31, 30, 29, 28 });

    // The half contours the canthus vertical crosses, the WFLW landmarks the constructions
    // and the pose cues read and the landmark count: defined once in CheekOutline (whose
    // tracer seeds on the same half contours) and re-exported here.
    public const int NumLandmarks = CheekOutline.NumLandmarks;
    /// <summary>The right half contour (0..16), the canthus vertical's polyline.</summary>
    public static readonly IReadOnlyList<int> RightContour = CheekOutline.RightFarContour;
    /// <summary>The left half contour (32..16), the canthus vertical's polyline.</summary>
    public static readonly IReadOnlyList<int> LeftContour = CheekOutline.LeftFarContour;
    public const int LmCanthusR = CheekOutline.LmCanthusR, LmCanthusL = CheekOutline.LmCanthusL;
    public const int LmAlaR = CheekOutline.LmAlaR, LmAlaL = CheekOutline.LmAlaL;
    public const int LmNoseTip = CheekOutline.LmNoseTip, LmChin = CheekOutline.LmChin;
    /// <summary>The inner-lip midpoints whose mean level is the stomion level.</summary>
    public static readonly (int Upper, int Lower) LmStomion = CheekOutline.LmStomion;
    /// <summary>The contour's top points: the nose margin is read from the nearer of them.</summary>
    public const int LmContourTopR = 0, LmContourTopL = 32;

    /// <summary>zy.*.clip: the level crossed the contour.</summary>
    public const int ClipNone = 0;
    /// <summary>zy.*.clip: the level passed above point 0/32.</summary>
    public const int ClipTop = 1;
    /// <summary>zy.*.clip: the level passed below point 4/28.</summary>
    public const int ClipBottom = 2;
    /// <summary>|cheek.yaw| for zy.r.ok, zy.l.ok, cr.ok, sr.ok.</summary>
    public const double FrontalYawGate = 5.0;
    /// <summary>|cheek.yaw| for zy.ratio.ok (the ratio moves ~3% per degree).</summary>
    public const double ZygionRatioYawGate = 2.0;
    /// <summary>cheek.yaw and cheek.yaw.model must agree within this for any yaw gate.</summary>
    public const double YawAgreeDeg = 5.0;
    /// <summary>|raw SPIGA pitch| above this fails cheek.pitch.ok, and nothing else.</summary>
    public const double PitchGrossRaw = 20.0;
    /// <summary>apex.ok needs |cheek.yaw| within <see cref="ObliqueYaw"/> +/- <see cref="ObliqueYawGate"/>.</summary>
    public const double ObliqueYaw = 45.0, ObliqueYawGate = 5.0;
    /// <summary>|cheek.yaw| (the landmark yaw) that the pipeline calls a quarter face.</summary>
    public static readonly (double Low, double High) QuarterYawRange = (35.0, 55.0);
    /// <summary>A face that was a quarter face on the previous frame stays one this far past both edges of the range.</summary>
    public const double QuarterYawHold = 3.0;
    /// <summary>The model's calibrated |yaw| must reach this, with the landmark yaw's sign.</summary>
    public const double QuarterMinModelYaw = 20.0;
    /// <summary>... and stay at or below this (widened by <see cref="QuarterYawHold"/> with the hold).</summary>
    public const double QuarterMaxModelYaw = 55.0;
    /// <summary>A nose past the far contour by more than this (<see cref="NoseMargin(Vec2d[], Frame)"/>): profile logic instead.</summary>
    public const double QuarterMarginMin = -0.03;

    /// <summary>
    /// (raw SPIGA yaw, true yaw) on the synthetic renders, the level head's reading first;
    /// <see cref="CalibrateYaw"/> reads it less <see cref="YawRawOffset"/>.
    /// </summary>
    public static readonly IReadOnlyList<(double Raw, double TrueYaw)> YawCalibration = Array.AsReadOnly(new[]
    {
        (1.5, 0.0),
        (3.04, 5.0),
        (5.87, 10.0),
        (7.82, 15.0),
        (10.41, 20.0),
        (12.89, 25.0),
        (15.57, 30.0),
        (19.59, 35.0),
        (23.73, 40.0),
        (25.77, 42.0),
        (28.28, 45.0),
        (30.54, 48.0),
        (32.63, 50.0),
        (37.79, 55.0),
        (43.04, 60.0),
        (44.31, 65.0),
    });
    /// <summary>Degrees of true turn per reported degree past the table.</summary>
    public const double YawSlopeBeyond = 1.0 / 0.62;
    /// <summary>
    /// The constant SPIGA's yaw reads over the truth on every render, degrees;
    /// <see cref="CalibrateYaw"/> subtracts it before the table. A reading of exactly this
    /// is a level head.
    /// </summary>
    public const double YawRawOffset = 1.23;
    /// <summary>
    /// (|nose offset| in canthus spans, true |yaw|) on the same renders: the sweep's
    /// roll-corrected <see cref="NoseOffset(Vec2d[], Frame)"/>, read by
    /// <see cref="LandmarkYaw(Vec2d[], Frame)"/>. Stops at 60 degrees.
    /// </summary>
    public static readonly IReadOnlyList<(double Offset, double TrueYaw)> NoseOffsetCalibration = Array.AsReadOnly(new[]
    {
        (0.0, 0.0),
        (0.0458, 5.0),
        (0.0908, 10.0),
        (0.122, 15.0),
        (0.167, 20.0),
        (0.2069, 25.0),
        (0.2573, 30.0),
        (0.3289, 35.0),
        (0.3861, 40.0),
        (0.4181, 42.0),
        (0.4449, 45.0),
        (0.514, 48.0),
        (0.5431, 50.0),
        (0.6842, 55.0),
        (0.7696, 60.0),
    });
    /// <summary>Degrees per canthus span past the table (its last segment reads 58.5).</summary>
    public const double NoseOffsetSlopeBeyond = 57.0;

    // The tables as np.interp's xp and fp (both strictly increasing); the yaw table's cue
    // column is the raw reading less the offset.
    static readonly double[] YawCalReported = Column(YawCalibration.Count, i => YawCalibration[i].Raw - YawRawOffset);
    static readonly double[] YawCalTrue = Column(YawCalibration.Count, i => YawCalibration[i].TrueYaw);
    static readonly double[] NoseCalOffset = Column(NoseOffsetCalibration.Count, i => NoseOffsetCalibration[i].Offset);
    static readonly double[] NoseCalTrue = Column(NoseOffsetCalibration.Count, i => NoseOffsetCalibration[i].TrueYaw);

    static double[] Column(int count, Func<int, double> value)
    {
        var column = new double[count];
        for (int i = 0; i < count; i++) column[i] = value(i);
        return column;
    }

    // ------------------------------------------------------------------ the frame

    /// <summary>A face frame on the pupil axis, so head roll never enters a number.</summary>
    public sealed class Frame
    {
        /// <summary>The pupil midpoint, px.</summary>
        public Vec2d Origin { get; }
        /// <summary>Unit vector from the right pupil (96) toward the left (97), the subject's left.</summary>
        public Vec2d Across { get; }
        /// <summary>Its normal toward the chin, (-Across.Y, Across.X): image-down for a level head.</summary>
        public Vec2d Down { get; }
        /// <summary>The pupil distance, px.</summary>
        public double Ipd { get; }

        public Frame(Vec2d origin, Vec2d across, Vec2d down, double ipd)
        {
            Origin = origin;
            Across = across;
            Down = down;
            Ipd = ipd;
        }

        /// <summary>How far below the interpupillary line the point is, px.</summary>
        public double Level(Vec2d p) => Vec2d.Dot(p - Origin, Down);

        /// <summary>How far toward the subject's left of the midline the point is, px.</summary>
        public double Offset(Vec2d p) => Vec2d.Dot(p - Origin, Across);

        /// <summary>The point at this offset and level, px.</summary>
        public Vec2d Point(double offset, double level) => Origin + Across * offset + Down * level;
    }

    /// <summary>
    /// The frame of the 98 landmarks on the pupil axis (landmarks 96 and 97), or null when
    /// the pupils coincide, there are fewer than 98 landmarks or any landmark isn't finite.
    /// </summary>
    public static Frame? PupilFrame(Vec2d[]? landmarks)
    {
        Vec2d[]? pts = CheekOutline.AsLandmarks(landmarks);
        if (pts == null) return null;
        Vec2d right = pts[DynafaceConstants.LmRightPupil];
        Vec2d left = pts[DynafaceConstants.LmLeftPupil];
        Vec2d axis = left - right;
        double ipd = Hypot(axis.X, axis.Y);
        if (!IsFinite(ipd) || ipd <= 0.0) return null;
        Vec2d across = axis / ipd;
        var down = new Vec2d(-across.Y, across.X);
        return new Frame((right + left) / 2.0, across, down, ipd);
    }

    /// <summary><see cref="PupilFrame(Vec2d[])"/> of float landmarks.</summary>
    public static Frame? PupilFrame(Vec2[]? landmarks) => PupilFrame(CheekOutline.ToVec2d(landmarks));

    // ------------------------------------------------------------------ polylines

    /// <summary>
    /// Where a polyline reaches values == target: the first segment whose ends bracket it,
    /// interpolated; status 0. When none does, the end nearer the target, clipped: status
    /// -1 (the first point) or +1 (the last).
    /// </summary>
    /// <param name="points">The polyline's vertices, in order.</param>
    /// <param name="values">A value at each vertex, such as its level in a <see cref="Frame"/>.</param>
    /// <param name="target">The value to reach.</param>
    public static (Vec2d Point, int Status) CrossPolyline(Vec2d[] points, double[] values, double target)
    {
        for (int k = 0; k < points.Length - 1; k++)
        {
            double a = values[k], b = values[k + 1];
            if ((a - target) * (b - target) <= 0 && a != b)
            {
                double t = (target - a) / (b - a);
                return (points[k] + t * (points[k + 1] - points[k]), 0);
            }
        }
        bool nearerFirst = Math.Abs(values[0] - target) <= Math.Abs(values[values.Length - 1] - target);
        return nearerFirst ? (points[0], -1) : (points[points.Length - 1], 1);
    }

    /// <summary>
    /// Where the vertical (pupil-axis normal) at this offset crosses the contour polyline:
    /// the crossing with the greatest level (the mandibular border, not a crossing near the
    /// contour's top on a turned or hair-occluded face), interpolated. The vertex nearest
    /// in offset when nothing crosses.
    /// </summary>
    /// <param name="points">The half contour's vertices, px, in order.</param>
    /// <param name="offsets">Each vertex's <see cref="Frame.Offset"/>.</param>
    /// <param name="levels">Each vertex's <see cref="Frame.Level"/>.</param>
    /// <param name="offset">The vertical's offset, such as the lateral canthus's.</param>
    public static Vec2d ContourBelow(Vec2d[] points, double[] offsets, double[] levels, double offset)
    {
        Vec2d? best = null;
        double bestLevel = double.NegativeInfinity;
        for (int k = 0; k < points.Length - 1; k++)
        {
            double a = offsets[k], b = offsets[k + 1];
            if ((a - offset) * (b - offset) <= 0 && a != b)
            {
                double t = (offset - a) / (b - a);
                double level = levels[k] + t * (levels[k + 1] - levels[k]);
                if (level > bestLevel)
                {
                    bestLevel = level;
                    best = points[k] + t * (points[k + 1] - points[k]);
                }
            }
        }
        if (best != null) return best.Value;
        var distance = new double[offsets.Length];
        for (int i = 0; i < offsets.Length; i++) distance[i] = Math.Abs(offsets[i] - offset);
        return points[ArgMin(distance)];
    }

    /// <summary>
    /// The constructed zygion of one side: the level levelIpd * ipd below the
    /// interpupillary line, crossed with the contour's top polyline (0-4, or 32-28),
    /// interpolated by <see cref="CrossPolyline"/>. The clip code says what happened when
    /// the level missed the polyline: <see cref="ClipNone"/> when it crossed,
    /// <see cref="ClipTop"/> when it passed above point 0/32 (the point is 0/32) or
    /// <see cref="ClipBottom"/> when it passed below point 4/28 (the point is 4/28).
    /// </summary>
    /// <param name="landmarks">The 98 WFLW landmarks, px.</param>
    /// <param name="frame">Their pupil-axis frame.</param>
    /// <param name="left">The subject's left side (contour 32-28), else the right.</param>
    /// <param name="levelIpd">The level below the interpupillary line, in IPD units.</param>
    public static (Vec2d Point, int Clip) Zygion(Vec2d[] landmarks, Frame frame, bool left,
        double levelIpd = ZygionLevelIpd)
    {
        if (landmarks == null) throw new ArgumentNullException(nameof(landmarks));
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        var idx = left ? LeftContourTop : RightContourTop;
        var top = new Vec2d[idx.Count];
        var levels = new double[idx.Count];
        for (int i = 0; i < idx.Count; i++)
        {
            top[i] = landmarks[idx[i]];
            levels[i] = frame.Level(top[i]);
        }
        var (point, status) = CrossPolyline(top, levels, levelIpd * frame.Ipd);
        int clip = status == 0 ? ClipNone : status < 0 ? ClipTop : ClipBottom;
        return (point, clip);
    }

    /// <summary><see cref="Zygion(Vec2d[], Frame, bool, double)"/> of float landmarks.</summary>
    public static (Vec2d Point, int Clip) Zygion(Vec2[] landmarks, Frame frame, bool left,
        double levelIpd = ZygionLevelIpd) =>
        Zygion(CheekOutline.ToVec2d(landmarks)!, frame, left, levelIpd);

    // ------------------------------------------------------------------ frontal constructions

    /// <summary>
    /// The six frontal constructions and the lines they are drawn with, in top-left pixels
    /// of the standard image. Millimeters are applied by the measure (* pix2mm), never here.
    /// </summary>
    public sealed class FrontalCheek
    {
        /// <summary>The pupil-axis frame they were built in.</summary>
        public Frame Frame { get; set; } = null!;
        /// <summary>The zygion level, px below the interpupillary line (for the level line).</summary>
        public double ZygionLevel { get; set; }
        /// <summary>The right zygion, px.</summary>
        public Vec2d ZygionR { get; set; }
        /// <summary>The left zygion, px.</summary>
        public Vec2d ZygionL { get; set; }
        /// <summary><see cref="ClipNone"/>, <see cref="ClipTop"/> or <see cref="ClipBottom"/>.</summary>
        public int ZygionRClip { get; set; }
        /// <summary><see cref="ClipNone"/>, <see cref="ClipTop"/> or <see cref="ClipBottom"/>.</summary>
        public int ZygionLClip { get; set; }
        /// <summary>The right zygion's distance from the midline, px (positive).</summary>
        public double ZygionRPx { get; set; }
        /// <summary>The left zygion's distance from the midline, px (positive).</summary>
        public double ZygionLPx { get; set; }
        /// <summary>The right lateral canthus (60), the top of its vertical.</summary>
        public Vec2d CanthusR { get; set; }
        /// <summary>The left lateral canthus (72), the top of its vertical.</summary>
        public Vec2d CanthusL { get; set; }
        /// <summary>The right vertical's foot on the contour.</summary>
        public Vec2d FootR { get; set; }
        /// <summary>The left vertical's foot on the contour.</summary>
        public Vec2d FootL { get; set; }
        /// <summary>The right cheek reference point, px.</summary>
        public Vec2d CheekRefR { get; set; }
        /// <summary>The left cheek reference point, px.</summary>
        public Vec2d CheekRefL { get; set; }
        /// <summary>The right cheek reference point's depth below the canthus along the vertical, px.</summary>
        public double CheekRefRPx { get; set; }
        /// <summary>The left cheek reference point's depth below the canthus along the vertical, px.</summary>
        public double CheekRefLPx { get; set; }
        /// <summary>The right submalar reference point, px.</summary>
        public Vec2d SubmalarRefR { get; set; }
        /// <summary>The left submalar reference point, px.</summary>
        public Vec2d SubmalarRefL { get; set; }
        /// <summary>The stomion level, px below the interpupillary line.</summary>
        public double StomionLevel { get; set; }
    }

    /// <summary>One side's canthus vertical and the two reference points on it, px.</summary>
    readonly struct CanthusVertical
    {
        public readonly Vec2d Canthus, Foot, CheekRef, SubmalarRef;
        public readonly double Depth;

        public CanthusVertical(Vec2d canthus, Vec2d foot, Vec2d cheekRef, double depth, Vec2d submalarRef)
        {
            Canthus = canthus;
            Foot = foot;
            CheekRef = cheekRef;
            Depth = depth;
            SubmalarRef = submalarRef;
        }
    }

    /// <summary>Drop one side's vertical from its lateral canthus to the contour and place the two reference points on it.</summary>
    static CanthusVertical CanthusVerticalOf(Vec2d[] pts, Frame frame, int canthus,
        IReadOnlyList<int> contour, double stomionLevel)
    {
        Vec2d ex = pts[canthus];
        var cont = new Vec2d[contour.Count];
        var offsets = new double[contour.Count];
        var levels = new double[contour.Count];
        for (int i = 0; i < contour.Count; i++)
        {
            cont[i] = pts[contour[i]];
            Vec2d rel = cont[i] - frame.Origin;
            offsets[i] = Vec2d.Dot(rel, frame.Across);
            levels[i] = Vec2d.Dot(rel, frame.Down);
        }
        double exOffset = frame.Offset(ex);
        Vec2d crossing = ContourBelow(cont, offsets, levels, exOffset);
        // The crossing's level at the canthus's own offset: exactly on the vertical.
        Vec2d foot = frame.Point(exOffset, frame.Level(crossing));
        Vec2d cheekRef = ex + (foot - ex) * CheekRefFraction;
        double depth = frame.Level(cheekRef) - frame.Level(ex);
        Vec2d submalarRef = cheekRef + frame.Down * ((stomionLevel - frame.Level(cheekRef)) * SubmalarRefFraction);
        return new CanthusVertical(ex, foot, cheekRef, depth, submalarRef);
    }

    /// <summary>
    /// All six frontal points (dynaface-lib's frontal_constructions). Per side, in the
    /// pupil-axis frame: the zygion is <see cref="Zygion(Vec2d[], Frame, bool, double)"/>'s
    /// crossing of the <see cref="ZygionLevelIpd"/> level with the contour top, and its
    /// midline distance is its offset, made positive on both sides. The canthus vertical
    /// runs from the lateral canthus (60 / 72) down the frame's normal to its foot on the
    /// half contour, <see cref="ContourBelow"/>'s lowest crossing, taken at the canthus's
    /// own offset; the cheek reference point is <see cref="CheekRefFraction"/> of the way
    /// down that vertical, and its depth is the level difference along it. The stomion
    /// level is the mean level of the inner-lip midpoints (90 and 94), and the submalar
    /// reference point lies <see cref="SubmalarRefFraction"/> of the way from the cheek
    /// reference point down to it, on the same vertical.
    /// </summary>
    /// <returns>The constructions, or null when the pupils coincide, there are fewer than 98 landmarks or any landmark isn't finite.</returns>
    public static FrontalCheek? FrontalConstructions(Vec2d[]? landmarks)
    {
        Vec2d[]? pts = CheekOutline.AsLandmarks(landmarks);
        if (pts == null) return null;
        Frame? frame = PupilFrame(pts);
        if (frame == null) return null;
        var (zygionR, clipR) = Zygion(pts, frame, left: false);
        var (zygionL, clipL) = Zygion(pts, frame, left: true);
        double stomionLevel = (frame.Level(pts[LmStomion.Upper]) + frame.Level(pts[LmStomion.Lower])) / 2.0;
        CanthusVertical right = CanthusVerticalOf(pts, frame, LmCanthusR, RightContour, stomionLevel);
        CanthusVertical left = CanthusVerticalOf(pts, frame, LmCanthusL, LeftContour, stomionLevel);
        return new FrontalCheek
        {
            Frame = frame,
            ZygionLevel = ZygionLevelIpd * frame.Ipd,
            ZygionR = zygionR,
            ZygionL = zygionL,
            ZygionRClip = clipR,
            ZygionLClip = clipL,
            ZygionRPx = -frame.Offset(zygionR),
            ZygionLPx = frame.Offset(zygionL),
            CanthusR = right.Canthus,
            CanthusL = left.Canthus,
            FootR = right.Foot,
            FootL = left.Foot,
            CheekRefR = right.CheekRef,
            CheekRefL = left.CheekRef,
            CheekRefRPx = right.Depth,
            CheekRefLPx = left.Depth,
            SubmalarRefR = right.SubmalarRef,
            SubmalarRefL = left.SubmalarRef,
            StomionLevel = stomionLevel,
        };
    }

    /// <summary><see cref="FrontalConstructions(Vec2d[])"/> of float landmarks.</summary>
    public static FrontalCheek? FrontalConstructions(Vec2[]? landmarks) =>
        FrontalConstructions(CheekOutline.ToVec2d(landmarks));

    // ------------------------------------------------------------------ yaw

    /// <summary>
    /// A cue's magnitude through a calibration table: np.interp inside it, linear at
    /// slopeBeyond past its last row; NaN stays NaN.
    /// </summary>
    static double ThroughTable(double magnitude, double[] xp, double[] fp, double slopeBeyond)
    {
        if (double.IsNaN(magnitude)) return double.NaN;
        if (magnitude > xp[xp.Length - 1])
            return fp[fp.Length - 1] + (magnitude - xp[xp.Length - 1]) * slopeBeyond;
        return Interp(magnitude, xp, fp);
    }

    /// <summary>
    /// SPIGA's yaw as a true turn: the reading less <see cref="YawRawOffset"/> (so a reading
    /// of the offset is a level head and one below it a turn toward the subject's right),
    /// then interpolated on its magnitude through <see cref="YawCalibration"/>, linear at
    /// <see cref="YawSlopeBeyond"/> past the table's last point, with that difference's
    /// sign; NaN stays NaN. A level head is 0.0, never -0.0.
    /// </summary>
    /// <param name="reported">The model's yaw, degrees.</param>
    public static double CalibrateYaw(double reported)
    {
        double shifted = reported - YawRawOffset;
        if (double.IsNaN(shifted)) return double.NaN;
        double trueYaw = ThroughTable(Math.Abs(shifted), YawCalReported, YawCalTrue, YawSlopeBeyond);
        return trueYaw != 0.0 ? CopySign(trueYaw, shifted) : 0.0;
    }

    /// <summary>math.copysign: magnitude's size with sign's sign bit.</summary>
    static double CopySign(double magnitude, double sign)
    {
        double m = Math.Abs(magnitude);
        return BitConverter.DoubleToInt64Bits(sign) < 0 ? -m : m;
    }

    /// <summary>
    /// The nose tip's offset from the midpoint of the lateral canthi, along the pupil axis,
    /// in canthus-to-canthus spans: positive toward the subject's left (image right).
    /// Roll-corrected; the symmetric yaw cue.
    /// </summary>
    /// <returns>The offset, in canthus spans; NaN with fewer than 98 landmarks, a landmark that isn't finite, or when the canthi don't span the axis.</returns>
    public static double NoseOffset(Vec2d[]? landmarks, Frame frame)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        Vec2d[]? pts = CheekOutline.AsLandmarks(landmarks);
        if (pts == null) return double.NaN;
        double canthusR = frame.Offset(pts[LmCanthusR]);
        double canthusL = frame.Offset(pts[LmCanthusL]);
        double span = canthusL - canthusR;
        if (!(span > 0.0)) return double.NaN;
        return (frame.Offset(pts[LmNoseTip]) - (canthusR + canthusL) / 2.0) / span;
    }

    /// <summary><see cref="NoseOffset(Vec2d[], Frame)"/> of float landmarks.</summary>
    public static double NoseOffset(Vec2[]? landmarks, Frame frame) =>
        NoseOffset(CheekOutline.ToVec2d(landmarks), frame);

    /// <summary>
    /// The turn measured from the landmarks: <see cref="NoseOffset(Vec2d[], Frame)"/> through
    /// <see cref="NoseOffsetCalibration"/> (interpolated on its magnitude,
    /// <see cref="NoseOffsetSlopeBeyond"/> past the table's last point, 60 degrees), signed
    /// like SPIGA's yaw. This is cheek.yaw and the yaw every gate uses;
    /// <see cref="CalibrateYaw"/> of the model's yaw is cheek.yaw.model, the independent check.
    /// </summary>
    /// <returns>The yaw, degrees, positive when the nose points to the image right; NaN when the nose offset is.</returns>
    public static double LandmarkYaw(Vec2d[]? landmarks, Frame frame)
    {
        double offset = NoseOffset(landmarks, frame);
        if (double.IsNaN(offset)) return double.NaN;
        double trueYaw = ThroughTable(Math.Abs(offset), NoseCalOffset, NoseCalTrue, NoseOffsetSlopeBeyond);
        return CopySign(trueYaw, offset);
    }

    /// <summary><see cref="LandmarkYaw(Vec2d[], Frame)"/> of float landmarks.</summary>
    public static double LandmarkYaw(Vec2[]? landmarks, Frame frame) =>
        LandmarkYaw(CheekOutline.ToVec2d(landmarks), frame);

    /// <summary>
    /// The nose tip's margin from the nearer contour top (0 or 32), along the pupil axis, as
    /// a fraction of the contour width: on the renders 0.47 face-on, 0.05 at 45 degrees, 0
    /// at 50 where the tip meets the far silhouette, negative past it.
    /// </summary>
    /// <returns>The margin; NaN with fewer than 98 landmarks, a landmark that isn't finite, or when the contour tops don't span the axis.</returns>
    public static double NoseMargin(Vec2d[]? landmarks, Frame frame)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        Vec2d[]? pts = CheekOutline.AsLandmarks(landmarks);
        if (pts == null) return double.NaN;
        double topR = frame.Offset(pts[LmContourTopR]);
        double topL = frame.Offset(pts[LmContourTopL]);
        double width = topL - topR;
        if (!(width > 0.0)) return double.NaN;
        double tip = frame.Offset(pts[LmNoseTip]);
        double toLeft = topL - tip, toRight = tip - topR;
        // Python's min(): the first unless the second is smaller.
        return (toRight < toLeft ? toRight : toLeft) / width;
    }

    /// <summary><see cref="NoseMargin(Vec2d[], Frame)"/> of float landmarks.</summary>
    public static double NoseMargin(Vec2[]? landmarks, Frame frame) =>
        NoseMargin(CheekOutline.ToVec2d(landmarks), frame);

    /// <summary>
    /// True when the nose points to the image's right (nose offset &gt; 0): the camera sees
    /// the subject's right side, so the far cheek is the left. Falls back to the yaw's sign
    /// without a frame or when the canthi don't span the pupil axis.
    /// </summary>
    /// <param name="landmarks">The 98 WFLW landmarks, px.</param>
    /// <param name="yawReported">The model's yaw, degrees, for the fallback; NaN (the default) falls back to false.</param>
    public static bool FarLeftOf(Vec2d[]? landmarks, double yawReported = double.NaN)
    {
        Frame? frame = PupilFrame(landmarks);
        if (frame != null)
        {
            double offset = NoseOffset(landmarks, frame);
            if (!double.IsNaN(offset)) return offset > 0.0;
        }
        return yawReported > 0.0;
    }

    /// <summary><see cref="FarLeftOf(Vec2d[], double)"/> of float landmarks.</summary>
    public static bool FarLeftOf(Vec2[]? landmarks, double yawReported = double.NaN) =>
        FarLeftOf(CheekOutline.ToVec2d(landmarks), yawReported);

    /// <summary>
    /// (IsQuarter, FarLeft) from the raw landmarks and head pose, before any crop or flip.
    /// Quarter when |<see cref="LandmarkYaw(Vec2d[], Frame)"/>| is within
    /// <see cref="QuarterYawRange"/>, widened by <see cref="QuarterYawHold"/> at both edges
    /// when previousQuarter (hysteresis), the calibrated model yaw has the same sign,
    /// reaches <see cref="QuarterMinModelYaw"/> and stays at or below
    /// <see cref="QuarterMaxModelYaw"/> (widened by the same hold), and the nose margin is
    /// at least <see cref="QuarterMarginMin"/> unless requireMargin is false. A NaN yaw
    /// fails every comparison.
    /// </summary>
    /// <param name="landmarks">The 98 WFLW landmarks of the uncropped image.</param>
    /// <param name="yawReported">The model's yaw, degrees.</param>
    /// <param name="previousQuarter">Whether the previous kept frame of a video was a quarter face.</param>
    /// <param name="requireMargin">Whether the nose margin rule applies; false is for a face the profile test has already declined.</param>
    /// <returns>Whether the face is a quarter face, and whether its far cheek is the subject's left (false unless a quarter face).</returns>
    public static (bool IsQuarter, bool FarLeft) ClassifyQuarter(Vec2d[]? landmarks, double yawReported,
        bool previousQuarter = false, bool requireMargin = true)
    {
        Frame? frame = PupilFrame(landmarks);
        if (frame == null) return (false, false);
        double yaw = LandmarkYaw(landmarks, frame);
        double modelYaw = CalibrateYaw(yawReported);
        double low = QuarterYawRange.Low, high = QuarterYawRange.High;
        double modelHigh = QuarterMaxModelYaw;
        if (previousQuarter)
        {
            low -= QuarterYawHold;
            high += QuarterYawHold;
            modelHigh += QuarterYawHold;
        }
        bool inWindow = low <= Math.Abs(yaw) && Math.Abs(yaw) <= high;
        bool modelAgrees = QuarterMinModelYaw <= Math.Abs(modelYaw) && Math.Abs(modelYaw) <= modelHigh
            && (modelYaw > 0.0) == (yaw > 0.0);
        bool marginOk = !requireMargin || NoseMargin(landmarks, frame) >= QuarterMarginMin;
        if (!(inWindow && modelAgrees && marginOk)) return (false, false);
        return (true, FarLeftOf(landmarks));
    }

    /// <summary><see cref="ClassifyQuarter(Vec2d[], double, bool, bool)"/> of float landmarks.</summary>
    public static (bool IsQuarter, bool FarLeft) ClassifyQuarter(Vec2[]? landmarks, double yawReported,
        bool previousQuarter = false, bool requireMargin = true) =>
        ClassifyQuarter(CheekOutline.ToVec2d(landmarks), yawReported, previousQuarter, requireMargin);

    // ------------------------------------------------------------------ locating

    /// <summary>
    /// The far cheek's observed points as the cheek landmarks take them: null without a
    /// pupil frame (fewer than 98 landmarks, a landmark that isn't finite, or coincident
    /// pupils), else <see cref="CheekOutline.ObservedPoints(Vec2d[], float[], int[], byte[], bool)"/>.
    /// The outline goes in whole or not at all (a partial or ragged one means the contour
    /// fallback), and a search that fails is a face without observed points, never an
    /// exception: it is written to <see cref="System.Diagnostics.Trace"/> as a warning with
    /// its text (dynaface-lib logs it) and the result is null.
    /// </summary>
    public static CheekOutline.ObliqueCheek? ObserveFarCheek(Vec2d[]? landmarks, float[]? outlineX,
        int[]? outlineY, byte[]? outlineSrc, bool farLeft)
    {
        Vec2d[]? pts = CheekOutline.AsLandmarks(landmarks);
        if (pts == null || PupilFrame(pts) == null) return null;
        if (outlineX == null || outlineY == null || outlineSrc == null ||
            outlineX.Length != outlineY.Length || outlineX.Length != outlineSrc.Length)
        {
            outlineX = null;
            outlineY = null;
            outlineSrc = null;
        }
        try
        {
            return CheekOutline.ObservedPoints(pts, outlineX, outlineY, outlineSrc, farLeft);
        }
        catch (Exception exc)
        {
            System.Diagnostics.Trace.TraceWarning(
                "The far-cheek search failed, so the face has no observed cheek points: {0}: {1}",
                exc.GetType().Name, exc.Message);
            return null;
        }
    }

    /// <summary><see cref="ObserveFarCheek(Vec2d[], float[], int[], byte[], bool)"/> of float landmarks.</summary>
    public static CheekOutline.ObliqueCheek? ObserveFarCheek(Vec2[]? landmarks, float[]? outlineX,
        int[]? outlineY, byte[]? outlineSrc, bool farLeft) =>
        ObserveFarCheek(CheekOutline.ToVec2d(landmarks), outlineX, outlineY, outlineSrc, farLeft);

    /// <summary>
    /// The ten cheek landmarks from what was located: rows 0-5 from the frontal
    /// constructions when there are any, the far side's rows 6-9 from the observed points
    /// when there are any (the submalar point only when its depth isn't NaN), NaN
    /// elsewhere. A new array every call.
    /// </summary>
    /// <param name="frontal">The frontal constructions, or null.</param>
    /// <param name="observed">The far cheek's observed points, or null.</param>
    /// <param name="farLeft">Whether the far cheek is the subject's left: the side whose rows the observed points fill.</param>
    /// <returns>The ten points in <see cref="CheekLandmarkNames"/> order, px, NaN where absent.</returns>
    public static Vec2d[] LandmarkArray(FrontalCheek? frontal, CheekOutline.ObliqueCheek? observed, bool farLeft)
    {
        var points = new Vec2d[CheekLandmarkNames.Count];
        for (int i = 0; i < points.Length; i++) points[i] = Vec2d.NaN;
        if (frontal != null)
        {
            points[CheekLmZygionR] = frontal.ZygionR;
            points[CheekLmZygionL] = frontal.ZygionL;
            points[CheekLmCheekRefR] = frontal.CheekRefR;
            points[CheekLmCheekRefL] = frontal.CheekRefL;
            points[CheekLmSubmalarRefR] = frontal.SubmalarRefR;
            points[CheekLmSubmalarRefL] = frontal.SubmalarRefL;
        }
        if (observed != null)
        {
            points[farLeft ? CheekLmApexL : CheekLmApexR] = observed.Apex;
            if (!double.IsNaN(observed.SubmalarDepth))
                points[farLeft ? CheekLmSubmalarL : CheekLmSubmalarR] = observed.Submalar;
        }
        return points;
    }

    /// <summary>
    /// What a face's landmarks and view give the cheek landmarks: the far cheek's observed
    /// points (<see cref="ObserveFarCheek(Vec2d[], float[], int[], byte[], bool)"/>) on a
    /// quarter face, the frontal constructions (<see cref="FrontalConstructions(Vec2d[])"/>)
    /// on any other face that isn't a profile, nothing on a profile; the other result is
    /// always null, and so is either with fewer than 98 landmarks, a landmark that isn't
    /// finite or coincident pupils. Never throws.
    /// </summary>
    /// <param name="landmarks">The 98 WFLW landmarks, px of the standard image.</param>
    /// <param name="quarter">Whether the face is a quarter face.</param>
    /// <param name="lateral">Whether it is a profile; read only when quarter is false.</param>
    /// <param name="farLeft">On a quarter face, whether the far cheek is the subject's left.</param>
    /// <param name="outlineX">The far cheek's outline, or null for the contour fallback.</param>
    /// <param name="outlineY">Its rows.</param>
    /// <param name="outlineSrc">Each row's source.</param>
    public static (FrontalCheek? Frontal, CheekOutline.ObliqueCheek? Observed) Locate(Vec2d[]? landmarks,
        bool quarter, bool lateral, bool farLeft, float[]? outlineX, int[]? outlineY, byte[]? outlineSrc)
    {
        if (quarter) return (null, ObserveFarCheek(landmarks, outlineX, outlineY, outlineSrc, farLeft));
        if (lateral) return (null, null);
        return (FrontalConstructions(landmarks), null);
    }

    /// <summary><see cref="Locate(Vec2d[], bool, bool, bool, float[], int[], byte[])"/> of float landmarks.</summary>
    public static (FrontalCheek? Frontal, CheekOutline.ObliqueCheek? Observed) Locate(Vec2[]? landmarks,
        bool quarter, bool lateral, bool farLeft, float[]? outlineX, int[]? outlineY, byte[]? outlineSrc) =>
        Locate(CheekOutline.ToVec2d(landmarks), quarter, lateral, farLeft, outlineX, outlineY, outlineSrc);

    /// <summary>
    /// The ten cheek landmarks: <see cref="LandmarkArray"/> of
    /// <see cref="Locate(Vec2d[], bool, bool, bool, float[], int[], byte[])"/>'s two results,
    /// with a pose for the view (<see cref="FacePose.Quarter"/> is a quarter face,
    /// <see cref="FacePose.Frontal"/> a frontal one, and <see cref="FacePose.Lateral"/> or an
    /// undecided <see cref="FacePose.Detect"/> gives all NaN, as does a quarter or frontal
    /// face whose landmarks give no frame). Never throws.
    /// </summary>
    public static Vec2d[] LocateAll(Vec2d[]? landmarks, FacePose pose, bool farLeft,
        float[]? outlineX, int[]? outlineY, byte[]? outlineSrc)
    {
        var (frontal, observed) = Locate(landmarks, pose == FacePose.Quarter, pose != FacePose.Frontal,
            farLeft, outlineX, outlineY, outlineSrc);
        return LandmarkArray(frontal, observed, farLeft);
    }

    /// <summary><see cref="LocateAll(Vec2d[], FacePose, bool, float[], int[], byte[])"/> of float landmarks.</summary>
    public static Vec2d[] LocateAll(Vec2[]? landmarks, FacePose pose, bool farLeft,
        float[]? outlineX, int[]? outlineY, byte[]? outlineSrc) =>
        LocateAll(CheekOutline.ToVec2d(landmarks), pose, farLeft, outlineX, outlineY, outlineSrc);
}
