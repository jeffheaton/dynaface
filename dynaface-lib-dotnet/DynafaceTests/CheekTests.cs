using System.Globalization;
using System.Text;
using static Cheek;
using static DynafaceTests.CheekFixtures;

namespace DynafaceTests;

// Ports dynaface-lib-python's tests/test_cheek.py: the pupil frame, the frontal
// constructions, the calibration tables, the pose cues and the quarter classification,
// on hand-placed points and on the landmarks recorded in tests_data/cheek_landmarks.json.
// TestLocateAll's cases stub cheek_outline.observed_points with unittest.mock; static
// methods can't be stubbed here, so those cases run on the real search (an outline whose
// points are known, or one it declines) and check the same rows, and the one that only
// checks the swallowed exception's log line is left out. SharedConstants checks every
// constant of cheek.py and cheek_outline.py against the Python source itself.
public class CheekTests
{
    static readonly string[] Sweep =
    {
        "yaw0", "yaw5", "yaw10", "yaw15", "yaw20", "yaw25", "yaw30", "yaw35", "yaw40",
        "yaw42", "yaw45", "yaw48", "yaw50", "yaw55", "yaw60",
    };
    // The negative turns: the tables were measured on the positive sweep.
    static readonly string[] Mirrored = { "yaw-15", "yaw-30", "yaw-45", "yaw-50" };

    static readonly Vec2d PupilR = new(380.0, 480.0), PupilL = new(640.0, 480.0);
    const double Ipd = 260.0;

    // The render's true turn from its name: yaw-45 -> -45, yaw45_hollow3 -> 45.
    static double TrueYaw(string name) =>
        double.Parse(name.Substring(3).Split('_')[0], CultureInfo.InvariantCulture);

    // A symmetric 98-point face: pupils at (380, 480) and (640, 480), lateral canthi at
    // (340, 480) and (680, 480), a half-ellipse contour (semi-axes 200 across and 300 down
    // from the pupil line) as points 0-32 with the chin (16) at (510, 780), the nose tip
    // (54) and the inner-lip midpoints (90, 94) on the midline, and every other landmark
    // parked at the center.
    static Vec2d[] StickFace()
    {
        double cx = 510.0, cy = 480.0, a = 200.0, b = 300.0;
        var pts = Enumerable.Repeat(new Vec2d(cx, cy + 100.0), 98).ToArray();
        for (int i = 0; i < 17; i++)
        {
            double phi = Math.PI + (Math.PI / 2.0) * i / 16.0;
            pts[i] = new(cx + a * Math.Cos(phi), cy - b * Math.Sin(phi));
        }
        for (int i = 0; i < 16; i++) pts[32 - i] = new(2.0 * cx - pts[i].X, pts[i].Y);
        pts[LmCanthusR] = new(340.0, 480.0);
        pts[LmCanthusL] = new(680.0, 480.0);
        pts[96] = PupilR;
        pts[97] = PupilL;
        pts[LmNoseTip] = new(cx, 620.0);
        pts[LmAlaR] = new(480.0, 640.0);
        pts[LmAlaL] = new(540.0, 640.0);
        pts[LmStomion.Upper] = new(cx, 700.0);
        pts[LmStomion.Lower] = new(cx, 700.0);
        return pts;
    }

    // Every point rotated about center, as a photo rolled by that angle.
    static Vec2d Rotated(Vec2d p, double degrees, Vec2d center)
    {
        double theta = degrees * (Math.PI / 180.0);
        double c = Math.Cos(theta), s = Math.Sin(theta);
        Vec2d d = p - center;
        return new Vec2d(d.X * c + d.Y * -s, d.X * s + d.Y * c) + center;
    }

    static Vec2d[] Rotated(Vec2d[] points, double degrees, Vec2d? center = null) =>
        points.Select(p => Rotated(p, degrees, center ?? new Vec2d(512.0, 512.0))).ToArray();

    // The landmarks with the nose tip slid along the pupil axis so that LandmarkYaw reads
    // target: the inverse of NoseOffsetCalibration gives the offset in canthus spans.
    static Vec2d[] WithLandmarkYaw(Vec2d[] landmarks, double target)
    {
        var pts = (Vec2d[])landmarks.Clone();
        var frame = PupilFrame(pts)!;
        double[] offsets = NoseOffsetCalibration.Select(r => r.Offset).ToArray();
        double[] trues = NoseOffsetCalibration.Select(r => r.TrueYaw).ToArray();
        double wanted = Math.Abs(CheekNumerics.Interp(Math.Abs(target), trues, offsets)) * Math.Sign(target);
        double canthusR = frame.Offset(pts[LmCanthusR]), canthusL = frame.Offset(pts[LmCanthusL]);
        double span = canthusL - canthusR;
        double tipOffset = (canthusR + canthusL) / 2.0 + wanted * span;
        pts[LmNoseTip] += frame.Across * (tipOffset - frame.Offset(pts[LmNoseTip]));
        return pts;
    }

    static bool AllNaN(IEnumerable<Vec2d> points) => points.All(p => double.IsNaN(p.X) && double.IsNaN(p.Y));

    // ------------------------------------------------------------ TestFrame

    public class FrameRule
    {
        [Fact]
        public void LevelFrame()
        {
            var frame = PupilFrame(StickFace())!;
            Close(new Vec2d(510.0, 480.0), frame.Origin);
            Close(new Vec2d(1.0, 0.0), frame.Across);
            Close(new Vec2d(0.0, 1.0), frame.Down);
            Near(Ipd, frame.Ipd);
            Near(100.0, frame.Level(new Vec2d(700.0, 580.0)));
            Near(190.0, frame.Offset(new Vec2d(700.0, 580.0)));
            Near(-110.0, frame.Offset(new Vec2d(400.0, 580.0)));
            Close(new Vec2d(700.0, 580.0), frame.Point(190.0, 100.0));
        }

        [Fact]
        public void RolledFrame()
        {
            // A 30 degree roll: the frame follows the pupils, the numbers don't.
            var center = new Vec2d(510.0, 480.0);
            var frame = PupilFrame(Rotated(StickFace(), 30.0, center))!;
            Close(new Vec2d(510.0, 480.0), frame.Origin, atol: 1e-9);
            Close(new Vec2d(Math.Cos(Math.PI / 6), Math.Sin(Math.PI / 6)), frame.Across);
            Near(0.0, Vec2d.Dot(frame.Across, frame.Down));
            Near(Ipd, frame.Ipd);
            Vec2d rolled = Rotated(new Vec2d(700.0, 580.0), 30.0, center);
            Near(100.0, frame.Level(rolled));
            Near(190.0, frame.Offset(rolled));
            Close(rolled, frame.Point(190.0, 100.0));
        }

        [Fact]
        public void CoincidentPupilsGiveNone()
        {
            var pts = StickFace();
            pts[97] = pts[96];
            Assert.Null(PupilFrame(pts));
            Assert.Null(FrontalConstructions(pts));
        }

        [Fact]
        public void FewerThan98LandmarksGiveNone()
        {
            var pts = StickFace().Take(97).ToArray();
            Assert.Null(PupilFrame(pts));
            Assert.Null(FrontalConstructions(pts));
            Assert.Null(PupilFrame(Array.Empty<Vec2d>()));
            Assert.Null(FrontalConstructions(Array.Empty<Vec2d>()));
        }

        [Fact]
        public void NonFinitePupilsGiveNone()
        {
            var pts = StickFace();
            pts[96] = new(double.NaN, 480.0);
            Assert.Null(PupilFrame(pts));
            pts[96] = new(double.PositiveInfinity, 480.0);
            Assert.Null(PupilFrame(pts));
        }

        [Fact]
        public void NonFiniteLandmarkGivesNone()
        {
            // A NaN or infinity anywhere means no frame and no point: the cues read NaN,
            // the classification says frontal, and the far side falls back to the yaw.
            var good = PupilFrame(StickFace())!;
            foreach (var (index, value) in new[] { (LmChin, double.NaN), (LmCanthusL, double.PositiveInfinity), (33, double.NegativeInfinity) })
            {
                var pts = StickFace();
                pts[index].Y = value;
                Assert.Null(PupilFrame(pts));
                Assert.Null(FrontalConstructions(pts));
                Assert.True(double.IsNaN(NoseOffset(pts, good)));
                Assert.True(double.IsNaN(LandmarkYaw(pts, good)));
                Assert.True(double.IsNaN(NoseMargin(pts, good)));
                Assert.Equal((false, false), ClassifyQuarter(pts, 28.0));
                Assert.True(FarLeftOf(pts, yawReported: 28.0));
                Assert.False(FarLeftOf(pts));
                foreach (var pose in new[] { FacePose.Frontal, FacePose.Quarter })
                {
                    var points = LocateAll(pts, pose, true, null, null, null);
                    Assert.Equal(10, points.Length);
                    Assert.True(AllNaN(points));
                }
            }
        }

        [Fact]
        public void AcceptsIntegerFloatLandmarks()
        {
            // The pipeline's landmarks are floats (whole pixels in Python's integer tuples).
            var pts = StickFace().Select(p => new Vec2((int)p.X, (int)p.Y)).ToArray();
            var frame = PupilFrame(pts);
            Assert.NotNull(frame);
            Near(Ipd, frame!.Ipd);
        }
    }

    // ------------------------------------------------------------ TestPolylines

    public class Polylines
    {
        static readonly Vec2d[] Line = { new(0.0, 0.0), new(0.0, 10.0), new(0.0, 20.0) };
        static readonly Vec2d[] Zigzag = { new(0.0, 0.0), new(20.0, 10.0), new(0.0, 20.0), new(20.0, 30.0) };
        static double[] D(params double[] v) => v;

        [Fact]
        public void CrossPolylineInterpolates()
        {
            var (point, status) = CrossPolyline(Line, D(0.0, 10.0, 20.0), 5.0);
            Assert.Equal(0, status);
            Close(new Vec2d(0.0, 5.0), point);
            (point, status) = CrossPolyline(Line, D(0.0, 10.0, 20.0), 17.5);
            Assert.Equal(0, status);
            Close(new Vec2d(0.0, 17.5), point);
        }

        [Fact]
        public void CrossPolylineClipsToNearerEnd()
        {
            var (point, status) = CrossPolyline(Line, D(0.0, 10.0, 20.0), -5.0);
            Assert.Equal(-1, status);
            Close(new Vec2d(0.0, 0.0), point);
            (point, status) = CrossPolyline(Line, D(0.0, 10.0, 20.0), 25.0);
            Assert.Equal(1, status);
            Close(new Vec2d(0.0, 20.0), point);
        }

        [Fact]
        public void CrossPolylineSkipsFlatSegment()
        {
            var (point, status) = CrossPolyline(Line, D(5.0, 5.0, 10.0), 5.0);
            Assert.Equal(0, status);
            Close(new Vec2d(0.0, 10.0), point);
            // A polyline that is flat everywhere clips rather than dividing by zero.
            (point, status) = CrossPolyline(Line, D(5.0, 5.0, 5.0), 5.0);
            Assert.Equal(-1, status);
            Close(new Vec2d(0.0, 0.0), point);
        }

        [Fact]
        public void ContourBelowTakesTheLowerCrossing()
        {
            double[] offsets = Zigzag.Select(p => p.X).ToArray(), levels = Zigzag.Select(p => p.Y).ToArray();
            Close(new Vec2d(10.0, 25.0), ContourBelow(Zigzag, offsets, levels, 10.0));
            // Reversed order: the same lowest crossing, so it isn't "the last one".
            Close(new Vec2d(10.0, 25.0), ContourBelow(Zigzag.Reverse().ToArray(), offsets.Reverse().ToArray(),
                levels.Reverse().ToArray(), 10.0));
        }

        [Fact]
        public void ContourBelowFallsBackToNearestVertex()
        {
            double[] offsets = Zigzag.Select(p => p.X).ToArray(), levels = Zigzag.Select(p => p.Y).ToArray();
            Close(new Vec2d(20.0, 10.0), ContourBelow(Zigzag, offsets, levels, 50.0));
            Close(new Vec2d(0.0, 0.0), ContourBelow(Zigzag, offsets, levels, -3.0));
        }

        [Fact]
        public void ContourBelowInterpolates()
        {
            Vec2d[] points = { new(0.0, 0.0), new(10.0, 10.0), new(10.0, 30.0) };
            Close(new Vec2d(5.0, 5.0), ContourBelow(points, points.Select(p => p.X).ToArray(),
                points.Select(p => p.Y).ToArray(), 5.0));
        }
    }

    // ------------------------------------------------------------ TestStickFace

    public class StickFaceConstructions
    {
        readonly Vec2d[] pts = StickFace();
        readonly FrontalCheek fc;
        readonly Cheek.Frame frame;

        public StickFaceConstructions()
        {
            fc = FrontalConstructions(pts)!;
            Assert.NotNull(fc);
            frame = fc.Frame;
        }

        [Fact]
        public void ZygionLevelIsExactlyTheIpdFraction()
        {
            Near(ZygionLevelIpd * Ipd, fc.ZygionLevel);
            Near(78.0, frame.Level(fc.ZygionR));
            Near(78.0, frame.Level(fc.ZygionL));
        }

        [Fact]
        public void ZygionCrossesUnclipped()
        {
            Assert.Equal(ClipNone, fc.ZygionRClip);
            Assert.Equal(ClipNone, fc.ZygionLClip);
            // The level 78 lies between contour points 2 (58.5) and 3 (87.2).
            Assert.True(pts[2].Y < fc.ZygionR.Y && fc.ZygionR.Y < pts[3].Y);
            Assert.True(pts[2].X < fc.ZygionR.X && fc.ZygionR.X < pts[3].X);
        }

        [Fact]
        public void SidesMirror()
        {
            Near(fc.ZygionRPx, fc.ZygionLPx, 6);
            Assert.True(fc.ZygionRPx > 0.0);
            Near(fc.ZygionR.X, 1020.0 - fc.ZygionL.X, 6);
            Near(fc.ZygionR.Y, fc.ZygionL.Y, 6);
            Near(fc.CheekRefRPx, fc.CheekRefLPx, 6);
            Near(fc.FootR.Y, fc.FootL.Y, 6);
            Near(fc.FootR.X, 1020.0 - fc.FootL.X, 6);
            Near(fc.SubmalarRefR.X, 1020.0 - fc.SubmalarRefL.X, 6);
            Near(fc.SubmalarRefR.Y, fc.SubmalarRefL.Y, 6);
        }

        [Fact]
        public void CanthusVerticalEnds()
        {
            Close(new Vec2d(340.0, 480.0), fc.CanthusR);
            Close(new Vec2d(680.0, 480.0), fc.CanthusL);
            // The feet sit exactly on the verticals, on the contour below the pupils.
            Near(340.0, fc.FootR.X, 9);
            Near(680.0, fc.FootL.X, 9);
            Assert.True(fc.FootR.Y > 620.0 && fc.FootR.Y < 640.0);
        }

        [Fact]
        public void ExactThirdsAndHalves()
        {
            double drop = frame.Level(fc.FootR) - frame.Level(fc.CanthusR);
            Near(drop / 3.0, fc.CheekRefRPx, 9);
            Near(340.0, fc.CheekRefR.X, 9);
            Near(480.0 + drop / 3.0, fc.CheekRefR.Y, 9);
            Near(220.0, fc.StomionLevel);
            Near(340.0, fc.SubmalarRefR.X, 9);
            Near((frame.Level(fc.CheekRefR) + 220.0) / 2.0, frame.Level(fc.SubmalarRefR), 9);
        }

        [Fact]
        public void ClipTopWhenTheContourTopsAreBelowTheLevel()
        {
            var moved = StickFace();
            foreach (int i in RightContourTop.Concat(LeftContourTop)) moved[i].Y += 100.0;
            var c = FrontalConstructions(moved)!;
            Assert.Equal(ClipTop, c.ZygionRClip);
            Assert.Equal(ClipTop, c.ZygionLClip);
            Close(moved[0], c.ZygionR);
            Close(moved[32], c.ZygionL);
        }

        [Fact]
        public void ClipBottomWhenPoint4IsAboveTheLevel()
        {
            var moved = StickFace();
            foreach (int i in RightContourTop.Concat(LeftContourTop)) moved[i].Y -= 100.0;
            var c = FrontalConstructions(moved)!;
            Assert.Equal(ClipBottom, c.ZygionRClip);
            Assert.Equal(ClipBottom, c.ZygionLClip);
            Close(moved[4], c.ZygionR);
            Close(moved[28], c.ZygionL);
        }

        [Fact]
        public void ZygionLevelParameter()
        {
            var (point, clip) = Zygion(pts, frame, left: true, levelIpd: 0.2);
            Assert.Equal(ClipNone, clip);
            Near(52.0, frame.Level(point));
            // Nearer the pupil line the half-ellipse is wider: further out on each side.
            Assert.True(point.X > fc.ZygionL.X);
            (point, clip) = Zygion(pts, frame, left: false, levelIpd: 0.2);
            Assert.Equal(ClipNone, clip);
            Near(52.0, frame.Level(point));
            Assert.True(point.X < fc.ZygionR.X);
        }

        [Fact]
        public void PoseCuesOnTheMidlineNose()
        {
            Near(0.0, NoseOffset(pts, frame));
            Near(0.0, LandmarkYaw(pts, frame));
            Near(0.5, NoseMargin(pts, frame));
            Assert.False(FarLeftOf(pts));
        }

        [Fact]
        public void PoseCuesWithTheNoseMoved()
        {
            // A tip 0.25 of the canthus span toward image right.
            var moved = StickFace();
            moved[LmNoseTip].X += 0.25 * 340.0;
            var f = PupilFrame(moved)!;
            Near(0.25, NoseOffset(moved, f));
            Assert.True(FarLeftOf(moved));
            // The contour spans 310 to 710: the tip at 595 keeps 115 / 400 of the width.
            Near(115.0 / 400.0, NoseMargin(moved, f));
            moved[LmNoseTip].X = 730.0;
            Near(-20.0 / 400.0, NoseMargin(moved, f));
            moved[LmNoseTip].X = 510.0 - 0.25 * 340.0;
            Near(-0.25, NoseOffset(moved, f));
            Assert.False(FarLeftOf(moved));
        }
    }

    // ------------------------------------------------------------ TestImg4

    // The pipeline landmarks of img4 (yaw 1.1) reproduce the frontal reference readings in
    // millimeters through the fixture's pix2mm.
    public class Img4
    {
        static readonly Vec2d[] Pts = PipelineLandmarks("img4");
        static readonly double Mm = Entry("img4").GetProperty("pipeline").GetProperty("pix2mm").GetDouble();
        static readonly FrontalCheek Fc = FrontalConstructions(Pts)!;

        [Fact]
        public void ZygionWidth()
        {
            Assert.Equal(ClipNone, Fc.ZygionRClip);
            Assert.Equal(ClipNone, Fc.ZygionLClip);
            Within(18.9, Fc.ZygionLevel * Mm, 0.3);
            Within(65.5, Fc.ZygionRPx * Mm, 0.3);
            Within(65.1, Fc.ZygionLPx * Mm, 0.3);
            Within(130.6, (Fc.ZygionRPx + Fc.ZygionLPx) * Mm, 0.3);
        }

        [Fact]
        public void ZygionCrossings()
        {
            var frame = Fc.Frame;
            // Segment 1-2 at t 0.78 on the right, 31-30 at t 0.83 on the left.
            double tR = (Fc.ZygionLevel - frame.Level(Pts[1])) / (frame.Level(Pts[2]) - frame.Level(Pts[1]));
            double tL = (Fc.ZygionLevel - frame.Level(Pts[31])) / (frame.Level(Pts[30]) - frame.Level(Pts[31]));
            Within(0.78, tR, 0.01);
            Within(0.83, tL, 0.01);
            Close(Pts[1] + tR * (Pts[2] - Pts[1]), Fc.ZygionR);
            Close(Pts[31] + tL * (Pts[30] - Pts[31]), Fc.ZygionL);
            // The contour tops sit 2.7 and 3.1 mm below the pupil line.
            Within(2.7, frame.Level(Pts[0]) * Mm, 0.1);
            Within(3.1, frame.Level(Pts[32]) * Mm, 0.1);
        }

        [Fact]
        public void ReferencePoints()
        {
            var frame = Fc.Frame;
            Within(28.8, Fc.CheekRefRPx * Mm, 0.3);
            Within(28.8, Fc.CheekRefLPx * Mm, 0.3);
            foreach (var (canthus, submalar) in new[] { (Fc.CanthusR, Fc.SubmalarRefR), (Fc.CanthusL, Fc.SubmalarRefL) })
                Within(48.3, (frame.Level(submalar) - frame.Level(canthus)) * Mm, 0.3);
            foreach (var (canthus, cheekRef, submalar) in new[]
                     { (Fc.CanthusR, Fc.CheekRefR, Fc.SubmalarRefR), (Fc.CanthusL, Fc.CheekRefL, Fc.SubmalarRefL) })
            {
                Near(frame.Offset(canthus), frame.Offset(cheekRef), 9);
                Near(frame.Offset(canthus), frame.Offset(submalar), 9);
            }
        }

        [Fact]
        public void PoseCues()
        {
            var frame = Fc.Frame;
            Within(0.0, NoseOffset(Pts, frame), 0.02);
            Assert.True(Math.Abs(LandmarkYaw(Pts, frame)) < 2.0);
            Within(0.49, NoseMargin(Pts, frame), 0.02);
        }
    }

    // ------------------------------------------------------------ TestRollInvariance

    public class RollInvariance
    {
        [Fact]
        public void FrontalConstructions15Degrees()
        {
            var pts = PipelineLandmarks("img4");
            var fc = FrontalConstructions(pts)!;
            var fr = FrontalConstructions(Rotated(pts, 15.0))!;
            Within(fc.ZygionLevel, fr.ZygionLevel, 1e-6);
            Within(fc.ZygionRPx, fr.ZygionRPx, 1e-6);
            Within(fc.ZygionLPx, fr.ZygionLPx, 1e-6);
            Within(fc.CheekRefRPx, fr.CheekRefRPx, 1e-6);
            Within(fc.CheekRefLPx, fr.CheekRefLPx, 1e-6);
            Within(fc.StomionLevel, fr.StomionLevel, 1e-6);
            Assert.Equal(fc.ZygionRClip, fr.ZygionRClip);
            Assert.Equal(fc.ZygionLClip, fr.ZygionLClip);
            var center = new Vec2d(512.0, 512.0);
            foreach (var (a, b) in new[]
                     {
                         (fc.ZygionR, fr.ZygionR), (fc.ZygionL, fr.ZygionL), (fc.CanthusR, fr.CanthusR),
                         (fc.CanthusL, fr.CanthusL), (fc.FootR, fr.FootR), (fc.FootL, fr.FootL),
                         (fc.CheekRefR, fr.CheekRefR), (fc.CheekRefL, fr.CheekRefL),
                         (fc.SubmalarRefR, fr.SubmalarRefR), (fc.SubmalarRefL, fr.SubmalarRefL),
                     })
                Close(Rotated(a, 15.0, center), b, atol: 1e-6);
            Within(fc.Frame.Ipd, fr.Frame.Ipd, 1e-6);
        }

        [Fact]
        public void PoseCues()
        {
            foreach (string name in new[] { "img4", "yaw45", "yaw-45" })
            {
                var pts = RawLandmarks(name);
                var frame = PupilFrame(pts)!;
                var rolled = Rotated(pts, 15.0, new Vec2d(400.0, 400.0));
                var frameR = PupilFrame(rolled)!;
                Near(NoseOffset(pts, frame), NoseOffset(rolled, frameR), 9);
                Near(LandmarkYaw(pts, frame), LandmarkYaw(rolled, frameR), 9);
                Near(NoseMargin(pts, frame), NoseMargin(rolled, frameR), 9);
                Assert.Equal(ClassifyQuarter(pts, RawYaw(name)), ClassifyQuarter(rolled, RawYaw(name)));
            }
        }
    }

    // ------------------------------------------------------------ TestCalibrationTables

    public class CalibrationTables
    {
        [Fact]
        public void TablesAreIncreasing()
        {
            foreach (var table in new[]
                     {
                         YawCalibration.Select(r => (r.Raw, r.TrueYaw)).ToArray(),
                         NoseOffsetCalibration.Select(r => (r.Offset, r.TrueYaw)).ToArray(),
                     })
            {
                var xs = table.Select(r => r.Item1).ToArray();
                var ys = table.Select(r => r.Item2).ToArray();
                Assert.Equal(xs.OrderBy(x => x).ToArray(), xs);
                Assert.Equal(xs.Length, xs.Distinct().Count());
                Assert.Equal(ys.OrderBy(y => y).ToArray(), ys);
                Assert.Equal(0.0, table[0].Item2);
            }
            Assert.Equal((1.5, 0.0), YawCalibration[0]);
            Assert.True(YawRawOffset < YawCalibration[0].Raw);
            Assert.True(YawRawOffset > 0.0);
            Assert.Equal((0.0, 0.0), NoseOffsetCalibration[0]);
        }

        [Fact]
        public void TableRows()
        {
            Assert.Contains((28.28, 45.0), YawCalibration);
            Assert.Contains((44.31, 65.0), YawCalibration);
            Assert.Contains((25.77, 42.0), YawCalibration);
            Assert.Contains((30.54, 48.0), YawCalibration);
            Assert.Equal((44.31, 65.0), YawCalibration[^1]);
            Assert.Contains((0.4449, 45.0), NoseOffsetCalibration);
            Assert.Contains((0.4181, 42.0), NoseOffsetCalibration);
            Assert.Contains((0.514, 48.0), NoseOffsetCalibration);
            Assert.Equal((0.7696, 60.0), NoseOffsetCalibration[^1]);
        }

        [Fact]
        public void CalibrateYawReadsTheTable()
        {
            foreach (var (reported, trueYaw) in YawCalibration)
            {
                Near(trueYaw, CalibrateYaw(reported), 9);
                // The mirrored reading: as far below the offset as this is above it.
                Near(-trueYaw, CalibrateYaw(2.0 * YawRawOffset - reported), 9);
            }
            Near(45.0, CalibrateYaw(28.28));
            Near(65.0, CalibrateYaw(44.31));
            Near(0.0, CalibrateYaw(1.5));
            Near(46.5, CalibrateYaw((28.28 + 30.54) / 2));
        }

        [Fact]
        public void CalibrateYawBeyondTheTable()
        {
            Near(65.0 + (50.0 - 44.31) * YawSlopeBeyond, CalibrateYaw(50.0));
            Near(-(65.0 + (50.0 + 2.0 * YawRawOffset - 44.31) * YawSlopeBeyond), CalibrateYaw(-50.0));
            // The offset itself is a level head; a reading of 0 is a turn toward the
            // subject's right, read on the table's first segment.
            Assert.Equal(0.0, CalibrateYaw(YawRawOffset));
            Near(-5.0 * (YawRawOffset - (1.5 - YawRawOffset)) / (3.04 - 1.5), CalibrateYaw(0.0));
            Within(-3.12, CalibrateYaw(0.0), 0.01);
        }

        [Fact]
        public void CalibrateYawIsOddAboutTheOffsetAndMonotonic()
        {
            for (int i = 0; i <= 120; i++)
            {
                double reported = i * 0.5; // np.linspace(0, 60, 121)
                Near(-CalibrateYaw(YawRawOffset + reported), CalibrateYaw(YawRawOffset - reported));
            }
            var values = Enumerable.Range(0, 1201).Select(i => CalibrateYaw(-60.0 + i * 0.1)).ToArray();
            for (int i = 1; i < values.Length; i++) Assert.True(values[i] >= values[i - 1]);
            Assert.True(CalibrateYaw(0.0) < 0.0);
            Assert.Equal(0.0, CalibrateYaw(1.5)); // the level head's row
            Assert.True(CalibrateYaw(2.0) > 0.0);
            // Never -0.0: a level head has no sign.
            Assert.False(double.IsNegative(CalibrateYaw(1.08)));
        }

        [Fact]
        public void NaNPassesThrough()
        {
            Assert.True(double.IsNaN(CalibrateYaw(double.NaN)));
            var pts = StickFace();
            var frame = PupilFrame(pts)!;
            pts[LmNoseTip] = new(double.NaN, double.NaN);
            Assert.True(double.IsNaN(NoseOffset(pts, frame)));
            Assert.True(double.IsNaN(LandmarkYaw(pts, frame)));
            Assert.True(double.IsNaN(NoseMargin(pts, frame)));
            Assert.Equal((false, false), ClassifyQuarter(pts, 28.28));
            Assert.False(FarLeftOf(pts));
        }

        [Fact]
        public void LandmarkYawReadsTheTable()
        {
            var baseFace = StickFace();
            foreach (var (offset, trueYaw) in NoseOffsetCalibration)
                foreach (double sign in new[] { 1.0, -1.0 })
                {
                    var pts = (Vec2d[])baseFace.Clone();
                    pts[LmNoseTip].X += sign * offset * 340.0;
                    var frame = PupilFrame(pts)!;
                    Near(sign * offset, NoseOffset(pts, frame));
                    Near(sign * trueYaw, LandmarkYaw(pts, frame), 9);
                }
            var p = (Vec2d[])baseFace.Clone();
            p[LmNoseTip].X += 0.4449 * 340.0;
            Near(45.0, LandmarkYaw(p, PupilFrame(p)!));
        }

        [Fact]
        public void LandmarkYawBeyondTheTable()
        {
            var pts = StickFace();
            pts[LmNoseTip].X -= 0.9 * 340.0;
            Near(-(60.0 + (0.9 - 0.7696) * NoseOffsetSlopeBeyond), LandmarkYaw(pts, PupilFrame(pts)!));
        }

        [Fact]
        public void LandmarkYawIsMonotonic()
        {
            var baseFace = StickFace();
            var values = new List<double>();
            for (int i = 0; i <= 200; i++)
            {
                var pts = (Vec2d[])baseFace.Clone();
                pts[LmNoseTip].X += i / 200.0 * 340.0;
                values.Add(LandmarkYaw(pts, PupilFrame(pts)!));
            }
            for (int i = 1; i < values.Count; i++) Assert.True(values[i] >= values[i - 1]);
        }
    }

    // ------------------------------------------------------------ TestRenderSweep

    public class RenderSweep
    {
        static (Vec2d[] Pts, Cheek.Frame Frame) Cue(string name)
        {
            var pts = RawLandmarks(name);
            var frame = PupilFrame(pts);
            Assert.NotNull(frame);
            return (pts, frame!);
        }

        [Fact]
        public void NoseOffsetReproducesTheFixtureCue()
        {
            foreach (string name in ImageNames)
            {
                var (pts, frame) = Cue(name);
                Within(Entry(name).GetProperty("cues").GetProperty("nose_offset").GetDouble(), NoseOffset(pts, frame), 5e-5, name);
            }
        }

        [Fact]
        public void LandmarkYawAgainstTheTrueTurn()
        {
            foreach (string name in Sweep.Skip(1))
            {
                var (pts, frame) = Cue(name);
                Within(TrueYaw(name), LandmarkYaw(pts, frame), 0.05, name);
            }
            foreach (string name in Mirrored.Concat(new[] { "yaw45_hollow1.5", "yaw45_hollow3", "yaw45_hollow4.5", "yaw50_hollow3" }))
            {
                var (pts, frame) = Cue(name);
                Within(TrueYaw(name), LandmarkYaw(pts, frame), 1.5, name);
            }
            var (p0, f0) = Cue("yaw0");
            Within(2.26, LandmarkYaw(p0, f0), 0.05);
            var (pb, fb) = Cue("yaw45_bump3");
            Within(42.45, LandmarkYaw(pb, fb), 0.05);
        }

        [Fact]
        public void CalibrateYawAgainstTheTrueTurn()
        {
            foreach (string name in Sweep.Concat(new[] { "yaw65" }))
                Within(TrueYaw(name), CalibrateYaw(RawYaw(name)), 0.05, name);
            foreach (string name in Mirrored.Concat(new[] { "yaw45_hollow1.5", "yaw45_hollow3", "yaw45_hollow4.5", "yaw50_hollow3", "yaw45_bump3" }))
                Within(TrueYaw(name), CalibrateYaw(RawYaw(name)), 1.5, name);
            Within(-31.41, CalibrateYaw(RawYaw("yaw-30")), 0.05);
        }

        [Fact]
        public void YawOffsetIsFittedToTheLevelHeadAndTheMirroredPairs()
        {
            var estimates = new List<double> { RawYaw("yaw0") };
            foreach (int turn in new[] { 15, 30, 45, 50 })
                estimates.Add((RawYaw($"yaw{turn}") + RawYaw($"yaw-{turn}")) / 2.0);
            Within(1.5, estimates[0], 0.005);
            foreach (double estimate in estimates.Skip(1)) Assert.True(estimate > 0.5 && estimate < 1.5);
            Within(estimates.Average(), YawRawOffset, 0.005);
        }

        [Fact]
        public void Mirrored45Pair()
        {
            var (pp, fp) = Cue("yaw45");
            var (pn, fn) = Cue("yaw-45");
            double yawP = LandmarkYaw(pp, fp), yawN = LandmarkYaw(pn, fn);
            Assert.True(yawP > 0.0);
            Assert.True(yawN < 0.0);
            Within(-0.06, Math.Abs(yawP) - Math.Abs(yawN), 0.05);
            double modelP = CalibrateYaw(RawYaw("yaw45")), modelN = CalibrateYaw(RawYaw("yaw-45"));
            Within(0.23, Math.Abs(modelP) - Math.Abs(modelN), 0.05);
            Assert.True(Math.Abs(Math.Abs(NoseOffset(pp, fp)) - Math.Abs(NoseOffset(pn, fn))) < 0.02);
        }

        [Fact]
        public void LandmarkAndModelYawsAgree()
        {
            var excluded = new HashSet<string> { "yaw65", "img2", "img3", "yaw45_pitch10" };
            var gaps = new Dictionary<string, double>();
            foreach (string name in ImageNames)
            {
                var (pts, frame) = Cue(name);
                gaps[name] = Math.Abs(LandmarkYaw(pts, frame) - CalibrateYaw(RawYaw(name)));
                if (!excluded.Contains(name)) Assert.True(gaps[name] <= YawAgreeDeg, name);
            }
            Within(0.74, gaps["yaw-15"], 0.05);
            Within(0.29, gaps["yaw-45"], 0.05);
            double widest = gaps.Where(g => !excluded.Contains(g.Key)).Max(g => g.Value);
            Within(gaps["yaw45_pitch5"], widest, 1e-9);
            Within(3.76, widest, 0.05);
            Within(2.56, gaps["yaw45_bump3"], 0.05);
            double unpitched = gaps.Where(g => !excluded.Contains(g.Key) && !g.Key.Contains("_pitch")).Max(g => g.Value);
            Within(gaps["yaw45_bump3"], unpitched, 1e-9);
            Assert.True(gaps["yaw65"] > YawAgreeDeg);
        }

        [Fact]
        public void Mirrored45sAgreeWithMargin()
        {
            foreach (string name in new[] { "yaw45", "yaw-45" })
            {
                var (pts, frame) = Cue(name);
                double gap = Math.Abs(LandmarkYaw(pts, frame) - CalibrateYaw(RawYaw(name)));
                Assert.True(gap <= YawAgreeDeg - 1.5, name);
            }
        }

        [Fact]
        public void CuesAreMonotonicOver0To60()
        {
            var offsets = new List<double>();
            var reported = new List<double>();
            var ratios = new List<double>();
            foreach (string name in Sweep)
            {
                var (pts, frame) = Cue(name);
                offsets.Add(Math.Abs(NoseOffset(pts, frame)));
                reported.Add(RawYaw(name));
                ratios.Add(Entry(name).GetProperty("cues").GetProperty("eye_width_ratio").GetDouble());
            }
            for (int i = 1; i < Sweep.Length; i++)
            {
                Assert.True(offsets[i] >= offsets[i - 1]);
                Assert.True(reported[i] >= reported[i - 1]);
                Assert.True(ratios[i] <= ratios[i - 1]);
            }
            Assert.Equal(ratios[0], ratios[1]);
        }

        [Fact]
        public void NoseMarginOnTheRenders()
        {
            var expected = new Dictionary<string, double>
            {
                ["yaw0"] = 0.47, ["yaw45"] = 0.05, ["yaw-45"] = 0.05, ["yaw50"] = 0.00,
                ["yaw55"] = -0.06, ["yaw60"] = -0.09, ["img2"] = -0.15, ["img3"] = -0.13,
            };
            foreach (var (name, margin) in expected)
            {
                var (pts, frame) = Cue(name);
                Within(margin, NoseMargin(pts, frame), 0.01, name);
            }
        }

        [Fact]
        public void FarSide()
        {
            foreach (string name in new[] { "yaw45", "yaw48", "yaw50", "img2" }) Assert.True(FarLeftOf(RawLandmarks(name)), name);
            foreach (string name in new[] { "yaw-45", "yaw-50", "img3", "quarter_sketch" }) Assert.False(FarLeftOf(RawLandmarks(name)), name);
        }

        [Fact]
        public void FarSideFallsBackToTheYaw()
        {
            var shortLm = RawLandmarks("yaw45").Take(50).ToArray();
            Assert.True(FarLeftOf(shortLm, 28.0));
            Assert.False(FarLeftOf(shortLm, -28.0));
            Assert.False(FarLeftOf(shortLm));
            Assert.False(FarLeftOf(shortLm, double.NaN));
        }
    }

    // ------------------------------------------------------------ TestSensitivity

    public class Sensitivity
    {
        [Fact]
        public void Guards()
        {
            double mm = Renders.GetProperty("mm_per_px").GetDouble();
            var at0 = FrontalConstructions(RawLandmarks("yaw0"))!;
            var at10 = FrontalConstructions(RawLandmarks("yaw10"))!;
            foreach (var fc in new[] { at0, at10 })
            {
                Assert.Equal(ClipNone, fc.ZygionRClip);
                Assert.Equal(ClipNone, fc.ZygionLClip);
            }
            foreach (var (value, limit, what) in new[]
                     {
                         (at10.ZygionRPx - at0.ZygionRPx, 1.5, "zygion_r_px"), (at10.ZygionLPx - at0.ZygionLPx, 1.5, "zygion_l_px"),
                         (at10.CheekRefRPx - at0.CheekRefRPx, 0.3, "cheek_ref_r_px"), (at10.CheekRefLPx - at0.CheekRefLPx, 0.3, "cheek_ref_l_px"),
                     })
                Assert.True(Math.Abs(value * mm / 10.0) < limit, what);
            // The turn widens the near side and narrows the far side.
            Assert.True(at10.ZygionRPx > at0.ZygionRPx);
            Assert.True(at10.ZygionLPx < at0.ZygionLPx);
            Within(75.2, at0.ZygionRPx * mm, 0.1);
            Within(67.8, at0.ZygionLPx * mm, 0.1);
        }
    }

    // ------------------------------------------------------------ TestPitchedRenders

    public class PitchedRenders
    {
        static readonly string[] Level = { "yaw0_pitch-5", "yaw0_pitch5", "yaw0_pitch10" };
        static readonly string[] Turned = { "yaw45_pitch-5", "yaw45_pitch5", "yaw45_pitch10" };

        static double PitchOf(string name) =>
            double.Parse(name.Split("_pitch")[1], CultureInfo.InvariantCulture);

        [Fact]
        public void FrontalConstructionsUnderPitch()
        {
            double mm = Renders.GetProperty("mm_per_px").GetDouble();
            var level = FrontalConstructions(RawLandmarks("yaw0"))!;
            foreach (string name in Level)
            {
                var fc = FrontalConstructions(RawLandmarks(name))!;
                Assert.Equal(ClipNone, fc.ZygionRClip);
                Assert.Equal(ClipNone, fc.ZygionLClip);
                double pitch = PitchOf(name);
                foreach (var (value, limit) in new[]
                         {
                             (fc.ZygionRPx - level.ZygionRPx, 1.0), (fc.ZygionLPx - level.ZygionLPx, 1.0),
                             (fc.CheekRefRPx - level.CheekRefRPx, 0.6), (fc.CheekRefLPx - level.CheekRefLPx, 0.6),
                         })
                    Assert.True(Math.Abs(value * mm / pitch) < limit, name);
            }
        }

        [Fact]
        public void GrossPitchClipsTheLevelAtTheTop()
        {
            var pts = RawLandmarks("yaw0_pitch25");
            var fc = FrontalConstructions(pts)!;
            Assert.Equal(ClipTop, fc.ZygionLClip);
            Close(pts[32], fc.ZygionL);
            Assert.Equal(ClipNone, fc.ZygionRClip);
            Assert.True(fc.Frame.Level(pts[32]) - fc.ZygionLevel > 1.0);
            Assert.True(Math.Abs(fc.Frame.Level(pts[0]) - fc.ZygionLevel) < 1.0);
        }

        [Fact]
        public void ModelPitchIsMonotonicAndCompressed()
        {
            string[] names = { "yaw0_pitch-5", "yaw0", "yaw0_pitch5", "yaw0_pitch10", "yaw0_pitch25" };
            var raw = names.Select(n => Entry(n).GetProperty("raw").GetProperty("pitch").GetDouble()).ToArray();
            for (int i = 1; i < raw.Length; i++) Assert.True(raw[i] > raw[i - 1]);
            Within(5.39, raw[1], 0.05);
            double slope = (raw[^1] - raw[0]) / 30.0;
            Assert.True(slope > 0.25 && slope < 0.5);
            Assert.All(raw, value => Assert.True(Math.Abs(value) <= PitchGrossRaw));
        }

        [Fact]
        public void LevelHeadStaysFrontalUnderPitch()
        {
            foreach (string name in Level.Concat(new[] { "yaw0_pitch25" }))
            {
                var pts = RawLandmarks(name);
                var frame = PupilFrame(pts)!;
                Assert.True(Math.Abs(LandmarkYaw(pts, frame)) < FrontalYawGate, name);
                Assert.True(Math.Abs(CalibrateYaw(RawYaw(name))) < FrontalYawGate, name);
                Assert.Equal((false, false), ClassifyQuarter(pts, RawYaw(name)));
            }
        }

        [Fact]
        public void TurnedHeadStaysQuarterUnderPitch()
        {
            var yaws = new List<double>();
            var models = new List<double>();
            foreach (string name in Turned)
            {
                var pts = RawLandmarks(name);
                var frame = PupilFrame(pts)!;
                Assert.Equal((true, true), ClassifyQuarter(pts, RawYaw(name)));
                Assert.True(FarLeftOf(pts, RawYaw(name)));
                yaws.Add(LandmarkYaw(pts, frame));
                models.Add(CalibrateYaw(RawYaw(name)));
                Within(45.0, models[^1], 2.5, name);
            }
            Within(45.02, yaws[0], 0.05);
            Within(47.79, yaws[1], 0.05);
            Within(49.55, yaws[2], 0.05);
            for (int i = 0; i < 2; i++)
            {
                Assert.True(Math.Abs(yaws[i] - ObliqueYaw) <= ObliqueYawGate);
                Assert.True(Math.Abs(yaws[i] - models[i]) <= YawAgreeDeg);
            }
            Assert.True(Math.Abs(yaws[2] - models[2]) > YawAgreeDeg);
        }
    }

    // ------------------------------------------------------------ TestClassifyQuarter

    public class ClassifyQuarterRule
    {
        static (bool, bool) Classify(string name, bool previousQuarter = false) =>
            ClassifyQuarter(RawLandmarks(name), RawYaw(name), previousQuarter: previousQuarter);

        [Fact]
        public void FrontalFaces()
        {
            foreach (string name in new[]
                     {
                         "img4", "img1", "quarter_sketch", "quarter_sketch_forced", "yaw0", "yaw5", "yaw10", "yaw15",
                         "yaw20", "yaw25", "yaw30", "yaw-15", "yaw-30", "yaw0_pitch-5", "yaw0_pitch5", "yaw0_pitch10",
                         "yaw0_pitch25",
                     })
                Assert.True(Classify(name) == (false, false), name);
        }

        [Fact]
        public void QuarterFaces()
        {
            foreach (string name in new[] { "yaw35", "yaw40", "yaw42", "yaw45", "yaw48", "yaw50" })
                Assert.True(Classify(name) == (true, true), name);
            foreach (string name in new[] { "yaw-45", "yaw-50" })
                Assert.True(Classify(name) == (true, false), name);
            foreach (string name in new[]
                     {
                         "yaw45_hollow1.5", "yaw45_hollow3", "yaw45_hollow4.5", "yaw50_hollow3", "yaw45_bump3",
                         "yaw45_pitch-5", "yaw45_pitch5", "yaw45_pitch10",
                     })
                Assert.True(Classify(name) == (true, true), name);
            // The float overload classifies the pipeline's landmarks the same way.
            var floats = RawLandmarks("yaw45").Select(p => p.ToVec2()).ToArray();
            Assert.Equal((true, true), ClassifyQuarter(floats, RawYaw("yaw45")));
        }

        [Fact]
        public void NotQuarter()
        {
            foreach (string name in new[] { "yaw55", "yaw60", "yaw65", "img2", "img3" })
                Assert.True(Classify(name) == (false, false), name);
            // The window alone would admit yaw55: its nose is past the far contour.
            var pts = RawLandmarks("yaw55");
            var frame = PupilFrame(pts)!;
            Within(55.0, LandmarkYaw(pts, frame), 0.05);
            Assert.True(NoseMargin(pts, frame) < QuarterMarginMin);
        }

        // The raw model yaw that CalibrateYaw reads as this many degrees.
        static double RawFor(double calibrated)
        {
            double low = 0.0, high = 80.0;
            for (int i = 0; i < 80; i++)
            {
                double middle = (low + high) / 2;
                if (CalibrateYaw(middle) < calibrated) low = middle; else high = middle;
            }
            return (low + high) / 2;
        }

        [Fact]
        public void ModelReadingPastTheWindow()
        {
            var pts = RawLandmarks("yaw45");
            double limit = QuarterMaxModelYaw;
            double inside = RawFor(limit - 0.1), outside = RawFor(limit + 0.1);
            Assert.Equal((true, true), ClassifyQuarter(pts, inside));
            Assert.Equal((false, false), ClassifyQuarter(pts, outside));
            Assert.Equal((true, true), ClassifyQuarter(pts, outside, previousQuarter: true));
            double pastHold = RawFor(limit + QuarterYawHold + 0.1);
            Assert.Equal((false, false), ClassifyQuarter(pts, pastHold, previousQuarter: true));
        }

        [Fact]
        public void TheMarginRuleCanBeWaived()
        {
            var pts = RawLandmarks("yaw55");
            double raw = RawYaw("yaw55");
            Assert.True(Math.Abs(CalibrateYaw(raw)) <= QuarterMaxModelYaw);
            Assert.Equal((false, false), ClassifyQuarter(pts, raw));
            Assert.Equal((true, true), ClassifyQuarter(pts, raw, requireMargin: false));
            foreach (string name in new[] { "img4", "yaw30", "yaw60" })
                Assert.True(ClassifyQuarter(RawLandmarks(name), RawYaw(name), requireMargin: false) == (false, false), name);
        }

        [Fact]
        public void ForgedModelYaw()
        {
            Assert.Equal((false, false), ClassifyQuarter(RawLandmarks("img4"), 28.0));
            Assert.Equal((false, false), ClassifyQuarter(RawLandmarks("yaw10"), 28.28));
        }

        [Fact]
        public void DisplacedNoseWithoutTheModel()
        {
            var pts = RawLandmarks("yaw45");
            Assert.Equal((false, false), ClassifyQuarter(pts, 5.0));
            Assert.Equal((false, false), ClassifyQuarter(pts, 0.0));
            Assert.Equal((false, false), ClassifyQuarter(pts, double.NaN));
            // The model's calibrated minimum, 20 degrees (raw 10.41), with the sign.
            Assert.Equal((true, true), ClassifyQuarter(pts, 10.41));
            Assert.Equal((false, false), ClassifyQuarter(pts, 10.0));
        }

        [Fact]
        public void CuesDisagreeingInSign()
        {
            Assert.Equal((false, false), ClassifyQuarter(RawLandmarks("yaw45"), -RawYaw("yaw45")));
            Assert.Equal((false, false), ClassifyQuarter(RawLandmarks("yaw-45"), -RawYaw("yaw-45")));
        }

        [Fact]
        public void NoFrame()
        {
            Assert.Equal((false, false), ClassifyQuarter(Array.Empty<Vec2d>(), 28.28));
            Assert.Equal((false, false), ClassifyQuarter(RawLandmarks("yaw45").Take(97).ToArray(), 28.28));
            var pts = RawLandmarks("yaw45");
            pts[97] = pts[96];
            Assert.Equal((false, false), ClassifyQuarter(pts, 28.28));
        }

        [Fact]
        public void HoldKeepsOneView()
        {
            var baseLm = RawLandmarks("yaw35");
            double model = RawYaw("yaw35");
            foreach (double target in new[] { 34.0, 36.0, 31.0, 33.0 })
            {
                var pts = WithLandmarkYaw(baseLm, target);
                Near(target, LandmarkYaw(pts, PupilFrame(pts)!), 6);
            }
            bool previous = false;
            var seen = new List<bool>();
            foreach (double target in new[] { 34.0, 36.0, 34.0, 36.0 })
            {
                (previous, _) = ClassifyQuarter(WithLandmarkYaw(baseLm, target), model, previousQuarter: previous);
                seen.Add(previous);
            }
            Assert.Equal(new[] { false, true, true, true }, seen);
            previous = true;
            seen.Clear();
            foreach (double target in new[] { 34.0, 36.0, 34.0, 36.0 })
            {
                (previous, _) = ClassifyQuarter(WithLandmarkYaw(baseLm, target), model, previousQuarter: previous);
                seen.Add(previous);
            }
            Assert.Equal(new[] { true, true, true, true }, seen);
        }

        [Fact]
        public void HoldLeavesBelowTheWidenedEdge()
        {
            var baseLm = RawLandmarks("yaw35");
            double model = RawYaw("yaw35");
            var at36 = WithLandmarkYaw(baseLm, 36.0);
            var at33 = WithLandmarkYaw(baseLm, 33.0);
            var at31 = WithLandmarkYaw(baseLm, 31.0);
            Assert.True(ClassifyQuarter(at36, model).IsQuarter);
            Assert.Equal((false, false), ClassifyQuarter(at31, model, previousQuarter: true));
            Assert.Equal((true, true), ClassifyQuarter(at33, model, previousQuarter: true));
            Assert.Equal((false, false), ClassifyQuarter(at33, model));
            // The hold never relaxes the margin rule or the model's minimum.
            Assert.Equal((false, false), ClassifyQuarter(at36, 5.0, previousQuarter: true));
            Assert.Equal((false, false), ClassifyQuarter(RawLandmarks("yaw55"), RawYaw("yaw55"), previousQuarter: true));
        }

        [Fact]
        public void HoldAtTheTopEdge()
        {
            var baseFace = StickFace();
            for (int i = 0; i < 33; i++) baseFace[i].X = 510.0 + (baseFace[i].X - 510.0) * 1.5;
            double model = RawYaw("yaw55"); // 37.79, calibrated 55: past the minimum
            var at56 = WithLandmarkYaw(baseFace, 56.0);
            var at59 = WithLandmarkYaw(baseFace, 59.0);
            foreach (var pts in new[] { at56, at59 }) Assert.True(NoseMargin(pts, PupilFrame(pts)!) > 0.05);
            Assert.Equal((false, false), ClassifyQuarter(at56, model));
            Assert.Equal((true, true), ClassifyQuarter(at56, model, previousQuarter: true));
            Assert.Equal((false, false), ClassifyQuarter(at59, model, previousQuarter: true));
            Assert.Equal((true, true), ClassifyQuarter(WithLandmarkYaw(baseFace, 54.0), model));
        }
    }

    // ------------------------------------------------------------ TestLocateAll

    public class LocateAllRule
    {
        static readonly int[] FrontalRows = { 0, 1, 2, 3, 4, 5 };
        static readonly int[] ObliqueRows = { 6, 7, 8, 9 };

        static CheekOutline.ObliqueCheek Stub(double depth = 0.01) => new()
        {
            Apex = new Vec2d(101.0, 202.0),
            Submalar = new Vec2d(303.0, 404.0),
            SubmalarDepth = depth,
        };

        [Fact]
        public void FrontalRowsFromTheConstructions()
        {
            var pts = PipelineLandmarks("img4");
            var points = LocateAll(pts, FacePose.Frontal, false, null, null, null);
            Assert.Equal(10, points.Length);
            var fc = FrontalConstructions(pts)!;
            Close(fc.ZygionR, points[CheekLmZygionR]);
            Close(fc.ZygionL, points[CheekLmZygionL]);
            Close(fc.CheekRefR, points[CheekLmCheekRefR]);
            Close(fc.CheekRefL, points[CheekLmCheekRefL]);
            Close(fc.SubmalarRefR, points[CheekLmSubmalarRefR]);
            Close(fc.SubmalarRefL, points[CheekLmSubmalarRefL]);
            Assert.True(AllNaN(Rows(points, ObliqueRows)));
            Assert.Equal(10, CheekLandmarkNames.Count);
            // The pipeline's float landmarks (whole pixels here) give the same rows.
            Assert.Equal(points, LocateAll(pts.Select(p => p.ToVec2()).ToArray(), FacePose.Frontal, false, null, null, null));
        }

        [Fact]
        public void LateralIsAllNaN()
        {
            var pts = PipelineLandmarks("img4");
            Assert.True(AllNaN(LocateAll(pts, FacePose.Lateral, false, null, null, null)));
            Assert.True(AllNaN(LocateAll(pts, FacePose.Detect, false, null, null, null)));
        }

        [Fact]
        public void DegenerateLandmarksAreAllNaN()
        {
            var pts = PipelineLandmarks("img4");
            foreach (var pose in new[] { FacePose.Frontal, FacePose.Quarter })
            {
                Assert.True(AllNaN(LocateAll(pts.Take(97).ToArray(), pose, false, null, null, null)));
                Assert.True(AllNaN(LocateAll(Array.Empty<Vec2d>(), pose, false, null, null, null)));
                var coincident = (Vec2d[])pts.Clone();
                coincident[97] = coincident[96];
                Assert.True(AllNaN(LocateAll(coincident, pose, true, null, null, null)));
            }
        }

        [Fact]
        public void QuarterRowsFollowTheFarSide()
        {
            // The synthetic face with a submalar notch, and its mirror image: the observed
            // apex and submalar point land in the far side's rows (7 and 9 for the left,
            // 6 and 8 for the right), and nothing else is filled.
            var lm = SyntheticFace.BaseLandmarks();
            var (xs, rows, src) = SyntheticFace.Outline(new[] { (0.78, 0.03, 20.0), (0.55, -0.01, 30.0) });
            const int width = 1000;
            var cases = new[]
            {
                (FarLeft: true, Lm: lm, Xs: xs, ApexRow: 7, SubRow: 9),
                (FarLeft: false, Lm: SyntheticFace.MirrorLandmarks(lm, width), Xs: xs.Select(x => width - 1 - x).ToArray(), ApexRow: 6, SubRow: 8),
            };
            foreach (var c in cases)
            {
                var observed = CheekOutline.ObservedPoints(c.Lm, c.Xs, rows, src, c.FarLeft)!;
                Assert.False(double.IsNaN(observed.SubmalarDepth));
                var points = LocateAll(c.Lm, FacePose.Quarter, c.FarLeft, c.Xs, rows, src);
                Assert.Equal(observed.Apex, points[c.ApexRow]);
                Assert.Equal(observed.Submalar, points[c.SubRow]);
                var other = Enumerable.Range(0, 10).Where(r => r != c.ApexRow && r != c.SubRow).ToArray();
                Assert.True(AllNaN(Rows(points, other)));
            }
        }

        [Fact]
        public void QuarterSubmalarNaNWhenItsDepthIs()
        {
            var points = LandmarkArray(null, Stub(double.NaN), true);
            Close(new Vec2d(101.0, 202.0), points[CheekLmApexL]);
            Assert.True(AllNaN(new[] { points[CheekLmSubmalarL] }));
            // On the real search: the contour fallback reads no depth, so only the apex row.
            var pts = RawLandmarks("yaw45");
            var observed = ObserveFarCheek(pts, null, null, null, true)!;
            Assert.True(double.IsNaN(observed.SubmalarDepth));
            var located = LocateAll(pts, FacePose.Quarter, true, null, null, null);
            Assert.Equal(observed.Apex, located[CheekLmApexL]);
            Assert.True(AllNaN(Rows(located, 0, 1, 2, 3, 4, 5, 6, 8, 9)));
        }

        [Fact]
        public void QuarterWithoutObservedPoints()
        {
            // A search that declines (a band of two rows) leaves every row NaN.
            var lm = SyntheticFace.BaseLandmarks();
            var (xs, rows, src) = SyntheticFace.Outline(null, 300, 301);
            Assert.Null(CheekOutline.ObservedPoints(lm, xs, rows, src, true));
            Assert.True(AllNaN(LocateAll(lm, FacePose.Quarter, true, xs, rows, src)));
        }

        [Fact]
        public void QuarterRaggedOutlineMeansTheFallback()
        {
            var pts = RawLandmarks("yaw45");
            var fallback = LocateAll(pts, FacePose.Quarter, true, null, null, null);
            var xs = Enumerable.Range(0, 5).Select(i => (float)i).ToArray();
            var ys = Enumerable.Range(0, 4).ToArray();
            var src = new byte[5];
            Assert.Equal(fallback, LocateAll(pts, FacePose.Quarter, true, xs, ys, src));
            Assert.Equal(fallback, LocateAll(pts, FacePose.Quarter, true, xs, null, src));
            Assert.Equal(ObserveFarCheek(pts, null, null, null, true)!.Apex,
                ObserveFarCheek(pts, xs, ys, src, true)!.Apex);
        }

        [Fact]
        public void LandmarkArrayFillsByOneRule()
        {
            var pts = PipelineLandmarks("img4");
            var fc = FrontalConstructions(pts)!;
            var frontal = LocateAll(pts, FacePose.Frontal, false, null, null, null);
            var both = LandmarkArray(fc, Stub(), true);
            Assert.Equal(10, both.Length);
            Assert.Equal(Rows(frontal, FrontalRows), Rows(both, FrontalRows));
            Assert.True(Rows(both, FrontalRows).All(p => p.IsFinite));
            Close(new Vec2d(101.0, 202.0), both[CheekLmApexL]);
            Close(new Vec2d(303.0, 404.0), both[CheekLmSubmalarL]);
            Assert.True(AllNaN(Rows(both, 6, 8)));
            Assert.Equal(frontal, LandmarkArray(fc, null, true));
            // The right side's rows, and no submalar row when its depth is NaN.
            var right = LandmarkArray(null, Stub(double.NaN), false);
            Close(new Vec2d(101.0, 202.0), right[CheekLmApexR]);
            Assert.True(AllNaN(Rows(right, 0, 1, 2, 3, 4, 5, 7, 8, 9)));
            // Nothing located: all NaN, and a new array every time.
            var none = LandmarkArray(null, null, false);
            Assert.True(AllNaN(none));
            Assert.NotSame(none, LandmarkArray(null, null, false));
        }

        [Fact]
        public void ObserveFarCheekNeedsAPupilFrame()
        {
            var pts = RawLandmarks("yaw45");
            Assert.NotNull(ObserveFarCheek(pts, null, null, null, true));
            var coincident = (Vec2d[])pts.Clone();
            coincident[97] = coincident[96];
            var broken = (Vec2d[])pts.Clone();
            broken[LmChin].Y = double.NaN;
            foreach (var landmarks in new[] { coincident, broken, pts.Take(97).ToArray(), Array.Empty<Vec2d>() })
                Assert.Null(ObserveFarCheek(landmarks, null, null, null, true));
            // A face whose canthi coincide has a pupil frame but no level frame: no points.
            var flatCanthi = (Vec2d[])pts.Clone();
            flatCanthi[LmCanthusL] = flatCanthi[LmCanthusR];
            Assert.Null(ObserveFarCheek(flatCanthi, null, null, null, true));
            // LocateAll is LandmarkArray of its result.
            var ob = ObserveFarCheek(pts, null, null, null, true);
            Assert.Equal(LandmarkArray(null, ob, true), LocateAll(pts, FacePose.Quarter, true, null, null, null));
        }

        [Fact]
        public void QuarterPathRunsOnTheRealModule()
        {
            var pts = RawLandmarks("yaw45");
            var points = LocateAll(pts, FacePose.Quarter, true, null, null, null);
            Assert.Equal(10, points.Length);
            Assert.True(AllNaN(Rows(points, FrontalRows)));
            Assert.True(AllNaN(Rows(points, 6, 8)));
            foreach (var (index, value) in new[] { (LmChin, double.NaN), (LmCanthusL, double.PositiveInfinity) })
            {
                var broken = (Vec2d[])pts.Clone();
                broken[index].Y = value;
                Assert.True(AllNaN(LocateAll(broken, FacePose.Quarter, true, null, null, null)));
            }
        }

        [Fact]
        public void LocateGivesOneResultPerView()
        {
            var pts = RawLandmarks("yaw45");
            var (frontal, observed) = Locate(pts, quarter: true, lateral: false, farLeft: true, null, null, null);
            Assert.Null(frontal);
            Assert.NotNull(observed);
            (frontal, observed) = Locate(pts, quarter: false, lateral: true, farLeft: true, null, null, null);
            Assert.Null(frontal);
            Assert.Null(observed);
            (frontal, observed) = Locate(pts, quarter: false, lateral: false, farLeft: true, null, null, null);
            Assert.NotNull(frontal);
            Assert.Null(observed);
        }
    }

    // ------------------------------------------------------------ shared constants

    // Every module-level constant of dynaface/cheek.py and dynaface/cheek_outline.py (the
    // test project links both into TestData/python), checked against the .NET copy, as
    // SharedConstantsTests does for SPIGA's inputs: a constant added or changed on either
    // side fails here until the other follows.
    public class SharedConstants
    {
        static string PythonSource(string file) =>
            File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", "python", file));

        static readonly Lazy<Dictionary<string, object>> OutlineConstants =
            new(() => PyConstants.Parse(PythonSource("cheek_outline.py"), null));
        static readonly Lazy<Dictionary<string, object>> CheekConstants =
            new(() => PyConstants.Parse(PythonSource("cheek.py"), OutlineConstants.Value));

        static object L(params object[] items) => items.ToList();
        static object Ints(IEnumerable<int> items) => items.Select(i => (object)(double)i).ToList();

        // The .NET value of each cheek_outline.py constant, by its Python name.
        static Dictionary<string, object> OutlineExpected() => new()
        {
            ["NUM_LANDMARKS"] = CheekOutline.NumLandmarks,
            ["LM_CHIN"] = CheekOutline.LmChin,
            ["LM_CANTHUS_R"] = CheekOutline.LmCanthusR,
            ["LM_CANTHUS_L"] = CheekOutline.LmCanthusL,
            ["LM_ALA_R"] = CheekOutline.LmAlaR,
            ["LM_ALA_L"] = CheekOutline.LmAlaL,
            ["LM_NOSE_TIP"] = CheekOutline.LmNoseTip,
            ["LM_SUBNASALE"] = CheekOutline.LmSubnasale,
            ["LM_STOMION"] = L(CheekOutline.LmStomion.Upper, CheekOutline.LmStomion.Lower),
            ["RIGHT_FAR_CONTOUR"] = Ints(CheekOutline.RightFarContour),
            ["LEFT_FAR_CONTOUR"] = Ints(CheekOutline.LeftFarContour),
            ["APEX_H_PRIOR"] = CheekOutline.ApexHPrior,
            ["APEX_H_WINDOW"] = L(CheekOutline.ApexHWindow.Low, CheekOutline.ApexHWindow.High),
            ["APEX_MIN_DEV"] = CheekOutline.ApexMinDev,
            ["APEX_NEAR_NOSE_H"] = CheekOutline.ApexNearNoseH,
            ["NOSE_ROWS"] = L(CheekOutline.NoseRows.Tip, CheekOutline.NoseRows.Subnasale),
            ["NOSE_STEP"] = CheekOutline.NoseStep,
            ["OUTLINE_TOL"] = CheekOutline.OutlineTol,
            ["OUTLINE_MIN_MASK"] = CheekOutline.OutlineMinMask,
            ["SMOOTH_ROWS"] = CheekOutline.SmoothRows,
            ["_HALO_ROWS"] = CheekOutline.HaloRows,
            ["SUBMALAR_MIN_DEPTH"] = CheekOutline.SubmalarMinDepth,
            ["VANISH_MIN_SIN"] = CheekOutline.VanishMinSin,
            ["MASK_ALPHA"] = CheekOutline.MaskAlpha,
            ["ALPHA_SEARCH"] = CheekOutline.AlphaSearch,
            ["EDGE_GRAD_MIN"] = CheekOutline.EdgeGradMin,
            ["EDGE_MAX_H"] = CheekOutline.EdgeMaxH,
            ["_GRAY_WEIGHTS"] = L(CheekOutline.GrayWeights.R, CheekOutline.GrayWeights.G, CheekOutline.GrayWeights.B),
            ["_GRAY_SHIFT"] = CheekOutline.GrayShift,
            ["OUTLINE_SRC_CONTOUR"] = CheekOutline.OutlineSrcContour,
            ["OUTLINE_SRC_MASK"] = CheekOutline.OutlineSrcMask,
            ["OUTLINE_SRC_NOSE"] = CheekOutline.OutlineSrcNose,
            ["OUTLINE_SRC_EDGE"] = CheekOutline.OutlineSrcEdge,
            ["_BINOMIAL_5"] = CheekOutline.Binomial5.Cast<object>().ToList(),
            ["_SOBEL_ROWS"] = CheekOutline.SobelRows.Cast<object>().ToList(),
        };

        // The .NET value of each cheek.py constant, by its Python name.
        static Dictionary<string, object> CheekExpected() => new()
        {
            ["CHEEK_LANDMARK_NAMES"] = CheekLandmarkNames.Cast<object>().ToList(),
            ["CHEEK_LANDMARK_CODES"] = CheekLandmarkCodes.Cast<object>().ToList(),
            ["CHEEK_LM_ZYGION_R"] = CheekLmZygionR,
            ["CHEEK_LM_ZYGION_L"] = CheekLmZygionL,
            ["CHEEK_LM_CHEEK_REF_R"] = CheekLmCheekRefR,
            ["CHEEK_LM_CHEEK_REF_L"] = CheekLmCheekRefL,
            ["CHEEK_LM_SUBMALAR_REF_R"] = CheekLmSubmalarRefR,
            ["CHEEK_LM_SUBMALAR_REF_L"] = CheekLmSubmalarRefL,
            ["CHEEK_LM_APEX_R"] = CheekLmApexR,
            ["CHEEK_LM_APEX_L"] = CheekLmApexL,
            ["CHEEK_LM_SUBMALAR_R"] = CheekLmSubmalarR,
            ["CHEEK_LM_SUBMALAR_L"] = CheekLmSubmalarL,
            ["ZYGION_LEVEL_IPD"] = ZygionLevelIpd,
            ["CHEEK_REF_FRACTION"] = CheekRefFraction,
            ["SUBMALAR_REF_FRACTION"] = SubmalarRefFraction,
            ["RIGHT_CONTOUR_TOP"] = Ints(RightContourTop),
            ["LEFT_CONTOUR_TOP"] = Ints(LeftContourTop),
            ["NUM_LANDMARKS"] = Cheek.NumLandmarks,
            ["RIGHT_CONTOUR"] = Ints(RightContour),
            ["LEFT_CONTOUR"] = Ints(LeftContour),
            ["LM_CANTHUS_R"] = LmCanthusR,
            ["LM_CANTHUS_L"] = LmCanthusL,
            ["LM_ALA_R"] = LmAlaR,
            ["LM_ALA_L"] = LmAlaL,
            ["LM_NOSE_TIP"] = LmNoseTip,
            ["LM_CHIN"] = LmChin,
            ["LM_STOMION"] = L(LmStomion.Upper, LmStomion.Lower),
            ["LM_CONTOUR_TOP_R"] = LmContourTopR,
            ["LM_CONTOUR_TOP_L"] = LmContourTopL,
            ["CLIP_NONE"] = ClipNone,
            ["CLIP_TOP"] = ClipTop,
            ["CLIP_BOTTOM"] = ClipBottom,
            ["FRONTAL_YAW_GATE"] = FrontalYawGate,
            ["ZYGION_RATIO_YAW_GATE"] = ZygionRatioYawGate,
            ["YAW_AGREE_DEG"] = YawAgreeDeg,
            ["PITCH_GROSS_RAW"] = PitchGrossRaw,
            ["OBLIQUE_YAW"] = ObliqueYaw,
            ["OBLIQUE_YAW_GATE"] = ObliqueYawGate,
            ["QUARTER_YAW_RANGE"] = L(QuarterYawRange.Low, QuarterYawRange.High),
            ["QUARTER_YAW_HOLD"] = QuarterYawHold,
            ["QUARTER_MIN_MODEL_YAW"] = QuarterMinModelYaw,
            ["QUARTER_MAX_MODEL_YAW"] = QuarterMaxModelYaw,
            ["QUARTER_MARGIN_MIN"] = QuarterMarginMin,
            ["YAW_CALIBRATION"] = YawCalibration.Select(r => L(r.Raw, r.TrueYaw)).ToList(),
            ["YAW_SLOPE_BEYOND"] = YawSlopeBeyond,
            ["YAW_RAW_OFFSET"] = YawRawOffset,
            ["NOSE_OFFSET_CALIBRATION"] = NoseOffsetCalibration.Select(r => L(r.Offset, r.TrueYaw)).ToList(),
            ["NOSE_OFFSET_SLOPE_BEYOND"] = NoseOffsetSlopeBeyond,
        };

        static void AssertSame(object expected, object actual, string name)
        {
            switch (expected)
            {
                case string s:
                    Assert.True(actual is string a && a == s, $"{name}: Python {expected}, .NET {actual}");
                    break;
                case List<object> list:
                    var other = Assert.IsType<List<object>>(actual);
                    Assert.True(list.Count == other.Count, $"{name}: Python has {list.Count} items, .NET {other.Count}");
                    for (int i = 0; i < list.Count; i++) AssertSame(list[i], other[i], $"{name}[{i}]");
                    break;
                default:
                    double py = Convert.ToDouble(expected, CultureInfo.InvariantCulture);
                    double net = Convert.ToDouble(actual, CultureInfo.InvariantCulture);
                    Assert.True(py == net, $"{name}: Python {py:R}, .NET {net:R}");
                    break;
            }
        }

        static void CheckAll(Dictionary<string, object> python, Dictionary<string, object> net)
        {
            Assert.Equal(python.Keys.OrderBy(k => k), net.Keys.OrderBy(k => k));
            foreach (var (name, value) in python) AssertSame(value, Normalize(net[name]), name);
        }

        // Ints and bytes as doubles, so they compare with Python's numbers.
        static object Normalize(object value) => value switch
        {
            List<object> list => list.Select(Normalize).ToList(),
            string s => s,
            _ => Convert.ToDouble(value, CultureInfo.InvariantCulture),
        };

        [Fact]
        public void CheekOutlineConstantsMatchPython() => CheckAll(OutlineConstants.Value, OutlineExpected());

        [Fact]
        public void CheekConstantsMatchPython() => CheckAll(CheekConstants.Value, CheekExpected());

        [Fact]
        public void ConstantsBeyondTheParserMatchPython()
        {
            // The ones that aren't plain literals: the far canthus by side, the outputs
            // comprehension and the interpolation columns.
            string outline = PythonSource("cheek_outline.py");
            Assert.Contains("FAR_CANTHUS = {False: LM_CANTHUS_R, True: LM_CANTHUS_L}", outline);
            Assert.Equal(CheekOutline.LmCanthusR, CheekOutline.FarCanthus(false));
            Assert.Equal(CheekOutline.LmCanthusL, CheekOutline.FarCanthus(true));
            string cheek = PythonSource("cheek.py");
            Assert.Contains("f\"{code}.{axis}\" for code in CHEEK_LANDMARK_CODES for axis in (\"x\", \"y\")", cheek);
            Assert.Equal(CheekLandmarkCodes.SelectMany(c => new[] { c + ".x", c + ".y" }), CheekLandmarkOutputs);
            Assert.Contains("[reported - YAW_RAW_OFFSET for reported, _ in YAW_CALIBRATION]", cheek);
            Assert.Contains("from dynaface.const import LM_LEFT_PUPIL, LM_RIGHT_PUPIL", cheek);
            Assert.Equal(97, DynafaceConstants.LmLeftPupil);
            Assert.Equal(96, DynafaceConstants.LmRightPupil);
        }

        [Fact]
        public void TheParserReadsEveryForm()
        {
            var values = PyConstants.Parse(string.Join("\n",
                "A = 3  # a comment",
                "B, C = 1.5, -2",
                "D = A // 2 + 1",
                "E = 1.0 / 4.0",
                "F = tuple(range(5, 1, -2))",
                "G = (",
                "    (1, 2.0),  # one",
                "    (3, 4e-3),",
                ")",
                "H = [\"x # y\", \"z\"]",
                "skipped = 1",
                "I = {False: 1}",
                "J = other.K",
                "L = tuple(range(32, 15, -1))"), new Dictionary<string, object> { ["K"] = 7L });
            Assert.Equal(17, ((List<object>)values["L"]).Count);
            Assert.Equal(3L, values["A"]);
            Assert.Equal(1.5, values["B"]);
            Assert.Equal(-2L, values["C"]);
            Assert.Equal(2L, values["D"]);
            Assert.Equal(0.25, values["E"]);
            Assert.Equal(new List<object> { 5L, 3L }, values["F"]);
            AssertSame(L(L(1.0, 2.0), L(3.0, 4e-3)), Normalize(values["G"]), "G");
            Assert.Equal(new List<object> { "x # y", "z" }, values["H"]);
            Assert.False(values.ContainsKey("skipped"));
            Assert.False(values.ContainsKey("I"));
            Assert.Equal(7L, values["J"]);
        }
    }
}

// A reader for the module-level constants of a Python source file: assignments at column 0
// to upper-case names (A = ..., A, B = ...), whose value is built from numbers, strings,
// tuples, lists, + - * / //, tuple(range(...)), and names assigned earlier (or, dotted, in
// the other module). An assignment it can't evaluate (a dict, a call, a comprehension) is
// left out, so the tests check those by hand. Values are long, double, string and
// List<object>.
internal static class PyConstants
{
    static readonly System.Text.RegularExpressions.Regex Assignment =
        new(@"^(_?[A-Z][A-Z0-9_]*(?:\s*,\s*_?[A-Z][A-Z0-9_]*)*)\s*=(?!=)\s*(.*)$");

    public static Dictionary<string, object> Parse(string source, Dictionary<string, object>? other)
    {
        var values = new Dictionary<string, object>();
        string[] lines = source.Replace("\r\n", "\n").Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            var m = Assignment.Match(lines[i]);
            if (!m.Success) continue;
            var text = new StringBuilder(StripComment(m.Groups[2].Value));
            while (Depth(text.ToString()) > 0 && i + 1 < lines.Length) text.Append(' ').Append(StripComment(lines[++i]));
            string[] names = m.Groups[1].Value.Split(',').Select(n => n.Trim()).ToArray();
            object? value;
            try
            {
                value = new Evaluator(text.ToString(), values, other).Evaluate();
            }
            catch (FormatException)
            {
                continue;
            }
            if (names.Length == 1) values[names[0]] = value;
            else if (value is List<object> items && items.Count == names.Length)
                for (int k = 0; k < names.Length; k++) values[names[k]] = items[k];
        }
        return values;
    }

    static string StripComment(string line)
    {
        bool inString = false;
        for (int i = 0; i < line.Length; i++)
        {
            if (line[i] == '"') inString = !inString;
            else if (line[i] == '#' && !inString) return line.Substring(0, i);
        }
        return line;
    }

    static int Depth(string text)
    {
        int depth = 0;
        bool inString = false;
        foreach (char c in text)
        {
            if (c == '"') inString = !inString;
            else if (!inString && "([{".IndexOf(c) >= 0) depth++;
            else if (!inString && ")]}".IndexOf(c) >= 0) depth--;
        }
        return depth;
    }

    sealed class Evaluator
    {
        readonly List<string> tokens = new();
        readonly Dictionary<string, object> known;
        readonly Dictionary<string, object>? other;
        int pos;

        public Evaluator(string text, Dictionary<string, object> known, Dictionary<string, object>? other)
        {
            this.known = known;
            this.other = other;
            var re = new System.Text.RegularExpressions.Regex(
                @"\s*(\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\.\d+(?:[eE][-+]?\d+)?|\d+|""[^""]*""|[A-Za-z_][A-Za-z0-9_.]*|//|[()\[\]{},+\-*/:])");
            int at = 0;
            while (at < text.Length)
            {
                if (char.IsWhiteSpace(text[at])) { at++; continue; }
                var m = re.Match(text, at);
                if (!m.Success || m.Index != at) throw new FormatException(text);
                tokens.Add(m.Groups[1].Value);
                at += m.Length;
            }
        }

        string? Peek => pos < tokens.Count ? tokens[pos] : null;
        string Next() => pos < tokens.Count ? tokens[pos++] : throw new FormatException("end");
        void Expect(string t) { if (Next() != t) throw new FormatException(t); }

        public object Evaluate()
        {
            object value = Expression();
            if (Peek == ",")
            {
                var items = new List<object> { value };
                while (Peek == ",") { Next(); if (Peek == null) break; items.Add(Expression()); }
                value = items;
            }
            if (Peek != null) throw new FormatException(Peek);
            return value;
        }

        object Expression()
        {
            object value = Term();
            while (Peek == "+" || Peek == "-")
                value = Arithmetic(Next(), value, Term());
            return value;
        }

        object Term()
        {
            object value = Unary();
            while (Peek == "*" || Peek == "/" || Peek == "//")
                value = Arithmetic(Next(), value, Unary());
            return value;
        }

        object Unary()
        {
            if (Peek == "-")
            {
                Next();
                object v = Unary();
                return v switch { long l => (object)(-l), double d => -d, _ => throw new FormatException("-") };
            }
            return Atom();
        }

        object Atom()
        {
            string t = Next();
            if (t == "(" || t == "[")
            {
                string close = t == "(" ? ")" : "]";
                var items = new List<object>();
                bool comma = false;
                while (Peek != close)
                {
                    items.Add(Expression());
                    if (Peek == ",") { Next(); comma = true; }
                    else if (Peek != close) throw new FormatException(Peek ?? "end");
                }
                Next();
                return t == "(" && items.Count == 1 && !comma ? items[0] : items;
            }
            if (t.StartsWith("\"")) return t.Substring(1, t.Length - 2);
            if (char.IsDigit(t[0]) || t[0] == '.')
                return t.Contains('.') || t.Contains('e') || t.Contains('E')
                    ? double.Parse(t, CultureInfo.InvariantCulture)
                    : (object)long.Parse(t, CultureInfo.InvariantCulture);
            if (t == "tuple" && Peek == "(")
            {
                Next();
                object inner = Expression();
                Expect(")");
                return inner as List<object> ?? throw new FormatException("tuple");
            }
            if (t == "range" && Peek == "(")
            {
                Next();
                var args = new List<long>();
                while (Peek != ")")
                {
                    args.Add(Expression() is long l ? l : throw new FormatException("range"));
                    if (Peek == ",") Next();
                }
                Next();
                long start = args.Count > 1 ? args[0] : 0, stop = args.Count > 1 ? args[1] : args[0];
                long step = args.Count > 2 ? args[2] : 1;
                var list = new List<object>();
                for (long i = start; step > 0 ? i < stop : i > stop; i += step) list.Add(i);
                return list;
            }
            if (char.IsLetter(t[0]) || t[0] == '_')
            {
                if (Peek == "(") throw new FormatException("call");
                if (known.TryGetValue(t, out var v)) return v;
                int dot = t.LastIndexOf('.');
                if (dot > 0 && other != null && other.TryGetValue(t.Substring(dot + 1), out var o)) return o;
                throw new FormatException(t);
            }
            throw new FormatException(t);
        }

        static object Arithmetic(string op, object a, object b)
        {
            if (a is long x && b is long y)
                return op switch
                {
                    "+" => (object)(x + y),
                    "-" => x - y,
                    "*" => x * y,
                    "/" => (double)x / y,
                    "//" => (long)Math.Floor((double)x / y),
                    _ => throw new FormatException(op),
                };
            double p = Convert.ToDouble(a is List<object> ? throw new FormatException(op) : a, CultureInfo.InvariantCulture);
            double q = Convert.ToDouble(b is List<object> ? throw new FormatException(op) : b, CultureInfo.InvariantCulture);
            return op switch
            {
                "+" => p + q,
                "-" => p - q,
                "*" => p * q,
                "/" => p / q,
                "//" => Math.Floor(p / q),
                _ => throw new FormatException(op),
            };
        }
    }
}
