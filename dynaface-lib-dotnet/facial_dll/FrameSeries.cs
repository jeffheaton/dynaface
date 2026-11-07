using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

// Measures a video frame by frame: the values behind its chart, a CSV file of them, the
// frame where an output peaks, and the evaluation the Dynaface apps report. One row per
// frame, one column per output. Mirrors dynaface-lib's dynaface.video.FrameSeries
// (without its rows() helper for pandas). FromStates also takes frames saved without
// their image, which dynaface-lib's states never are.
public sealed class FrameSeries
{
    readonly List<string> _names = new List<string>();
    readonly List<int> _frames = new List<int>();
    readonly Dictionary<string, List<double>> _values = new Dictionary<string, List<double>>();

    // Starts an empty series for the outputs of these measures, the same ones the frames
    // are analyzed with. fps: frames per second, for each frame's time; 0 if unknown.
    public FrameSeries(IEnumerable<FaceMeasureBase> measures, float fps = 0f)
    {
        foreach (string name in DynafaceMeasures.GetAllItems(measures))
        {
            if (_values.ContainsKey(name)) continue;
            _names.Add(name);
            _values[name] = new List<double>();
        }
        Fps = fps;
    }

    // The outputs, in column order: the enabled items of the enabled measures
    // (DynafaceMeasures.GetAllItems).
    public IReadOnlyList<string> Names => _names;

    // Each row's frame number.
    public IReadOnlyList<int> Frames => _frames;

    // Frames per second, for each frame's time; 0 if unknown.
    public float Fps { get; }

    public int Count => _frames.Count;

    // One output's value in every row: NaN where the frame had no face or the value
    // couldn't be determined. Throws KeyNotFoundException for an output not in the series.
    public IReadOnlyList<double> this[string name] => _values[name];

    // Measures saved analyses again, frame by frame, without drawing. Each row is numbered
    // by its state's position in states, so the frames of a trimmed range keep the numbers
    // they have in the whole video. measures: the ones to run, all of them
    // (DynafaceMeasures.AllMeasures()) if null. step: measure every step-th state, such as
    // every 10th for a quick chart of a long video. When frames without a face were
    // skipped, as VideoAnalyzer skips them, pass the kept frames' numbers in the video as
    // frameNumbers (VideoAnalyzer.Frames), so each row's time is right.
    //
    // A measure's outputs are NaN on a frame whose view it doesn't suit
    // (FaceMeasureBase.Suits: a frontal measure on a quarter or lateral frame, a lateral
    // measure on a frontal one), and so are an item's on a frame whose view the item
    // doesn't suit (MeasureItemInfo.Suits), so a mixed-view video never reports
    // millimeters measured on a turned head; Evaluate and Peak therefore never pick a
    // turned or profile frame. A state is a quarter face when its Pose is
    // FacePose.Quarter and it has landmarks, and a profile when IsLateral.
    //
    // A state saved without its image (PostCropImage not valid) is measured on its
    // landmarks alone: the measures that need the image (FaceMeasureBase.NeedsImage) are
    // left out for it, so their outputs are NaN. That suits consumers that keep only
    // landmarks, such as the Unity app's video frames.
    public static FrameSeries FromStates(
        IReadOnlyList<FaceAnalysisState> states, IEnumerable<FaceMeasureBase> measures = null,
        int start = 0, int? stop = null, int step = 1, float fps = 0f,
        IReadOnlyList<int> frameNumbers = null)
    {
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), step, "The step must be at least 1.");
        if (frameNumbers != null && frameNumbers.Count != states.Count)
            throw new ArgumentException("frameNumbers needs one number per state.", nameof(frameNumbers));

        var all = new List<FaceMeasureBase>(measures ?? DynafaceMeasures.AllMeasures());
        var series = new FrameSeries(all, fps);
        List<FaceMeasureBase> landmarksOnly = null;
        FaceImage blank = default;

        int end = stop ?? states.Count;
        for (int index = start; index < end; index += step)
        {
            FaceAnalysisState state = states[index];
            int frame = frameNumbers == null ? index : frameNumbers[index];
            FaceMeasureContext ctx;
            List<FaceMeasureBase> run;
            if (state.PostCropImage.IsValid)
            {
                // With render: false nothing is drawn, so the state's image needs no copy.
                ctx = FaceAnalysisPersistence.Restore(state, copyImage: false);
                run = all;
            }
            else
            {
                if (landmarksOnly == null)
                {
                    landmarksOnly = all.FindAll(m => !m.NeedsImage);
                    int size = DynafaceConstants.StyleganWidth;
                    blank = new FaceImage(new Rgba32[size * size], size, size);
                }
                // The measures left run on landmarks alone, and with render: false nothing
                // is drawn, so one blank image stands in for every frame.
                ctx = FaceAnalysisPersistence.Restore(state.WithImage(blank), copyImage: false);
                run = landmarksOnly;
            }
            // The restored context's view is the state's: IsLateral as saved, and
            // IsQuarter for a Quarter pose with landmarks.
            series.Add(frame, ResultsInView(all, ctx.Analyze(run, render: false), ctx.IsLateral, ctx.IsQuarter));
        }
        return series;
    }

    // The results with NaN for every output (FaceMeasureBase.Outputs) of a measure that
    // doesn't suit the view (FaceMeasureBase.Suits): a frontal measure's on a quarter or
    // lateral face, and a lateral measure's on a frontal one; and for every output of an
    // item that doesn't suit it (MeasureItemInfo.Suits, after SyncItems). Analyze runs
    // every enabled measure whatever the view, so without this a millimeter measure's
    // numbers from a turned head would be reported. Null stays null. Mirrors
    // dynaface-lib's video._results_in_view.
    internal static Dictionary<string, double> ResultsInView(
        IEnumerable<FaceMeasureBase> measures, IReadOnlyDictionary<string, double> results,
        bool lateral, bool quarter)
    {
        if (results == null) return null;
        var inView = new Dictionary<string, double>();
        foreach (var kv in results) inView[kv.Key] = kv.Value;
        foreach (var measure in measures)
        {
            if (!measure.Suits(lateral, quarter))
            {
                foreach (string name in measure.Outputs()) inView[name] = double.NaN;
                continue;
            }
            measure.SyncItems();   // the items' flags follow the measure's as they are
            foreach (var item in measure.Items)
            {
                if (item.Suits(lateral, quarter)) continue;
                foreach (string name in item.OutputNames()) inView[name] = double.NaN;
            }
        }
        return inView;
    }

    // Adds a row: one frame's Analyze() results, or null for a frame without a face, whose
    // values are all NaN. frame: its number, such as its position in the video.
    public void Add(int frame, IReadOnlyDictionary<string, double> results)
    {
        _frames.Add(frame);
        foreach (string name in _names)
            _values[name].Add(results != null && results.TryGetValue(name, out double value) ? value : double.NaN);
    }

    // A frame's time in seconds, or NaN if the series has no frame rate.
    public double Time(int frame) => Fps > 0f ? frame / (double)Fps : double.NaN;

    // The frame where these outputs, added together, are largest. The widest smile is
    // Peak("dental_area"), and the widest eye opening is Peak("eye.left", "eye.right").
    // Rows where any of them is NaN are skipped, and of equal rows, the first wins. Null if
    // no row has all of them. Throws KeyNotFoundException for an output that isn't in the
    // series, and ArgumentException when none is named.
    public int? Peak(params string[] names)
    {
        int? row = PeakRow(names);
        return row.HasValue ? _frames[row.Value] : (int?)null;
    }

    // Finds the widest smile and the widest eye opening, and the eyes and mouth at each,
    // from the dental_area, eye.left and eye.right outputs: build the series with
    // MeasureDentalArea and MeasureEyeArea, with those items enabled, to have them. Null if
    // the series lacks one of those outputs or no frame has a value for it.
    public Evaluation Evaluate()
    {
        if (!_values.ContainsKey("dental_area") || !_values.ContainsKey("eye.left") || !_values.ContainsKey("eye.right"))
            return null;
        int? smile = PeakRow(new[] { "dental_area" });
        int? eyes = PeakRow(new[] { "eye.left", "eye.right" });
        if (!smile.HasValue || !eyes.HasValue) return null;
        return new Evaluation(MomentAt(smile.Value), MomentAt(eyes.Value));
    }

    // Writes the series as CSV: a header of frame, time and the outputs, then one row per
    // frame, each line ending in "\n". The time is in seconds, to 2 decimal places, and is
    // left empty without a frame rate. Values have at most 4 decimal places, and one that
    // isn't a finite number is left empty. Same text as dynaface-lib's write_csv().
    public void WriteCsv(TextWriter writer)
    {
        writer.Write("frame,time");
        foreach (string name in _names)
        {
            writer.Write(',');
            writer.Write(name);
        }
        writer.Write('\n');

        for (int row = 0; row < _frames.Count; row++)
        {
            int frame = _frames[row];
            double time = Time(frame);
            writer.Write(frame.ToString(CultureInfo.InvariantCulture));
            writer.Write(',');
            if (!double.IsNaN(time))
                writer.Write(Math.Round(time, 2, MidpointRounding.ToEven).ToString("0.00", CultureInfo.InvariantCulture));
            foreach (string name in _names)
            {
                writer.Write(',');
                writer.Write(CsvValue(_values[name][row]));
            }
            writer.Write('\n');
        }
    }

    // The series as CSV text (see WriteCsv).
    public string ToCsv()
    {
        using var writer = new StringWriter(CultureInfo.InvariantCulture);
        WriteCsv(writer);
        return writer.ToString();
    }

    // At most 4 decimal places, without trailing zeros, rounding half to even like Python's
    // formatting; empty when the value isn't a finite number.
    static string CsvValue(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value)) return string.Empty;
        string text = Math.Round(value, 4, MidpointRounding.ToEven).ToString("0.####", CultureInfo.InvariantCulture);
        return text == "-0" ? "0" : text;
    }

    int? PeakRow(IReadOnlyList<string> names)
    {
        if (names == null || names.Count == 0)
            throw new ArgumentException("Name at least one output.", nameof(names));

        var columns = new List<double>[names.Count];
        for (int i = 0; i < names.Count; i++) columns[i] = _values[names[i]];

        int? best = null;
        double bestTotal = double.NegativeInfinity;
        for (int row = 0; row < _frames.Count; row++)
        {
            double total = 0;
            bool complete = true;
            foreach (List<double> column in columns)
            {
                if (double.IsNaN(column[row])) { complete = false; break; }
                total += column[row];
            }
            if (complete && total > bestTotal)
            {
                best = row;
                bestTotal = total;
            }
        }
        return best;
    }

    Moment MomentAt(int row)
    {
        int frame = _frames[row];
        return new Moment(frame, Time(frame),
            _values["dental_area"][row], _values["eye.left"][row], _values["eye.right"][row]);
    }

    // A frame that the evaluation picks out, with the outputs it compares there.
    public readonly struct Moment
    {
        public Moment(int frame, double time, double dentalArea, double eyeLeft, double eyeRight)
        {
            Frame = frame;
            Time = time;
            DentalArea = dentalArea;
            EyeLeft = eyeLeft;
            EyeRight = eyeRight;
        }

        public int Frame { get; }            // the frame's number in the series
        public double Time { get; }          // seconds (Frame / Fps); NaN without a frame rate
        public double DentalArea { get; }    // the dental display, in mm²
        public double EyeLeft { get; }       // the left eye's area, in mm²
        public double EyeRight { get; }      // the right eye's area, in mm²

        public double EyeTotal => EyeLeft + EyeRight;

        // One eye's area over the other's, or NaN when the other's is 0.
        public double EyeLeftOverRight => Ratio(EyeLeft, EyeRight);
        public double EyeRightOverLeft => Ratio(EyeRight, EyeLeft);

        static double Ratio(double a, double b) => b != 0 ? a / b : double.NaN;
    }

    // The eyes at the widest smile and the mouth at the widest eye opening: the comparison
    // used in studies of oral-ocular synkinesis, which the Dynaface apps show as their
    // Evaluation report.
    public sealed class Evaluation
    {
        public Evaluation(Moment widestSmile, Moment widestEyes)
        {
            WidestSmile = widestSmile;
            WidestEyes = widestEyes;
        }

        public Moment WidestSmile { get; }   // the frame with the largest dental_area
        public Moment WidestEyes { get; }    // the frame with the largest eye.left + eye.right
    }
}
