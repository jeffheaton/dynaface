using System;
using System.IO;

class Program
{
    static int Main(string[] args)
    {
        if (args.Length < 4)
        {
            Console.WriteLine("Usage: FacialDllConsole <image> <blazeface.onnx> <spiga.onnx> <u2net.onnx> [options]");
            Console.WriteLine();
            Console.WriteLine("  --list-tensors   Print detected tensor names and exit.");
            return 1;
        }

        string imagePath     = args[0];
        string blazeFacePath = args[1];
        string spigaPath     = args[2];
        string u2netPath     = args[3];
        bool listTensors     = Array.IndexOf(args, "--list-tensors") >= 0;

        if (!File.Exists(imagePath))     { Console.Error.WriteLine($"Image not found: {imagePath}");              return 1; }
        if (!File.Exists(blazeFacePath)) { Console.Error.WriteLine($"BlazeFace model not found: {blazeFacePath}"); return 1; }
        if (!File.Exists(spigaPath))     { Console.Error.WriteLine($"SPIGA model not found: {spigaPath}");         return 1; }
        if (!File.Exists(u2netPath))     { Console.Error.WriteLine($"U2Net model not found: {u2netPath}");         return 1; }

        Console.WriteLine("Initializing models...");
        using var inference = new OnnxDynafaceInference(blazeFacePath, spigaPath, u2netPath);

        if (listTensors)
        {
            Console.WriteLine(inference.DescribeTensors());
            return 0;
        }

        FacePipeline.Initialize(inference);
        return RunPipeline(imagePath, inference);
    }

    static int RunPipeline(string imagePath, IDynafaceInference inference)
    {
        Console.WriteLine("Loading image...");
        var photo = ImageLoader.Load(imagePath);

        Console.WriteLine("Running face detection...");
        FacePipelineResult? result = null;
        foreach (int rot in new[] { 0, 90, 180, 270 })
        {
            var candidate = FacePipeline.Run(photo, rotationAngle: rot, flipHorizontal: false);
            if (candidate == null) continue;
            if (result == null) result = candidate;
            if (FacePipeline.LastDetectionEyesOk)
            {
                result = candidate;
                Console.WriteLine($"Face detected at {rot}°.");
                break;
            }
            Console.WriteLine($"Face found at {rot}° but eye keypoints failed — trying next rotation.");
        }

        if (result == null)
        {
            Console.Error.WriteLine("No face detected.");
            return 1;
        }

        var crop = result.Value.AlignedCrop;
        Console.WriteLine($"Pose: {result.Value.Pose}{(result.Value.IsLateral && result.Value.Flipped ? " (flipped to face left)" : "")}");

        Vec2[] lateralLandmarks = null;
        if (result.Value.IsLateral)
        {
            Console.WriteLine("Running lateral background-removal analysis...");
            var lateral = LateralAnalyzer.Analyze(inference, crop, result.Value.Wflw98);
            if (lateral == null)
                Console.WriteLine("  Lateral analysis failed (no sagittal profile) — lateral measures will be skipped.");
            else
                lateralLandmarks = lateral.Value.LateralLandmarks;
        }

        Console.WriteLine("Running measurements...");
        var ctx = new FaceMeasureContext(
            crop, result.Value.Wflw98, result.Value.Pix2mm,
            isLateral: result.Value.IsLateral, lateralLandmarks: lateralLandmarks,
            headPose: result.Value.HeadPose);

        foreach (var m in BuildMeasures())
            if (m.Enabled) m.Calc(ctx, render: true);

        FaceRenderer.DisplayMode = FaceRenderer.LandmarkDisplayMode.Lm;
        FaceRenderer.DrawLandmarksOnto(crop.Pixels, crop.Width, crop.Height, result.Value.Wflw98);

        string dir     = Path.GetDirectoryName(Path.GetFullPath(imagePath)) ?? ".";
        string outPath = Path.Combine(dir, Path.GetFileNameWithoutExtension(imagePath) + "_annotated.png");
        Console.WriteLine($"Saving → {outPath}");
        ImageLoader.Save(ctx.ToImage(), outPath);

        Console.WriteLine();
        Console.WriteLine("=== Measurements ===");
        foreach (var line in ctx.TextLines)
            Console.WriteLine(line);

        if (ctx.Values.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("=== Structured values ===");
            foreach (var kv in ctx.Values)
                Console.WriteLine($"{kv.Key}={kv.Value}");
        }

        return 0;
    }

    static FaceMeasureBase[] BuildMeasures() => new FaceMeasureBase[]
    {
        new MeasurePose                 { Enabled = true },
        new MeasureIntercanthalDistance { Enabled = true },
        new MeasureOuterEyeCorners      { Enabled = true },
        new MeasureEyeArea              { Enabled = true },
        new MeasureBrows                { Enabled = true },
        new MeasureNoseFrontal          { Enabled = true },
        new MeasureMouthLength          { Enabled = true },
        new MeasureDentalArea           { Enabled = true },
        new MeasureOCE                  { Enabled = true },
        new MeasureFAI                  { Enabled = true },
        new MeasurePosition             { Enabled = true },
        new MeasureLateral              { Enabled = true },
        new MeasureSkinTone             { Enabled = true },
        new MeasureLandmarks            { Enabled = true },
    };
}
