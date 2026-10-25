using System;
using System.IO;

namespace DynafaceTests;

// Resolves where the 3 real ONNX models (blaze_face_short_range.onnx, spiga_wflw.onnx,
// u2net.onnx) live on this machine, for the model-gated integration tests. The models
// total ~400MB and are gitignored — they can never live in the repo — so tests look for
// a pre-populated directory instead of downloading anything.
public static class ModelPathResolver
{
    public const string BlazeFaceFileName = "blaze_face_short_range.onnx";
    public const string SpigaFileName = "spiga_wflw.onnx";
    public const string U2NetFileName = "u2net.onnx";

    public const string ModelsEnvVar = "DYNAFACE_ONNX_MODELS";

    // When DYNAFACE_ONNX_MODELS is set it is the only place looked at (so CI, or a
    // deliberate "no models" run, gets exactly what it asked for). Otherwise: dynaface-lib's
    // own default download location, then this repo's sibling dynaface-app/data (known to
    // have all 3 models on this dev machine).
    public static string[] CandidateDirectories()
    {
        var fromEnv = Environment.GetEnvironmentVariable(ModelsEnvVar);
        if (!string.IsNullOrEmpty(fromEnv)) return new[] { fromEnv };
        return new[]
        {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dynaface", "models"),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "dynaface-app", "data")),
        };
    }

    public static string? FindModelDirectory()
    {
        foreach (var dir in CandidateDirectories())
            if (HasAllModels(dir)) return dir;
        return null;
    }

    public static bool HasAllModels(string? dir)
    {
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return false;
        return File.Exists(Path.Combine(dir, BlazeFaceFileName))
            && File.Exists(Path.Combine(dir, SpigaFileName))
            && File.Exists(Path.Combine(dir, U2NetFileName));
    }
}
