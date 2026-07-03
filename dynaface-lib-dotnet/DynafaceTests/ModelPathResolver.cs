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
    public const string SpigaFileName     = "spiga_wflw.onnx";
    public const string U2NetFileName     = "u2net.onnx";

    // Checked in order: DYNAFACE_ONNX_MODELS env var, dynaface-lib's own default
    // download location, then this repo's sibling dynaface-app/data (known to have
    // all 3 models on this dev machine).
    public static string? FindModelDirectory()
    {
        var candidates = new[]
        {
            Environment.GetEnvironmentVariable("DYNAFACE_ONNX_MODELS"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dynaface", "models"),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "dynaface-app", "data")),
        };

        foreach (var dir in candidates)
        {
            if (string.IsNullOrEmpty(dir)) continue;
            if (HasAllModels(dir)) return dir;
        }
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
