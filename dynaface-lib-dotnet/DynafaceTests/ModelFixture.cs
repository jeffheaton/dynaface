using System;
using System.IO;

namespace DynafaceTests;

// Shared, once-per-test-collection ONNX model load for the model-gated integration
// tests below (loading all 3 sessions per-test would be slow and pointless — the
// models themselves are stateless). Each model-gated test is a [SkippableFact] /
// [SkippableTheory] that starts with RequireModels(): when the 3 .onnx files aren't
// found (see ModelPathResolver for the search order) the test is reported as Skipped,
// or fails if DYNAFACE_REQUIRE_MODELS is set (CI's publishing build sets it), so a
// missing prerequisite is never mistaken for a pass. No network calls are ever made.
public class ModelFixture : IDisposable
{
    public const string RequireEnvVar = "DYNAFACE_REQUIRE_MODELS";

    public bool Available { get; }
    public string? ModelDir { get; }

    // Skips the calling test when the models are missing, or fails it in require mode.
    public void RequireModels()
    {
        if (Available) return;
        string message = "ONNX models not found (looked in: " +
            string.Join(", ", ModelPathResolver.CandidateDirectories()) +
            $"). Set {ModelPathResolver.ModelsEnvVar} to a folder with " +
            $"{ModelPathResolver.BlazeFaceFileName}, {ModelPathResolver.SpigaFileName} and " +
            $"{ModelPathResolver.U2NetFileName}.";
        var require = Environment.GetEnvironmentVariable(RequireEnvVar);
        if (!string.IsNullOrEmpty(require) && require != "0" && !require.Equals("false", StringComparison.OrdinalIgnoreCase))
            Assert.Fail($"{RequireEnvVar} is set but {message}");
        Skip.If(true, message);
    }

    readonly IDynafaceInference? _inference;

    public ModelFixture()
    {
        ModelDir = ModelPathResolver.FindModelDirectory();
        Available = ModelDir != null;
        if (Available)
        {
            _inference = new OnnxDynafaceInference(
                Path.Combine(ModelDir!, ModelPathResolver.BlazeFaceFileName),
                Path.Combine(ModelDir!, ModelPathResolver.SpigaFileName),
                Path.Combine(ModelDir!, ModelPathResolver.U2NetFileName));
            FacePipeline.Initialize(_inference);
        }
    }

    public IDynafaceInference Inference => _inference!;

    public void Dispose() => _inference?.Dispose();
}

[CollectionDefinition("ModelTests")]
public class ModelTestsCollection : ICollectionFixture<ModelFixture>
{
}
