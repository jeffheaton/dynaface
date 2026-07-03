using System;
using System.IO;

namespace DynafaceTests;

// Shared, once-per-test-collection ONNX model load for the model-gated integration
// tests below (loading all 3 sessions per-test would be slow and pointless — the
// models themselves are stateless). Tests check Available and return early (a no-op
// pass, logged via the test's own output) when the 3 .onnx files aren't found —
// see ModelPathResolver for the search order. No network calls are ever made.
public class ModelFixture : IDisposable
{
    public bool Available { get; }
    public string? ModelDir { get; }

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
