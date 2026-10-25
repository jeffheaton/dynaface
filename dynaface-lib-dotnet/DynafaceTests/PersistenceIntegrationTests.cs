using System.IO;

namespace DynafaceTests;

// Model-gated persistence round-trip, mirroring dynaface-lib's test_persist.py:
// capture -> restore -> re-run measures -> same numbers as a fresh run.
[Collection("ModelTests")]
public class PersistenceIntegrationTests
{
    readonly ModelFixture _fixture;

    public PersistenceIntegrationTests(ModelFixture fixture) => _fixture = fixture;

    static string TestImage(string name) => Path.Combine(AppContext.BaseDirectory, "TestData", name);

    [SkippableFact]
    public void CaptureThenRestore_ReproducesTheSameMeasurements()
    {
        _fixture.RequireModels();

        var photo = ImageLoader.Load(TestImage("img1-512.jpg"));
        var result = FacialPipelineIntegrationTests.RunWithRotationRetry(photo);
        Assert.NotNull(result);

        // Capture before anything draws: the context below draws on the result's own
        // pixel array, and a capture taken afterwards would save the overlays too (which
        // MeasureSkinTone, sampling the restored image, would then measure).
        var state = FaceAnalysisPersistence.Capture(result.Value);

        var originalCtx = FacialPipelineIntegrationTests.BuildContext(result.Value);
        var originalValues = FacialPipelineIntegrationTests.RunAllFrontalMeasures(originalCtx);

        var restoredCtx = FaceAnalysisPersistence.Restore(state);
        var restoredValues = FacialPipelineIntegrationTests.RunAllFrontalMeasures(restoredCtx);

        foreach (var kv in originalValues)
        {
            Assert.True(restoredValues.ContainsKey(kv.Key), $"missing key after restore: {kv.Key}");
            Assert.Equal(kv.Value, restoredValues[kv.Key], 3);
        }
    }

    [SkippableFact]
    public void Capture_ClonesPixelArray_IndependentFromLiveContext()
    {
        _fixture.RequireModels();

        var photo = ImageLoader.Load(TestImage("img1-512.jpg"));
        var result = FacialPipelineIntegrationTests.RunWithRotationRetry(photo);
        Assert.NotNull(result);

        var state = FaceAnalysisPersistence.Capture(result.Value);

        // A live FaceMeasureContext built from the SAME result shares its pixel
        // array (per its constructor's own contract) and measures draw directly
        // onto it — Capture must clone so that drawing can never retroactively
        // mutate an already-captured snapshot.
        Assert.NotSame(result.Value.AlignedCrop.Pixels, state.PostCropImage.Pixels);
    }
}
