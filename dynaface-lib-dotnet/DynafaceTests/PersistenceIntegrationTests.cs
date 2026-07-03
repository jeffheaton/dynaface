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

    [Fact]
    public void CaptureThenRestore_ReproducesTheSameMeasurements()
    {
        if (!_fixture.Available) return;

        var photo = ImageLoader.Load(TestImage("img1-512.jpg"));
        var result = FacialPipelineIntegrationTests.RunWithRotationRetry(photo);
        Assert.NotNull(result);

        var originalCtx = FacialPipelineIntegrationTests.BuildContext(result.Value);
        var originalValues = FacialPipelineIntegrationTests.RunAllFrontalMeasures(originalCtx);

        var state = FaceAnalysisPersistence.Capture(result.Value);
        var restoredCtx = FaceAnalysisPersistence.Restore(state);
        var restoredValues = FacialPipelineIntegrationTests.RunAllFrontalMeasures(restoredCtx);

        foreach (var kv in originalValues)
        {
            Assert.True(restoredValues.ContainsKey(kv.Key), $"missing key after restore: {kv.Key}");
            Assert.Equal(kv.Value, restoredValues[kv.Key], 3);
        }
    }

    [Fact]
    public void Capture_ClonesPixelArray_IndependentFromLiveContext()
    {
        if (!_fixture.Available) return;

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
