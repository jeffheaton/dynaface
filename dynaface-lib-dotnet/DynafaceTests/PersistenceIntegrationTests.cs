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

    // A profile is restored as a profile, flipped or not, so MeasureLateral measures it
    // as it measured the original. Mirrors test_persist_left_lateral and
    // test_flipped_profile; up to 2.0.4 a restored profile lost its lateral landmarks.
    [SkippableTheory]
    [InlineData("img3-1024-left-lateral.jpg", false)]
    [InlineData("img2-1024-right-lateral.jpg", true)]
    public void CaptureThenRestore_AProfileMeasuresTheSame(string image, bool flipped)
    {
        _fixture.RequireModels();

        var result = FacialPipelineIntegrationTests.RunWithRotationRetry(ImageLoader.Load(TestImage(image)));
        Assert.NotNull(result);
        Assert.True(result.Value.IsLateral);
        Assert.Equal(flipped, result.Value.Flipped);
        var measures = new FaceMeasureBase[] { new MeasureLateral(), new MeasurePose(), new MeasurePosition() };
        var expected = TestHelpers.ContextFor(result.Value).Analyze(measures, render: false);
        Assert.Contains("nfa", expected.Keys);

        var restored = FaceAnalysisPersistence.Restore(FaceAnalysisPersistence.Capture(result.Value));
        Assert.True(restored.IsLateral);
        Assert.Equal(flipped, restored.Flipped);
        Assert.Equal(expected, restored.Analyze(measures, render: false));
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
