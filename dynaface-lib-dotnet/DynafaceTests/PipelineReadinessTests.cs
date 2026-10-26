namespace DynafaceTests;

// Replaces dynaface-lib's test_fail_init_models: dynaface-lib raises ValueError when
// models.init_models() hasn't been called. The .NET pipeline uses a Try*-style
// not-ready contract instead (no exception) — verified here with a fake
// IDynafaceInference rather than by asserting on FacePipeline's global static state
// "before" initialization, since that would be test-order-dependent (other tests in
// the same run may already have called FacePipeline.Initialize()).
//
// Installing the fake replaces the inference FacePipeline shares with the model-gated
// tests, so this class runs in their collection, never at the same time as one of
// them, and puts the real inference back afterward. Run alongside them, it made every
// model-gated test after it fail on Windows CI.
[Collection("ModelTests")]
public class PipelineReadinessTests
{
    readonly ModelFixture _fixture;

    public PipelineReadinessTests(ModelFixture fixture) => _fixture = fixture;

    class NotReadyInference : IDynafaceInference
    {
        public bool IsReady => false;
        public (float[] regressors, float[] scores)? RunBlazeFace(float[] tensor) => null;
        public (float[] landmarks, float[] pose)? RunSpiga(float[] imageTensor) => null;
        public float[] RunU2Net(float[] imageTensor) => null!;
        public void Dispose() { }
    }

    [Fact]
    public void BlazeFaceDetector_NotReadyInference_TryDetectBboxReturnsFalse()
    {
        var detector = new BlazeFaceDetector(new NotReadyInference());
        Assert.False(detector.IsReady);

        var photo = new FaceImage(new Rgba32[100 * 100], 100, 100);
        bool found = detector.TryDetectBbox(photo, 0, out _, out _);
        Assert.False(found);
    }

    [Fact]
    public void SpigaLandmarkDetector_NotReadyInference_IsNotReady()
    {
        var detector = new SpigaLandmarkDetector(new NotReadyInference());
        Assert.False(detector.IsReady);
    }

    [Fact]
    public void FacePipeline_Initialize_WithNotReadyInference_StaysNotReady()
    {
        try
        {
            FacePipeline.Initialize(new NotReadyInference());
            Assert.False(FacePipeline.IsReady);

            var photo = new FaceImage(new Rgba32[100 * 100], 100, 100);
            var result = FacePipeline.Run(photo, 0, flipHorizontal: false);
            Assert.Null(result);
        }
        finally
        {
            if (_fixture.Available) FacePipeline.Initialize(_fixture.Inference);
        }
    }
}
