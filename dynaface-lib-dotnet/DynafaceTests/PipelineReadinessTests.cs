namespace DynafaceTests;

// Replaces dynaface-lib's test_fail_init_models: dynaface-lib raises ValueError when
// models.init_models() hasn't been called. The .NET pipeline uses a Try*-style
// not-ready contract instead (no exception) — verified here with a fake
// IDynafaceInference rather than by asserting on FacePipeline's global static state
// "before" initialization, since that would be test-order-dependent (other tests in
// the same run may already have called FacePipeline.Initialize()).
public class PipelineReadinessTests
{
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
        FacePipeline.Initialize(new NotReadyInference());
        Assert.False(FacePipeline.IsReady);

        var photo = new FaceImage(new Rgba32[100 * 100], 100, 100);
        var result = FacePipeline.Run(photo, 0, flipHorizontal: false);
        Assert.Null(result);
    }
}
