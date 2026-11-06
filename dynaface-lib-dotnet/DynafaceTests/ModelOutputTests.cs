namespace DynafaceTests;

// A model whose output no working model gives (none, the wrong size, values that aren't
// numbers, the same value everywhere, or landmarks all at one point) throws
// ModelOutputException, so a backend that computes the models wrongly isn't mistaken
// for a photo without a face. Mirrors dynaface-lib's tests/test_model_outputs.py.
public class ModelOutputTests
{
    internal sealed class FakeInference : IDynafaceInference
    {
        public (float[] regressors, float[] scores)? Face;
        public (float[] landmarks, float[] pose)? Landmarks;
        public float[]? Mask;

        public bool IsReady => true;
        public (float[] regressors, float[] scores)? RunBlazeFace(float[] tensor) => Face;
        public (float[] landmarks, float[] pose)? RunSpiga(float[] imageTensor) => Landmarks;
        public float[] RunU2Net(float[] imageTensor) => Mask!;
        public void Dispose() { }
    }

    static float[] Values(int count, int seed, Func<Random, float> next)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => next(random)).ToArray();
    }

    // Outputs like a working model's: scores that differ, none high enough for a face.
    static (float[] regressors, float[] scores) NoFace() => (
        Values(896 * 16, 1, r => (float)(r.NextDouble() * 10 - 5)),
        Values(896, 2, r => (float)(-8 - r.NextDouble() * 4)));

    static (float[] landmarks, float[] pose) Landmarks() => (
        Values(98 * 2, 3, r => (float)(0.3 + 0.4 * r.NextDouble())),
        new[] { 5f, -3f, 2f, 0f, 0f, 30f });

    static float[] Mask() => Values(320 * 320, 4, r => (float)r.NextDouble());

    static float[] WithNaN(float[] values)
    {
        var copy = (float[])values.Clone();
        copy[copy.Length / 2] = float.NaN;
        return copy;
    }

    static readonly FaceImage Photo = new(
        Enumerable.Repeat(new Rgba32(128, 128, 128, 255), 256 * 256).ToArray(), 256, 256);

    internal static FakeInference Working() => new() { Face = NoFace(), Landmarks = Landmarks(), Mask = Mask() };

    [Fact]
    public void WorkingModels()
    {
        var inference = Working();
        Assert.False(new BlazeFaceDetector(inference).TryDetectBbox(Photo, 0, out _, out _));   // no face
        Assert.NotNull(new SpigaLandmarkDetector(inference).Detect(Photo, (64, 64, 128, 128)));
        Assert.Equal(256 * 256, SagittalProfile.ProcessImage(inference, Photo).Length);
    }

    [Theory]
    [InlineData("no output")]
    [InlineData("wrong size")]
    [InlineData("NaN score")]
    [InlineData("infinite box")]
    [InlineData("every score the same")]
    public void FaceDetector(string fault)
    {
        var inference = Working();
        var (regressors, scores) = NoFace();
        inference.Face = fault switch
        {
            "no output" => null,
            "wrong size" => (regressors[..100], scores),
            "NaN score" => (regressors, WithNaN(scores)),
            "infinite box" => (Enumerable.Repeat(float.PositiveInfinity, regressors.Length).ToArray(), scores),
            "every score the same" => (regressors, new float[scores.Length]),
            _ => throw new ArgumentException(fault),
        };
        var error = Assert.Throws<ModelOutputException>(
            () => new BlazeFaceDetector(inference).TryDetectBbox(Photo, 0, out _, out _));
        Assert.Equal(ModelOutputCheck.FaceDetector, error.Model);
        // Output computed wrongly suggests another backend; no output or the wrong size
        // says what happened instead.
        string expected = fault switch
        {
            "no output" => "no output",
            "wrong size" => "wrong size",
            _ => "CPU",
        };
        Assert.Contains(expected, error.Message);
    }

    [Theory]
    [InlineData("no output")]
    [InlineData("wrong size")]
    [InlineData("NaN landmark")]
    [InlineData("NaN pose")]
    [InlineData("all at one point")]
    [InlineData("all zero")]
    public void LandmarkModel(string fault)
    {
        var inference = Working();
        var (landmarks, pose) = Landmarks();
        inference.Landmarks = fault switch
        {
            "no output" => null,
            "wrong size" => (landmarks[..50], pose),
            "NaN landmark" => (WithNaN(landmarks), pose),
            "NaN pose" => (landmarks, WithNaN(pose)),
            "all at one point" => (Enumerable.Repeat(0.5f, landmarks.Length).ToArray(), pose),
            "all zero" => (new float[landmarks.Length], new float[pose.Length]),
            _ => throw new ArgumentException(fault),
        };
        var error = Assert.Throws<ModelOutputException>(
            () => new SpigaLandmarkDetector(inference).Detect(Photo, (64, 64, 128, 128)));
        Assert.Equal(ModelOutputCheck.LandmarkModel, error.Model);
    }

    [Theory]
    [InlineData("no output")]
    [InlineData("wrong size")]
    [InlineData("NaN")]
    [InlineData("every pixel the same")]
    public void BackgroundModel(string fault)
    {
        var inference = Working();
        inference.Mask = fault switch
        {
            "no output" => null,
            "wrong size" => new float[100],
            "NaN" => WithNaN(Mask()),
            "every pixel the same" => new float[320 * 320],
            _ => throw new ArgumentException(fault),
        };
        var error = Assert.Throws<ModelOutputException>(() => SagittalProfile.ProcessImage(inference, Photo));
        Assert.Equal(ModelOutputCheck.BackgroundModel, error.Model);
    }
}

// The pipeline reports a broken backend instead of finding no face. It installs a fake
// inference, so it runs in the model-gated tests' collection and puts theirs back, as
// PipelineReadinessTests does.
[Collection("ModelTests")]
public class ModelOutputPipelineTests
{
    readonly ModelFixture _fixture;

    public ModelOutputPipelineTests(ModelFixture fixture) => _fixture = fixture;

    [Fact]
    public void RunReportsZeroedOutputs()
    {
        // Zeroed outputs, as from a GPU that runs the models but computes nothing: they
        // used to look like a photo without a face.
        var inference = ModelOutputTests.Working();
        inference.Face = (new float[896 * 16], new float[896]);
        var photo = new FaceImage(Enumerable.Repeat(new Rgba32(128, 128, 128, 255), 640 * 480).ToArray(), 640, 480);
        try
        {
            FacePipeline.Initialize(inference);
            Assert.True(FacePipeline.IsReady);
            Assert.Throws<ModelOutputException>(() => FacePipeline.Run(photo, 0, flipHorizontal: false));
        }
        finally
        {
            if (_fixture.Available) FacePipeline.Initialize(_fixture.Inference);
        }
    }
}
