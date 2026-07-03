using System.Collections.Generic;

namespace DynafaceTests;

// BlazeFaceDetector.GreedyNms has no Python test equivalent (dynaface_onnx.py's own
// _nms is exercised only indirectly, via test_facial.py's end-to-end numeric
// assertions) — this hand-rolled port needs its own direct coverage.
public class NmsTests
{
    [Fact]
    public void GreedyNms_NonOverlappingBoxes_KeepsBoth()
    {
        var x0 = new List<float> { 0f, 10f };
        var y0 = new List<float> { 0f, 0f };
        var x1 = new List<float> { 1f, 11f };
        var y1 = new List<float> { 1f, 1f };
        var scores = new List<float> { 0.9f, 0.8f };

        var keep = BlazeFaceDetector.GreedyNms(x0, y0, x1, y1, scores, 0.3f);

        Assert.Equal(2, keep.Count);
    }

    [Fact]
    public void GreedyNms_HeavilyOverlappingBoxes_KeepsOnlyHighestScore()
    {
        var x0 = new List<float> { 0f, 0.05f };
        var y0 = new List<float> { 0f, 0f };
        var x1 = new List<float> { 1f, 1.05f };
        var y1 = new List<float> { 1f, 1f };
        var scores = new List<float> { 0.5f, 0.9f };

        var keep = BlazeFaceDetector.GreedyNms(x0, y0, x1, y1, scores, 0.3f);

        Assert.Single(keep);
        Assert.Equal(1, keep[0]);
    }

    [Fact]
    public void GreedyNms_EmptyInput_ReturnsEmpty()
    {
        var empty = new List<float>();
        var keep = BlazeFaceDetector.GreedyNms(empty, empty, empty, empty, empty, 0.3f);
        Assert.Empty(keep);
    }

    [Fact]
    public void GreedyNms_HighestScoreListedFirst()
    {
        var x0 = new List<float> { 0f, 50f, 100f };
        var y0 = new List<float> { 0f, 0f, 0f };
        var x1 = new List<float> { 1f, 51f, 101f };
        var y1 = new List<float> { 1f, 1f, 1f };
        var scores = new List<float> { 0.2f, 0.95f, 0.5f };

        var keep = BlazeFaceDetector.GreedyNms(x0, y0, x1, y1, scores, 0.3f);

        Assert.Equal(3, keep.Count);
        Assert.Equal(1, keep[0]); // index 1 has the highest score, listed first
    }
}
