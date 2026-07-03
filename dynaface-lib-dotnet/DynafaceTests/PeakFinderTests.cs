namespace DynafaceTests;

// Verifies PeakFinder against scipy.signal.find_peaks reference cases (computed via
// scipy 1.17.1), covering plateau-aware plain peaks, a height filter, and a distance
// filter — the 3 modes lateral.py's own use of find_peaks actually exercises.
public class PeakFinderTests
{
    [Fact]
    public void FindPeaks_Plain_HandlesPlateausAsScipyDoes()
    {
        double[] x = { 0, 1, 3, 1, 0, 2, 5, 2, 0, 1, 1, 1, 0, 3, 3, 3, 0 };
        var (indices, _) = PeakFinder.FindPeaks(x);
        Assert.Equal(new[] { 2, 6, 10, 14 }, indices);
    }

    [Fact]
    public void FindPeaks_HeightFilter_MatchesScipy()
    {
        double[] x = { 0, 5, 0, 8, 0, 3, 0, 9, 0, 4, 0 };
        var (indices, heights) = PeakFinder.FindPeaks(x, heightThreshold: 4.0);
        Assert.Equal(new[] { 1, 3, 7, 9 }, indices);
        Assert.Equal(new double[] { 5, 8, 9, 4 }, heights);
    }

    [Fact]
    public void FindPeaks_DistanceFilter_KeepsHighestInEachCluster()
    {
        double[] x = { 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0 };
        var (indices, _) = PeakFinder.FindPeaks(x, minDistance: 3);
        Assert.Equal(new[] { 3, 7, 11 }, indices);
    }

    [Fact]
    public void FindMinimaIndices_NegatesAndFindsMaxima()
    {
        double[] x = { 5, 3, 5, 1, 5, 4, 5 };
        var minima = PeakFinder.FindMinimaIndices(x);
        Assert.Equal(new[] { 1, 3, 5 }, minima);
    }

    [Fact]
    public void FindPeaks_EmptyInput_ReturnsEmpty()
    {
        var (indices, heights) = PeakFinder.FindPeaks(System.Array.Empty<double>());
        Assert.Empty(indices);
        Assert.Empty(heights);
    }
}
