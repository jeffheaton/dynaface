using System.Globalization;
using System.Text.RegularExpressions;
using static LateralLandmarkFinder;

namespace DynafaceTests;

// Values the .NET port keeps its own copy of, checked against dynaface-lib-python's
// source itself (the test project links the two files into TestData/python), so the
// ports can't drift apart: SPIGA's two fixed model inputs, which every inference
// backend feeds alongside the image, and the lateral landmark names.
public class SharedConstantsTests
{
    static string PythonSource(string file) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", "python", file));

    // The text between a Python assignment's start and the first end marker after it.
    static string Between(string source, string start, string end)
    {
        int from = source.IndexOf(start, StringComparison.Ordinal);
        Assert.True(from >= 0, $"'{start}' isn't in the Python source");
        from += start.Length;
        int to = source.IndexOf(end, from, StringComparison.Ordinal);
        Assert.True(to >= 0, $"'{end}' doesn't follow '{start}'");
        return source.Substring(from, to - from);
    }

    // A NumPy float32 array literal's numbers as NumPy stores them: Python reads each
    // as a double, then NumPy rounds it to float32.
    static float[] Float32s(string literal) =>
        Regex.Matches(literal, @"-?\d+\.\d+(?:[eE][-+]?\d+)?")
            .Select(m => (float)double.Parse(m.Value, CultureInfo.InvariantCulture))
            .ToArray();

    [Fact]
    public void SpigaInputsMatchPython()
    {
        string source = PythonSource("dynaface_onnx.py");
        float[] model3d = SpigaModelInputs.Model3d();
        float[] cameraMatrix = SpigaModelInputs.CameraMatrix();

        Assert.Equal(SpigaLandmarkDetector.NumWflwLandmarks * 3, model3d.Length);
        Assert.Equal(9, cameraMatrix.Length);
        Assert.Equal(Float32s(Between(source, "_MODEL3D = np.array(", "dtype=")), model3d);
        Assert.Equal(Float32s(Between(source, "_CAM_MATRIX = np.array(", "dtype=")), cameraMatrix);
    }

    [Fact]
    public void SpigaInputsAreCopies()
    {
        // A backend can change its own array without changing anyone else's.
        float[] model3d = SpigaModelInputs.Model3d();
        model3d[0] = 42f;
        Assert.NotEqual(42f, SpigaModelInputs.Model3d()[0]);

        float[] cameraMatrix = SpigaModelInputs.CameraMatrix();
        cameraMatrix[0] = 42f;
        Assert.NotEqual(42f, SpigaModelInputs.CameraMatrix()[0]);
    }

    [Fact]
    public void LateralLandmarkNamesMatchPython()
    {
        string list = Between(PythonSource("lateral.py"), "LATERAL_LANDMARK_NAMES = [", "]");
        string[] python = Regex.Matches(list, "\"([^\"]*)\"").Select(m => m.Groups[1].Value).ToArray();
        Assert.Equal(python, LandmarkNames);
    }

    [Fact]
    public void LateralLandmarkNamesFollowTheIndexConstants()
    {
        Assert.Contains("Glabella", LandmarkNames[Glabella]);
        Assert.Contains("Nasion", LandmarkNames[Nasion]);
        Assert.Contains("Nasal Tip", LandmarkNames[NasalTip]);
        Assert.Contains("Subnasal", LandmarkNames[Subnasal]);
        Assert.Contains("Mento Labial", LandmarkNames[MentoLabial]);
        Assert.Contains("Pogonion", LandmarkNames[Pogonion]);

        // Read-only: no consumer can rename a point for everyone else.
        Assert.Throws<NotSupportedException>(() => ((IList<string>)LandmarkNames)[0] = "Renamed");
    }
}
