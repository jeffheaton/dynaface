namespace DynafaceTests;

// Mirrors dynaface-lib's test_measures_landmarks.py MockFace: each landmark is
// (i*2, i*2+1) for easy verification, letting measures be exercised without a real
// model run.
internal static class TestHelpers
{
    public static Vec2[] MakeLandmarks(int count = 98)
    {
        var lm = new Vec2[count];
        for (int i = 0; i < count; i++) lm[i] = new Vec2(i * 2, i * 2 + 1);
        return lm;
    }

    public static FaceMeasureContext BuildContext(
        int width = 1024, int height = 1024, Vec2[]? landmarks = null,
        bool isLateral = false, Vec2[]? lateralLandmarks = null, float pix2mm = 0.25f)
    {
        landmarks ??= MakeLandmarks();
        var photo = new FaceImage(new Rgba32[width * height], width, height);
        return new FaceMeasureContext(photo, landmarks, pix2mm, isLateral, lateralLandmarks);
    }
}
