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

    // A context with a pinned scale, so a measure's formulas can be checked with
    // round numbers (a real one measures its scale on its landmarks).
    public static FaceMeasureContext BuildContext(
        int width = 1024, int height = 1024, Vec2[]? landmarks = null,
        bool isLateral = false, Vec2[]? lateralLandmarks = null, float pix2mm = 0.25f)
    {
        landmarks ??= MakeLandmarks();
        var photo = new FaceImage(new Rgba32[width * height], width, height);
        return FaceMeasureContext.WithFixedScale(photo, landmarks, pix2mm, isLateral, lateralLandmarks);
    }

    // The context an app builds from a pipeline result: its scale measured on the
    // result's landmarks, with the pupillary distance the run assumed.
    public static FaceMeasureContext ContextFor(FacePipelineResult r, bool clonePixels = true, bool withLateralLandmarks = true) =>
        new FaceMeasureContext(
            clonePixels
                ? new FaceImage((Rgba32[])r.AlignedCrop.Pixels.Clone(), r.AlignedCrop.Width, r.AlignedCrop.Height)
                : r.AlignedCrop,
            r.Wflw98, isLateral: r.IsLateral,
            lateralLandmarks: withLateralLandmarks ? r.LateralAnalysis?.LateralLandmarks : null,
            headPose: r.HeadPose, pose: r.Pose, flipped: r.Flipped, pupilDistMm: r.PupilDistMm);
}
