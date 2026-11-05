// Persistence DTO mirroring dynaface-lib's AnalyzeFace.dump_state/load_state fields
// exactly: post-crop image, raw headpose, landmarks, pupillary distance, pix2mm, and
// face rotation. A typed class (the .NET-idiomatic replacement for Python's plain
// heterogeneous list) rather than a serialization format — this is an in-memory
// round-trip, same as dynaface-lib's own test_persist.py.
//
// Deliberately does NOT capture lateral_landmarks/IsLateral — dynaface-lib's
// dump_state doesn't either, so a restored lateral face never re-fires
// MeasureLateral correctly there either; Restore() reproduces that same limitation
// rather than improving on it.
public sealed class FaceAnalysisState
{
    public FaceImage PostCropImage;
    public float[] HeadPose;
    public Vec2[] Landmarks;
    public float PupillaryDistance;
    public float Pix2mm;
    public float? FaceRotationRad;
}

public static class FaceAnalysisPersistence
{
    // Snapshots a FacePipelineResult. Clones the pixel array so later drawing onto a
    // FaceMeasureContext built from the ORIGINAL result (which shares the same array)
    // can't retroactively mutate this captured state — matching dynaface-lib's own
    // original_img/render_img separation (original_img is never drawn on).
    public static FaceAnalysisState Capture(FacePipelineResult result)
    {
        var crop = result.AlignedCrop;
        var clonedPixels = (Rgba32[])crop.Pixels.Clone();

        return new FaceAnalysisState
        {
            PostCropImage = new FaceImage(clonedPixels, crop.Width, crop.Height),
            HeadPose = result.HeadPose != null ? (float[])result.HeadPose.Clone() : null,
            Landmarks = result.Wflw98 != null ? (Vec2[])result.Wflw98.Clone() : null,
            PupillaryDistance = result.PupillaryDistance,
            Pix2mm = result.Pix2mm,
            FaceRotationRad = result.FaceRotationRad,
        };
    }

    // Rebuilds a FaceMeasureContext from saved state, ready for measures to run
    // again and reproduce the same numbers. Like dynaface-lib's load_state(), it
    // keeps the pupillary distance the analysis assumed, not today's
    // DynafaceConfig.PupilDistMm: Pix2mm has always been computed as it over the
    // PupillaryDistance saved beside it, so their product gives it back. With it, the
    // context measures its scale on the saved landmarks, which also corrects states
    // saved when the scale was measured before the crop. A pair that holds no
    // pupillary distance (a profile's fixed scale, or no pupils) keeps its saved scale.
    public static FaceMeasureContext Restore(FaceAnalysisState state)
    {
        float savedPupilDistMm = state.PupillaryDistance * state.Pix2mm;
        bool havePupils = state.Landmarks != null && state.Landmarks.Length > DynafaceConstants.LmLeftPupil;
        if (havePupils && savedPupilDistMm > 0f && !float.IsInfinity(savedPupilDistMm))
            return new FaceMeasureContext(
                state.PostCropImage, state.Landmarks, headPose: state.HeadPose, pupilDistMm: savedPupilDistMm);
        return FaceMeasureContext.WithFixedScale(
            state.PostCropImage, state.Landmarks, state.Pix2mm, headPose: state.HeadPose);
    }
}
