// Persistence DTO mirroring dynaface-lib's AnalyzeFace.dump_state/load_state fields
// exactly: post-crop image, raw headpose, landmarks, pupillary distance, pix2mm, face
// rotation, and the view. A typed class (the .NET-idiomatic replacement for Python's
// plain heterogeneous list) rather than a serialization format — this is an in-memory
// round-trip, same as dynaface-lib's own test_persist.py.
public sealed class FaceAnalysisState
{
    public FaceImage PostCropImage;
    public float[] HeadPose;
    public Vec2[] Landmarks;
    public float PupillaryDistance;
    public float Pix2mm;
    public float? FaceRotationRad;

    // The view, as dynaface-lib's dump_state saves it from 2.1 on: the resolved pose,
    // whether a profile was mirrored to face left, and for a profile its 6 lateral
    // landmarks and sagittal outline (LateralAnalyzer.Result's), so MeasureLateral
    // measures a restored profile as it did the original. A state that leaves them at
    // their defaults (as every state did up to 2.0.4) restores as a frontal face.
    public FacePose Pose;
    public bool Flipped;
    public bool IsLateral;
    public Vec2[] LateralLandmarks;
    public float[] SagittalX;
    public float[] SagittalY;

    // A copy of this state that shows another image; the arrays are shared.
    internal FaceAnalysisState WithImage(FaceImage image)
    {
        var copy = (FaceAnalysisState)MemberwiseClone();
        copy.PostCropImage = image;
        return copy;
    }
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

        var state = new FaceAnalysisState
        {
            PostCropImage = new FaceImage(clonedPixels, crop.Width, crop.Height),
            HeadPose = result.HeadPose != null ? (float[])result.HeadPose.Clone() : null,
            Landmarks = result.Wflw98 != null ? (Vec2[])result.Wflw98.Clone() : null,
            PupillaryDistance = result.PupillaryDistance,
            Pix2mm = result.Pix2mm,
            FaceRotationRad = result.FaceRotationRad,
        };
        CaptureView(result, state);
        return state;
    }

    // Copies a result's view (pose, flip and a profile's lateral data) into a state.
    internal static void CaptureView(FacePipelineResult result, FaceAnalysisState state)
    {
        state.Pose = result.Pose;
        state.Flipped = result.Flipped;
        state.IsLateral = result.IsLateral;
        var lateral = result.IsLateral ? result.LateralAnalysis : null;
        state.LateralLandmarks = (Vec2[])lateral?.LateralLandmarks?.Clone();
        state.SagittalX = (float[])lateral?.SagittalX?.Clone();
        state.SagittalY = (float[])lateral?.SagittalY?.Clone();
    }

    // Rebuilds a FaceMeasureContext from saved state, ready for measures to run
    // again and reproduce the same numbers. Like dynaface-lib's load_state(), it
    // keeps the pupillary distance the analysis assumed, not today's
    // DynafaceConfig.PupilDistMm: Pix2mm has always been computed as it over the
    // PupillaryDistance saved beside it, so their product gives it back. With it, the
    // context measures its scale on the saved landmarks, which also corrects states
    // saved when the scale was measured before the crop. A pair that holds no
    // pupillary distance (a profile's fixed scale, or no pupils) keeps its saved scale.
    //
    // The context draws on a copy of the state's image, so measuring a restored state
    // with render on leaves the state as it was, as load_state() does.
    public static FaceMeasureContext Restore(FaceAnalysisState state) => Restore(state, copyImage: true);

    // copyImage: false lets the context use the state's own pixels, for callers that
    // never draw (FrameSeries.FromStates measures with render: false).
    internal static FaceMeasureContext Restore(FaceAnalysisState state, bool copyImage)
    {
        FaceImage image = copyImage && state.PostCropImage.IsValid
            ? new FaceImage((Rgba32[])state.PostCropImage.Pixels.Clone(),
                state.PostCropImage.Width, state.PostCropImage.Height)
            : state.PostCropImage;
        float savedPupilDistMm = state.PupillaryDistance * state.Pix2mm;
        bool havePupils = state.Landmarks != null && state.Landmarks.Length > DynafaceConstants.LmLeftPupil;
        if (!state.IsLateral && havePupils && savedPupilDistMm > 0f && !float.IsInfinity(savedPupilDistMm))
            return new FaceMeasureContext(
                image, state.Landmarks, headPose: state.HeadPose,
                pose: state.Pose, flipped: state.Flipped, pupilDistMm: savedPupilDistMm);
        return FaceMeasureContext.WithFixedScale(
            image, state.Landmarks, state.Pix2mm, state.IsLateral, state.LateralLandmarks,
            state.HeadPose, state.Pose, state.Flipped);
    }
}
