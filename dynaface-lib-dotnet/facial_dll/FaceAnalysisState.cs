// Persistence DTO mirroring dynaface-lib's AnalyzeFace.dump_state/load_state fields
// exactly: post-crop image, raw headpose, landmarks, pupillary distance, pix2mm, face
// rotation, and the view (with a quarter face's far side and far-cheek outline). A
// typed class (the .NET-idiomatic replacement for Python's plain heterogeneous list)
// rather than a serialization format — this is an in-memory round-trip, same as
// dynaface-lib's own test_persist.py.
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

    // A three-quarter (quarter) face's view, as dynaface-lib's dump_state saves it: whether
    // the far (measured) cheek is the subject's left, and that cheek's traced outline
    // (FacePipelineResult.CheekOutlineX/Y/Src: x, rows and each row's source), so the
    // cheek landmarks of a restored quarter face are found as the original's were. The
    // quarter view follows the pose: a state is a quarter face when Pose is
    // FacePose.Quarter and it has landmarks, as FaceMeasureContext.IsQuarter reads it.
    // Null for any other face. A quarter state whose FarLeft is null (one saved before
    // the quarter view was kept, such as a forced quarter state of an earlier 2.1 build)
    // takes the far side from its landmarks, as load_state does for a state without
    // far_left; one without an outline measures on the landmark contour. The mask the
    // outline was traced from is never saved.
    public bool? FarLeft;
    public float[] CheekOutlineX;
    public int[] CheekOutlineY;
    public byte[] CheekOutlineSrc;

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

    // Copies a result's view (pose, flip, a profile's lateral data and a quarter face's
    // far side and outline) into a state.
    internal static void CaptureView(FacePipelineResult result, FaceAnalysisState state)
    {
        state.Pose = result.Pose;
        state.Flipped = result.Flipped;
        state.IsLateral = result.IsLateral;
        var lateral = result.IsLateral ? result.LateralAnalysis : null;
        state.LateralLandmarks = (Vec2[])lateral?.LateralLandmarks?.Clone();
        state.SagittalX = (float[])lateral?.SagittalX?.Clone();
        state.SagittalY = (float[])lateral?.SagittalY?.Clone();
        bool quarter = result.IsQuarter;
        state.FarLeft = quarter ? result.FarLeft : (bool?)null;
        state.CheekOutlineX = quarter ? (float[])result.CheekOutlineX?.Clone() : null;
        state.CheekOutlineY = quarter ? (int[])result.CheekOutlineY?.Clone() : null;
        state.CheekOutlineSrc = quarter ? (byte[])result.CheekOutlineSrc?.Clone() : null;
    }

    // Rebuilds a FaceMeasureContext from saved state, ready for measures to run
    // again and reproduce the same numbers. Like dynaface-lib's load_state(), it
    // keeps the pupillary distance the analysis assumed, not today's
    // DynafaceConfig.PupilDistMm: Pix2mm has always been computed as it over the
    // PupillaryDistance saved beside it, so their product gives it back. With it, the
    // context measures its scale on the saved landmarks, which also corrects states
    // saved when the scale was measured before the crop. A pair that holds no
    // pupillary distance (a profile's fixed scale, or no pupils) keeps its saved scale.
    // The distance isn't checked against DynafaceConfig.CheckPupilDistMm's range: a
    // state restores the distance it was analyzed with.
    //
    // The context draws on a copy of the state's image, so measuring a restored state
    // with render on leaves the state as it was, as load_state() does.
    //
    // A quarter state (Pose Quarter, with landmarks) restores its far side and outline
    // through SetQuarterView, so the context finds the cheek landmarks as the original
    // did; its FarLeft, when null, comes from its landmarks (Cheek.FarLeftOf, with the
    // saved yaw for the fallback), and a missing outline leaves the landmark contour to
    // stand in. Mirrors dynaface-lib's _load_view().
    public static FaceMeasureContext Restore(FaceAnalysisState state) => Restore(state, copyImage: true);

    // copyImage: false lets the context use the state's own pixels, for callers that
    // never draw (FrameSeries.FromStates measures with render: false).
    internal static FaceMeasureContext Restore(FaceAnalysisState state, bool copyImage)
    {
        FaceImage image = copyImage && state.PostCropImage.IsValid
            ? new FaceImage((Rgba32[])state.PostCropImage.Pixels.Clone(),
                state.PostCropImage.Width, state.PostCropImage.Height)
            : state.PostCropImage;
        // Rounded, so the float error in the product doesn't change the distance: 63 mm
        // comes back as 63. dynaface-lib rounds the same way.
        float savedPupilDistMm = (float)System.Math.Round((double)state.PupillaryDistance * state.Pix2mm, 4);
        bool havePupils = state.Landmarks != null && state.Landmarks.Length > DynafaceConstants.LmLeftPupil;
        FaceMeasureContext ctx =
            !state.IsLateral && havePupils && savedPupilDistMm > 0f && !float.IsInfinity(savedPupilDistMm)
                ? new FaceMeasureContext(
                    image, state.Landmarks, headPose: state.HeadPose,
                    pose: state.Pose, flipped: state.Flipped, pupilDistMm: savedPupilDistMm)
                : FaceMeasureContext.WithFixedScale(
                    image, state.Landmarks, state.Pix2mm, state.IsLateral, state.LateralLandmarks,
                    state.HeadPose, state.Pose, state.Flipped);
        if (ctx.IsQuarter)
        {
            // A state saved without its far side takes it from its landmarks, as
            // _load_view() does; a saved head pose's yaw (0 without one, as Python's
            // saved head pose pads it) is the fallback when they give no frame.
            float yaw = state.HeadPose != null && state.HeadPose.Length > 0 ? state.HeadPose[0] : 0f;
            bool farLeft = state.FarLeft ?? (havePupils && Cheek.FarLeftOf(state.Landmarks, yaw));
            // Copies, so the context and the state don't share the outline.
            ctx.SetQuarterView(farLeft, (float[])state.CheekOutlineX?.Clone(),
                (int[])state.CheekOutlineY?.Clone(), (byte[])state.CheekOutlineSrc?.Clone());
        }
        return ctx;
    }
}
