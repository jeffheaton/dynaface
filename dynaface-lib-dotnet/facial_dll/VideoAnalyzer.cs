using System;
using System.Collections.Generic;

// Analyzes a video's frames one at a time, as the Dynaface apps import a video. Mirrors
// dynaface-lib's dynaface.video.VideoAnalyzer. Give it every frame in order, upright: a
// frame with a face is analyzed and kept as a FaceAnalysisState; a frame without one is
// skipped but still counted, so the kept frames keep their numbers in the video. With a
// smoothing above 1, each frame's landmarks are averaged with those of the frames before
// it (LandmarkSmoother), and the scale is measured again on the averaged landmarks. Only
// the first maxFrames frames are analyzed.
//
// .NET-only: keepImages: false keeps each frame's landmarks, head pose and scale but not
// its crop, for apps that can't hold thousands of 1024x1024 images. Those frames are
// analyzed uncropped and only their landmarks cropped (StyleGanCropper.CropLandmarks),
// which gives the same landmarks without building the image, and FrameSeries.FromStates
// measures such states on their landmarks alone. Thumbnail is the first kept frame's crop
// either way.
public sealed class VideoAnalyzer
{
    // The most frames the Dynaface apps import from one video.
    public const int DefaultMaxFrames = 5000;

    readonly List<FaceAnalysisState> _states = new List<FaceAnalysisState>();
    readonly List<int> _frames = new List<int>();
    readonly LandmarkSmoother _smoother;
    readonly float _tiltThreshold;
    readonly Func<FaceImage, int, bool, bool, FacePipelineResult?> _run;
    int _seen;
    bool? _lateral;

    // smoothing: how many frames' landmarks to average; 1, the default, turns it off.
    // pose: FacePose.Detect to classify each frame as frontal or lateral, or the pose to
    // use for all of them. The others are as for FacePipeline.Run.
    public VideoAnalyzer(
        int smoothing = 1, int maxFrames = DefaultMaxFrames, FacePose pose = FacePose.Detect,
        bool keepImages = true, float tiltThreshold = DynafaceConstants.DefaultTiltThreshold)
        : this(smoothing, maxFrames, keepImages, tiltThreshold,
            (frame, rotationAngle, flipHorizontal, crop) =>
                FacePipeline.Run(frame, rotationAngle, flipHorizontal, pose, tiltThreshold, crop))
    {
    }

    // For tests: runs a stand-in for FacePipeline.Run(frame, rotationAngle, flipHorizontal,
    // pose, tiltThreshold, crop).
    internal VideoAnalyzer(
        int smoothing, int maxFrames, bool keepImages, float tiltThreshold,
        Func<FaceImage, int, bool, bool, FacePipelineResult?> run)
    {
        _smoother = new LandmarkSmoother(smoothing);
        MaxFrames = maxFrames;
        KeepImages = keepImages;
        _tiltThreshold = tiltThreshold;
        _run = run;
    }

    // The kept frames' states, in order.
    public IReadOnlyList<FaceAnalysisState> States => _states;

    // Each kept frame's number in the video, counting from 0.
    public IReadOnlyList<int> Frames => _frames;

    public int MaxFrames { get; }
    public bool KeepImages { get; }

    // True once MaxFrames frames have been given; Add analyzes no more.
    public bool IsFull => _seen >= MaxFrames;

    // The first kept frame's crop, or null before one is kept.
    public FaceImage? Thumbnail { get; private set; }

    // Analyzes the next frame. True if it had a face and was kept; false if it had none,
    // or the analyzer was already full. rotationAngle and flipHorizontal are as for
    // FacePipeline.Run.
    public bool Add(FaceImage frame, int rotationAngle = 0, bool flipHorizontal = false)
    {
        if (IsFull) return false;
        int number = _seen++;

        // Tilt straightening needs the image, so only crop later without it.
        bool cropLater = !KeepImages && _tiltThreshold < 0f;
        FacePipelineResult? found = _run(frame, rotationAngle, flipHorizontal, !cropLater);
        if (found == null) return false;
        FacePipelineResult result = found.Value;

        Vec2[] landmarks = result.Wflw98;
        FaceImage crop = result.AlignedCrop;
        if (cropLater && !result.IsLateral) // a profile is always cropped
        {
            // AlignedCrop is the uncropped (working) image here.
            float yaw = result.HeadPose != null && result.HeadPose.Length > 0 ? result.HeadPose[0] : 0f;
            if (Thumbnail == null)
                crop = StyleGanCropper.Crop(result.AlignedCrop, landmarks, yaw, _tiltThreshold).crop;
            landmarks = StyleGanCropper.CropLandmarks(
                result.AlignedCrop.Width, result.AlignedCrop.Height, landmarks, yaw, _tiltThreshold);
        }

        if (result.IsLateral != _lateral)
        {
            // A profile's landmarks don't average with a frontal face's.
            _smoother.Reset();
            _lateral = result.IsLateral;
        }
        if (_smoother.Size > 1) landmarks = _smoother.Add(landmarks);

        float pupillaryDistance = result.PupillaryDistance;
        float pix2mm = result.Pix2mm;
        if (!result.IsLateral && (_smoother.Size > 1 || cropLater))
            (pupillaryDistance, pix2mm) = FaceMeasureContext.CalcPd(landmarks, result.PupilDistMm);

        var state = new FaceAnalysisState
        {
            PostCropImage = KeepImages ? crop : default,
            HeadPose = result.HeadPose,
            Landmarks = landmarks,
            PupillaryDistance = pupillaryDistance,
            Pix2mm = pix2mm,
            FaceRotationRad = result.FaceRotationRad,
        };
        FaceAnalysisPersistence.CaptureView(result, state);
        _states.Add(state);
        _frames.Add(number);
        if (Thumbnail == null) Thumbnail = crop;
        return true;
    }

    // Counts a frame that couldn't be read, so the frames after it keep their numbers.
    public void Skip()
    {
        if (!IsFull) _seen++;
    }

    // Measures the kept frames: one row per frame, numbered as in the video.
    public FrameSeries Series(IEnumerable<FaceMeasureBase> measures = null, float fps = 0f) =>
        FrameSeries.FromStates(_states, measures, fps: fps, frameNumbers: _frames);
}
