using System.Collections.Generic;

// Orchestrates the lateral numeric pipeline, replacing dynaface-lib's
// lateral.py analyze_lateral() minus its matplotlib chart rendering — the chart
// itself is never consumed by any measurement, only the 6 lateral landmarks +
// sagittal arrays are. See LateralChartRenderer for the pixel-buffer equivalent
// of that chart, composited onto the crop by FacePipeline.RunLateral (mirroring
// load_image's own _overlay_lateral_analysis call).
public static class LateralAnalyzer
{
    public struct Result
    {
        public Vec2[] LateralLandmarks; // 6 points: see LateralLandmarkFinder's index constants
        public float[] SagittalX;       // shift-corrected, full length
        public float[] SagittalY;
    }

    // croppedLateralImage: the lateral-cropped FaceImage (from LateralCropper.Crop).
    // frontalLandmarksInCropSpace: the 98 WFLW landmarks already remapped into that
    // same crop's pixel space (top-left semantic, matching the rest of the pipeline).
    public static Result? Analyze(
        IDynafaceInference inference, FaceImage croppedLateralImage, Vec2[] frontalLandmarksInCropSpace)
    {
        byte[] binary = SagittalProfile.ProcessImage(inference, croppedLateralImage);
        var (sagittalXInt, sagittalYInt) = SagittalProfile.ExtractSagittalProfile(
            binary, croppedLateralImage.Width, croppedLateralImage.Height);
        if (sagittalXInt.Length == 0) return null;

        var (sagittalXShifted, shiftX) = SagittalProfile.ShiftSagittalProfile(sagittalXInt);

        var sagittalXDouble = new double[sagittalXShifted.Length];
        for (int i = 0; i < sagittalXShifted.Length; i++) sagittalXDouble[i] = sagittalXShifted[i];

        var (maxIndices, _) = PeakFinder.FindPeaks(sagittalXDouble);
        int[] minIndices = PeakFinder.FindMinimaIndices(sagittalXDouble);

        var extremaSet = new HashSet<int>(maxIndices);
        foreach (int m in minIndices) extremaSet.Add(m);

        // Constants match analyze_lateral's own call site exactly (these differ from
        // _find_monotonic_corners' own function-default values — the call-site
        // values are what actually executes).
        int[] cornerIdxs = MonotonicCornerFinder.FindMonotonicCorners(
            sagittalXDouble,
            scales: new[] { 9, 13, 17 },
            polyOrder: 2,
            dxTol: 0.035, minRun: 10, distancePx: 32,
            anglePercentile: 93.0, angleMinDeg: 16.0,
            kappaPercentile: 92.0, mixWeightAngle: 0.75,
            excludeExtrema: extremaSet);

        var sagittalYFloat = new float[sagittalYInt.Length];
        for (int i = 0; i < sagittalYInt.Length; i++) sagittalYFloat[i] = sagittalYInt[i];

        Vec2[] lateralLandmarks = LateralLandmarkFinder.FindLateralLandmarks(
            sagittalXShifted, sagittalYFloat, maxIndices, minIndices, cornerIdxs,
            (int)shiftX, frontalLandmarksInCropSpace);

        var sagittalXFull = new float[sagittalXShifted.Length];
        for (int i = 0; i < sagittalXShifted.Length; i++) sagittalXFull[i] = sagittalXShifted[i] + shiftX;

        return new Result
        {
            LateralLandmarks = lateralLandmarks,
            SagittalX = sagittalXFull,
            SagittalY = sagittalYFloat,
        };
    }
}
