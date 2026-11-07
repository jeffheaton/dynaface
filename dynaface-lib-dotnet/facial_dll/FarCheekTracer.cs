using System;

// What a three-quarter (quarter) face's far-cheek outline is traced from, kept with the
// pipeline's result so the outline can be traced again on other landmarks (a video's
// smoothed ones) without a second background-model pass: the background model's closed
// foreground mask and the alpha it was thresholded from (SagittalProfile.ForegroundMask,
// the mask every profile is traced from), and the standard image's gray image
// (CheekOutline.GrayImage, the library's own conversion, so both ports read the same
// intensity edge). About 3 MB for the 1024x1024 standard image, all [row, column]
// top-left. Mirrors the mask, alpha and gray image dynaface-lib's
// AnalyzeFace._trace_far_cheek() keeps for retrace_far_cheek(). A saved state carries
// the outline, never this.
internal sealed class FarCheekTracer
{
    public bool[,] Foreground { get; }
    public byte[,] Alpha { get; }
    public byte[,] Gray { get; }

    public FarCheekTracer(bool[,] foreground, byte[,] alpha, byte[,] gray)
    {
        Foreground = foreground ?? throw new ArgumentNullException(nameof(foreground));
        Alpha = alpha ?? throw new ArgumentNullException(nameof(alpha));
        Gray = gray;
    }

    // Runs the background model on the standard image (the quarter crop, before anything
    // draws on it), through foregroundMask (SagittalProfile.ForegroundMask with the
    // pipeline's inference; tests pass a stand-in), and takes its gray image.
    public static FarCheekTracer FromImage(
        FaceImage crop, Func<FaceImage, (bool[,] Foreground, byte[,] Alpha)> foregroundMask)
    {
        if (foregroundMask == null) throw new ArgumentNullException(nameof(foregroundMask));
        var (foreground, alpha) = foregroundMask(crop);
        return new FarCheekTracer(foreground, alpha, CheekOutline.GrayImage(crop));
    }

    // The far cheek's outline on these landmarks (top-left pixels of the standard image):
    // CheekOutline.TraceFarCheek on the kept mask, alpha and gray image, so its rows, seed
    // and nose rows belong to the landmarks it will be measured with.
    public (float[] X, int[] Y, byte[] Src) Trace(Vec2[] landmarks, bool farLeft) =>
        CheekOutline.TraceFarCheek(Foreground, Alpha, landmarks, farLeft, gray: Gray);
}
