// Pure data constants shared across the pipeline and measures.
// Mirrors dynaface-lib's const.py.
public static class DynafaceConstants
{
    public const int StyleganWidth = 1024;

    public const float StyleganLeftPupilX = 640f;
    public const float StyleganLeftPupilY = 480f;
    public const float StyleganRightPupilX = 380f;
    public const float StyleganRightPupilY = 480f;
    public const float StyleganPupilDist = StyleganLeftPupilX - StyleganRightPupilX; // 260

    public const float StdPupilDistMm = 63f;

    // Tilt correction is off by default; crop_stylegan only rotates when a caller
    // explicitly passes a threshold >= 0.
    public const float DefaultTiltThreshold = -1f;

    // WFLW landmark indices for the two pupil centres.
    public const int LmRightPupil = 96;
    public const int LmLeftPupil = 97;

    public static readonly Rgba32 FillColorWhite = new Rgba32(255, 255, 255, 255);
    public static readonly Rgba32 FillColorBlack = new Rgba32(0, 0, 0, 255);

    // Lateral view has no reliable pupil-distance signal (face is turned), so
    // dynaface-lib hardcodes a fixed px->mm ratio instead of deriving one.
    public const float LateralPix2mm = 0.24f;

    public const float LateralPadTopRatio = 0.12f;
    public const float LateralPadBottomRatio = 0.10f;
    // Fraction of the 1024 crop width kept clear left of the profile's leading
    // edge (nose tip); must clear the analysis text column drawn at the left.
    public const float LateralLeftMarginRatio = 0.20f;
}
