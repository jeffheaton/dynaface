using System;

// Mutable global configuration, mirroring dynaface-lib's config.py module-level flag.
public static class DynafaceConfig
{
    // When false, pose classification never reports lateral, regardless of headpose.
    public static bool AutoLateral = true;

    // Pupillary distance (mm) used to derive every px→mm conversion. Mirrors
    // dynaface-lib's mutable AnalyzeFace.pd class attribute — the desktop app sets
    // it from user preferences so measurements can be calibrated to the actual
    // patient instead of the 63mm population-average default. It must be from
    // DynafaceConstants.MinPupilDistMm to MaxPupilDistMm (40–85 mm): setting another
    // value throws ArgumentOutOfRangeException and leaves it as it was.
    public static float PupilDistMm
    {
        get => _pupilDistMm;
        set => _pupilDistMm = CheckPupilDistMm(value, nameof(PupilDistMm));
    }

    static float _pupilDistMm = DynafaceConstants.StdPupilDistMm;

    // Returns pupilDistMm if it's a pupillary distance Dynaface accepts, from
    // DynafaceConstants.MinPupilDistMm to MaxPupilDistMm, and throws
    // ArgumentOutOfRangeException otherwise (NaN included). Mirrors dynaface-lib's
    // facial.check_pupil_distance(); FaceMeasureContext.CalcPd checks every distance a
    // frontal scale is measured from with it.
    public static float CheckPupilDistMm(float pupilDistMm, string paramName = "pupilDistMm")
    {
        if (!(pupilDistMm >= DynafaceConstants.MinPupilDistMm && pupilDistMm <= DynafaceConstants.MaxPupilDistMm))
            throw new ArgumentOutOfRangeException(paramName, pupilDistMm,
                $"The pupillary distance must be from {DynafaceConstants.MinPupilDistMm} to "
                + $"{DynafaceConstants.MaxPupilDistMm} mm. Is it in millimeters?");
        return pupilDistMm;
    }

    // The size of the text the measures draw, mirroring dynaface-lib's
    // AnalyzeFace.text_size: 0.75 by default, as there, and twice that draws it twice as
    // large. Setting it sets FaceRenderer's two glyph scales, since its bitmap font only
    // scales by whole pixels: the measures' labels (TEXT_SIZE_MEASURE) to TextSize × 4,
    // 3 by default, and the landmark numbers (NumberScale) one less, 2 by default.
    // Setting a scale directly afterward overrides it. It must be above 0: setting
    // another value throws ArgumentOutOfRangeException.
    public static float TextSize
    {
        get => _textSize;
        set
        {
            if (!(value > 0f) || float.IsInfinity(value))
                throw new ArgumentOutOfRangeException(nameof(TextSize), value, "The text size must be above 0.");
            _textSize = value;
            int labels = Math.Max(1, (int)Math.Round(value * 4f, MidpointRounding.AwayFromZero));
            FaceRenderer.TEXT_SIZE_MEASURE = labels;
            FaceRenderer.NumberScale = Math.Max(1, labels - 1);
        }
    }

    static float _textSize = 0.75f;
}
