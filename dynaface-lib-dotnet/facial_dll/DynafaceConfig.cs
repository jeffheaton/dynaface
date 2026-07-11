// Mutable global configuration, mirroring dynaface-lib's config.py module-level flag.
public static class DynafaceConfig
{
    // When false, pose classification never reports lateral, regardless of headpose.
    public static bool AutoLateral = true;

    // Pupillary distance (mm) used to derive every px→mm conversion. Mirrors
    // dynaface-lib's mutable AnalyzeFace.pd class attribute — the desktop app sets
    // it from user preferences so measurements can be calibrated to the actual
    // patient instead of the 63mm population-average default.
    public static float PupilDistMm = DynafaceConstants.StdPupilDistMm;
}
