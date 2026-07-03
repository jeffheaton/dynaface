// Mutable global configuration, mirroring dynaface-lib's config.py module-level flag.
public static class DynafaceConfig
{
    // When false, pose classification never reports lateral, regardless of headpose.
    public static bool AutoLateral = true;
}
