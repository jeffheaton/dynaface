// Direct port of dynaface-lib's AnalyzeFace._is_lateral / _force_lateral.
// Operates on landmarks in top-left semantic pixel space (y=0 = visual top),
// matching the coordinate contract SpigaLandmarkDetector returns.
public static class PoseClassifier
{
    // yaw threshold (degrees) beyond which a face is considered lateral, provided
    // the nose-asymmetry ratio also agrees. Matches dynaface-lib's hardcoded 20.
    const float YawLateralThresholdDeg = 20f;

    // Auto-detects lateral vs frontal from headpose yaw + landmark asymmetry.
    // Returns (isLateral, facingLeft). Always (false, false) if DynafaceConfig.AutoLateral
    // is false or there are no landmarks.
    public static (bool isLateral, bool facingLeft) IsLateral(float[] headpose, Vec2[] landmarks, int imageWidth)
    {
        if (!DynafaceConfig.AutoLateral) return (false, false);
        if (landmarks == null || landmarks.Length == 0) return (false, false);

        float yaw = headpose[0];
        float noseDistanceRatio = NoseDistanceRatio(landmarks, imageWidth);

        bool isLateral = MathHelpers.Abs(yaw) > YawLateralThresholdDeg && noseDistanceRatio < 0f;
        bool facingLeft = yaw < 0f;
        return (isLateral, facingLeft);
    }

    // Forces lateral classification (used when the caller explicitly asks for a
    // lateral pose rather than auto-detecting), still resolving left/right facing
    // from yaw (falling back to nose-asymmetry if yaw is somehow not finite).
    public static (bool isLateral, bool facingLeft) ForceLateral(float[] headpose, Vec2[] landmarks, int imageWidth)
    {
        if (landmarks == null || landmarks.Length == 0) return (true, false);

        float yaw = headpose[0];
        if (!float.IsNaN(yaw)) return (true, yaw < 0f);

        float nd = NoseDistance(landmarks);
        return (true, nd < 0f);
    }

    static float NoseDistanceRatio(Vec2[] landmarks, int imageWidth) => NoseDistance(landmarks) / imageWidth;

    // landmarks[54] = nose tip, landmarks[6]/[26] = left/right brow-corner-ish contour
    // points; picks whichever side's distance to the nose tip is smaller in magnitude.
    static float NoseDistance(Vec2[] landmarks)
    {
        float nd1 = landmarks[54].X - landmarks[6].X;
        float nd2 = landmarks[26].X - landmarks[54].X;
        return MathHelpers.Abs(nd1) < MathHelpers.Abs(nd2) ? nd1 : nd2;
    }
}
