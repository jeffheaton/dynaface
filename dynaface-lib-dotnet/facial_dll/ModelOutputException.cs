using System;

// Thrown when a model gives output that no working model gives: none at all (a ready
// backend's IDynafaceInference call returned null), output of the wrong size, values
// that aren't numbers (NaN or infinity), the same value everywhere, or landmarks all
// at one point. The backend that runs the models failed, or isn't computing them
// correctly, as GPU compute on some virtual and software graphics adapters doesn't:
// without this check, the face detector's broken output looked like a photo without
// a face, and the landmark model's like a crop error. Mirrors dynaface-lib's
// ModelOutputError.
public sealed class ModelOutputException : Exception
{
    public ModelOutputException(string model, string message) : base(message)
    {
        Model = model;
    }

    // Which model gave it: ModelOutputCheck's FaceDetector, LandmarkModel or BackgroundModel.
    public string Model { get; }
}

// The checks the detectors run on each model's output, and their errors.
internal static class ModelOutputCheck
{
    internal const string FaceDetector = "face detector (BlazeFace)";
    internal const string LandmarkModel = "landmark model (SPIGA)";
    internal const string BackgroundModel = "background model (U2-Net)";

    // A ready backend that gave nothing: it failed to run the model.
    internal static ModelOutputException NoOutput(string model) => new ModelOutputException(model,
        $"The {model}'s backend gave no output: it failed to run the model, or the model isn't loaded.");

    internal static ModelOutputException WrongSize(string model) => new ModelOutputException(model,
        $"The {model} gave output of the wrong size: it isn't the model Dynaface expects, "
        + "or its backend didn't run it correctly.");

    // Output a working model never gives, computed wrongly by its backend.
    internal static ModelOutputException Broken(string model, string problem) => new ModelOutputException(model,
        $"The {model} gave {problem}, which a working model never does, so the backend it runs on "
        + "isn't computing it correctly. Run the models on another backend, such as the CPU.");

    // Throws unless the output has at least count values, all of them numbers.
    internal static void Require(string model, float[] values, int count)
    {
        if (values == null) throw NoOutput(model);
        if (values.Length < count) throw WrongSize(model);
        foreach (float value in values)
            if (float.IsNaN(value) || float.IsInfinity(value))
                throw Broken(model, "values that aren't numbers");
    }

    // True if the first count values are all the same. Even for a blank image, a working
    // model's scores and saliency differ from place to place.
    internal static bool AllSame(float[] values, int count)
    {
        for (int i = 1; i < count; i++)
            if (values[i] != values[0]) return false;
        return true;
    }
}
