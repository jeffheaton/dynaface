using System.Collections.Generic;

// The canonical measure registry — mirrors dynaface-lib's measures.py.
// Consumers get the full measurement set from here (and run it via
// FaceMeasureContext.Analyze()) instead of hand-building their own lists.
public static class DynafaceMeasures
{
    // Mirrors dynaface-lib's measures.all_measures(): same measures, same order
    // (order matters — it determines sidebar text and on-image label stacking).
    public static FaceMeasureBase[] AllMeasures() => new FaceMeasureBase[]
    {
        new MeasureFAI(),
        new MeasureOCE(),
        new MeasureBrows(),
        new MeasureDentalArea(),
        new MeasureEyeArea(),
        new MeasureIntercanthalDistance(),
        new MeasureMouthLength(),
        new MeasureNoseFrontal(),
        new MeasureOuterEyeCorners(),
        new MeasureLateral(),
        new MeasurePosition(),
        new MeasurePose(),
        new MeasureSkinTone(),
        new MeasureLandmarks(),
        // Last, as in dynaface-lib, so the outputs before it keep their order (CSV
        // columns included) and its drawing lands over the landmark dots.
        new MeasureCheekMeasures(),
    };

    // Mirrors dynaface-lib's measures.get_all_items(): the outputs to show or export
    // (e.g. CSV columns), the output names (MeasureItemInfo.OutputNames()) of the
    // enabled items of the enabled measures. A group's outputs are listed together in
    // its place, never its label, and a display-only item lists none.
    public static List<string> GetAllItems(IEnumerable<FaceMeasureBase> measures)
    {
        var result = new List<string>();
        foreach (var measure in measures)
        {
            if (!measure.Enabled) continue;
            foreach (var item in measure.Items)
                if (item.Enabled)
                    result.AddRange(item.OutputNames());
        }
        return result;
    }
}
