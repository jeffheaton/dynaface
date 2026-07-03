using System.Collections.Generic;

namespace DynafaceTests;

// Exercises the item-level enable/disable system (mirrors dynaface-lib's
// measures_base.py MeasureBase/MeasureItem contract) using MeasureDentalArea as a
// representative multi-item measure.
public class FaceMeasureBaseTests
{
    [Fact]
    public void SetEnabled_False_CascadesToAllItems()
    {
        var measure = new MeasureDentalArea();
        measure.SetEnabled(false);

        Assert.False(measure.Enabled);
        foreach (var item in measure.Items)
            Assert.False(item.Enabled);
    }

    [Fact]
    public void SetItemEnabled_TogglesOnlyNamedItem()
    {
        var measure = new MeasureDentalArea();
        measure.SetItemEnabled("dental_ratio", false);

        Assert.False(measure.IsEnabled("dental_ratio"));
        Assert.True(measure.IsEnabled("dental_area"));
        Assert.True(measure.IsEnabled("dental_left"));
    }

    [Fact]
    public void IsEnabled_UnknownItemName_DefaultsToTrue()
    {
        var measure = new MeasureDentalArea();
        Assert.True(measure.IsEnabled("not_a_real_item"));
    }

    [Fact]
    public void UpdateForType_SetsEveryItemToTheMeasuresOwnFlag()
    {
        // Matches dynaface-lib's update_for_type exactly: every item gets the
        // MEASURE's own IsLateral/IsFrontal flag, not each item's individual one.
        var measure = new MeasurePosition(); // IsFrontal=true, IsLateral=true
        measure.UpdateForType(lateral: true);
        foreach (var item in measure.Items) Assert.True(item.Enabled);

        measure.UpdateForType(lateral: false);
        foreach (var item in measure.Items) Assert.True(item.Enabled);
    }

    [Fact]
    public void DisablingAnItem_SuppressesItsResultLine_ButStillReturnsAllValues()
    {
        // Matches dynaface-lib's filter_measurements: the returned dict always
        // includes every declared item regardless of its enabled state — only
        // rendering (pixels/sidebar text) is gated by IsEnabled().
        var measure = new MeasureFAI();
        measure.SetItemEnabled("fai", false);
        var ctx = TestHelpers.BuildContext();

        Dictionary<string, double> result = measure.Calc(ctx, render: true);

        Assert.True(result.ContainsKey("fai"));
        Assert.Empty(ctx.TextLines); // suppressed since the "fai" item is disabled
    }

    [Fact]
    public void FaceMeasureBase_DefaultsEnabledTrue()
    {
        // dynaface-lib's MeasureBase.__init__ defaults self.enabled=True.
        var measure = new MeasureFAI();
        Assert.True(measure.Enabled);
    }
}
