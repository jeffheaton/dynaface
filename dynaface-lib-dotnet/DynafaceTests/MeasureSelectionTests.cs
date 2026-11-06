using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace DynafaceTests;

// MeasureSelection keeps a user's choice of measures and items when the face switches
// between frontal and lateral, which UpdateForType can't, and saves and loads it by
// name. Mirrors dynaface-lib's tests/test_measure_selection.py.
public class MeasureSelectionTests
{
    // The outputs of the measures that Ticked() leaves on, for each view.
    static readonly string[] Frontal = { "fai", "eye.left", "eye.right", "eye.ratio", "pitch", "roll", "yaw" };
    static readonly string[] Lateral = { "nfa", "at", "nt", "at_nt", "pitch", "roll", "yaw" };
    static readonly string[] AllEyes = { "eye.left", "eye.right", "eye.diff", "eye.ratio" };

    static T Find<T>(IEnumerable<FaceMeasureBase> measures) where T : FaceMeasureBase =>
        measures.OfType<T>().Single();

    // The measures as a user might tick them: FAI, the eye area without eye.diff, the
    // head pose, and the lateral measures without nla.
    static FaceMeasureBase[] Ticked()
    {
        var measures = DynafaceMeasures.AllMeasures();
        var keep = new HashSet<string> { "FAI", "EYE AREA", "POSE", "LATERAL" };
        foreach (var measure in measures) measure.SetEnabled(keep.Contains(measure.Label));
        Find<MeasureEyeArea>(measures).SetItemEnabled("eye.diff", false);
        Find<MeasureLateral>(measures).SetItemEnabled("nla", false);
        return measures;
    }

    [Fact]
    public void SwitchingViewsKeepsTheChoices()
    {
        var measures = Ticked();
        var selection = MeasureSelection.Of(measures);
        foreach (var (lateral, expected) in new[] { (false, Frontal), (true, Lateral), (false, Frontal) })
        {
            selection.Apply(measures, lateral);
            Assert.Equal(expected, DynafaceMeasures.GetAllItems(measures));
        }
    }

    // What MeasureSelection is for: UpdateForType turns eye.diff back on.
    [Fact]
    public void UpdateForTypeLosesThem()
    {
        var measures = Ticked();
        foreach (bool lateral in new[] { true, false })
            foreach (var measure in measures) measure.UpdateForType(lateral);
        Assert.Contains("eye.diff", DynafaceMeasures.GetAllItems(measures));
    }

    [Fact]
    public void AMeasureThatDoesNotSuitTheFaceIsOff()
    {
        var measures = Ticked();
        MeasureSelection.Of(measures).Apply(measures, lateral: true);
        var eyes = Find<MeasureEyeArea>(measures);
        Assert.False(eyes.Enabled);
        Assert.DoesNotContain(eyes.Items, item => item.Enabled);
    }

    [Fact]
    public void UnmentionedMeasuresAreOffAndUnmentionedItemsOn()
    {
        var measures = DynafaceMeasures.AllMeasures();
        new MeasureSelection().Apply(measures, lateral: false);
        Assert.Empty(DynafaceMeasures.GetAllItems(measures));

        var selection = new MeasureSelection();
        selection.Choose(Find<MeasureEyeArea>(measures));
        selection.Apply(measures, lateral: false);
        Assert.Equal(AllEyes, DynafaceMeasures.GetAllItems(measures));
    }

    [Fact]
    public void ChooseAndUnchoose()
    {
        var eyes = new MeasureEyeArea();
        var selection = new MeasureSelection();
        Assert.False(selection.IsChosen(eyes));
        Assert.True(selection.IsItemChosen(eyes, "eye.diff"));

        selection.Choose(eyes);
        selection.ChooseItem(eyes, "eye.diff", false);
        Assert.True(selection.IsChosen(eyes));
        Assert.False(selection.IsItemChosen(eyes, "eye.diff"));

        selection.Choose(eyes, false);
        Assert.False(selection.IsChosen(eyes));
        Assert.False(selection.IsItemChosen(eyes, "eye.diff"));
    }

    [Fact]
    public void ASwitchedOffMeasureShowsAllItsItemsWhenChosenAgain()
    {
        var measures = DynafaceMeasures.AllMeasures();
        var eyes = Find<MeasureEyeArea>(measures);
        eyes.SetEnabled(false);   // which switches off every item too
        var selection = MeasureSelection.Of(measures);
        selection.Choose(eyes);
        selection.Apply(measures, lateral: false);
        Assert.Equal(AllEyes, eyes.Items.Where(item => item.Enabled).Select(item => item.Name));
    }

    [Fact]
    public void SavedByName()
    {
        var entries = MeasureSelection.Of(Ticked()).ToEntries();
        Assert.Equal(new[] { "FAI", "ORAL CE" }, entries.Take(2).Select(entry => entry.Label));

        var eyes = entries.Single(entry => entry.Label == "EYE AREA");
        Assert.True(eyes.Chosen);
        Assert.Equal(AllEyes, eyes.ItemNames);
        Assert.Equal(new[] { true, true, false, true }, eyes.ItemChosen);

        var brows = entries.Single(entry => entry.Label == "BROW");
        Assert.False(brows.Chosen);
        Assert.Empty(brows.ItemNames);
        Assert.Empty(brows.ItemChosen);
    }

    [Fact]
    public void SaveAndLoad()
    {
        var selection = MeasureSelection.Of(Ticked());
        var options = new JsonSerializerOptions { IncludeFields = true };
        string json = JsonSerializer.Serialize(selection.ToEntries(), options);
        var loaded = MeasureSelection.FromEntries(
            JsonSerializer.Deserialize<MeasureSelection.Entry[]>(json, options));
        Assert.Equal(json, JsonSerializer.Serialize(loaded.ToEntries(), options));

        var measures = DynafaceMeasures.AllMeasures();   // every measure and item on, as made
        foreach (var (lateral, expected) in new[] { (false, Frontal), (true, Lateral) })
        {
            loaded.Apply(measures, lateral);
            Assert.Equal(expected, DynafaceMeasures.GetAllItems(measures));
        }
    }

    [Fact]
    public void ItemsMentionedWithoutTheirMeasure()
    {
        var eyes = new MeasureEyeArea();
        var selection = new MeasureSelection();
        selection.ChooseItem(eyes, "eye.diff", false);
        var entry = Assert.Single(selection.ToEntries());
        Assert.Equal("EYE AREA", entry.Label);
        Assert.False(entry.Chosen);
        Assert.Equal(new[] { "eye.diff" }, entry.ItemNames);
        Assert.Equal(new[] { false }, entry.ItemChosen);
        Assert.False(selection.IsChosen(eyes));
    }

    // .NET-only: the entries are parallel arrays, so a damaged file can pair them up
    // wrongly. It loads what it can.
    [Fact]
    public void ADamagedSaveLoadsWhatItCan()
    {
        var eyes = new MeasureEyeArea();
        var fai = new MeasureFAI();
        var selection = MeasureSelection.FromEntries(new[]
        {
            null,
            new MeasureSelection.Entry { Label = null, Chosen = true },
            new MeasureSelection.Entry
            {
                Label = "EYE AREA", Chosen = true,
                ItemNames = new[] { "eye.left", "eye.diff" }, ItemChosen = new[] { false },
            },
            new MeasureSelection.Entry { Label = "FAI", Chosen = true },
        });
        Assert.True(selection.IsChosen(eyes));
        Assert.True(selection.IsItemChosen(eyes, "eye.left"));
        Assert.True(selection.IsItemChosen(eyes, "eye.diff"));
        Assert.True(selection.IsChosen(fai));
        Assert.Equal(new[] { "EYE AREA", "FAI" }, selection.ToEntries().Select(entry => entry.Label));
    }
}
