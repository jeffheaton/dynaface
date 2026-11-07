using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace DynafaceTests;

// MeasureSelection keeps a user's choice of measures and items when the face switches
// between frontal, quarter (three-quarter) and lateral, which UpdateForType can't, and
// saves and loads it by name, and keeps an item off on a view it doesn't suit. Mirrors
// dynaface-lib's tests/test_measure_selection.py, its cheek-measure cases included
// (MeasureGroupsTests covers group items and item views on toy measures).
public class MeasureSelectionTests
{
    // The outputs of the measures that Ticked() leaves on, for each view.
    static readonly string[] Frontal = { "fai", "eye.left", "eye.right", "eye.ratio", "pitch", "roll", "yaw" };
    static readonly string[] Lateral = { "nfa", "at", "nt", "at_nt", "pitch", "roll", "yaw" };
    // On a quarter face, only the pose suits it among the ticked measures.
    static readonly string[] Quarter = { "pitch", "roll", "yaw" };
    static readonly string[] AllEyes = { "eye.left", "eye.right", "eye.diff", "eye.ratio" };
    // The measures a quarter face can use: no millimeter among their outputs.
    static readonly string[] QuarterMeasures = { "POSITION", "POSE", "LANDMARKS", "CHEEK MEASURES" };

    // The outputs of these Cheek Measures items, in the measure's order.
    static string[] CheekOutputs(params string[] items) => MeasureCheekMeasuresTests.OutputsOf(items);

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

    // ---- The third view: Apply(measures, lateral: false, quarter: true) enables what is
    // chosen among the measures that suit a quarter face, and the two-argument calls
    // keep their meaning.

    [Fact]
    public void QuarterViewEnablesTheMeasuresThatSuitIt()
    {
        var measures = DynafaceMeasures.AllMeasures();   // every measure and item on, as made
        MeasureSelection.Of(measures).Apply(measures, lateral: false, quarter: true);
        Assert.Equal(QuarterMeasures, measures.Where(m => m.Enabled).Select(m => m.Label));
        foreach (var measure in measures)
        {
            Assert.Equal(measure.IsQuarter, measure.Suits(false, true));
            Assert.Equal(measure.IsQuarter, measure.Enabled);
            // Each item on where it suits the view: the measure's, narrowed to the item's.
            Assert.Equal(measure.Items.Select(item => measure.IsQuarter && item.IsQuarter),
                measure.Items.Select(item => item.Enabled));
        }
        var expected = new List<string>();
        expected.AddRange(Find<MeasurePosition>(measures).Outputs());
        expected.AddRange(Find<MeasurePose>(measures).Outputs());
        expected.AddRange(Find<MeasureLandmarks>(measures).Outputs());
        // The Landmarks measure's Cheek Landmarks group is listed as its outputs.
        Assert.Equal(new[] { "zy.r.x", "zy.r.y" }, expected.TakeLast(20).Take(2));
        string[] quarterItems = { "Far cheek outline", "Malar apex", "Submalar point", "Head turn" };
        expected.AddRange(CheekOutputs(quarterItems));
        Assert.Equal(expected, DynafaceMeasures.GetAllItems(measures));
        // The frontal items are off on a quarter face, so neither their outputs nor any
        // item's label is listed.
        var cheek = Find<MeasureCheekMeasures>(measures);
        Assert.True(cheek.Enabled);
        Assert.Equal(41, cheek.Outputs().Count);
        Assert.Equal(quarterItems, cheek.Items.Where(item => item.Enabled).Select(item => item.Name));
        foreach (string name in CheekOutputs("Zygion width").Concat(cheek.Items.Select(item => item.Name)))
            Assert.DoesNotContain(name, DynafaceMeasures.GetAllItems(measures));
    }

    [Fact]
    public void ASelectionNamingTheCheekMeasuresRoundTrips()
    {
        var cheek = new MeasureCheekMeasures();
        var selection = new MeasureSelection();
        selection.Choose(new MeasurePose());
        selection.Choose(cheek);
        selection.ChooseItem(cheek, "Malar apex", false);
        var options = new JsonSerializerOptions { IncludeFields = true };
        string json = JsonSerializer.Serialize(selection.ToEntries(), options);
        var saved = JsonSerializer.Deserialize<MeasureSelection.Entry[]>(json, options)!;
        var entry = saved.Single(e => e.Label == "CHEEK MEASURES");
        Assert.True(entry.Chosen);
        Assert.Equal(new[] { "Malar apex" }, entry.ItemNames);
        Assert.Equal(new[] { false }, entry.ItemChosen);
        var loaded = MeasureSelection.FromEntries(saved);
        Assert.Equal(json, JsonSerializer.Serialize(loaded.ToEntries(), options));
        Assert.True(loaded.IsChosen(cheek));
        Assert.False(loaded.IsItemChosen(cheek, "Malar apex"));
        Assert.True(loaded.IsItemChosen(cheek, "Submalar point"));

        var measures = DynafaceMeasures.AllMeasures();
        var onQuarter = CheekOutputs("Far cheek outline", "Submalar point", "Head turn");
        var onFrontal = CheekOutputs("Zygion width", "Reference points", "Head turn");
        // The cheek measures suit a frontal face and a quarter face, and never a profile;
        // each item only the views it suits, and the unchosen apex neither.
        foreach (var (lateral, quarter, expected) in new[]
        {
            (false, true, Quarter.Concat(onQuarter)),
            (false, false, Quarter.Concat(onFrontal)),
            (true, false, (IEnumerable<string>)Quarter),
            (false, true, Quarter.Concat(onQuarter)),
        })
        {
            loaded.Apply(measures, lateral, quarter);
            Assert.Equal(expected, DynafaceMeasures.GetAllItems(measures));
        }
        Assert.Equal(new[]
        {
            ("Zygion width", false), ("Reference points", false), ("Far cheek outline", true),
            ("Malar apex", false), ("Submalar point", true), ("Head turn", true),
        }, Find<MeasureCheekMeasures>(measures).Items.Select(item => (item.Name, item.Enabled)));
        // The choice of the unsuited items is kept for the next frontal face.
        Assert.True(loaded.IsItemChosen(cheek, "Zygion width"));
    }

    [Fact]
    public void ASelectionThatNeverMentionsThemLeavesThemUnchosen()
    {
        // A selection saved before the cheek measures existed: Of() mentions every measure,
        // so the entry is dropped to make such a file.
        var entries = MeasureSelection.Of(Ticked()).ToEntries().ToList();
        var cheekEntry = entries.Single(e => e.Label == "CHEEK MEASURES");
        Assert.False(cheekEntry.Chosen);
        Assert.Empty(cheekEntry.ItemNames);
        entries.Remove(cheekEntry);
        var loaded = MeasureSelection.FromEntries(entries);
        Assert.False(loaded.IsChosen(new MeasureCheekMeasures()));
        var measures = DynafaceMeasures.AllMeasures();
        loaded.Apply(measures, lateral: false, quarter: true);
        Assert.Equal(Quarter, DynafaceMeasures.GetAllItems(measures));
        var cheek = Find<MeasureCheekMeasures>(measures);
        Assert.False(cheek.Enabled);
        Assert.DoesNotContain(cheek.Items, item => item.Enabled);
    }

    // dynaface-lib's test_one_selection_for_a_quarter_face runs the forced three-quarter
    // sketch and a frontal photo with the models; the fixture faces' landmarks do here:
    // the ticked selection plus the cheek measures (its apex unchosen) runs the pose and
    // the cheek measures on the quarter face, and the frontal measures on the frontal one.
    [Fact]
    public void OneSelectionWithTheCheekMeasures()
    {
        var chosen = Ticked();
        var cheek = Find<MeasureCheekMeasures>(chosen);
        cheek.SetEnabled(true);
        cheek.SetItemEnabled("Malar apex", false);
        var selection = MeasureSelection.Of(chosen);
        var allCheek = cheek.Outputs();

        var quarter = CheekFaces.Sketch();
        Assert.True(quarter.IsQuarter);
        Assert.False(quarter.IsLateral);
        selection.Apply(chosen, quarter.IsLateral, quarter.IsQuarter);
        var results = quarter.Analyze(chosen, render: false)!;
        Assert.Equal(Quarter.Concat(allCheek).OrderBy(n => n), results.Keys.OrderBy(n => n));
        Assert.Equal(Quarter.Concat(CheekOutputs("Far cheek outline", "Submalar point", "Head turn")),
            DynafaceMeasures.GetAllItems(chosen));
        Assert.Equal(0.0, results["cheek.far"]);
        // The unchosen item, the measures that don't suit the view and the cheek measure's
        // frontal items are off; every output is still returned (the frontal ones NaN on a
        // quarter face).
        foreach (string name in new[] { "apex.ok", "zy.w", "cr.ok", "fai", "eye.left" })
            Assert.DoesNotContain(name, DynafaceMeasures.GetAllItems(chosen));
        Assert.True(double.IsNaN(results["zy.w"]));
        Assert.False(double.IsNaN(results["apex.r.h"]));

        var frontal = CheekFaces.Frontal();
        Assert.False(frontal.IsQuarter || frontal.IsLateral);
        selection.Apply(chosen, frontal.IsLateral, frontal.IsQuarter);
        results = frontal.Analyze(chosen, render: false)!;
        Assert.Equal(Frontal.Append("eye.diff").Concat(allCheek).OrderBy(n => n), results.Keys.OrderBy(n => n));
        Assert.Equal(Frontal.Concat(CheekOutputs("Zygion width", "Reference points", "Head turn")),
            DynafaceMeasures.GetAllItems(chosen));
    }

    [Fact]
    public void TwoArgumentCallsGiveTheOldLists()
    {
        var measures = Ticked();
        var selection = MeasureSelection.Of(measures);
        foreach (var (lateral, quarter, expected) in new[]
        {
            (false, (bool?)null, Frontal),
            (true, null, Lateral),
            (false, true, Quarter),
            (true, true, Lateral),     // quarter is ignored
            (false, null, Frontal),    // the choices came back
            (false, false, Frontal),
        })
        {
            if (quarter.HasValue) selection.Apply(measures, lateral, quarter.Value);
            else selection.Apply(measures, lateral);
            Assert.Equal(expected, DynafaceMeasures.GetAllItems(measures));
        }
        // UpdateForType keeps its one-argument meaning too, and with quarter: true
        // overwrites the items for the quarter view.
        foreach (var measure in measures)
        {
            measure.UpdateForType(false);
            Assert.Equal(measure.IsFrontal && measure.Items.Count > 0, measure.Items.Any(item => item.Enabled));
            measure.UpdateForType(false, true);
            Assert.Equal(measure.IsQuarter && measure.Items.Count > 0, measure.Items.Any(item => item.Enabled));
        }
    }

    // Apply decides which measures Analyze runs: on a quarter face only the chosen ones
    // that suit it, and the same selection runs the frontal measures on a frontal face.
    // dynaface-lib runs this on photos with the models; landmarks alone do here.
    [Fact]
    public void OneSelectionForAQuarterFaceAndAFrontalOne()
    {
        var chosen = Ticked();
        var selection = MeasureSelection.Of(chosen);
        var photo = new FaceImage(new Rgba32[1024 * 1024], 1024, 1024);
        var landmarks = TestHelpers.MakeLandmarks();
        landmarks[DynafaceConstants.LmRightPupil] = new Vec2(380f, 480f);
        landmarks[DynafaceConstants.LmLeftPupil] = new Vec2(640f, 480f);

        var quarter = new FaceMeasureContext(photo, landmarks, pose: FacePose.Quarter);
        Assert.True(quarter.IsQuarter);
        selection.Apply(chosen, quarter.IsLateral, quarter.IsQuarter);
        Assert.Equal(Quarter.OrderBy(n => n), quarter.Analyze(chosen, render: false)!.Keys.OrderBy(n => n));
        Assert.Equal(Quarter, DynafaceMeasures.GetAllItems(chosen));

        var frontal = new FaceMeasureContext(photo, landmarks);
        Assert.False(frontal.IsQuarter || frontal.IsLateral);
        selection.Apply(chosen, frontal.IsLateral, frontal.IsQuarter);
        // Each measure that runs returns all its outputs, items on or off.
        Assert.Equal(Frontal.Append("eye.diff").OrderBy(n => n),
            frontal.Analyze(chosen, render: false)!.Keys.OrderBy(n => n));
        Assert.Equal(Frontal, DynafaceMeasures.GetAllItems(chosen));
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
