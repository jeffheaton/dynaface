namespace DynafaceTests;

// Grouped and view-limited measure items (the contract at the top of
// FaceMeasureBase.cs): a group item is a switch named for what it draws that stands for
// several outputs, and an item declared with views suits only those views of its
// measure. Toy measures, no models. Mirrors dynaface-lib-python's
// tests/test_measure_groups.py.
public class MeasureGroupsTests
{
    // A frontal and quarter measure with a plain output, two groups (one for each view),
    // a group for both views and a drawing switch.
    sealed class Grouped : FaceMeasureBase
    {
        public List<string> Drawn = new();

        public Grouped()
        {
            Items.Add(new MeasureItemInfo("plain"));
            Items.Add(new MeasureItemInfo("Width", outputs: new[] { "w.r", "w.l", "w" }, views: new[] { FacePose.Frontal }));
            Items.Add(new MeasureItemInfo("Apex", outputs: new[] { "a.h", "a.ok" }, views: new[] { FacePose.Quarter }));
            Items.Add(new MeasureItemInfo("Turn", outputs: new[] { "yaw", "pitch" },
                views: new[] { FacePose.Frontal, FacePose.Quarter }));
            Items.Add(new MeasureItemInfo("switch", displayOnly: true));
            IsFrontal = true;
            IsQuarter = true;
            SyncItems();
        }

        public override string Label => "Grouped";

        public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true)
        {
            Drawn = Items.Where(i => render && i.Enabled).Select(i => i.Name).ToList();
            return Outputs().ToDictionary(name => name, _ => 1.0);
        }
    }

    // A frontal measure with plain items only, as every measure was before groups.
    sealed class Plain : FaceMeasureBase
    {
        public Plain()
        {
            Items.Add(new MeasureItemInfo("a"));
            Items.Add(new MeasureItemInfo("b"));
            Items.Add(new MeasureItemInfo("t", true, true));
            IsFrontal = true;
            SyncItems();
        }

        public override string Label => "Plain";

        public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true) =>
            Outputs().ToDictionary(name => name, _ => 1.0);
    }

    // A frontal measure that sets its flags but never calls SyncItems, as a plugin
    // measure written before item views might.
    sealed class Unsynced : FaceMeasureBase
    {
        public Unsynced()
        {
            Items.Add(new MeasureItemInfo("a"));
            Items.Add(new MeasureItemInfo("b"));
            IsFrontal = true;
        }

        public override string Label => "Unsynced";

        public override Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true) =>
            Outputs().ToDictionary(name => name, _ => 1.0);
    }

    static readonly string[] GroupedOutputs = { "plain", "w.r", "w.l", "w", "a.h", "a.ok", "yaw", "pitch" };

    static List<string> AllItems(params FaceMeasureBase[] measures) => DynafaceMeasures.GetAllItems(measures);

    static (bool frontal, bool quarter, bool lateral) Flags(MeasureItemInfo item) =>
        (item.IsFrontal, item.IsQuarter, item.IsLateral);

    static Dictionary<string, double> Results(params (string name, double value)[] values) =>
        values.ToDictionary(v => v.name, v => v.value);

    // ---- Output names ------------------------------------------------------------

    [Fact]
    public void KindsOfItem()
    {
        Assert.Equal(new[] { "fai" }, new MeasureItemInfo("fai").OutputNames());
        Assert.Empty(new MeasureItemInfo("tip", displayOnly: true).OutputNames());
        var group = new MeasureItemInfo("Width", outputs: new[] { "w.r", "w.l" });
        Assert.Equal(new[] { "w.r", "w.l" }, group.OutputNames());
        // A copy: changing it doesn't change the item.
        group.OutputNames().Add("x");
        Assert.Equal(new[] { "w.r", "w.l" }, group.OutputNames());
        // The group's name is a label, never an output.
        Assert.DoesNotContain("Width", group.OutputNames());
        Assert.Empty(new MeasureItemInfo("Empty", outputs: new string[0]).OutputNames());
    }

    // .NET-only: the item keeps its own copy of the outputs and views it's given.
    [Fact]
    public void AnItemCopiesItsOutputsAndViews()
    {
        var outputs = new List<string> { "w.r", "w.l" };
        var views = new List<FacePose> { FacePose.Frontal };
        var group = new MeasureItemInfo("Width", outputs: outputs, views: views);
        outputs.Add("x");
        views.Add(FacePose.Quarter);
        Assert.Equal(new[] { "w.r", "w.l" }, group.Outputs);
        Assert.Equal(new[] { FacePose.Frontal }, group.Views);
    }

    [Fact]
    public void MeasureOutputsInOrder()
    {
        Assert.Equal(GroupedOutputs, new Grouped().Outputs());
        Assert.Equal(new[] { "a", "b" }, new Plain().Outputs());
    }

    // dynaface-lib's select_outputs: every output of every declared item, a group's
    // expanded, whether or not the item is enabled. Here Outputs() lists them.
    [Fact]
    public void OutputsExpandGroupsEnabledOrNot()
    {
        var measure = new Grouped();
        foreach (var item in measure.Items) item.Enabled = false;
        var outputs = measure.Outputs();
        Assert.Equal(GroupedOutputs, outputs);
        Assert.DoesNotContain("Width", outputs);
        Assert.DoesNotContain("switch", outputs);
        Assert.Equal(GroupedOutputs, measure.Calc(TestHelpers.BuildContext(), render: false).Keys);
    }

    // ---- GetAllItems -------------------------------------------------------------

    [Fact]
    public void GroupsExpand() => Assert.Equal(GroupedOutputs, AllItems(new Grouped()));

    [Fact]
    public void AnOffGroupDropsAllItsOutputsButCalcReturnsThem()
    {
        var measure = new Grouped();
        measure.SetItemEnabled("Width", false);
        Assert.False(measure.IsEnabled("Width"));
        Assert.Equal(new[] { "plain", "a.h", "a.ok", "yaw", "pitch" }, AllItems(measure));
        var result = measure.Calc(TestHelpers.BuildContext(), render: true);
        Assert.Equal(GroupedOutputs, result.Keys);
        Assert.DoesNotContain("Width", measure.Drawn); // an off group isn't drawn
    }

    [Fact]
    public void ADisabledMeasureListsNothing()
    {
        var measure = new Grouped();
        measure.SetEnabled(false);
        Assert.Empty(AllItems(measure));
    }

    // ---- Views -------------------------------------------------------------------

    [Fact]
    public void SyncItemsNarrowsToTheItemsViews()
    {
        var measure = new Grouped();
        var flags = measure.Items.ToDictionary(item => item.Name, Flags);
        Assert.Equal((true, true, false), flags["plain"]);
        Assert.Equal((true, false, false), flags["Width"]);
        Assert.Equal((false, true, false), flags["Apex"]);
        Assert.Equal((true, true, false), flags["Turn"]);
        Assert.Equal((true, true, false), flags["switch"]);
        // Idempotent, and never wider than the measure: a view the measure doesn't suit
        // stays off for an item that names it.
        measure.SyncItems();
        Assert.Equal(flags, measure.Items.ToDictionary(item => item.Name, Flags));
        measure.IsQuarter = false;
        measure.SyncItems();
        Assert.Equal((false, false, false), Flags(measure.Items[2]));
        measure.IsQuarter = true;
        measure.SyncItems();
        Assert.Equal(flags, measure.Items.ToDictionary(item => item.Name, Flags));
    }

    [Fact]
    public void ItemSuits()
    {
        var items = new Grouped().Items;
        MeasureItemInfo width = items[1], apex = items[2], turn = items[3];
        Assert.True(width.Suits(false));
        Assert.False(width.Suits(false, quarter: true));
        Assert.False(width.Suits(true));
        Assert.False(width.Suits(true, quarter: true)); // lateral first
        Assert.False(apex.Suits(false));
        Assert.True(apex.Suits(false, quarter: true));
        Assert.True(turn.Suits(false) && turn.Suits(false, quarter: true));
    }

    [Fact]
    public void UpdateForType()
    {
        var measure = new Grouped();
        measure.UpdateForType(lateral: false, quarter: true);
        Assert.Equal(
            new Dictionary<string, bool>
            {
                ["plain"] = true, ["Width"] = false, ["Apex"] = true, ["Turn"] = true, ["switch"] = true,
            },
            measure.Items.ToDictionary(item => item.Name, item => item.Enabled));
        Assert.Equal(new[] { "plain", "a.h", "a.ok", "yaw", "pitch" }, AllItems(measure));
        measure.UpdateForType(lateral: false);
        Assert.False(measure.IsEnabled("Apex"));
        Assert.True(measure.IsEnabled("Width"));
        measure.UpdateForType(lateral: true);
        Assert.DoesNotContain(measure.Items, item => item.Enabled);
    }

    [Fact]
    public void SelectionApply()
    {
        var measure = new Grouped();
        var selection = new MeasureSelection();
        selection.Choose(measure);
        selection.ChooseItem(measure, "Turn", false);
        selection.Apply(new[] { measure }, lateral: false, quarter: true);
        Assert.True(measure.Enabled);
        Assert.Equal(
            new Dictionary<string, bool>
            {
                ["plain"] = true,
                ["Width"] = false,   // chosen, but not on a quarter face
                ["Apex"] = true,
                ["Turn"] = false,    // unchosen
                ["switch"] = true,
            },
            measure.Items.ToDictionary(item => item.Name, item => item.Enabled));
        selection.Apply(new[] { measure }, lateral: false);
        Assert.True(measure.IsEnabled("Width"));
        Assert.False(measure.IsEnabled("Apex"));
        Assert.False(measure.IsEnabled("Turn"));
        // The choices stay as they were, keyed by the group's label.
        Assert.True(selection.IsItemChosen(measure, "Width"));
        Assert.False(selection.IsItemChosen(measure, "Turn"));
        var entry = Assert.Single(selection.ToEntries());
        Assert.Equal(("Grouped", true), (entry.Label, entry.Chosen));
        Assert.Equal(new[] { "Turn" }, entry.ItemNames);
        Assert.Equal(new[] { false }, entry.ItemChosen);
    }

    [Fact]
    public void FrameSeriesNaNsAnUnsuitedItemsOutputs()
    {
        var measure = new Grouped();
        var results = measure.Calc(TestHelpers.BuildContext(), render: false);
        var inView = FrameSeries.ResultsInView(new[] { measure }, results, lateral: false, quarter: true)!;
        foreach (string name in new[] { "w.r", "w.l", "w" })
            Assert.True(double.IsNaN(inView[name]), name);
        foreach (string name in new[] { "plain", "a.h", "a.ok", "yaw", "pitch" })
            Assert.Equal(1.0, inView[name]);
        inView = FrameSeries.ResultsInView(new[] { measure }, results, lateral: false, quarter: false)!;
        Assert.True(double.IsNaN(inView["a.h"]));
        Assert.Equal(1.0, inView["w"]);
        Assert.Null(FrameSeries.ResultsInView(new[] { measure }, null, false, false));
    }

    // .NET-only: the same through FromStates, from saved states of each view.
    [Fact]
    public void FromStatesNaNsAnUnsuitedItemsOutputs()
    {
        static FaceAnalysisState State(FacePose pose) =>
            new() { Landmarks = TestHelpers.MakeLandmarks(), Pose = pose };

        var series = FrameSeries.FromStates(
            new[] { State(FacePose.Quarter), State(FacePose.Frontal) }, new FaceMeasureBase[] { new Grouped() });
        Assert.Equal(GroupedOutputs, series.Names);
        foreach (string name in new[] { "w.r", "w.l", "w" })
            Assert.Equal(new[] { double.NaN, 1.0 }, series[name]);
        foreach (string name in new[] { "a.h", "a.ok" })
            Assert.Equal(new[] { 1.0, double.NaN }, series[name]);
        foreach (string name in new[] { "plain", "yaw", "pitch" })
            Assert.Equal(new[] { 1.0, 1.0 }, series[name]);
    }

    // ---- A measure of plain items behaves exactly as before ----------------------

    [Fact]
    public void WithoutGroupsUnchanged()
    {
        var measure = new Plain();
        foreach (var item in measure.Items)
        {
            Assert.Null(item.Outputs);
            Assert.Null(item.Views);
            Assert.Equal((true, false, false), Flags(item));
        }
        Assert.Equal(new[] { "a", "b" }, AllItems(measure));
        Assert.Equal(new[] { "a", "b" }, measure.Outputs());
        measure.UpdateForType(lateral: false, quarter: true);
        Assert.Equal(new[] { false, false, false }, measure.Items.Select(i => i.Enabled));
        measure.UpdateForType(lateral: false);
        Assert.Equal(new[] { true, true, true }, measure.Items.Select(i => i.Enabled));
        var selection = new MeasureSelection();
        selection.Choose(measure);
        selection.ChooseItem(measure, "b", false);
        selection.Apply(new[] { measure }, lateral: false);
        Assert.Equal(new[] { true, false, true }, measure.Items.Select(i => i.Enabled));
        selection.Apply(new[] { measure }, lateral: false, quarter: true);
        Assert.False(measure.Enabled);
        Assert.Equal(new[] { false, false, false }, measure.Items.Select(i => i.Enabled));
    }

    // ---- A measure that never calls SyncItems, or gains an item after it, still has
    // its items follow its view: UpdateForType, MeasureSelection.Apply and the series'
    // view rule sync the items before reading their flags.

    [Fact]
    public void WithoutSyncItemsUpdateForType()
    {
        var measure = new Unsynced();
        measure.UpdateForType(lateral: false);
        Assert.Equal(new[] { true, true }, measure.Items.Select(i => i.Enabled));
        measure.UpdateForType(lateral: false, quarter: true);
        Assert.Equal(new[] { false, false }, measure.Items.Select(i => i.Enabled));
    }

    [Fact]
    public void WithoutSyncItemsSelectionApply()
    {
        var measure = new Unsynced();
        var selection = new MeasureSelection();
        selection.Choose(measure);
        selection.Apply(new[] { measure }, lateral: false);
        Assert.True(measure.Enabled);
        Assert.Equal(new[] { true, true }, measure.Items.Select(i => i.Enabled));
        Assert.Equal(new[] { "a", "b" }, AllItems(measure));
    }

    [Fact]
    public void WithoutSyncItemsFrameSeries()
    {
        var inView = FrameSeries.ResultsInView(new[] { new Unsynced() }, Results(("a", 1.0), ("b", 2.0)), false, false);
        Assert.Equal(Results(("a", 1.0), ("b", 2.0)), inView);
    }

    [Fact]
    public void AnUnsuitedMeasureNaNsEveryOutput()
    {
        // A disabled item's outputs too, not only the enabled items'.
        var measure = new Plain();
        measure.SetItemEnabled("b", false);
        var inView = FrameSeries.ResultsInView(new[] { measure }, Results(("a", 1.0), ("b", 2.0)),
            lateral: false, quarter: true)!;
        Assert.True(double.IsNaN(inView["a"]));
        Assert.True(double.IsNaN(inView["b"]));
    }

    [Fact]
    public void AnItemAddedAfterSyncItems()
    {
        var measure = new Plain();
        measure.Items.Add(new MeasureItemInfo("c"));
        measure.UpdateForType(lateral: false);
        Assert.Equal(new[] { "a", "b", "t", "c" }, measure.Items.Where(i => i.Enabled).Select(i => i.Name));
        var inView = FrameSeries.ResultsInView(new[] { measure }, Results(("c", 3.0)), false, false)!;
        Assert.Equal(3.0, inView["c"]);
    }
}
