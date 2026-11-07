using System.Collections.Generic;
using System.Linq;

// One of a measure's items: an output, a group of outputs or a drawing switch, with its
// enabled state and the views it suits (e.g. MeasureDentalArea has 5 outputs:
// dental_area/dental_left/dental_right/dental_ratio/dental_diff). Mirrors dynaface-lib's
// measures_base.py MeasureItem.
//   - DisplayOnly marks a drawing switch with no value of its own (MeasureLateral's
//     tip_proj): Calc never returns it and DynafaceMeasures.GetAllItems never lists it.
//   - Outputs makes the item a group: a switch named for what it draws, whose Name is a
//     label (shown in lists and kept by MeasureSelection), never an output. It stands for
//     its Outputs, which are returned, listed and exported together (OutputNames()).
//   - Views limits the item to some of its measure's views (FacePose.Frontal, Quarter
//     and Lateral); null for every view its measure suits. FaceMeasureBase.SyncItems
//     sets the item's IsFrontal/IsLateral/IsQuarter from the measure's, narrowed to them.
public class MeasureItemInfo
{
    // The output's name, or for a group or a drawing switch, its label.
    public string Name;
    public bool Enabled = true;
    public bool DisplayOnly;

    // For a group, the outputs it stands for, in order; its Name is then a label, never
    // an output. Null for an item that is its own output.
    public IReadOnlyList<string> Outputs;

    // The views of its measure the item suits (some of FacePose.Frontal, Quarter and
    // Lateral); null for every view its measure suits.
    public IReadOnlyList<FacePose> Views;

    // The views the item suits, as FaceMeasureBase.SyncItems last set them.
    public bool IsLateral;
    public bool IsFrontal;
    public bool IsQuarter;

    // outputs: for a group, the outputs it stands for (copied); views: the views it suits
    // (copied), null for all of its measure's.
    public MeasureItemInfo(string name, bool enabled = true, bool displayOnly = false,
        IEnumerable<string> outputs = null, IEnumerable<FacePose> views = null)
    {
        Name = name;
        Enabled = enabled;
        DisplayOnly = displayOnly;
        Outputs = outputs == null ? null : new List<string>(outputs).AsReadOnly();
        Views = views == null ? null : new List<FacePose>(views).AsReadOnly();
    }

    // The outputs the item stands for: none for a display-only item, a group's outputs in
    // order, otherwise its own Name. A new list each call, so changing it leaves the item
    // as it was.
    public List<string> OutputNames()
    {
        if (DisplayOnly) return new List<string>();
        if (Outputs != null) return new List<string>(Outputs);
        return new List<string> { Name };
    }

    // Whether the item suits the view, by FaceMeasureBase.Suits' rule (lateral, else
    // quarter, else frontal) on the item's own flags, which FaceMeasureBase.SyncItems
    // sets (UpdateForType, MeasureSelection.Apply and FrameSeries.FromStates call it
    // before asking). quarter is ignored when lateral is true.
    public bool Suits(bool lateral, bool quarter = false) =>
        lateral ? IsLateral : quarter ? IsQuarter : IsFrontal;

    public override string ToString() => $"(name={Name},enabled={Enabled})";
}

// Abstract base for all face measurements. Each subclass draws shapes/text directly
// onto FaceMeasureContext (via ctx.Pixels/ctx.TextLines/ctx.Values) and returns its
// named numeric results, mirroring dynaface-lib's measures_base.py MeasureBase /
// AnalyzeFace.analyze()'s `result.update(calc.calc(self))` contract:
//   - Calc always returns every declared output (Outputs()), whether or not its item is
//     enabled and whether or not render is true. A value that can't be determined is NaN.
//   - An item's Enabled flag gates its drawing and, for callers that honor it, which
//     outputs are shown or exported: DynafaceMeasures.GetAllItems lists the outputs of
//     the enabled items.
//   - DisplayOnly items are drawing switches, never returned or listed as outputs.
//   - A group item (MeasureItemInfo.Outputs) is a switch named for what it draws; its
//     name is a label, never an output. It stands for its outputs, which are returned,
//     listed and exported together, and turning it off drops all of them from
//     GetAllItems but not from Calc.
//   - A measure declares the views it suits with IsFrontal, IsLateral and IsQuarter (a
//     three-quarter face, FacePose.Quarter; no millimeter is defined on a turned head, so
//     only measures without millimeters suit it). Suits(lateral, quarter) reads them with
//     one rule, "lateral, else quarter, else frontal", and UpdateForType and
//     MeasureSelection.Apply enable a measure through it. Each item carries the same
//     three flags, copied by SyncItems, which UpdateForType, MeasureSelection.Apply and
//     FrameSeries.FromStates call before reading them, so a measure that sets its flags
//     without calling it, or adds an item afterward, still has its items follow its
//     view. An item declared with Views suits only those views of its measure, so
//     UpdateForType and MeasureSelection.Apply leave it off on a view it doesn't suit,
//     and FrameSeries reports its outputs as NaN there.
public abstract class FaceMeasureBase
{
    public bool Enabled = true;
    public bool IsFrontal;
    public bool IsLateral;

    // True for a measure that suits a three-quarter face (FacePose.Quarter): one with no
    // millimeter among its outputs. False unless the measure says so.
    public bool IsQuarter;

    public List<MeasureItemInfo> Items = new();

    public abstract string Label { get; }

    // True for a measure that samples the image's pixels (MeasureSkinTone), rather than
    // measuring the landmarks alone. FrameSeries.FromStates leaves these out for frames
    // saved without their image. .NET-only: dynaface-lib's saved states always hold it.
    public virtual bool NeedsImage => false;

    public abstract Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true);

    // Every output the measure declares, in order: its items' OutputNames(), a group's
    // outputs listed in its place. Calc returns all of them.
    public List<string> Outputs()
    {
        var names = new List<string>();
        foreach (var item in Items) names.AddRange(item.OutputNames());
        return names;
    }

    // Whether the measure suits the view: IsLateral for a lateral face, IsQuarter for a
    // quarter (three-quarter) face, IsFrontal otherwise. quarter is ignored when lateral
    // is true.
    public bool Suits(bool lateral, bool quarter = false) =>
        lateral ? IsLateral : quarter ? IsQuarter : IsFrontal;

    // Overwrites every item's Enabled flag with MeasureItemInfo.Suits(lateral, quarter):
    // the measure's view, narrowed to the item's own Views when it declares any (it calls
    // SyncItems first, so the items' flags follow the measure's as they are now). A
    // user's choice of items is lost; to keep the choices, use MeasureSelection.Apply.
    // quarter: false, the default, keeps the one-argument calls as they were.
    public void UpdateForType(bool lateral, bool quarter = false)
    {
        SyncItems();
        foreach (var item in Items) item.Enabled = item.Suits(lateral, quarter);
    }

    public void SetItemEnabled(string name, bool enabled)
    {
        foreach (var item in Items)
            if (item.Name == name) item.Enabled = enabled;
    }

    // Returns true if no item with this name is registered (matches dynaface-lib's
    // is_enabled default-to-enabled fallback).
    public bool IsEnabled(string name)
    {
        foreach (var item in Items)
            if (item.Name == name) return item.Enabled;
        return true;
    }

    public void SetEnabled(bool enabled)
    {
        Enabled = enabled;
        foreach (var item in Items) item.Enabled = enabled;
    }

    // Sets each item's view flags from the measure's: each is the measure's flag, and for
    // an item declared with Views, only where that view is among them. Idempotent.
    public void SyncItems()
    {
        foreach (var item in Items)
        {
            var views = item.Views;
            item.IsLateral = IsLateral && (views == null || views.Contains(FacePose.Lateral));
            item.IsFrontal = IsFrontal && (views == null || views.Contains(FacePose.Frontal));
            item.IsQuarter = IsQuarter && (views == null || views.Contains(FacePose.Quarter));
        }
    }
}
