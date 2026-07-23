using System.Collections.Generic;

// A single named, independently enable-able output field within a FaceMeasureBase
// (e.g. AnalyzeDentalArea has 5: dental_area/dental_left/dental_right/dental_ratio/dental_diff).
// Mirrors dynaface-lib's measures_base.py MeasureItem.
public class MeasureItemInfo
{
    public string Name;
    public bool Enabled = true;
    public bool IsLateral;
    public bool IsFrontal;

    public MeasureItemInfo(string name, bool enabled = true)
    {
        Name = name;
        Enabled = enabled;
    }

    public override string ToString() => $"(name={Name},enabled={Enabled})";
}

// Abstract base for all face measurements. Each subclass draws shapes/text directly
// onto FaceMeasureContext (via ctx.Pixels/ctx.TextLines/ctx.Values) and returns its
// named numeric results, mirroring dynaface-lib's measures_base.py MeasureBase /
// AnalyzeFace.analyze()'s `result.update(calc.calc(self))` contract.
public abstract class FaceMeasureBase
{
    public bool Enabled = true;
    public bool IsFrontal;
    public bool IsLateral;
    public List<MeasureItemInfo> Items = new();

    public abstract string Label { get; }

    public abstract Dictionary<string, double> Calc(FaceMeasureContext ctx, bool render = true);

    // Sets every item's Enabled to this measure's own IsLateral/IsFrontal flag
    // (not the item's own IsLateral/IsFrontal) — matches dynaface-lib's
    // update_for_type exactly, including that apparent asymmetry.
    public void UpdateForType(bool lateral)
    {
        bool value = lateral ? IsLateral : IsFrontal;
        foreach (var item in Items) item.Enabled = value;
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

    public void SyncItems()
    {
        foreach (var item in Items)
        {
            item.IsLateral = IsLateral;
            item.IsFrontal = IsFrontal;
        }
    }
}
