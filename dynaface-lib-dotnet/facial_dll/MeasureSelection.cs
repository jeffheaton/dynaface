using System;
using System.Collections.Generic;

// Which measures, and which of their items, a user has chosen, kept apart from the
// measures themselves, so that switching between a frontal, a quarter (three-quarter)
// and a lateral face never loses a choice. Mirrors dynaface-lib's
// measures.MeasureSelection.
//
// FaceMeasureBase.UpdateForType overwrites every item's Enabled flag with whether the
// item suits the face (MeasureItemInfo.Suits: its measure's view, narrowed to the item's
// own Views). Apply instead enables what was chosen and suits the face, and leaves the
// choices as they were. Save a selection with ToEntries, one Entry per measure keyed by
// its Label and its items' names (a group item by its label), and restore it with
// FromEntries. A measure that a selection doesn't mention isn't chosen; an item it
// doesn't mention is, so an item added to a measure later is shown.
public sealed class MeasureSelection
{
    // One measure's choices, as ToEntries saves them. Public fields, so Unity's
    // JsonUtility (inside a wrapper object) and other serializers can write it.
    [Serializable]
    public sealed class Entry
    {
        public string Label;
        public bool Chosen;

        // The items' names and whether each is chosen, in pairs. Null or empty when
        // the selection mentions none of them, which leaves every item chosen.
        public string[] ItemNames;
        public bool[] ItemChosen;
    }

    sealed class Choices
    {
        public bool Chosen;
        public readonly List<string> ItemNames = new List<string>();   // first mentioned first
        public readonly Dictionary<string, bool> Items = new Dictionary<string, bool>();
    }

    readonly List<string> _labels = new List<string>();   // first mentioned first
    readonly Dictionary<string, Choices> _choices = new Dictionary<string, Choices>();

    // The choices that the measures' Enabled flags hold now, such as after a user ticks
    // them: each measure's own flag, and its items' flags if it is enabled. A measure
    // switched off with SetEnabled(false) has every item switched off too, which isn't
    // a choice of items, so choosing it again later shows all of them. Use it on
    // measures whose flags hold the user's choices, not on ones that Apply has set for
    // a face, which hold only what suits that face.
    public static MeasureSelection Of(IEnumerable<FaceMeasureBase> measures)
    {
        if (measures == null) throw new ArgumentNullException(nameof(measures));
        var selection = new MeasureSelection();
        foreach (var measure in measures)
        {
            selection.Choose(measure, measure.Enabled);
            if (!measure.Enabled) continue;
            foreach (var item in measure.Items)
                selection.ChooseItem(measure, item.Name, item.Enabled);
        }
        return selection;
    }

    // Whether the measure is chosen; a measure never mentioned isn't.
    public bool IsChosen(FaceMeasureBase measure) =>
        _choices.TryGetValue(LabelOf(measure), out var choices) && choices.Chosen;

    // Whether one of the measure's items is chosen; an item never mentioned is.
    public bool IsItemChosen(FaceMeasureBase measure, string item) =>
        !_choices.TryGetValue(LabelOf(measure), out var choices)
        || !choices.Items.TryGetValue(item, out bool chosen)
        || chosen;

    // Chooses a measure, or with chosen: false, unchooses it.
    public void Choose(FaceMeasureBase measure, bool chosen = true) =>
        For(LabelOf(measure)).Chosen = chosen;

    // Chooses one of a measure's items, or with chosen: false, unchooses it.
    public void ChooseItem(FaceMeasureBase measure, string item, bool chosen = true)
    {
        if (item == null) throw new ArgumentNullException(nameof(item));
        SetItem(For(LabelOf(measure)), item, chosen);
    }

    // Sets the measures' and items' Enabled flags for a frontal, a quarter
    // (three-quarter) or a lateral face: on where chosen and the measure suits the face
    // (FaceMeasureBase.Suits), and for an item, where it suits the face too
    // (MeasureItemInfo.Suits, which also honors an item's own Views), off everywhere
    // else. It calls each measure's SyncItems first, so the items' flags follow the
    // measure's as they are now. The choices stay as they were, so applying another view
    // later brings them back. quarter: true for a quarter face
    // (FaceMeasureContext.IsQuarter), ignored when lateral is true; false, the default,
    // keeps the two-argument calls as they were.
    public void Apply(IEnumerable<FaceMeasureBase> measures, bool lateral, bool quarter = false)
    {
        if (measures == null) throw new ArgumentNullException(nameof(measures));
        foreach (var measure in measures)
        {
            measure.SyncItems();
            measure.Enabled = measure.Suits(lateral, quarter) && IsChosen(measure);
            foreach (var item in measure.Items)
                item.Enabled = measure.Enabled
                    && item.Suits(lateral, quarter)
                    && IsItemChosen(measure, item.Name);
        }
    }

    // The selection as one Entry for each measure it mentions, ready to save.
    public Entry[] ToEntries()
    {
        var entries = new Entry[_labels.Count];
        for (int i = 0; i < entries.Length; i++)
        {
            var choices = _choices[_labels[i]];
            var itemChosen = new bool[choices.ItemNames.Count];
            for (int j = 0; j < itemChosen.Length; j++)
                itemChosen[j] = choices.Items[choices.ItemNames[j]];
            entries[i] = new Entry
            {
                Label = _labels[i],
                Chosen = choices.Chosen,
                ItemNames = choices.ItemNames.ToArray(),
                ItemChosen = itemChosen,
            };
        }
        return entries;
    }

    // A selection saved with ToEntries. An entry without a label is skipped, and so are
    // an entry's item choices when its names and flags don't pair up, which leaves its
    // items chosen, so a damaged file loads what it can.
    public static MeasureSelection FromEntries(IEnumerable<Entry> entries)
    {
        if (entries == null) throw new ArgumentNullException(nameof(entries));
        var selection = new MeasureSelection();
        foreach (var entry in entries)
        {
            if (entry?.Label == null) continue;
            var choices = selection.For(entry.Label);
            choices.Chosen = entry.Chosen;
            if (entry.ItemNames == null || entry.ItemChosen == null
                || entry.ItemNames.Length != entry.ItemChosen.Length)
                continue;
            for (int i = 0; i < entry.ItemNames.Length; i++)
                if (entry.ItemNames[i] != null)
                    SetItem(choices, entry.ItemNames[i], entry.ItemChosen[i]);
        }
        return selection;
    }

    static string LabelOf(FaceMeasureBase measure)
    {
        if (measure == null) throw new ArgumentNullException(nameof(measure));
        return measure.Label;
    }

    Choices For(string label)
    {
        if (!_choices.TryGetValue(label, out var choices))
        {
            choices = new Choices();
            _choices[label] = choices;
            _labels.Add(label);
        }
        return choices;
    }

    static void SetItem(Choices choices, string item, bool chosen)
    {
        if (!choices.Items.ContainsKey(item)) choices.ItemNames.Add(item);
        choices.Items[item] = chosen;
    }
}
