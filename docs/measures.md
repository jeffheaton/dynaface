# Measures

Each measure calculates one or more values from the landmarks. `AnalyzeFace.analyze()` returns every value in a dictionary, by the output names below, and draws the enabled ones on the image.

```python exec="on"
# Builds the table from the measure classes, so it always matches the library.
import inspect

from dynaface import facial  # noqa: F401 (must be imported before dynaface.measures)
from dynaface.measures import all_measures

print("| Measure | View | Outputs | Description |")
print("|---|---|---|---|")
for measure in all_measures():
    cls = type(measure)
    view = "frontal" if measure.is_frontal else "lateral"
    outputs = [item.name for item in measure.items if not item.display_only]
    if len(outputs) > 6:
        shown = f"`{outputs[0]}` to `{outputs[-1]}` ({len(outputs)} values)"
    else:
        shown = ", ".join(f"`{name}`" for name in outputs)
    summary = " ".join((inspect.getdoc(cls) or "").split("\n\n")[0].split())
    summary = summary.replace("|", "\\|")
    link = f"[{measure.abbrev()}][{cls.__module__}.{cls.__name__}]"
    print(f"| {link} | {view} | {shown} | {summary} |")
```

Frontal measures run on photos taken from the front, and lateral measures on side profiles. A value that can't be determined, such as a distance to a landmark that wasn't found, is NaN.

`analyze()` returns every output whether or not its item is enabled; enabling an item only controls whether it's drawn. To choose which values to show or export, use `AnalyzeFace.get_all_items()`, which lists the enabled outputs.
