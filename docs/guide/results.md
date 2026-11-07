# Working with results

## The results

[`analyze()`][dynaface.facial.AnalyzeFace.analyze] returns a dictionary that maps each output's name to its value:

```python
results = face.analyze()
print(results["fai"])  # in millimeters
print(results["oce.l"])
```

The [measures](../measures.md) page lists every output and its units. `analyze()` returns `None` if Dynaface found no face in the image, which `face.is_no_face()` also tells you. Within the results, a value that couldn't be determined is NaN, and the outputs of a measure that couldn't run at all are `None`.

## Choosing measures

`load_face_image()` and `AnalyzeFace` take the list of measures to run. `measures.all_measures()` includes [Landmarks](../measures.md#landmarks), which adds 216 coordinates to the results, the 98 landmarks' and the ten cheek landmarks', and [Cheek Measures](../measures.md#cheek-measures), which adds 41 values, about half of them NaN in any one view, and draws its constructions on every frontal and three-quarter face; leave out what you don't need:

```python
from dynaface import facial, measures
from dynaface.measures_frontal import AnalyzeLandmarks

chosen = [m for m in measures.all_measures() if not isinstance(m, AnalyzeLandmarks)]
face = facial.load_face_image("subject.jpg", measures=chosen)
```

Or list only the ones you want:

```python
face = facial.load_face_image(
    "subject.jpg",
    measures=[measures.AnalyzeFAI(), measures.AnalyzeOralCommissureExcursion()],
)
```

## Choosing what to draw and export

Each output has an item that controls whether it's drawn. Switching an item off doesn't remove its value from the results, but [`get_all_items()`][dynaface.facial.AnalyzeFace.get_all_items] lists only the outputs whose items are on, which is how the apps choose their chart and CSV columns. Some items are groups: a switch named for what it draws that stands for several outputs, such as the Landmarks measure's "Cheek Landmarks" (the 20 cheek landmark coordinates) or the Cheek Measures' "Zygion width" (`zy.r` to `zy.ok`). A group's name is only a label, never an output; switching it off stops its drawing and drops all its outputs from `get_all_items()`, and a measure's `outputs()` lists every output it returns, groups expanded:

```python
eyes = measures.AnalyzeEyeArea()
eyes.set_item_enabled("eye.diff", False)  # don't draw or list the difference

face = facial.load_face_image("subject.jpg", measures=[eyes])
results = face.analyze()  # still includes eye.diff
print(face.get_all_items())  # ['eye.left', 'eye.right', 'eye.ratio']

cheeks = measures.AnalyzeCheekMeasures()
cheeks.set_item_enabled("Far cheek outline", False)  # cheek.far to cheek.scale
```

## Frontal photos, three-quarter photos and profiles

When it loads an image, Dynaface decides whether it shows the face from the front, in profile or, when `config.AUTO_QUARTER` is on, turned three-quarters, and sets `face.lateral` to True for a profile and `face.quarter` to True for a three-quarter face; `face.pose` holds the view as a `Pose`. `analyze()` runs every measure you gave it whatever the view, and frontal measures don't mean anything on a profile, nor millimeter measures on a three-quarter face. To run only the measures that fit the view, apply a [`MeasureSelection`][dynaface.measures.MeasureSelection] before analyzing:

```python
face = facial.load_face_image("photo.jpg")
selection = measures.MeasureSelection.of(face.measures)  # as you set them
selection.apply(face.measures, lateral=face.lateral, quarter=face.quarter)

results = face.analyze()  # only the measures that fit the view
row = {name: results.get(name) for name in face.get_all_items()}
```

The selection holds your choice of measures and items apart from the measures themselves. `apply()` switches off what doesn't fit the face without forgetting it, so you can apply the same selection to frontal photos, three-quarter photos and profiles in turn, and an item you switched off, such as `eye.diff`, stays off. Each measure declares the views it suits with `is_frontal`, `is_quarter` and `is_lateral`, which its `suits()` reads; on a three-quarter face only Position, Pose, Landmarks and Cheek Measures suit. An item can suit fewer views than its measure: each item carries the same flags, narrowed by the views it was declared with, which its own `suits()` reads, and `apply()` leaves an item off on a view it doesn't suit, so the Cheek Measures' Zygion width and Reference points stay off on a three-quarter face and its Far cheek outline, Malar apex and Submalar point on a frontal one. `quarter` defaults to False, so a call that passes only `lateral` works as before. Take the selection with `of()` before applying it, since afterwards the measures hold only what fits the last face. `choose()` and `choose_item()` change it, and `to_dict()` gives it as a dictionary of the measures' and items' names, ready for `json.dump()`, which `MeasureSelection.from_dict()` turns back into a selection. A selection saved by an earlier version never mentions "Cheek Measures", so it leaves that measure unchosen; `choose()` it to add it. An item a selection never mentions is chosen, so such a selection keeps the Landmarks measure's new "Cheek Landmarks" item on wherever it keeps Landmarks.

[`FrameSeries.from_states()`][dynaface.video.FrameSeries.from_states] applies the same rule by itself: a measure's outputs are NaN on a frame whose view it doesn't suit, and so are an item's, so a mixed-view video never reports millimeters measured on a turned head, and `peak()` and `evaluate()` never pick a turned or profile frame.

Each measure also has `update_for_type()`, which takes the same `lateral` and `quarter` arguments and switches its items on or off to fit a view, but it switches every item that suits the view back on, so it loses the items you switched off.

If Dynaface gets the view wrong, give it the view yourself when you load the image. `Pose.QUARTER` forces the three-quarter view, the only way to get it while `AUTO_QUARTER` is off. In earlier versions a forced three-quarter face was cropped like a frontal one; now it takes the three-quarter path, with the centered crop and the far cheek's outline, and the millimeter measures no longer suit it.

```python
import cv2
from dynaface.const import Pose

image = cv2.cvtColor(cv2.imread("profile.jpg"), cv2.COLOR_BGR2RGB)  # OpenCV loads BGR
face = facial.AnalyzeFace(measures=measures.all_measures())
face.load_image(image, crop=True, force_pose=Pose.LATERAL)
```

## The image

`face.render_img` is the standard image with the measures drawn on it, as an RGB NumPy array, and `face.save("measured.jpg")` writes it to a file. `face.original_img` is the same image without the drawings. `face.text_size` sets the size of the text the measures draw: 0.75 by default, and 1.5 draws it twice as large.

## Measuring a folder of photos

This measures every photo in a folder and writes one row per photo to a CSV file:

```python
import csv
from pathlib import Path

from dynaface import facial, measures, models
from dynaface.measures_frontal import AnalyzeLandmarks

models.init_models(models.download_models(), models.detect_device())
chosen = [m for m in measures.all_measures() if not isinstance(m, AnalyzeLandmarks)]
selection = measures.MeasureSelection.of(chosen)

rows = []
for path in sorted(Path("photos").glob("*.jpg")):
    face = facial.load_face_image(str(path), measures=chosen)
    if face.is_no_face():
        print(f"No face found in {path.name}")
        continue
    selection.apply(face.measures, lateral=face.lateral, quarter=face.quarter)
    results = face.analyze()
    rows.append({"file": path.name, **{n: results.get(n) for n in face.get_all_items()}})

# Each view has its own outputs, so gather every column.
columns = list(dict.fromkeys(name for row in rows for name in row))
with open("measures.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
```
