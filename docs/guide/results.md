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

`load_face_image()` and `AnalyzeFace` take the list of measures to run. `measures.all_measures()` includes [Landmarks](../measures.md#landmarks), which adds 196 landmark coordinates to the results, so leave it out if you don't need them:

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

Each output has an item that controls whether it's drawn. Switching an item off doesn't remove its value from the results, but [`get_all_items()`][dynaface.facial.AnalyzeFace.get_all_items] lists only the outputs whose items are on, which is how the apps choose their chart and CSV columns:

```python
eyes = measures.AnalyzeEyeArea()
eyes.set_item_enabled("eye.diff", False)  # don't draw or list the difference

face = facial.load_face_image("subject.jpg", measures=[eyes])
results = face.analyze()  # still includes eye.diff
print(face.get_all_items())  # ['eye.left', 'eye.right', 'eye.ratio']
```

## Frontal photos and profiles

When it loads an image, Dynaface decides whether it shows the face from the front or in profile, and sets `face.lateral` to True for a profile. `analyze()` runs every measure you gave it either way, and frontal measures don't mean anything on a profile. To keep only the outputs that fit the view, switch the others off before analyzing:

```python
face = facial.load_face_image("photo.jpg")
for measure in face.measures:
    measure.update_for_type(face.lateral)

results = face.analyze()
row = {name: results.get(name) for name in face.get_all_items()}
```

If Dynaface gets the view wrong, give it the view yourself when you load the image:

```python
import cv2
from dynaface.const import Pose

image = cv2.cvtColor(cv2.imread("profile.jpg"), cv2.COLOR_BGR2RGB)  # OpenCV loads BGR
face = facial.AnalyzeFace(measures=measures.all_measures())
face.load_image(image, crop=True, force_pose=Pose.LATERAL)
```

## The image

`face.render_img` is the standard image with the measures drawn on it, as an RGB NumPy array, and `face.save("measured.jpg")` writes it to a file. `face.original_img` is the same image without the drawings.

## Measuring a folder of photos

This measures every photo in a folder and writes one row per photo to a CSV file:

```python
import csv
from pathlib import Path

from dynaface import facial, measures, models
from dynaface.measures_frontal import AnalyzeLandmarks

models.init_models(models.download_models(), models.detect_device())
chosen = [m for m in measures.all_measures() if not isinstance(m, AnalyzeLandmarks)]

rows = []
for path in sorted(Path("photos").glob("*.jpg")):
    face = facial.load_face_image(str(path), measures=chosen)
    if face.is_no_face():
        print(f"No face found in {path.name}")
        continue
    for measure in face.measures:
        measure.update_for_type(face.lateral)
    results = face.analyze()
    rows.append({"file": path.name, **{n: results.get(n) for n in face.get_all_items()}})

# Frontal photos and profiles have different outputs, so gather every column.
columns = list(dict.fromkeys(name for row in rows for name in row))
with open("measures.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
```
