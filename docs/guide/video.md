# Analyzing video

Dynaface analyzes a video one frame at a time, each frame just like a photo. Read the frames with OpenCV, convert each one from BGR to RGB, and give it to [`load_image()`][dynaface.facial.AnalyzeFace.load_image]:

```python
import cv2
from dynaface import facial, measures, models
from dynaface.measures_frontal import AnalyzeLandmarks

models.init_models(models.download_models(), models.detect_device())
chosen = [
    m
    for m in measures.all_measures()
    if m.is_frontal and not isinstance(m, AnalyzeLandmarks)
]

video = cv2.VideoCapture("smile.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
rows = []
while True:
    ok, frame = video.read()
    if not ok:
        break
    face = facial.AnalyzeFace(measures=chosen)
    face.load_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), crop=True)
    row = {"frame": len(rows), "time": len(rows) / fps}
    results = face.analyze()
    if results is not None:  # None for a frame without a face
        row.update({name: results.get(name) for name in face.get_all_items()})
    rows.append(row)
video.release()
```

A frame without a face, such as one where the subject turns away, gets a row with only its frame number and time. Write the rows to a CSV file as in [Measuring a folder of photos](results.md#measuring-a-folder-of-photos), or load them into pandas with `pandas.DataFrame(rows)`.

## Finding a moment

The rows make it easy to find a moment in the video, such as the peak of a smile, where the dental display is largest, and to look at the other measures at that moment:

```python
with_face = [row for row in rows if "dental_area" in row]
smile = max(with_face, key=lambda row: row["dental_area"])
print(f"Widest smile at {smile['time']:.2f} s, eye.ratio = {smile['eye.ratio']:.2f}")
```

Comparing the eyes at the widest smile with the mouth at the widest eye opening is the kind of comparison used in studies of oral-ocular synkinesis. The Evaluation report in the Dynaface apps makes it for you.

## Keeping the analysis

Finding the face and its landmarks is the slow part. [`dump_state()`][dynaface.facial.AnalyzeFace.dump_state] captures a frame's analysis, including its standard image, landmarks and scale, and [`load_state()`][dynaface.facial.AnalyzeFace.load_state] restores it without running the models again. To keep every frame's state, create a list before the loop, `states = []`, and add `states.append(face.dump_state())` after `load_image()`. You can then calculate other measures later:

```python
for state in states:
    face = facial.AnalyzeFace(measures=[measures.AnalyzeEyeArea()])
    face.load_state(state)
    results = face.analyze()
```

Restore each state into a new `AnalyzeFace`, as above, so that every frame is drawn on its own image. Each state holds its frame's 1024 by 1024 image, about 3 MB, so the states of a long video take a lot of memory. The desktop app keeps them in its `.dyfc` project files.

## Speed

Every frame goes through the face detector and the landmark model, so analyzing a video takes much longer than playing it, especially on a CPU. Trim videos to the parts you need, and use a GPU if you have one (see [Models and devices](models.md#choosing-a-device)).

## The example script

[`examples/process_media.py`](https://github.com/jeffheaton/dynaface/blob/main/dynaface-lib-python/examples/process_media.py) measures every frame of a video and writes an annotated copy of the video, a chart of each measure over time and a CSV file of the values. The [examples](../python/examples.md) page shows how to run it.
