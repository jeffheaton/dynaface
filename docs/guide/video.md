# Analyzing video

Dynaface analyzes a video one frame at a time, each frame just like a photo. Read the frames with OpenCV, convert each one from BGR to RGB, and give it to [`load_image()`][dynaface.facial.AnalyzeFace.load_image]. A [`FrameSeries`][dynaface.video.FrameSeries] keeps each frame's measurements:

```python
import cv2
from dynaface import facial, measures, models
from dynaface.measures_frontal import AnalyzeLandmarks
from dynaface.video import FrameSeries

models.init_models(models.download_models(), models.detect_device())
chosen = [
    m
    for m in measures.all_measures()
    if m.is_frontal and not isinstance(m, AnalyzeLandmarks)
]

video = cv2.VideoCapture("smile.mp4")
series = FrameSeries(chosen, fps=video.get(cv2.CAP_PROP_FPS))
while True:
    ok, frame = video.read()
    if not ok:
        break
    face = facial.AnalyzeFace(measures=chosen)
    face.load_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), crop=True)
    series.add(len(series), face.analyze(render=False))  # None for a frame without a face
video.release()

series.write_csv("smile.csv")
```

The series has a row for every frame and a column for every output. A frame without a face, such as one where the subject turns away, gets a row whose values are NaN, and the CSV file leaves them empty. `series.values` holds each output's values in frame order, and `pandas.DataFrame(series.rows())` loads them into pandas. `render=False` skips drawing the measures, which a series doesn't need.

## Finding a moment

The series makes it easy to find a moment in the video, such as the peak of a smile, where the dental display is largest, and to look at the other measures at that moment:

```python
smile = series.peak("dental_area")
row = series.rows()[smile]  # the frames are numbered from 0, so this is its row too
print(f"Widest smile at {row['time']:.2f} s, eye.ratio = {row['eye.ratio']:.2f}")
```

`peak()` adds up the outputs you name, so `series.peak("eye.left", "eye.right")` finds the widest eye opening. Comparing the eyes at the widest smile with the mouth at the widest eye opening is the kind of comparison used in studies of oral-ocular synkinesis. [`evaluate()`][dynaface.video.FrameSeries.evaluate] makes it, and the Evaluation report in the Dynaface apps shows it:

```python
evaluation = series.evaluate()
smile, eyes = evaluation.widest_smile, evaluation.widest_eyes
print(f"Widest smile: eyes {smile.eye_left:.1f} and {smile.eye_right:.1f} mm²")
print(f"Widest eye opening: dental display {eyes.dental_area:.1f} mm²")
```

## Keeping the analysis

Finding the face and its landmarks is the slow part. [`dump_state()`][dynaface.facial.AnalyzeFace.dump_state] captures a frame's analysis, including its standard image, landmarks, head pose and scale, and [`load_state()`][dynaface.facial.AnalyzeFace.load_state] restores it without running the models again. To keep every frame's state, create a list before the loop, `states = []`, and add `states.append(face.dump_state())` after `load_image()`. You can then calculate other measures later, for every frame at once or for a range of them:

```python
series = FrameSeries.from_states(states, [measures.AnalyzeEyeArea()], fps=30)
```

Each row keeps its state's position in the list as its frame number, so measuring a range, such as `start=100, stop=200`, numbers its frames as they are in the whole video. To work with one frame, restore it yourself:

```python
face = facial.AnalyzeFace(measures=[measures.AnalyzeEyeArea()])
face.load_state(states[0])
results = face.analyze()
```

`load_state()` restores the frame's image too, so `analyze()` draws on that frame. In Dynaface 2.0.3 and earlier it didn't, so call `face.render_reset()` after `load_state()` to draw on the restored frame.

A restored frame is measured with the pupillary distance it was analyzed with, and its scale is measured again on its landmarks, so states saved by versions up to 2.0.2, which measured the scale before the crop, come out right. States saved by Dynaface 2.0.4 and earlier didn't keep the head pose, though: they restore with a yaw, pitch and roll of 0.

Each state holds its frame's 1024 by 1024 image, about 3 MB, so the states of a long video take a lot of memory. The desktop app keeps them in its `.dyfc` project files.

## Speed

Every frame goes through the face detector and the landmark model, so analyzing a video takes much longer than playing it, especially on a CPU. Trim videos to the parts you need, and use a GPU if you have one (see [Models and devices](models.md#choosing-a-device)).

## The example script

[`examples/process_media.py`](https://github.com/jeffheaton/dynaface/blob/main/dynaface-lib-python/examples/process_media.py) measures every frame of a video and writes an annotated copy of the video, a chart of each measure over time and a CSV file of the values. The [examples](../python/examples.md) page shows how to run it.
