# How it works

Dynaface turns a photo or a video frame into measurements in five steps. The Python and .NET libraries follow the same steps and formulas.

## 1. Find the face

[BlazeFace](https://doi.org/10.48550/arXiv.1907.05047) finds the face in the image.

## 2. Place the landmarks

[SPIGA](https://doi.org/10.48550/arXiv.2210.07233) places 98 landmarks on the face, in the WFLW layout, and estimates the head pose: its yaw, pitch and roll. SPIGA combines a convolutional network with graph attention, which models the face's overall structure. That makes it less likely than many landmark models to pull an asymmetric face's landmarks toward a symmetric one, which matters for facial paralysis.

## 3. Decide the pose

From the head pose and the landmarks, Dynaface decides whether the photo shows the face from the front (frontal), from the side (lateral) or, when `config.AUTO_QUARTER` is on (it also needs `config.AUTO_LATERAL`, the pose-detection switch, which is on by default), turned about 45 degrees (a three-quarter face, `Pose.QUARTER`, the view the [Cheek Measures](../measures.md#cheek-measures) read the far cheek in). The three-quarter decision comes first, from the turn measured on the landmarks before any crop (35 to 55 degrees), with the head pose agreeing (the same direction, and a turn of 20 to 55 degrees once calibrated, so a profile whose landmarks crowd into the window is not taken for one), and only while the nose tip stays at, or within about 3% of the face width past, the far side of the face; otherwise the profile test runs as before, and a turn in the window that the profile test declines is still a three-quarter face, its nose flag set, rather than a frontal one. On 816 photos of the Face Research Lab London set, 204 frontal (at rest and smiling), 408 three-quarter and 204 profile, detection with both switches on called every frontal photo frontal, 92% of the three-quarter photos three-quarter (29 of the 32 others read a turn under 35 degrees and went frontal) and 94% of the profiles lateral (3% read as three-quarter and 3% as frontal), with the far side right on every three-quarter photo it took. `AUTO_QUARTER` ships off as a library default so that a default changes no caller's video imports: with it on, a frame whose turn lands in the window takes the three-quarter crop, pays a background-model pass and loses its millimeter measures. The Dynaface desktop app offers it as its "Detect three-quarter view" preference, and a caller that smooths landmarks across frames must restart its smoothing when the view changes, as the library's `VideoAnalyzer` and the desktop app's import do; unless you turn it on, a turned head is frontal or, past about 50 degrees, lateral, as in earlier versions. You can also tell Dynaface which view to use. A forced three-quarter view now takes the three-quarter path below; earlier versions cropped it like a frontal face.

## 4. Crop to a standard size

- **Frontal:** the face is scaled and cropped to 1024 by 1024 pixels with the pupils at fixed positions, 260 pixels apart. Every face ends up the same size and in the same place, so measurements are comparable.
- **Lateral:** the face is turned to face left and cropped to 1024 by 1024 pixels. [U²-Net](https://doi.org/10.1016/j.patcog.2020.107404) removes the background, and Dynaface follows the outline of the profile to find six landmarks on it: the soft tissue glabella, the soft tissue nasion, the nasal tip, the subnasal point, the mentolabial point and the soft tissue pogonion.
- **Three-quarter:** the face is cropped like a profile, to 1024 by 1024 pixels with the landmarks fitting from top to bottom and centered, and never flipped, so the two mirrored turns of one face come out alike and the chin stays in the frame. U²-Net removes the background, and Dynaface traces the outline of the far cheek, the one away from the camera, from the eye corner to the chin, for the cheek landmarks.

## 5. Measure

Each [measure](../measures.md) calculates its values from the landmarks and draws them on the image. Distances are converted from pixels to millimeters by assuming the pupils are 63 mm apart, the average for adults. If you know the person's actual pupillary distance, set it before loading the image for more accurate millimeters (see [Units and scale](units.md)):

```python
from dynaface.facial import AnalyzeFace

AnalyzeFace.pd = 60  # this person's pupillary distance, in millimeters
```

Frontal distances are in millimeters, areas in square millimeters and angles in degrees. The lateral tip projection distances are in pixels. A three-quarter face's cheek heights and depths are fractions of its chin-to-eye-corner distance, and no millimeter is reported for it, because the turn foreshortens the pupils (see [Turned heads](units.md#turned-heads)).

## Videos

A video is analyzed one frame at a time. Each frame's analysis can be kept with `AnalyzeFace.dump_state()` and restored with `load_state()`, so a video can be analyzed once and its measures recalculated later. [Analyzing video](video.md) shows how, and the [examples](../python/examples.md) include a script that measures every frame of a video.
