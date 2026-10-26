# How it works

Dynaface turns a photo or a video frame into measurements in five steps. The Python and .NET libraries follow the same steps and formulas.

## 1. Find the face

[BlazeFace](https://doi.org/10.48550/arXiv.1907.05047) finds the face in the image.

## 2. Place the landmarks

[SPIGA](https://doi.org/10.48550/arXiv.2210.07233) places 98 landmarks on the face, in the WFLW layout, and estimates the head pose: its yaw, pitch and roll. SPIGA combines a convolutional network with graph attention, which models the face's overall structure. That makes it less likely than many landmark models to pull an asymmetric face's landmarks toward a symmetric one, which matters for facial paralysis.

## 3. Decide the pose

From the head pose and the landmarks, Dynaface decides whether the photo shows the face from the front (frontal) or from the side (lateral). You can also tell it which to use.

## 4. Crop to a standard size

- **Frontal:** the face is scaled and cropped to 1024 by 1024 pixels with the pupils at fixed positions, 260 pixels apart. Every face ends up the same size and in the same place, so measurements are comparable.
- **Lateral:** the face is turned to face left and cropped to 1024 by 1024 pixels. [U²-Net](https://doi.org/10.1016/j.patcog.2020.107404) removes the background, and Dynaface follows the outline of the profile to find six landmarks on it: the soft tissue glabella, the soft tissue nasion, the nasal tip, the subnasal point, the mentolabial point and the soft tissue pogonion.

## 5. Measure

Each [measure](measures.md) calculates its values from the landmarks and draws them on the image. Distances are converted from pixels to millimeters by assuming the pupils are 63 mm apart, the average for adults. If you know the person's actual pupillary distance, set it before analyzing for more accurate millimeters:

```python
from dynaface.facial import AnalyzeFace

AnalyzeFace.pd = 60  # this person's pupillary distance, in millimeters
```

Frontal distances are in millimeters, areas in square millimeters and angles in degrees. The lateral tip projection distances are in pixels.

## Videos

A video is analyzed one frame at a time. Each frame's analysis can be kept with `AnalyzeFace.dump_state()` and restored with `load_state()`, so a video can be analyzed once and its measures recalculated later. The [examples](python/examples.md) include a script that measures every frame of a video.
