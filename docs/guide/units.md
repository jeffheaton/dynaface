# Units and scale

## The standard image

Before measuring, Dynaface scales and crops each frontal face to a standard image, 1024 by 1024 pixels, with the pupils 260 pixels apart and the subject's right pupil at (380, 480). Every face ends up the same size and in the same place, so measurements are comparable. A profile is cropped to 1024 by 1024 pixels too, scaled so the whole profile fits.

Coordinates, such as those in `face.landmarks`, are pixels in the standard image, with the origin at the top left, x increasing to the right and y increasing downward.

## From pixels to millimeters

Dynaface converts pixels to millimeters by assuming the pupils are 63 mm apart, the adult average:

```text
px2mm = pupillary distance (mm) / distance between the pupils (pixels)
```

In the standard image the pupils are 260 pixels apart, so a pixel is about 0.24 mm. Distances are multiplied by `px2mm`, and areas by its square. The [position measure](../measures.md#position) reports the values used, as `pd` and `px2mm`.

Because the scale comes from the pupils, every millimeter value is in effect a fraction of the pupillary distance. The assumed 63 mm doesn't affect comparisons of the same person over time, as long as it stays the same, but it does affect absolute sizes and comparisons between people.

### Using the subject's pupillary distance

If you've measured the subject's pupillary distance, set it before loading the image:

```python
from dynaface import facial

facial.AnalyzeFace.pd = 60  # the subject's pupillary distance, in millimeters
face = facial.load_face_image("subject.jpg")
results = face.analyze()
```

The scale is fixed when the image is loaded, so set `pd` first. It's a class attribute, so it applies to every face loaded afterward: set it again for each subject. Ratios and angles don't depend on it.

## Turned heads

The scale assumes the face is turned toward the camera. When the head turns, the pupils look closer together than they are, which distorts the scale. Keep the head facing the camera, and check the yaw from the [pose measure](../measures.md#pose).

## Profiles

Both pupils can't be seen in a profile, so Dynaface uses a fixed scale of 0.24 mm per pixel, close to a frontal image's, and reports the tip projection distances AT and NT in pixels. Their ratio and the angles have no units.

## Units of each output

| Outputs | Units |
|---|---|
| `fai`, `oce.l`, `oce.r`, `brow.d`, `id`, `oe`, `ml`, `nose.alar`, `nose.nostril` | millimeters |
| `eye.left`, `eye.right`, `eye.diff`, `dental_area`, `dental_left`, `dental_right`, `dental_diff` | square millimeters |
| `tilt`, `pitch`, `roll`, `yaw`, `nfa`, `nla`, `hue` | degrees |
| `sat`, `brightness` | percent |
| `pd`, `at`, `nt`, and the landmark coordinates | pixels of the standard image |
| `px2mm` | millimeters per pixel |
| `eye.ratio`, `dental_ratio`, `at_nt` | none |
