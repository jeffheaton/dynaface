# Units and scale

## The standard image

Before measuring, Dynaface scales and crops each frontal face to a standard image, 1024 by 1024 pixels, with the pupils 260 pixels apart and the subject's right pupil at (380, 480). Every face ends up the same size and in the same place, so measurements are comparable. A profile is cropped to 1024 by 1024 pixels too, scaled so the whole profile fits. A three-quarter face is cropped like a profile, scaled so the landmarks fit from top to bottom with the chin in the frame and centered, so the two mirrored turns of one face are cropped alike; it's never flipped.

Coordinates, such as those in `face.landmarks`, are pixels in the standard image, with the origin at the top left, x increasing to the right and y increasing downward.

## From pixels to millimeters

Dynaface converts pixels to millimeters by assuming the pupils are 63 mm apart, the adult average:

```text
px2mm = pupillary distance (mm) / distance between the pupils (pixels)
```

In the standard image the pupils are 260 pixels apart, so a pixel is about 0.24 mm. Distances are multiplied by `px2mm`, and areas by its square. The [position measure](../measures.md#position) reports the values used, as `pd` and `px2mm`.

Dynaface measures `px2mm` on the same landmarks it's about to measure, every time it measures. A scale taken from other landmarks, such as the face before it was cropped, would convert every distance wrongly, so it's never carried over from one set of landmarks to another.

Because the scale comes from the pupils, every millimeter value is in effect a fraction of the pupillary distance. The assumed 63 mm doesn't affect comparisons of the same person over time, as long as it stays the same, but it does affect absolute sizes and comparisons between people.

### Using the subject's pupillary distance

If you've measured the subject's pupillary distance, set it before loading the image:

```python
from dynaface import facial

facial.AnalyzeFace.pd = 60  # the subject's pupillary distance, in millimeters
face = facial.load_face_image("subject.jpg")
results = face.analyze()
```

The pupillary distance is fixed when the image is loaded, so set `pd` first. It's a class attribute, so it applies to every face loaded afterward: set it again for each subject. Each face keeps the one it was loaded with, and a face saved with `dump_state()` and restored with `load_state()` keeps the one it was analyzed with. To measure a face again with another, set it on that face (`face.pd = 58`) and call `analyze()`. Ratios and angles don't depend on it.

The pupillary distance must be from 40 to 85 mm (`const.MIN_PUPIL_DIST` and `const.MAX_PUPIL_DIST`). No person's pupils are closer or farther apart, so a value outside the range is a mistake, such as 6.3, in centimeters, for 63 mm: loading an image with `AnalyzeFace.pd` set to one raises `ValueError`. A distance set on one face, or restored with a saved state, isn't checked, so check one a user typed with [`check_pupil_distance()`][dynaface.facial.check_pupil_distance] first.

## Turned heads

The scale assumes the face is turned toward the camera. When the head turns, the pupils look closer together than they are, which distorts the scale. Keep the head facing the camera, and check the yaw from the [pose measure](../measures.md#pose).

A three-quarter face (`face.quarter`, the head turned about 45 degrees for the [Cheek Measures](../measures.md#cheek-measures)) keeps the pupil scale in `face.pix2mm`, measured on its own landmarks like any face's, but the turn foreshortens the pupillary distance by about the cosine of the yaw, so the scale over-reads by about 1.4 at 45 degrees and no millimeter is defined on it. Dynaface doesn't put that number anywhere: the [position measure](../measures.md#position) reports `px2mm` as NaN on a three-quarter face, the millimeter measures don't suit the view (`MeasureSelection.apply(..., quarter=True)` switches them off), and the Cheek Measures' three-quarter outputs are fractions of the chin-to-canthus distance, `cheek.scale`, in pixels. For millimeters, use the frontal photo of the same session.

## Profiles

Both pupils can't be seen in a profile, so Dynaface uses a fixed scale of 0.24 mm per pixel, close to a frontal image's, and reports the tip projection distances AT and NT in pixels. Their ratio and the angles have no units.

## Units of each output

| Outputs | Units |
|---|---|
| `fai`, `oce.l`, `oce.r`, `brow.d`, `id`, `oe`, `ml`, `nose.alar`, `nose.nostril`, `zy.r`, `zy.l`, `zy.w`, `cr.r.d`, `cr.l.d` | millimeters |
| `eye.left`, `eye.right`, `eye.diff`, `dental_area`, `dental_left`, `dental_right`, `dental_diff` | square millimeters |
| `tilt`, `pitch`, `roll`, `yaw`, `nfa`, `nla`, `hue`, `cheek.yaw`, `cheek.yaw.model`, `cheek.pitch` | degrees |
| `sat`, `brightness` | percent |
| `pd`, `at`, `nt`, `cheek.scale`, `sub.r.depth.px`, `sub.l.depth.px`, and the Landmarks measure's coordinates, the cheek landmarks' (`zy.r.x` to `sub.l.y`) among them | pixels of the standard image |
| `px2mm` | millimeters per pixel (NaN on a three-quarter face) |
| `apex.r.h`, `apex.l.h`, `apex.r.dev`, `apex.l.dev`, `sub.r.h`, `sub.l.h`, `sub.r.depth`, `sub.l.depth`, `cheek.nose.gap` | fractions of the chin-to-canthus distance, `cheek.scale` |
| `eye.ratio`, `dental_ratio`, `at_nt`, `zy.ratio`, `cheek.mask` | none |
| `zy.ratio.ok`, `zy.r.ok`, `zy.l.ok`, `zy.ok`, `cr.ok`, `sr.ok`, `apex.r.prior`, `apex.l.prior`, `apex.r.near_nose`, `apex.l.near_nose`, `apex.ok`, `sub.r.present`, `sub.l.present`, `sub.ok`, `cheek.far`, `cheek.nose`, `cheek.pitch.ok` | flags, 0 or 1 |
| `zy.r.clip`, `zy.l.clip` | codes, 0, 1 or 2 |
