# Taking photos and videos

Dynaface measures whatever is in the image, so the way a photo is taken affects every measurement. These guidelines keep measurements comparable between people and between visits.

## Frontal photos

- **Face the camera directly,** with the head level and the eyes looking into the lens. A turned or tilted head changes distances and areas. The [pose](../measures.md#pose) and [position](../measures.md#position) measures report how far the head is turned and tilted.
- **Center the face** and keep the camera at the same distance and height from one session to the next. Dynaface scales every face to the same size, but perspective still changes with distance.
- **Use even, front-facing light.** A shadow across one side of the face can move landmarks, and lighting changes the skin tone.
- **Keep the brows, eyes and mouth in view.** Hair, hands or a thick glasses frame over them can hide the landmarks the measures use.
- **Use enough resolution.** Dynaface resizes each face so the pupils are 260 pixels apart. A face much smaller than that is enlarged and loses detail, so let the face fill a good part of the frame.
- **Don't mirror the image.** In an ordinary photo, the subject's right side is on the left of the image, which is what Dynaface expects. Some phones save front-camera selfies mirrored, which swaps left and right in every measurement.

## Expressions

Capture the expressions your work needs, such as rest, a full smile, raised brows and gently closed eyes, and capture them the same way each time. Some measures are most informative as a comparison, such as oral commissure excursion and dental display at rest and during a smile.

## Profile photos

- Photograph the side of the face with the head level and the whole profile, from forehead to chin, in view. The [lateral measures](../measures.md#lateral) are taken relative to the image's horizontal and vertical axes, so a tilted head changes them.
- A plain background helps. Dynaface removes the background and traces the outline of the face, so hair or clothing over the forehead, nose, lips or chin can shift the points it finds.
- Either side works. Dynaface flips a profile that faces right.

## Three-quarter photos

The [Cheek Measures](../measures.md#cheek-measures) observe the malar apex and the submalar point on the outline of the far cheek, which only a turned head shows. By default Dynaface treats a turned head as frontal or, past about 50 degrees, as a profile (three-quarter detection ships off as a library default, so that a default changes no caller's video imports; the desktop app offers it as its "Detect three-quarter view" preference and resets its landmark smoothing when a video's view changes; see [How it works](how-it-works.md#3-decide-the-pose)): turn it on with `config.AUTO_QUARTER = True` before loading the image, or force the view with `force_pose=Pose.QUARTER` (see [Working with results](results.md#frontal-photos-three-quarter-photos-and-profiles)).

- **Turn 45 degrees, with the whole body.** Rotate the subject on a chair or floor marks rather than turning the head on the neck, which tilts and pitches it; marks reproduce the turn to about 2 degrees. The outline point moves with the turn, so the measure reports the yaw it measured as `cheek.yaw` and accepts a turn within 5 degrees of 45 in `apex.ok`. Told "three-quarter" with no mark, a study's subjects turned anywhere from 23 to 56 degrees (mean 44, sd 5), and only two thirds landed inside the 40-50 degree window; a floor mark or a turntable is what puts a photo inside it.
- **Take one photo per side, turned the same amount each way.** Each photo measures one cheek, the far one: the photo taken from the subject's right measures the left cheek. Match the two turns to within about 3 degrees.
- **Keep the head level,** as for a profile. Pitch is reported as `cheek.pitch` but isn't calibrated yet.
- **Don't line up the nose tip with the far cheek.** That common alignment turns the head 47 to 50 degrees, where the nose joins the outline (`cheek.nose` is 1) and Dynaface may see a profile.
- **Keep the far cheek clear of hair and the chin in the frame.** Dynaface removes the background and traces the far cheek's edge from the eye corner to the chin. Hair against the cheek has no edge in the mask, so those rows fall back to the landmarks, or to an intensity edge beside them, and don't count as traced (`cheek.mask` reports the fraction that were). A plain background helps. Even with the hair back and a plain background the background model occasionally drops the far cheek or eats into it (9 of 408 studio photos in the calibration set, 7 dropped and 2 eroded, on six subjects, one of them on three of four turns): the outline then draws yellow where an intensity edge stood in, or jagged cyan inside the skin, and `cheek.mask` reads low (0 to 0.7), so look at the drawn image and retake rather than keep the points.
- **Rest expression.** The points are defined at rest; nothing is validated on a smile.
- **Check `cheek.yaw` and `apex.ok`** before using a value.

## Videos

- The guidelines for photos apply to every frame. Keep the camera still, ideally on a tripod, and keep the face in the frame: a frame without a face has no measurements.
- Every frame goes through the AI models, so analysis takes much longer than the video plays. Trim videos to the parts you need before analyzing them.
- A three-quarter video runs the background model on every frame, as a profile video does, so it takes longer still. Videos are best imported with a forced pose, as the Dynaface mobile app does, so every frame is cropped the same way; with detection, a view can change partway through a video, and the library's `VideoAnalyzer` restarts its landmark smoothing at each change, as the desktop app's import does.

[Analyzing video](video.md) shows the code.

## Check the results

Dynaface is a research tool, not a medical device. Image quality, lighting, pose, expression and camera angle all affect its measurements. Before relying on a result, look at the image with the measures drawn on it and check that the landmarks sit where they should.

## Patient privacy

Photos and videos of patients identify them. Store and share them under your institution's privacy rules, and don't post them in public issues or examples.
