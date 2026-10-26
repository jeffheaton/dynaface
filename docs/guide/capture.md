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

## Videos

- The guidelines for photos apply to every frame. Keep the camera still, ideally on a tripod, and keep the face in the frame: a frame without a face has no measurements.
- Every frame goes through the AI models, so analysis takes much longer than the video plays. Trim videos to the parts you need before analyzing them.

[Analyzing video](video.md) shows the code.

## Check the results

Dynaface is a research tool, not a medical device. Image quality, lighting, pose, expression and camera angle all affect its measurements. Before relying on a result, look at the image with the measures drawn on it and check that the landmarks sit where they should.

## Patient privacy

Photos and videos of patients identify them. Store and share them under your institution's privacy rules, and don't post them in public issues or examples.
