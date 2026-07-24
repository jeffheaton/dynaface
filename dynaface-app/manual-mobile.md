# Dynaface User Guide

Dynaface is an iOS app that uses on-device artificial intelligence to detect a face in a photo and measure aspects of its symmetry, such as eye spacing, mouth width, brow height, and more. You can take a new photo with your camera or import one from your Photo Library, and Dynaface will automatically find the face, align it, and display a set of measurements you can turn on or off.

All photos and measurements stay on your device — nothing is uploaded anywhere. For questions about privacy and how Dynaface handles your data, see the [Dynaface FAQ](https://github.com/jeffheaton/dynaface/blob/main/FAQ.md).

---

## Getting Started

When you open Dynaface, you'll see your **Photo Library** — a grid of every photo you've previously captured or imported. The first time you launch the app, this screen will be empty.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/manual/2.0/dynaface-catalog.jpg" width="512">

Two buttons sit at the bottom of the screen:

- **Camera** — Opens the live camera view so you can take a new photo.
- **Import** — Opens your device's Photo Library so you can choose an existing picture.

## Taking a Photo

Tapping the Camera button opens a live preview from your device's camera.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/manual/2.0/dynaface-camera.jpg" width="512">

From this screen you can:

- **Capture** (center button) — Takes a photo and sends it to Dynaface for analysis.
- **Flip Camera** (right button) — Switches between the front and back camera.
- **Import** (left button) — Lets you pick a photo from your library instead.
- **Back** (top-left arrow) — Returns to the Photo Library.
- **Pinch to zoom** — Pinch with two fingers on the preview to zoom in before capturing.

The first time you use the camera, Dynaface will ask for permission to access it. If you decline, you can still import photos from your library, or re-enable camera access later from your device's Settings.

For best results, look directly at the camera, keep your face centered in the frame, and try to keep the camera steady.

## Importing a Photo

Tapping **Import** (from either the Photo Library screen or the camera screen) opens your device's photo picker. Choose any photo containing a face, and Dynaface will analyze it the same way it would a freshly captured photo.

## Automatic Face Detection

After you capture or import a photo, Dynaface automatically locates the face, straightens it, and crops it to a consistent size before measuring it — you don't need to do any manual cropping or alignment.

If Dynaface can't find a clear, forward-facing face in the photo, it will show a **Face Not Found** message with the option to try again.

## Viewing Your Measurements

Once a face is detected, Dynaface displays the photo along with a panel of measurements below or beside it.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/manual/2.0/dynaface-measure.jpg" width="512">

Along the bottom of this screen are five buttons:

- **Share/Export** — Opens export options for the photo and its measurements (see below).
- **Measurement Settings** (chart icon) — Opens a list of every available measurement so you can turn individual ones on or off. Your choices are remembered for that photo.
- **Landmarks Toggle** (labeled OFF/LM) — Shows or hides the underlying facial reference points the AI used to calculate the measurements.
- **Recalculate** (circular arrows) — Re-runs the measurement analysis on the photo, useful if you've changed which measurements are enabled or want a fresh calculation.
- **Delete** (trash icon) — Removes the photo from your library, after a confirmation prompt.

The **Back** arrow in the top-left returns you to the Photo Library (or the camera, if you just took the photo).

## Choosing Which Measurements to Show

Tap the measurement-settings button to see the full list of available measurements, each with its own on/off switch:

- **Position** — Head tilt, pupil distance, and the pixel-to-millimeter scale used for other measurements.
- **Pose** — Estimated pitch, roll, and yaw of the head in the photo.
- **Intercanthal Distance** — The distance between the inner corners of the eyes.
- **Mouth Length** — The width of the mouth.
- **Outer Eye Corners** — The distance between the outer corners of the eyes.
- **FAI (Facial Asymmetry Index)** — A single score that summarizes how closely the left and right sides of the face match one another, based on published facial-symmetry research.
- **Oral CE** — Compares the position of the left and right corners of the mouth.
- **Brow** — Compares eyebrow height between the left and right sides.
- **Eye Area** — Compares the visible eye-opening area between the left and right sides.
- **Dental Display** — Compares how much of the teeth are visible on the left and right sides of a smile.
- **Nose Frontal** — Measurements describing the nose as seen from the front.

Turning a measurement on adds it to both the on-screen overlay and the measurement panel; turning it off hides it. These settings are saved per photo, so returning to an earlier photo later will show the same selections you left it with.

## Facial Landmarks

Tapping the landmarks toggle switches between:

- **OFF** — Just the photo and your selected measurements.
- **LM** — The photo with the underlying reference points (used to calculate every measurement) drawn on top, so you can see exactly what the AI detected.

## Recalculating

If you'd like Dynaface to reprocess a photo — for example after enabling a measurement you'd previously left off — tap the **Recalculate** button. This re-runs face detection and measurement on the already-cropped photo.

## Exporting Your Results

Tap the **Share/Export** button to see your export options:

- **Image to Clipboard** — Copies the annotated photo (with visible measurements/landmarks) so you can paste it into another app.
- **Image to Photos** — Saves the annotated photo to your device's Photos app.
- **Text to Clipboard** — Copies the measurement values as plain text so you can paste them into notes, messages, or a spreadsheet.

## Managing Your Photo Library

Every photo you capture or import is saved to your on-device Photo Library, shown as a scrolling grid with the date each photo was taken.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/manual/2.0/dynaface-catalog-1.jpg" width="512">

Tap any thumbnail to reopen it and view or adjust its measurements. To remove a photo permanently, open it and tap the **Delete** button, then confirm.

---

*Dynaface is provided for educational and informational purposes only under the Apache 2.0 License. It is not a registered medical device and is not intended to diagnose, treat, cure, or prevent any disease or medical condition.*
