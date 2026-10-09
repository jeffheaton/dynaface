# Dynaface User Guide — iPhone, iPad & Android

Dynaface is an app for iPhone, iPad, and Android that uses on-device artificial intelligence to detect a face in a photo, or in every frame of a video, and measure aspects of its symmetry, such as eye spacing, mouth width, brow height, smile, and more. Take a new photo with your camera or import a photo or video from your library, and Dynaface will automatically find the face, align it, and display a set of measurements you can turn on or off. For videos, Dynaface also charts how each measurement changes over time and can export the results as a spreadsheet or, on iPhone and iPad, as an annotated video.

All photos, videos, and measurements stay on your device — nothing is uploaded anywhere. For questions about privacy and how Dynaface handles your data, see the [Dynaface FAQ](https://github.com/jeffheaton/dynaface/blob/main/FAQ.md).

> **Platform differences** are called out throughout this guide as **iPhone & iPad** or **Android**. Using Dynaface on a computer instead? See the [Windows & macOS guide](https://github.com/jeffheaton/dynaface/blob/main/dynaface-app/manual-app.md).

**What you need:**

- **iPhone & iPad** — iOS or iPadOS 15 or later.
- **Android** — Android 8.0 or later on a 64-bit device. Importing videos requires Android 9 or later; photos work on every supported version.

---

## Getting Started

When you open Dynaface, you'll see your **Photo Library** — a grid of every photo and video you've previously analyzed, newest first, each labeled with the date it was added. The first time you launch the app, this screen will be empty.

<!-- SCREENSHOT 1 — Empty Photo Library (retake: the current screen also has the ☰ MENU button in the top-right corner). Portrait iPhone, empty grid with the "No photos yet" message, Camera and Import buttons along the bottom. -->
<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/manual/2.0/dynaface-catalog.jpg" width="512">

Two buttons sit at the bottom of the screen:

- **Camera** (camera icon) — Opens the live camera view so you can take a new photo.
- **Import** (photo-library icon) — Opens your device's photo picker so you can choose an existing photo or video.

The **☰ menu** button in the top-right corner opens a short menu:

- **Settings** — App settings, such as pupillary distance (see [Settings](#settings)).
- **Dynaface Manual** — Opens the online user guide in your browser.
- **About** — Shows the app's version number. Tap anywhere to close it.
- **Close** — Closes the menu.

## Taking a Photo

Tapping the **Camera** button opens a live preview from your device's camera.

<!-- SCREENSHOT 2 — Camera screen (retake: the current camera screen has no Import button). Live preview with the Capture button bottom-center, Flip Camera bottom-right, and the Back arrow top-left. -->
<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/manual/2.0/dynaface-camera.jpg" width="512">

From this screen you can:

- **Capture** (center button) — Takes a photo and sends it to Dynaface for analysis.
- **Flip Camera** (right button) — Switches between the front and back cameras.
- **Back** (top-left arrow) — Returns to the Photo Library.

The in-app camera takes **still photos only**. To analyze a video, record it with your device's own camera app, then bring it in with **Import** (see [Working with Videos](#working-with-videos)).

The first time you use the camera, Dynaface asks for permission to access it. If you decline, you can still import photos and videos from your library. To turn camera access back on later:

- **iPhone & iPad** — Tap **Open Settings** on the camera screen, or allow camera access for Dynaface in the Settings app.
- **Android** — Open your phone's Settings app and allow the camera permission for Dynaface in its app settings.

For best results, look directly at the camera, keep your face centered in the frame, use even, front-facing light, and hold the device steady.

## Importing Photos and Videos

Tap **Import** on the Photo Library screen to open your device's photo picker, then choose a photo or video containing a face. Dynaface analyzes it the same way it would a freshly captured photo.

- **iPhone & iPad** — Import opens the system photo picker, showing both photos and videos.
- **Android** — On Android 11 and later, Import opens Android's photo picker, which also shows items stored in Google Photos; anything stored only in the cloud is downloaded when you select it. Older Android versions show the standard picker instead. Videos can be imported on Android 9 or later.

Dynaface reads videos in **MP4**, **MOV**, and **M4V** format — the formats phone cameras normally record.

While a photo is loading, Dynaface shows its progress ("Loading image...", "Analyzing...") with a **Cancel** button. Sideways (landscape) photos also get a quick "Detecting orientation..." step, which helps Dynaface find a face that isn't upright.

## Automatic Face Detection

After you capture or import a photo, Dynaface automatically locates the face and crops it to a consistent size before measuring it — you don't need to do any manual cropping or alignment.

If Dynaface can't find a clear face, it shows a **Face Not Found** screen. For an imported photo, the screen shows the image Dynaface examined; tap the **Back** arrow to return to the library and try a different photo. For a camera capture, a **Try Again** button takes you straight back to the camera.

If a file can't be opened at all — for example, a video that uses an unsupported codec or is damaged — you'll see an **Import Failed** message instead.

## Viewing Your Measurements

Once a face is detected, Dynaface displays the analyzed photo with the enabled measurements drawn on it, along with a panel listing each measurement's values. Hold your device upright to see the photo on top and the values below it, or turn it sideways to see the photo on the left and the values on the right.

<!-- SCREENSHOT 3 — Photo detail view (retake: the button row now has six buttons — Export, Measurements, OFF/LM, green Rotate arrow, red Delete, and the ☰ Photo Tools button at the far right). Portrait iPhone, a frontal photo with FAI and Eye Area drawn on it and the values panel below. -->
<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/manual/2.0/dynaface-measure.jpg" width="512">

Along the bottom of the screen is a row of buttons:

- **Export** (blue, share icon) — Opens export options for the photo and its measurements (see [Exporting Your Results](#exporting-your-results)).
- **Measurements** (chart icon) — Opens the measurements panel so you can choose which measurements to show. Your choices are remembered for that photo.
- **Landmarks Toggle** (labeled OFF/LM) — Shows or hides the facial reference points the AI used to calculate the measurements.
- **Rotate** (green circular arrows) — Turns the photo 90° clockwise and re-analyzes it (see [Rotating a Photo](#rotating-a-photo)).
- **Delete** (red, trash icon) — Removes the entry from your library, after a confirmation prompt.
- **Photo Tools** (☰, far right) — A small menu with **Recalculate** and **Rotate (90 CW)**.

**Rotate** and **Photo Tools** apply to photos only, so they don't appear when you're viewing a video — the row simply has fewer buttons.

The **Back** arrow in the top-left returns you to the Photo Library. On **Android**, the system Back gesture or button does the same thing; on the Photo Library screen, it sends Dynaface to the background.

### Zooming and Panning

- **Pinch** with two fingers on the photo to zoom in (up to 4×) and back out.
- **Drag** with one finger to move around while zoomed in.

Each photo opens fully zoomed out.

### Facial Landmarks

Tapping the landmarks toggle switches between:

- **OFF** — Just the photo and your selected measurements.
- **LM** — The photo with the underlying reference points (used to calculate every measurement) drawn on top, so you can see exactly what the AI detected.

The toggle starts at **OFF** each time you open a photo or video. Exported images and videos include the landmarks whenever they're showing.

## Choosing Which Measurements to Show

Tap the **Measurements** button to open the **MEASUREMENTS** panel. It lists every available measurement with its own on/off switch, plus **All** and **None** buttons at the top.

<!-- SCREENSHOT 4 — Measurements panel open on a photo. Portrait iPhone; show All/None, several measure toggles with EYE AREA enabled so its indented sub-items (eye.left, eye.right, eye.diff, eye.ratio) are visible; if it fits, scroll to show the POSE and TEXT SIZE sections and the Done button. -->

The measurements, as they appear in the panel:

- **Position** (POSITION) — Head tilt, pupil distance, and the pixel-to-millimeter scale used for other measurements.
- **Pose** (POSE) — Estimated pitch, roll, and yaw of the head in the photo.
- **Intercanthal Distance** (INTERCANTHAL) — The distance between the inner corners of the eyes.
- **Mouth Length** (MOUTH LENGTH) — The width of the mouth.
- **Outer Eye Corners** (OUTER EYE) — The distance between the outer corners of the eyes.
- **FAI — Facial Asymmetry Index** (FAI) — A single score that summarizes how closely the left and right sides of the face match one another, based on published facial-symmetry research.
- **Oral CE** (ORAL CE) — Compares the position of the left and right corners of the mouth.
- **Brow** (BROW) — Compares eyebrow height between the left and right sides.
- **Eye Area** (EYE AREA) — Compares the visible eye-opening area between the left and right sides.
- **Dental Display** (DENTAL) — Compares how much of the teeth are visible on the left and right sides of a smile.
- **Nose Frontal** (NOSE FRONTAL) — Measurements describing the nose as seen from the front.
- **Skin Tone** (SKIN TONE) — Samples the color of both cheeks and shows the average hue, saturation, and brightness as a swatch in the corner of the photo.
- **Lateral** (LATERAL) — Profile measurements for side-view photos (see [Side-View (Profile) Photos](#side-view-profile-photos)).

For detailed definitions of each measurement, see [Dynaface Measurements](https://www.dynaface.org/measurements/).

New photos and videos start with just **FAI** and **Eye Area** turned on (side-view photos also start with **Lateral**). Turn on any others you'd like to see.

Some measurements are groups: turning a group on reveals indented rows beneath it, so you can pick exactly which of its values appear. Turning a measurement on adds it to both the photo overlay and the values panel; turning it off hides it. Changes apply immediately, and they're saved per entry, so returning to an earlier photo or video later shows the same selections you left it with.

Two more sections live in the same panel:

- **POSE** (photos only) — Dynaface automatically detects whether a photo is a front view or a side profile. If it guesses wrong, choose **Force Frontal** or **Force Lateral** here; **Auto Detect** restores the automatic choice. The current choice is highlighted. Picking a different option closes the panel and re-analyzes the photo; if no face can be found that way, the photo is left as it was. **Force Lateral** stays dimmed until the side-view model has finished loading.
- **TEXT SIZE** — A **−** / **+** stepper from 1 to 5 that makes the measurement labels drawn on the photo larger or smaller. You see the change right away. This setting applies to every photo and video, and to exported images and videos too.

Tap **Done**, or tap outside the panel, to close it.

## Side-View (Profile) Photos

If you capture or import a photo taken from the side, Dynaface recognizes it as a profile and switches to its profile measurements: it traces the outline of the face from forehead to chin and marks six anatomical points along it — Soft Tissue Glabella, Soft Tissue Nasion, Nasal Tip, Subnasal Point, Mento Labial Point, and Soft Tissue Pogonion. The **Lateral** measurements report the nasofrontal angle (NFA), the nasolabial angle (NLA), and nasal tip projection. For profile photos, the **Measures CSV** export contains the face outline (see below).

<!-- SCREENSHOT 5 — Side-view (profile) photo with the LATERAL measures drawn (NFA and NLA angles visible) and the values panel below. Portrait iPhone. -->

If a profile is mistaken for a front view (or the other way around), use **Force Lateral** or **Force Frontal** in the measurements panel's POSE section.

Profile analysis applies to photos only — videos are always measured as front views.

## Working with Videos

Videos are analyzed the same way as photos — just one frame at a time. Record a video with your device's camera app, then tap **Import** and choose it. Dynaface first copies the video, then works through it frame by frame, showing its progress (for example "Frame 240 of 1,800 (13%) — about 2:05 left") with a **Cancel** button. Frames where no face can be found are simply skipped. Video entries show a small camcorder badge in the corner of their library thumbnail.

> **Long videos:** Dynaface analyzes at most 5,000 frames (about 2¾ minutes of video recorded at 30 frames per second). For anything longer, a **Long Video** message offers to **Analyze First 5,000** frames. To study a specific gesture in a long recording, trim the video down to that clip first in your phone's photo or video app, then import the trimmed copy.

> **Stay in the app:** analysis runs only while Dynaface is on screen. If you switch to another app or your screen locks, the analysis pauses and continues when you return. For long videos, keep Dynaface open until the import finishes.

> **Android:** Video import requires Android 9 or later.

### Video Layouts

The video screen adapts to how you hold your device:

- **Upright (portrait)** — The current frame is on top, with the transport bar (see below) right under it and the values panel below that. Tap **Chart** on the transport bar to switch the lower panel between the values and the measurement chart.
- **Phone turned sideways** — The frame fills the left side; the values (or the chart) and the transport bar sit on the right. **Chart** switches between them.
- **Tablet in landscape** (an iPad, for example, when there's room) — The Back arrow and action buttons move into a bar across the top. The frame sits at the top-left with the values beside it, and the chart runs the full width of the screen below them, with the transport bar along the bottom. The values and the chart are shown together, so there's no Chart button.

<!-- SCREENSHOT 6 — Video detail view on iPhone, portrait. Annotated frame on top, transport bar under it (previous/play/next, frame counter, slider, ☰, Chart), with Chart switched ON so the chart with its colored lines, legend, and red frame marker fills the lower panel. Use a clip with 2–3 measurements enabled. -->

<!-- SCREENSHOT 7 — Video detail view on iPad, landscape. Top bar with Back and the action buttons, annotated frame top-left, values top-right, chart full-width below, and the transport bar along the bottom with the inline scissors/restore buttons and ☰. This is the money shot — use a clip with 2–3 measurements enabled so the chart has multiple colored lines. -->

### Playing and Scrubbing

When a video is open, a transport bar appears with:

- **Previous / Play–Pause / Next** buttons — Step one frame at a time or play the clip.
- A **slider** to scrub anywhere in the video, with a **frame counter** (for example "127 / 480").

Every measurement updates as the current frame changes, exactly as if each frame were its own photo. Playback never skips frames, so it may run slower than real time on some devices. It stops at the last frame; tap Play again to start over from the beginning.

### The Measurement Chart

The chart plots each enabled measurement value across the whole clip, one colored line per value, with a legend, the lowest and highest values marked on the left, and a red marker showing the current frame. Scrub the slider to move the marker; the chart makes it easy to spot the moment a smile peaks or an eye closes. (Skin Tone isn't included in the chart or in CSV exports.)

### Trimming

You can trim the clip to the part you care about:

- **Cut Left of Frame** — Removes everything before the current frame.
- **Cut Right of Frame** — Removes everything after the current frame.
- **Restore All Frames** — Brings the whole video back.

These commands are in the **Video Tools** menu (below). On a tablet in landscape, they also appear as buttons right on the transport bar: a scissors button on each side of the slider, plus a restore button.

Trimming is never destructive — the discarded frames are only hidden, so Restore always brings back the full clip. Your trim is saved with the video. The frame counter, chart, Jump To, Evaluation, and exports cover only the kept frames. A command is dimmed when it wouldn't change anything — for example, a cut when you're already on the first or last kept frame.

### Video Tools

The **☰** button on the transport bar opens the **Video Tools** menu:

<!-- SCREENSHOT 8 — Video Tools menu open over a video on iPhone: Jump to Max Dental, Jump to Max Ocular, Evaluation, Cut Left of Frame, Cut Right of Frame, Restore All Frames (dimmed if the clip isn't trimmed), Cancel. -->

- **Jump to Max Dental** — Jumps straight to the frame with the widest smile (the most visible teeth).
- **Jump to Max Ocular** — Jumps to the frame with the widest eye opening.
- **Evaluation** — Opens a short report comparing the key moments of the clip: the eye measurements at maximum dental display, and the dental measurements at maximum eye opening, each with its frame number and time. **Copy Text** puts the report on the clipboard so you can paste it into notes or an email; **Done** closes it.
- **Cut Left of Frame**, **Cut Right of Frame**, and **Restore All Frames** — The trimming commands described above.
- **Cancel** — Closes the menu.

Jump To and Evaluation don't change which measurements you have turned on.

<!-- SCREENSHOT 9 — Evaluation report open over a video on iPhone, showing the Max Dental Display and Max Eye Opening sections with the Copy Text and Done buttons. -->

## Exporting Your Results

Tap the **Export** button to see your export options:

<!-- SCREENSHOT 10 — EXPORT panel on iPhone with a video open, so all rows show: Image to Clipboard, Image to Photos, Text to Clipboard, Measures CSV, Video with Measures, Cancel. -->

- **Image to Clipboard** (iPhone & iPad) — Copies the annotated photo (with visible measurements/landmarks) so you can paste it into another app.
- **Image to Photos** — Saves the annotated photo to your Photos library (iPhone & iPad) or your phone's photo gallery (Android). The first time, your device may ask for permission to save photos.
- **Text to Clipboard** — Copies the measurement values as plain text so you can paste them into notes, messages, or a document.
- **Measures CSV** — A spreadsheet-ready table of the measurements: for a photo, one row of values; for a video, one row per kept frame with frame and time columns. For a profile photo, this exports the face-outline points instead.
  - **iPhone & iPad** — Opens the share sheet with a `.csv` file, so you can save it to Files, send it with Mail or AirDrop, or open it in a spreadsheet app such as Numbers or Excel.
  - **Android** — Copies the table to the clipboard as comma-separated text, ready to paste into another app.
- **Video with Measures** (iPhone & iPad, videos only) — Renders a new MP4 of the clip with the enabled measurements drawn on every frame — exactly what you see during playback — and saves it to your Photos library. Dynaface shows its progress while rendering, with a **Cancel** button. The exported video has no sound.

When a video is open, the image and text options use the frame currently showing.

| Export option | iPhone & iPad | Android |
|---|---|---|
| Image to Clipboard | Yes | Not available |
| Image to Photos | Yes | Yes (saves to the gallery) |
| Text to Clipboard | Yes | Yes |
| Measures CSV | Share sheet (`.csv` file) | Copied to the clipboard |
| Video with Measures | Yes (videos only) | Not available |

## Rotating a Photo

If a photo imports sideways or upside down, tap the green **Rotate** button. Dynaface asks you to confirm; tap **Rotate** to turn the photo 90° clockwise and re-analyze it. Tap again to keep turning — four taps bring it all the way around. Your measurement selections are kept.

Rotating changes Dynaface's stored copy of the photo; the original in your photo library isn't touched. If Dynaface can't find a face at the new angle, the screen stays as it was — tap **Rotate** again to try the next quarter-turn.

The **☰ Photo Tools** button at the far right of the button row opens a small menu:

<!-- SCREENSHOT 11 — Photo Tools menu open over a photo on iPhone: Recalculate, Rotate (90 CW), Cancel. -->

- **Recalculate** — Re-runs the analysis on the original photo from scratch, for example after changing the pupillary-distance setting.
- **Rotate (90 CW)** — The same rotation as the green button.
- **Cancel** — Closes the menu.

Rotate and Photo Tools are available for photos only.

## Settings

Open Settings from the **☰ menu** button in the top-right corner of the Photo Library screen, then tap **Settings**.

<!-- SCREENSHOT 12 — Settings card on iPhone: Pupillary Distance (mm) field showing 63, the help text, and the Reset to Default and Done buttons. -->

- **Pupillary Distance (mm)** — The real-world distance between the subject's pupils, in millimeters (40–85; the population average of 63 is the default). Dynaface uses this to convert pixels into millimeters, so entering the subject's actual measured value makes the millimeter measurements more accurate. It applies to new photos and video imports; use **Recalculate** (in the Photo Tools menu) to apply a change to an existing photo. Videos keep the value from when they were imported. Values outside 40–85 are adjusted to the nearest limit.
- **Reset to Default** — Returns the pupillary distance to 63.

Tap **Done**, or tap outside the card, to save and close.

The measurement **text size** setting lives in the measurements panel instead (see above), so you can see the labels change as you adjust it.

## Managing Your Library

Every photo and video you analyze is saved to your on-device library, shown as a scrolling grid with dates. Videos are marked with a camcorder badge.

<!-- SCREENSHOT 13 — Photo Library with several entries (retake: the current screen has the ☰ MENU button top-right, and video entries carry a camcorder badge in the thumbnail's top-left corner). Portrait iPhone, a mix of photos and at least one video. -->
<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/manual/2.0/dynaface-catalog-1.jpg" width="512">

Tap any thumbnail to reopen it with the same measurement selections and trim you left it with. To remove an entry permanently, open it, tap the **Delete** button, and confirm. Deleting removes only Dynaface's copy; the original photo or video in your photo library isn't affected.

Everything stays on your device: your library, the analysis, and your settings. Nothing is sent over the internet. Because your library is stored inside the app, removing the app also removes your library — export anything you want to keep first.

## Tips & Troubleshooting

- **Face Not Found** — Use a clear, well-lit image with one face looking toward the camera. Very small faces, heavy blur, or extreme angles can prevent detection.
- **The photo came in sideways** — Tap the green **Rotate** button until it's upright.
- **A side view was measured as a front view (or the reverse)** — Open the measurements panel and choose **Force Lateral** or **Force Frontal** under POSE.
- **Millimeter values look off** — Set the subject's real pupillary distance in **Settings**, then use **Photo Tools > Recalculate** on the photo. For a video, re-import it after changing the setting.
- **A long video only partly imported** — Dynaface analyzes at most the first 5,000 frames. Trim the recording to the section you need and import the trimmed copy.
- **A video won't import** — Make sure it's an MP4, MOV, or M4V file. On Android, video import requires Android 9 or later.
- **A video import seems stuck** — Analysis pauses whenever Dynaface isn't on screen. Return to the app (and unlock your screen) to let it continue.
- **Rotate or Photo Tools is missing** — These are available for photos only, so they don't appear for videos (or for some photos saved by older versions of Dynaface).
- **No Image to Clipboard or Video with Measures on Android** — These export options aren't available on Android yet. Use **Image to Photos** to save the annotated image, and **Measures CSV** for the measurement data.
- **The camera shows a "Camera access is required" message** — Camera permission was turned off. Re-enable it as described in [Taking a Photo](#taking-a-photo).
- **Android Back gesture doesn't respond** — It's ignored while Dynaface is analyzing or exporting; use the **Cancel** button instead.
- **Getting help** — **☰ menu > Dynaface Manual** opens the online user guide. **☰ menu > About** shows the version number to include if you report a problem.

---

*Dynaface is provided for educational and informational purposes only under the Apache 2.0 License. It is not a registered medical device and is not intended to diagnose, treat, cure, or prevent any disease or medical condition.*
