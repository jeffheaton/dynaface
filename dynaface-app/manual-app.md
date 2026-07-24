# Dynaface User Guide — Windows & macOS

Dynaface is a desktop app for Windows and macOS that uses on-device artificial intelligence to detect a face in a photo, or in every frame of a video, and measure aspects of its symmetry, such as eye spacing, mouth width, brow height, smile, and more. Open a picture or video from your computer, capture a new photo with your webcam, or paste an image from the clipboard, and Dynaface will automatically find the face, align it, and display a set of measurements you can turn on or off. For videos, Dynaface also charts how each measurement changes over time and can export the results as a spreadsheet or an annotated video.

All photos, videos, and measurements stay on your computer — nothing is uploaded anywhere.

> **Keyboard shortcuts** are written as `Ctrl+O / ⌘O`: use the **Ctrl** key on Windows and the **Command (⌘)** key on a Mac.

---

## Getting Started

When you open Dynaface, you'll see your **Photo Library** — a grid of every photo and video you've previously analyzed, newest first, each labeled with its date. The first time you launch the app, this screen will be empty.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_1_empty.jpg" width="1024">

Two buttons sit at the bottom of the screen:

- **Camera** — Opens a live view from your webcam so you can take a new photo.
- **Import** — Opens a file dialog so you can choose an existing picture or video.

A **MENU** button in the top-right corner opens **Settings**, this **Dynaface Manual**, and the **About** screen.

Along the top of the window (Windows) or the top of the screen (Mac) is a standard menu bar with **File**, **Edit**, and **Help** menus — these are covered throughout this guide.

## Opening a Photo or Video

There are four ways to get an image or video into Dynaface. All of them lead to the same automatic analysis:

1. **File > Open…** (`Ctrl+O / ⌘O`) — Browse to any supported file on your computer.
2. **Import button** — The Import button on the Photo Library screen opens the same file dialog.
3. **Drag and drop** — Drag a photo or video from File Explorer (Windows) or Finder (Mac) and drop it anywhere on the Dynaface window.
4. **Paste** — Copy an image in any other app, then choose **Edit > Paste** (`Ctrl+V / ⌘V`) in Dynaface.

Supported files:

- **Photos** — JPEG, PNG, HEIC/HEIF, and TIFF.
- **Videos** — MP4, MOV, and M4V.

> **HEIC on Windows:** iPhone photos in HEIC format need Microsoft's free *HEVC/HEIF Image Extensions* from the Microsoft Store. If they aren't installed, Dynaface will tell you when you try to open a HEIC file. Macs can open HEIC photos out of the box.

## Taking a Photo with Your Webcam

Clicking the **Camera** button opens a live preview from your computer's camera.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_2_camera.jpg" width="1024">

From this screen you can:

- **Capture** (center button) — Takes a photo and sends it to Dynaface for analysis.
- **Flip Camera** (right button) — Switches cameras, if your computer has more than one.
- **Import** (left button) — Lets you pick a file instead.
- **Back** (top-left arrow) — Returns to the Photo Library.

The first time you use the camera, your operating system will ask for permission to access it. If you decline, you can still open files — or re-enable camera access later in your system's privacy settings.

For best results, face the camera directly, keep your face centered in the frame, and use even, front-facing light.

## Automatic Face Detection

After you open or capture a photo, Dynaface automatically locates the face, straightens it, and crops it to a consistent size before measuring it — you don't need to do any manual cropping or alignment.

If Dynaface can't find a clear face in the image, it shows a **Face Not Found** message with the option to try again with a different photo.

## Viewing Your Measurements

Once a face is detected, Dynaface displays the analyzed photo with the enabled measurements drawn on it, and a panel listing each measurement's values beside it.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_3_detail.jpg" width="1024">

Along the bottom are five buttons:

- **Share/Export** — Opens export options for the photo and its measurements (see [Exporting, Saving, and Printing](#exporting-saving-and-printing)).
- **Measurement Settings** (chart icon) — Opens the list of available measurements so you can turn individual ones on or off. Your choices are remembered for that photo.
- **Landmarks Toggle** (labeled OFF/LM) — Shows or hides the facial reference points the AI used to calculate the measurements.
- **Recalculate** (circular arrows) — Re-runs the analysis on the photo from scratch, for example after changing the pupillary-distance setting.
- **Delete** (trash icon) — Removes the entry from your library, after a confirmation prompt.

The **Back** arrow in the top-left returns you to the Photo Library.

### Zooming and Panning

- **Scroll** with your mouse wheel or trackpad over the image to zoom in and out.
- **Drag** the image to pan around while zoomed.
- Three buttons in the image's top-right corner **zoom in**, **zoom out**, and **fit** the whole photo back into view.

## Choosing Which Measurements to Show

Click the measurement-settings button to see the full list of available measurements, each with its own on/off switch, plus **All** / **None** shortcuts at the top.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_4_measures.jpg" width="1024">

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

Some measurements are groups: turning a group on reveals indented rows beneath it, so you can pick exactly which of its numbers appear. Turning a measurement on adds it to both the photo overlay and the values panel; turning it off hides it. These choices are saved per photo, so returning to an earlier photo later shows the same selections you left it with.

Two more sections live in the same panel:

- **POSE** — Dynaface automatically detects whether a photo is a front view or a side profile. If it guesses wrong, choose **Force Frontal** or **Force Lateral** here (**Auto Detect** restores the automatic choice).
- **TEXT SIZE** — A 1–5 stepper that makes the measurement labels drawn on the photo larger or smaller. Changes apply immediately, so you can see the effect as you adjust it.

## Side-View (Profile) Photos

If you open a photo taken from the side, Dynaface recognizes it as a profile and switches to its profile measurements: it traces the outline of the face from forehead to chin and marks six anatomical points along it — Soft Tissue Glabella, Soft Tissue Nasion, Nasal Tip, Subnasal Point, Mento Labial Point, and Soft Tissue Pogonion. For profile photos, the CSV export contains this outline (see below).

## Working with Videos

Videos are analyzed the same way as photos — just one frame at a time. Open a video the same way you'd open a photo (Open, Import, or drag and drop), and Dynaface will work through it frame by frame, showing its progress with the option to cancel. Frames where no face can be found are simply skipped.

> **Long videos:** Dynaface analyzes at most 5,000 frames (a few minutes of typical video). For anything longer, it offers to analyze just the first 5,000 — to study a specific gesture in a long recording, trim the video down to that clip first in your video player and open the trimmed copy.

<!-- SCREENSHOT 5 — Video detail view. A video entry in a landscape window: annotated frame top-left, values top-right, transport bar (prev/play/next, slider, frame counter, scissors, hamburger) across the middle, and the measurement chart full-width along the bottom with its legend and red frame marker. This is the money shot — use a clip with 2–3 measurements enabled so the chart has multiple colored lines. -->
<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_5_video.jpg" width="1024">

### Playing and Scrubbing

When a video is open, a transport bar appears with:

- **Previous / Play–Pause / Next** buttons — step one frame at a time or play the clip.
- A **slider** to scrub anywhere in the video, with a **frame counter** (for example "127 / 480").

Every measurement updates as the current frame changes, exactly as if each frame were its own photo.

### The Measurement Chart

Below the video, a chart plots each enabled measurement across the whole clip, one colored line per measurement, with a legend and a red marker showing the current frame. Scrub the slider to move the marker; the chart makes it easy to spot the moment a smile peaks or an eye closes.

### Trimming

The scissors buttons on the transport bar trim the clip to the part you care about:

- **Cut Left of Frame** — Removes everything before the current frame.
- **Cut Right of Frame** — Removes everything after the current frame.
- **Restore All Frames** — Brings the whole video back.

Trimming is never destructive — the discarded frames are only hidden, so Restore always recovers the full clip. The chart and exports cover only the kept frames. In smaller windows, these commands live in the **Video Tools** menu instead (below).

### Video Tools

The **☰** button on the transport bar opens the **Video Tools** menu:

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_6_video_tools.jpg" width="700">

- **Jump to Max Dental** — Jumps straight to the frame with the widest smile (the most visible teeth).
- **Jump to Max Ocular** — Jumps to the frame with the widest eye opening.
- **Evaluation** — Opens a short report comparing the key moments of the clip: the eye measurements at maximum dental display, and the dental measurements at maximum eye opening. **Copy Text** puts the report on the clipboard so you can paste it into notes or an email.
- The **Cut / Restore** commands described above.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_7_evaluate.jpg" width="1024">

## Exporting, Saving, and Printing

Click the **Share/Export** button to see the export options:

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_8_export.jpg" width="1024">

- **Image to Clipboard** — Copies the annotated photo (with visible measurements/landmarks) so you can paste it into a document, email, or slide.
- **Text to Clipboard** — Copies the measurement values as plain text.
- **Measures CSV** — Copies a spreadsheet-ready table to the clipboard: for a photo, one row of values; for a video, one row per frame with a time column, ready to paste into Excel, Numbers, or Google Sheets. For a profile photo, this exports the face-outline points instead.
- **Video with Measures** (videos only) — Renders a new MP4 of the clip with the enabled measurements drawn on every frame — exactly what you see during playback — and saves it into your **Videos** (Windows) or **Movies** (Mac) folder, then opens that folder for you. A progress bar shows during rendering, with the option to cancel.

You can also use the menus:

- **File > Save As…** (`Ctrl+S / ⌘S`) — Saves what you're currently viewing as a file. Choose the format in the dialog: **PNG Image**, **JPEG Image**, or **CSV File** — plus **MP4 Video** when a video is open.
- **Edit > Copy** (`Ctrl+C / ⌘C`) — Same as Image to Clipboard.
- **File > Print…** (`Ctrl+P / ⌘P`) — Opens your system's print dialog with the annotated photo.

Save As and Print are available while you're viewing a photo or video.

## Settings

Open Settings from the **MENU** button on the Photo Library screen, or from the menu bar — **File > Settings…** on Windows, **Dynaface > Settings…** (`⌘,`) on a Mac.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_9_settings.jpg" width="1024">

- **Pupillary Distance** — The real-world distance between the subject's pupils, in millimeters (40–85; the population average of 63 is the default). Dynaface uses this to convert pixels into millimeters, so entering the subject's actual measured value makes the millimeter measurements more accurate. It applies to new photos and video imports; use **Recalculate** to apply a change to an existing photo (videos keep the value from when they were imported). **Reset to Default** returns it to 63.

The measurement **text size** setting lives in the measurement-settings panel instead (see above), so you can see the labels change as you adjust it.

## Managing Your Library

Every photo and video you analyze is saved to your on-device library, shown as a scrolling grid with dates.

<img src="https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/dynaface/manual/app/2/dynaface_10_catalog.jpg" width="1024">

Click any thumbnail to reopen it with the same measurement selections and trim you left it with. To remove an entry permanently, open it, click the **Delete** button, and confirm.

Everything stays on your computer: your originals, the analysis, and your settings. Nothing is sent over the internet.

## Tips & Troubleshooting

- **Face Not Found** — Use a clear, well-lit image with one face looking toward the camera. Very small faces, heavy blur, or extreme angles can prevent detection.
- **HEIC photos won't open (Windows)** — Install Microsoft's free *HEVC/HEIF Image Extensions* from the Microsoft Store, then try again.
- **Millimeter values look off** — Set the subject's real pupillary distance in Settings, then use **Recalculate** on the photo.
- **A long video only partly imported** — Dynaface analyzes at most the first 5,000 frames. Trim the recording to the section you need and open the trimmed copy.
- **Getting help** — **Help > Dynaface Manual** opens this guide. **Help > Open Support Logs** opens the app's log folder — attach the latest log if you report a problem. **About Dynaface** (Help menu on Windows, Dynaface menu on a Mac) shows the version number to include in your report.

---

*Dynaface is provided for educational and informational purposes only under the Apache 2.0 License. It is not a registered medical device and is not intended to diagnose, treat, cure, or prevent any disease or medical condition.*
