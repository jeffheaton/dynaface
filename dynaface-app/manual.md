# Dynaface Classic User Guide

> **Looking for the current app?** This guide covers Dynaface Classic, the original desktop app. The current Dynaface app has its own guides: [Windows & macOS](https://github.com/jeffheaton/dynaface/blob/main/dynaface-app/manual-app.md) and [iPhone, iPad & Android](https://github.com/jeffheaton/dynaface/blob/main/dynaface-app/manual-mobile.md).

Dynaface Classic is an application that measures key facial symmetry statistics using artificial intelligence. It can measure both individual images and recorded video, including asymmetry caused by Bell's palsy, cranial tumor resection, or other factors. We released this program under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0); this program is for educational and research purposes only. For the peer-reviewed research that uses Dynaface, and how to cite it, see [dynaface.org/research](https://www.dynaface.org/research/).

For questions about privacy and how Dynaface handles your data, see the [Dynaface FAQ](https://github.com/jeffheaton/dynaface/blob/main/FAQ.md).

Dynaface Classic is available for macOS and Windows. The macOS version requires an Apple silicon Mac (Intel Macs should use the current app). The Windows version requires 64-bit Windows 10 or 11 and can optionally utilize an NVIDIA CUDA-compatible GPU to speed processing.

When you launch Dynaface, you will see the following.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-splash.jpg?v=1" width="512">

This window provides the following options:

- **Open File** - Open a new video, image, or Dynaface document file for editing.
- **About** - Information about Dynaface.
- **Settings** - Allows you to customize the program.
- **Exit** - Quit the program.

Most likely you will begin by opening a video or image file.

# Analyzing a Video or Image

Whether you are analyzing a video or an image, you will see a window similar to what you see here.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/app-controls.jpg?v=1" width="512">

Let's review the controls and options that you see on this window. Some of these options will only be available when analyzing a video.

1. **Tab Bar** - Select which tabs/windows you have open.
2. **Landmarks Toggle** - Toggle display of the 98 facial landmarks; AI detects these points and forms the foundation of all measurements done by this program. On a frontal or three-quarter face the box also draws the cheek landmarks with their codes (zy, cr and sr on a frontal face; apex and sub on the far cheek of a three-quarter face, sub only when the outline dips into a hollow there). Their coordinates are exported by the "Cheek Landmarks" entry of the Landmarks measure in the measure list (control 18), which also draws them as orange dots, without codes, while it is checked.
3. **Measures Toggle** - Toggle display of any selected facial measures.
4. **Chart Toggle** (video only) - Toggles a chart displaying the change of selected facial measures over time in a video.
5. **Pose** - Sets the view the face is analyzed in: frontal, lateral (profile) or three-quarter (3/4). Dynaface detects the view when it opens a file (three-quarter detection is off unless you turn it on in Settings), and this button corrects a misclassified pose or forces the view you want. For an image the change applies at once; for a video the whole video is analyzed again in the chosen view. The button is available for images and for videos opened from a video file, not for a saved Dynaface document.
6. **Image Display Zoom** - Zooms the image/video area. You can also zoom in on this area by placing the mouse over it and using the mouse wheel or trackpad scroll gesture.
7. **Chart Display Zoom** (video only) - Zooms the chart area. You can also zoom in on this area by placing the mouse over it and using the mouse wheel or trackpad scroll gesture.
8. **Fit Image/Video and Graph** - Automatically expand or shrink the chart and video/image areas to fill the area defined by the chart/display slider.
9. **Cut Left** (video only) - Remove all video frames before the playhead.
10. **Cut Right** (video only) - Remove all video frames after the playhead.
11. **Restore** (video only) - Undo any prior cuts to the video and restore the video to its original length.
12. **Measurement Text Zoom** - Increase or decrease the text size used to label the measurements in the display area.
13. **Jump To** - For video, allows jump to max/min areas of the video.
14. **Eval** - Performs several evaluation measures for video.
15. **Select All Measures** - Enable all available measures.
16. **Select No Measures** - Disable all available measures.
17. **Display Area** - The area where the video or image is displayed.
18. **Measure Toggle** - A hierarchical list of all measures and sub-measures allows each to be toggled. Measures that are not defined on the view being shown (the millimeter measures and Skin Tone on a three-quarter face, the frontal measures on a profile) are grayed out, and so are the entries of a measure that belong to the other view (under Cheek Measures, the frontal entries on a three-quarter face and the three-quarter entries on a frontal one); the choices you made come back when a frame of their view shows again. A measure some of whose entries are off shows a partly filled box. Entries marked "(drawing)" only switch a drawing on or off and add no value of their own.
19. **Chart Display and Playhead** - Displays a chart of facial measures over time; the playhead, a red vertical line, represents the current location displayed.
20. **Chart/Display Slider** - This allows the relative size of the display area and the chart adjustment by dragging.
21. **Backward One Frame** (video only) - Move both playheads (chart and video slider) back by a frame.
22. **Play** (video only) - Play the video from the current playhead position until the end of the video.
23. **Forward One Frame** (video only) - Move both playheads (chart and video slider) forward by a frame.
24. **Current and Total Frames** (video only) - Indicates the current frame location of the playheads.
25. **Video Slider and Playhead** (video only) - Allows you to drag the current location of the playhead through the video.

# Automatic Cropping and Scaling

Whether you analyze an image or video, cropping and scaling are very important. Based on the patient's eyes, the program will automatically zoom and crop to a 1024 x 1024 rectangle. Ideally, it would help if you centered the patient in the picture/video, stayed a consistent distance from the patient, and minimized any shaking of the camera/mobile camera device. However, autocropping is reasonably resilient to small movements of the patient and recording equipment. The patient should always be looking directly at the camera.

Generally, video or images captured with a handheld mobile device should be sufficient for Dynaface. Consider the image below; you can see how the program cropped and zoomed the patient to a consistent size and location within the frame.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-crop.jpg" width="512">

# Analyzing Frontal Images

Dynaface allows you to open images of types: jpeg, png, and heif. To open an image, either select "Open" from the menu or drag an image file onto the application. Once the image is open, you can zoom and pan the image in the display area. The measures can be displayed or hidden. If you would like to utilize the individual measurements in a different application, you can use "Save As" to save the document as a CSV file. If you think you might have further analysis on an image, you can save the image as a Dynaface (.dyfc) document.

When analyzing a single image, the window will appear as follows.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-image.jpg?v=1" width="512">

You can select which measures you wish to see and notice that some measures have multiple sub-measures that you can turn on or off. You can copy and paste the measured image into other programs or save an image file with the measures.

# Analyzing Lateral Images

Dynaface can also measure lateral images.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-lateral.jpg?v=1" width="512">

The following underlying landmarks are detected:

- **Soft Tissue Glabella:** The most prominent midline point on the forehead above the nasion. It marks the superior boundary of the nasofrontal region.

- **Soft Tissue Nasion:** The deepest midline depression between the forehead and the nasal bridge. It serves as the junction where the glabellar contour transitions to the nasal dorsum.

- **Nasal Tip:** The most anterior projecting point of the nose in the sagittal profile. It defines overall nasal projection and tip position.

- **Subnasal Point:** The junction where the base of the nose meets the upper lip. It forms the inferior limit of the nose and the vertex of the nasolabial region.

- **Mento Labial Point:** The deepest midline depression between the lower lip and the chin. It characterizes the curvature of the mentolabial fold.

- **Soft Tissue Pogonion:** The most anterior midline point on the soft tissue chin. It represents the forward limit of the lower facial profile.

- **Sagittal Line:** The vertical midline plane dividing the face into left and right halves in lateral view. All lateral (profile) measurements are projected and measured relative to this line.

These landmarks can calculate the following measures:

### 1. NFA – Nasofrontal Angle

- **Definition:** Angle formed between the glabella–nasion line and the nasion–nasal tip line.
- **Landmarks:** Soft Tissue Glabella, Soft Tissue Nasion, Nasal Tip
- **Purpose:** Evaluates the depth of the nasion and the transition between the forehead and nose.
- **Typical Range:** 115°–135°
  - Larger angles → smoother, flatter nasal root
  - Smaller angles → deeper nasion depression

---

### 2. NLA – Nasolabial Angle

- **Definition:** Angle at the subnasal point between the line to the nasal tip and the line to the soft tissue pogonion.
- **Landmarks:** Nasal Tip, Subnasal Point, Soft Tissue Pogonion
- **Purpose:** Describes the angle of the nasal base relative to the lower face in profile.

---

### 3. AT/NT – Nasal Tip Projection

- **Definition:** Ratio of how far the nasal tip projects forward to the vertical length of the nose.
- **Formula:** AT / NT
  - **AT:** horizontal distance from the nasal tip to a vertical line through the soft tissue nasion
  - **NT:** vertical distance from the soft tissue nasion down to the level of the nasal tip
- **Purpose:** Quantifies how far the nasal tip extends from the face.
- **Units:** AT and NT are in pixels of the standardized image; the ratio has no units. Both distances follow the image's horizontal and vertical axes, so keep the head level in profile photos.
  - Higher ratio → more prominent nasal projection
  - Lower ratio → flatter or shorter nasal contour

# Analyzing Three-Quarter Images

Dynaface can also measure three-quarter images, where the subject has turned about 45 degrees away from the camera.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-quarter.jpg?v=1" width="512">

A three-quarter image shows the outline of the far cheek, the one away from the camera, which neither a frontal nor a lateral image shows. Dynaface removes the background, traces that outline from the outer corner of the eye down to the chin, and finds two points on it:

- **Malar Apex:** The most prominent point of the cheek on the outline. When no bulge stands out on the outline, Dynaface takes the point at the height where the apex usually sits and labels it "malar apex (prior)"; "(nose?)" marks an apex at the height of the nose tip, which is probably the nose. Compare apex heights only between two observed or two prior apexes.

- **Submalar Point:** The deepest point of the outline between the apex and the level of the mouth, reported whenever the outline below the apex was traced from the background (hair over the far cheek leaves it and its flag empty), with a flag that says whether the outline dips inward there enough to count as a hollow; the drawing marks it only when it does.

Both points are reported as fractions of the chin-to-eye-corner distance rather than in millimeters: a turned head foreshortens the distance between the pupils, so no millimeter measure is defined on it. For that reason the measures that need millimeters or a face-on view (FAI, Oral CE, Brow, Dental Display, Eye Area, Nose Frontal, the frontal distances and Skin Tone) are switched off, and grayed out in the measure list, while a three-quarter image is shown; Cheek Measures, Position, Pose and Landmarks stay available, with the Landmarks measure's "Cheek Landmarks" entry. Of the Cheek Measures entries, Far cheek outline, Malar apex, Submalar point and Head turn apply to a three-quarter image, and Zygion width and Reference points are grayed out: the frontal cheek landmarks (the zygion, cheek reference point and submalar reference point of each side) belong to frontal photos; see "Cheek Measures" under "Facial Measurements" below.

One photo measures one cheek, so take two, one turned each way, and compare them. For a good three-quarter photo:

- Turn about 45 degrees, and turn the whole body (a chair or a mark on the floor) rather than the head on the neck, which tilts it; turn the same amount for each side.
- Keep the expression at rest; the points are defined on a face at rest.
- Keep hair off the far cheek, and the chin in the frame. A plain background helps, because the outline is traced from the background removal.
- Keep the nose clear of the cheek outline. If the tip of the nose lines up with the far cheek, the head is turned too far, and the apex may be read on the nose instead of the cheek.

Three-quarter detection is off unless you turn it on in Settings. With it off, open the photo and use the "Pose" button to choose "Three-quarter (3/4)"; the button forces the view whether or not detection is on.

# Automatic Pose Detection

Dynaface uses AI to determine the pose. Lateral (profile) detection is on by default. Three-quarter detection is a preference, off unless you turn it on in Settings, so until you do, a turned head is analyzed as a frontal face (or, turned far enough, as a profile). With it on, a head turned about 35 to 55 degrees is analyzed as three-quarter: in a test on 816 research photos it recognized every frontal photo, 92% of the three-quarter photos (most of the rest were turned less than 35 degrees and opened as frontal) and 94% of the profiles. Whether detection is on or off, the "Pose" button always forces the view you choose, so if the pose is not classified correctly, or you want the three-quarter view without turning its detection on, you can set it with the "Pose" button.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-mis-pose.jpg?v=1" width="512">

# Settings

Choose "Settings..." from the menu to open the Preferences tab. Click "Save" to keep your changes, "Reset" to return every row to its default, or "Cancel" to leave them as they were. The rows are:

- **Pupillary Distance (PD)** - The subject's pupillary distance in millimeters, from 40 to 85; the default of 63 is the adult average. Every millimeter measure is scaled from it, so enter the subject's own distance when absolute sizes matter. Ratios and angles do not depend on it.
- **Correct tilt greater than** - The head tilt, in degrees, above which Dynaface straightens the face before measuring it; -1, the default, leaves every face as it was photographed.
- **Crop/zoom/tilt smoothing** - A field kept from earlier versions; the default is 2. It has no effect today: the crop, zoom and tilt correction are placed on each video frame's own pupils, so the value is not used.
- **Data smoothing** - For videos, the number of frames the facial landmarks are averaged over, which steadies their jitter; 1 disables it and 2 is the default. The averaging restarts whenever the view changes, so a turned or profile frame's landmarks are never averaged with a frontal frame's.
- **Log Level** - How much the program writes to its log file, from DEBUG (everything) to CRITICAL (almost nothing); INFO is the default.
- **Detect lateral (profile) view** - Whether Dynaface decides for itself that a photo or video frame shows a profile; on by default. With it off, every face is analyzed as frontal unless you force the view with the "Pose" button.
- **Detect three-quarter view (slows video import)** - Whether Dynaface decides for itself that a photo or video frame shows a head turned about 45 degrees; off by default. It needs lateral detection, so it is unavailable while that row is off. It slows video import because a turned frame also runs the background model, which takes about three times as long as a frontal frame. With it off, the "Pose" button still forces the three-quarter view.
- **Use Accelerator (GPU)** - Whether the AI models run on the computer's graphics hardware (Apple silicon on macOS, an NVIDIA CUDA GPU on Windows) instead of the CPU; on by default. If the accelerator fails to start, Dynaface falls back to the CPU and turns this off.

A change to either detection switch applies to the photos and videos you open from then on; files already open keep the view they were analyzed in. While a video is still importing, Dynaface asks you to wait for the import to finish before it saves a change to either switch, since the rest of that video would otherwise be analyzed with the new setting. Saving loads the AI models again, so a change to the accelerator takes effect right away, after a short wait while they load.

# Analyzing Videos

Videos work very similarly to images, except there is the element of time. The AI must analyze every video frame to find the facial features of a video. Processing a video can take considerable time; a 2-minute video can take around 20 minutes to analyze thoroughly. Because of this processing need, we suggest you trim your videos to just the clips you need to study. While Dynaface does have basic video editing capabilities that allow you to cut the video to smaller sizes, Dynaface is not designed to process large videos.

Videos add the element of time. In the video shown above, you can see a blink cycle from someone with facial paralysis. You can see that the patient's left eye does not close fully during the blink cycle, while the unaffected right eye has a normal blink cycle.

Each frame of a video is analyzed in the view Dynaface detects for it: frontal, lateral when lateral detection is on, or three-quarter when three-quarter detection is on (see Settings). A three-quarter video is therefore analyzed as one when that detection is on, or when you force the view with the "Pose" button, which analyzes the whole video again in the chosen view. A turned frame, like a profile frame, also runs the background model, so it takes about three times as long as a frontal frame. When the view changes partway through a video, data smoothing restarts at the change, so a turned frame's landmarks are never averaged with a frontal frame's. A measure that is not defined on a frame's view is empty on that frame in the chart and the CSV file: the millimeter measures, Skin Tone, Landmarks and Cheek Measures on a profile frame; the millimeter measures, Skin Tone and Cheek Measures' frontal entries on a turned frame; Lateral Measures on any other frame; and Cheek Measures' three-quarter entries on a frontal frame (a turned frame still carries Cheek Measures' three-quarter values and its head turn, the Pose and Landmarks values and Position's tilt, though not its millimeter scale, and a profile frame the Lateral Measures). The CSV file of a video holds that series whichever frame is shown when you save it; only a single profile photo saves its sagittal outline instead. A video saved as a video file draws each frame's own measures: the chosen frontal measures on its frontal frames and none of them on a turned or profile frame. The chart shows only the measures that have a value somewhere in the video, "Jump To" never lands on a turned or profile frame, and "Eval" compares frontal frames only. The measure list follows the frame being shown, and the choices you make come back when a frame of their view shows again.

Once you have loaded a video, you should save it to a Dynaface (.dyfc) document file so that you do not have to wait for the video to be processed the next time you use Dynaface. To save the document, choose "Save As" from the menu, and the program will present you with these options.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-save-options.jpg" width="512">

Analyzing a video is similar to an image. You can click the chart checkbox to see a graph of the selected measures. You can also cut sections of the video. Use the video slider to move around the video to see different sections. You can also save a video MP4 file with all of the measures visible. Sound is not preserved during the analysis.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-video-graph.jpg" width="512">

You can produce a variety of graphs with Dynaface by removing and adding the various measures. For example, the following chart shows multiple blink cycles and smiles over 2 minutes.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-2min-graph.jpg" width="1024">

Though you can produce a variety of graphs with Dynaface, it is not a graphing package. You may wish to export these measures to a CSV for detailed analysis in other software programs. Below is a Dynaface CSV file opened in Microsoft Excel. As you can see, the measures are presented for each of the frames.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-csv-file.jpg" width="512">

# Facial Measurements

Dynaface supports several predefined facial measures, which are based on 98 facial landmarks that neural networks are designed to detect. These landmarks are presented here, giving you an idea of the types of measures Dynaface can provide. Full definitions of every measurement, including units and CSV column names, are at [dynaface.org/measurements](https://www.dynaface.org/measurements/).

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/face-landmarks.jpg" width="512">

## FAI

The Facial Asymmetry Index (FAI) compares the distance from the inner corner of each eye (medial canthus) to the corner of the mouth (oral commissure) on the same side. FAI is the absolute difference between the left and right distances, in millimeters, so 0 means the two sides match. For the index's validation across smile restoration techniques, see [Malm et al., 2018](https://pubmed.ncbi.nlm.nih.gov/30041270/).

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-fai.jpg" width="512">

## Oral CE

Oral commissure excursion is the distance from each corner of the mouth to the midpoint of the lower lip, in millimeters, reported separately for the left and right sides.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-oral.jpg" width="512">

## Brow

The difference in height between the left and right eyebrows, in millimeters, measured at the middle of each brow along lines parallel to the line between the pupils.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-brow.jpg" width="512">

## Dental Display

The area of the mouth opening inside the inner edges of the lips, where the teeth show during a smile, in mm². The opening is split at the facial midline, and Dynaface reports the total, each side, the difference between the sides, and their symmetry ratio (smaller side divided by larger).

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-dental.jpg" width="512">

## Eye Area

The area of each eye opening (palpebral fissure), in mm², with the difference and symmetry ratio between the left and right sides.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-fissure.jpg" width="512">

## Cheek Measures

Three points on each cheek that no landmark model places. On a frontal photo, Dynaface constructs the zygion of each side, the widest point of the face over the cheekbone, and reports each zygion's distance from the midline and the width between them in millimeters, and their symmetry ratio, along with a cheek reference point and a submalar reference point on each side, which anchor the cheek region. On a three-quarter photo it observes the malar apex and the submalar point on the outline of the far cheek, and reports their heights and the depth of the hollow as fractions of the chin-to-eye-corner distance; the frontal values are empty on a three-quarter photo and the three-quarter values on a frontal one. Each group of values comes with a flag (the "ok" columns, 1 or 0: one per zygion, one for the width, one for the ratio, one each for the two reference points, the apex and the submalar point) that says whether the photo allowed the point: the head turned within the few degrees the point allows and, for the apex and the submalar point, the nose clear of the outline and the outline traced from the background; "cheek.yaw" reports the turn, so check the group's flag before using a value.

The measure has six entries in the measure list. Each switches one drawing on or off and, with it, the values about it in the chart and the CSV file:

- **Zygion width** (frontal photos): the zygion level line, the two zygion dots with their labels, and the zygion width line at the top of the image.
- **Reference points** (frontal photos): the vertical from each outer eye corner, the cheek and submalar reference points on it with their labels, and the reference depth line at the top of the image.
- **Far cheek outline** (three-quarter photos): the far cheek's outline, colored by how each part of it was traced, the eye-corner, mouth and chin levels, and the line between the outline at the eye-corner and mouth levels.
- **Malar apex** (three-quarter photos): the apex dot, its label and the apex line at the top of the image.
- **Submalar point** (three-quarter photos): the submalar dot and its label, drawn only when the outline dips into a hollow, and the submalar line at the top of the image.
- **Head turn** (both): the line at the top of the image that reports the turn and the pitch, marked "(flagged)" when the zygion width's flag on a frontal photo, the apex's on a three-quarter photo, or the pitch check is 0.

When a three-quarter photo's result is flagged, lines starting with "flagged:" under the values at the top of the image say why, one reason per line: for example "flagged: turned too far (50.1 degrees, needs 40 to 50)", "flagged: the nose hides the far cheek" or "flagged: the cheek's edge is unclear (hair or background)". Most of these mean the photo should be retaken: turn the head about 45 degrees, so a strip of the far cheek shows between the nose tip and the background, with the hair off that cheek.

The cheek landmarks' coordinates are the "Cheek Landmarks" entry of the Landmarks measure, and the "Landmarks" box (control 2) draws the cheek landmarks themselves, with their codes, beside the 98 facial landmarks. Full definitions are at [dynaface.org/measurements](https://www.dynaface.org/measurements/).

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-cheek.jpg" width="512">
