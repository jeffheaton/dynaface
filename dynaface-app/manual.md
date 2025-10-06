# Dynaface User Guide

Dynaface is an application that measures key facial symmetry statistics using artificial intelligence. Dynaface can measure both individual images and recorded video. This program calculates the degree of Facial paralysis and asymmetry caused by Bell's Palsy, cranial tumor resection, or other factors. We released this program under the [MIT License](https://opensource.org/license/mit/); this program is for educational and research purposes only. [[likely include paper citation]]

We have provided both Macintosh and Windows versions of Dynaface. The Macintosh version requires a Mac M1 or later. The Windows application requires Windows 11 or higher and can optionally utilize an NVIDIA CUDA-compatible GPU to speed processing.

When you launch Dynaface, you will see the following.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-splash.jpg?v=1" width="512">

This window provides the following options:

- **Open File** - Open a new video, image, or Dynaface document file for editing.
- **About** - Information about Dynaface or
- **Settings** - Allows you to customize the program.
- **Exit** - Quit the program.

Most likely you will begin by opening a video or image file.

# Analyzing a Video or Image

Whether you are analyzing a video or an image, you will see a window similar to what you see here.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/app-controls.jpg?v=1" width="512">

Let's review the controls and options that you see on this window. Some of these options will only be available when analyzing a video.

1. **Tab Bar** - Select which tabs/windows you have open.
2. **Landmarks Toggle** - Toggle display of the 97 facial landmarks; AI detects these points and forms the foundation of all measurements done by this program.
3. **Mesures Toggle** - Toggle display of any selected facial measures.
4. **Chart Toggle** (video only) - Toggles a chart displaying the change of selected facial measures over time in a video.
5. **Pose** - Inactive for videos; for images allows you to change between frontal and lateral if the program misclassifies the pose of the subject.
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
18. **Measure Toggle** - A hierarchical list of all measures and sub-measures allows each to be toggled.
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

You can select which messages you wish to see and notice that some measures have multiple sub-measures that you can turn on or off. You can copy and paste the measured image into other programs or save an image file with the measures.

# Analyzing Lateral Images

Dynafce can also measure lateral images.

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

- **Definition:** Angle formed between the columella and the upper-lip line at the subnasale.
- **Landmarks:** Columella Point, Subnasale, Labrale Superius
- **Purpose:** Assesses nasal base rotation and the relationship between the nose and upper lip.
- **Typical Range:** 90°–110°
  - Larger angles → more upturned nasal base or retrusive upper lip
  - Smaller angles → more downward-pointing nasal tip or protrusive upper lip

---

### 3. tip_proj – Nasal Tip Projection (Goode Method)

- **Definition:** Ratio of nasal tip projection to nasal length.
- **Formula:** tip_proj = AT / NT
  - **AT:** perpendicular distance from nasal tip to a vertical line through the alar base
  - **NT:** linear distance from nasion to nasal tip
- **Purpose:** Quantifies how far the nasal tip extends from the face.
- **Typical Ideal Ratio:** 0.55–0.60
  - Higher ratio → more prominent nasal projection
  - Lower ratio → flatter or shorter nasal contour

# Automatic Pose Detection

Dynaface uses AI to determine the pose; however, if the poses is not classified correctly, you can correct it with the "Pose" button.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-mis-pose.jpg?v=1" width="512">

# Analyzing Videos

Videos work very similarly to images, except there is the element of time. The AI must analyze every video frame to find the facial features of a video. Processing a video can take considerable time; a 2-minute video can take around 20 minutes to analyze thoroughly. Because of this processing need, we suggest you trim your videos to just the clips you need to study. While Dynaface does have basic video editing capabilities that allow you to cut the video to smaller sizes, Dynaface is not designed to process large videos.

Videos add the element of time. In the video shown above, you can see a blink cycle from someone with facial paralysis. You can see that the patient's left eye does not close fully during the blink cycle, while the unaffected right eye has a normal blink cycle.

Once you have loaded a video, you should save it to a Dynaface (.dyfc) document file so that you do not have to wait for the video to be processed the next time you use Dynaface. To save the document, choose "Save As" from the menu, and the program will present you with these options.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-save-options.jpg" width="512">

Analyzing a video is similar to an image. You can click the chart checkbox to see a graph of the selected measures. You can also cut sections of the video. Use the video slider to move around the video to see different sections. You can also save a video MP4 file with all of the measures visible. Sound is not preserved during the analysis.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-video-graph.jpg" width="512">

You can produce a variety of graphs with Dynaface by removing and adding the various measures. For example, the following chat shows multiple blink cycles and smiles over 2 minutes.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-2min-graph.jpg" width="1024">

Though you can produce a variety of graphs with Dynaface, it is not a graphing package. You may wish to export these measures to a CSV for detailed analysis in other software programs. Below is a Dynaface CSV file opened in Microsoft Excel. As you can see, the measures are presented for each of the frames.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-csv-file.jpg" width="512">

# Facial Measurements

Dynaface supports several predefined facial measures, which are based on 96 facial landmarks that neural networks are designed to detect. These landmarks are presented here, giving you an idea of the types of measures Dynaface can provide.

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/face-landmarks.jpg" width="512">

## FAI

The [Facial Asymmetry Index (FAI)](https://pubmed.ncbi.nlm.nih.gov/30041270/) is an objective measure of facial asymmetry and validate its use and reliability across numerous reanimation techniques.

[[We can fill in exact details for each of these, we may add additional for the paper]]

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-fai.jpg" width="512">

## Oral CE

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-oral.jpg" width="512">

## Brow

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-brow.jpg" width="512">

## Dental Display

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-dental.jpg" width="512">

## Eye Area

<img src="https://data.heatonresearch.com/images/facial/manual/1.0/dynaface-measure-fissure.jpg" width="512">
