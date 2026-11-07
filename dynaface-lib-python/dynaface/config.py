# Detect a profile (lateral) face from its landmarks; see AnalyzeFace._is_lateral().
AUTO_LATERAL = True
# Classify a 35 to 55 degree turn (cheek.classify_quarter()) as a three-quarter
# (quarter) face, the view the cheek landmarks' oblique module measures: its far
# cheek is traced from the background model, and the millimeter measures don't
# suit it. Off, such a face is frontal or, with its nose past the far contour,
# lateral, as before, and only a forced Pose.QUARTER takes the quarter path; it
# also needs AUTO_LATERAL. Ships off so that a library default changes no
# caller's video imports: with it on, a frame whose yaw lands in the window
# takes the band crop instead of the frontal one, pays a background-model pass,
# and suits the cheek measure rather than the millimeter measures. A caller turns
# it on; the desktop app exposes it as the "Detect three-quarter view"
# preference. A caller that smooths landmarks across frames must reset its
# smoothing when the view changes, as VideoAnalyzer does, so one crop's
# landmarks never average with another's.
AUTO_QUARTER = False
