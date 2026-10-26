# Examples

The [examples folder](https://github.com/jeffheaton/dynaface/tree/main/dynaface-lib-python/examples) has scripts that measure images and videos. From `dynaface-lib-python` in a clone of the repository, run them on the bundled sample files:

```bash
# Measure a frontal photo and save it with the measures drawn on it.
python examples/process_image.py tests_data/img4-1024-frontal.jpg frontal.jpg

# Measure a side profile.
python examples/process_media.py --crop tests_data/img2-1024-right-lateral.jpg lateral.jpg

# Measure every frame of a video. This writes the annotated video-analyze.mp4,
# a chart of each measure over time (video-graph.png) and the values for every
# frame (video-data.csv).
python examples/process_media.py tests_data/video1-512.mp4 video.mp4
```

`process_media.py` handles images and `.mp4` videos. Add `--points` to draw the landmarks as well, and `--device` to choose the device (`cpu`, `cuda` or `mps`) instead of detecting it.
