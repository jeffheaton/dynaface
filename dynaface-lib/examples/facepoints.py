from dynaface.facial import load_face_image
from dynaface import facial, measures, models

SOURCE_IMG = "https://data.heatonresearch.com/dynaface/sample/img4-1024-frontal.jpg"
DEST_IMG = "output.jpg"

# Detect device and download models
device = models.detect_device()
print(f"Detected device: {device}")
path = models.download_models()
models.init_models(path, device)

# Analyze the face
face = load_face_image(SOURCE_IMG)
face.draw_landmarks(numbers=True)
face.save(DEST_IMG)
