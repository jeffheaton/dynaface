import os
from facial_analysis.facial import load_face_image

subject = "2021-8-19.png"
face = load_face_image(subject)
face.draw_landmarks(numbers=True)
face.save("face_landmarks.jpg")


