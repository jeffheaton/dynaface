from dynaface.facial import load_face_image


SOURCE_IMG = "/Users/jeff/data/facial/samples/2021-8-19.png"
DEST_IMG = "/Users/jeff/data/facial/samples/2021-8-19-output.png"


face = load_face_image(SOURCE_IMG)
face.analyze()
face.draw_landmarks(numbers=True)
face.save(DEST_IMG)
