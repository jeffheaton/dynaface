# test_cv2.py
import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: test_cv2.exe <image_path>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
if img is None:
    print("Failed to load image:", sys.argv[1])
    sys.exit(1)

print(f"Image dimensions: {img.shape[1]}x{img.shape[0]}")
