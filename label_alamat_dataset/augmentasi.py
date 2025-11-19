import cv2
import os
import numpy as np

DIR = "train_ocr/W"

def rotate_same_size(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
    return rotated

rotation_angles = [
    # Rotasi kecil
    -10, 10, -20, 20, -30, 30,

    # Rotasi sedang
    -45, 45, -60, 60,

    # Rotasi 90-an
    90, -90, 
    270, -270,

    # Rotasi 180
    180, -180,

    # Rotasi besar lainnya
    120, -120, 150, -150,

    # Rotasi inverted + miring
    180 - 10, 180 + 10,
    180 - 20, 180 + 20,
    180 - 30, 180 + 30,
]

for filename in os.listdir(DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):

        img_path = os.path.join(DIR, filename)
        img = cv2.imread(img_path)

        base_name = os.path.splitext(filename)[0]

        # ROTASI
        for angle in rotation_angles:
            rotated_img = rotate_same_size(img, angle)
            cv2.imwrite(os.path.join(DIR, f"{base_name}_rot{angle}.png"), rotated_img)

        # FLIP HORIZONTAL
        flip_h = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(DIR, f"{base_name}_flipH.png"), flip_h)

        # FLIP VERTICAL
        flip_v = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(DIR, f"{base_name}_flipV.png"), flip_v)

print("Augmentasi Selesai!")
