import cv2
import os
import numpy as np

DIR = "train_ocr"

def rotate_bound(image, angle):
    """Rotasi gambar tanpa memotong bagian mana pun"""
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))

rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]

# Loop semua folder di dalam DIR
for root, dirs, files in os.walk(DIR):
    original_files = []  # simpan file asli
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            base_name = os.path.splitext(filename)[0]

            original_files.append(img_path)  # simpan path untuk dihapus nanti

            # ROTASI tanpa potong
            for angle in rotation_angles:
                rotated_img = rotate_bound(img, angle)
                cv2.imwrite(os.path.join(root, f"{base_name}_rot{angle}.png"), rotated_img)

            # FLIP DIAGONAL
            flip_diag = cv2.flip(img, -1)
            cv2.imwrite(os.path.join(root, f"{base_name}_flipDiag.png"), flip_diag)

    # Hapus file asli
    for file_path in original_files:
        os.remove(file_path)
        print(f"Dihapus: {file_path}")

print("Augmentasi selesai dan file asli dihapus!")
