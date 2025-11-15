import os

for letter in os.listdir("train_ocr"):
    folder = os.path.join("train_ocr", letter)
    if not os.path.isdir(folder):
        continue

    for img_file in os.listdir(folder):
        if img_file.endswith(".png"):
            img_path = os.path.join(folder, img_file)
            box_path = img_path.replace(".png", ".box")

            w, h = 80, 120  # ukuran karakter kamu

            with open(box_path, "w") as f:
                f.write(f"{letter} 0 0 {w} {h} 0\n")
