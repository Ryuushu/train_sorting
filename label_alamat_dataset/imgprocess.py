# library
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pytesseract
import csv

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Folder output crop dan hasil preprocessing
output_crop_folder = "hasil_crop"
output_pre_folder = "hasil_preprocessing"
os.makedirs(output_crop_folder, exist_ok=True)
os.makedirs(output_pre_folder, exist_ok=True)

# Folder YOLO disimpan
images_train_folder = "C:/Users/Taufiqur Rahman/OneDrive/Documents/label_alamat_dataset/label_alamat_dataset/images/train"
labels_train_folder = "C:/Users/Taufiqur Rahman/OneDrive/Documents/label_alamat_dataset/label_alamat_dataset/labels/train"
os.makedirs(images_train_folder, exist_ok=True)
os.makedirs(labels_train_folder, exist_ok=True)

# Variabel Global
start_point = None
end_point = None
dragging = False
offset = (0, 0)
current_img = None
current_path = None
fig = None
ax = None

# Fungsi penamaan file
def get_next_filename(folder, prefix="data", ext=".png"):
    files = [f for f in os.listdir(folder) if f.endswith(ext)]
    nums = []
    for f in files:
        name = os.path.splitext(f)[0]
        try:
            n = int(name.split("_")[-1])
            nums.append(n)
        except:
            continue
    next_num = max(nums) + 1 if nums else 1
    return f"{prefix}_{next_num:04d}{ext}"

def ocr_characters(chars, save_path=None):
    "Lakukan OCR pada setiap karakter dan simpan hasilnya ke file CSV."
    results = []
    for i, c in enumerate(chars):
        text = pytesseract.image_to_string(
        c,
        config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ).strip()

        results.append(text)

    # Gabungkan semua karakter
    full_text = "".join(results)

    if save_path:
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Char_Index", "Recognized"])
            for i, t in enumerate(results):
                writer.writerow([i + 1, t])
            writer.writerow([])
            writer.writerow(["Full_Text", full_text])
        print(f"Hasil OCR disimpan di {save_path}")

    print(f"HASIL OCR = {full_text}")
    return results

def segment_characters(thresh_img, base_filename="noname"):
    "Segmentasi karakter dan normalisasi ukuran."
    if np.mean(thresh_img) > 127:
        thresh_img = cv2.bitwise_not(thresh_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Temukan kontur karakter
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = clean.shape
    char_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 1 < w < w_img * 0.9 and 8 < h < h_img * 0.95:
            char_boxes.append((x, y, w, h))

    if not char_boxes:
        print("Tidak ada karakter terdeteksi.")
        return []

    # Urutkan kiri ke kanan
    char_boxes = sorted(char_boxes, key=lambda b: b[0])

    target_size = (80, 120)  # lebih besar, lebih detail
    chars = []

    # Normalisasi tiap karakter tanpa menyimpan file gambar
    for (x, y, w, h) in char_boxes:
        char_crop = clean[y:y + h, x:x + w]
        pad_x = max(0, (h - w) // 2)
        padded = cv2.copyMakeBorder(
            char_crop, 5, 5, pad_x + 2, pad_x + 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
        chars.append(resized)

    print(f"[{len(chars)} karakter tersegmentasi (tidak disimpan ke file).")

    # === Subplot hasil segmentasi ===
    fig, axs = plt.subplots(1, len(chars), figsize=(15, 4))
    if len(chars) == 1:
        axs = [axs]
    for i, char in enumerate(chars):
        axs[i].imshow(char, cmap="gray")
        axs[i].set_title(f"Char {i + 1}")
        axs[i].axis("off")
    plt.suptitle(f"Hasil Segmentasi Karakter ({base_filename})")
    plt.tight_layout()
    plt.show()

    # === OCR per karakter dan simpan CSV per gambar ===
    ocr_csv_path = os.path.join("hasil_segmentasi", f"{base_filename}_ocr.csv")
    os.makedirs("hasil_segmentasi", exist_ok=True)
    ocr_characters(chars, save_path=ocr_csv_path)

    return chars

# Fungsi preprocessing citra setelah crop
def preprocess_image(img, base_filename="noname"):
    "Preprocessing dilakukan setelah gambar di-crop."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Peningkatan kontras lokal (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Blur ringan
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 3. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 12
    )

    # 4. Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # === Visualisasi hasil tahap demi tahap ===
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].imshow(gray, cmap="gray")
    axs[0].set_title("GrayScale")
    axs[1].imshow(enhanced, cmap="gray")
    axs[1].set_title("Kontras")
    axs[2].imshow(blur, cmap="gray")
    axs[2].set_title("Gaussian Blur")
    axs[3].imshow(opened, cmap="gray")
    axs[3].set_title("operasi morfologi")
    plt.tight_layout()
    plt.show()

    # === Segmentasi karakter + OCR untuk file ini ===
    segment_characters(opened, base_filename)
    return opened

# Deteksi ROI otomatis
def auto_detect_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 300 < w < img.shape[1] and 40 < h < 300 and 150 < y < img.shape[0] // 2:
            boxes.append((x, y, w, h))

    if not boxes:
        return None

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    x, y, w, h = boxes[0]
    x = max(0, x + 10)
    y = max(0, y + 375)
    w = min(img.shape[1] - x, w - 20)
    h = min(img.shape[0] - y, h - 15)
    return (x, y, x + w, y + h)

# Gambar bounding box
def draw_box():
    img_show = current_img.copy()
    if start_point and end_point:
        cv2.rectangle(img_show, start_point, end_point, (0, 255, 0), 3)
    ax.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    fig.canvas.draw()

# Mouse handlers
def on_press(event):
    global dragging, offset
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    x1, y1 = start_point
    x2, y2 = end_point
    if x1 <= x <= x2 and y1 <= y <= y2:
        dragging = True
        offset = (x - x1, y - y1)

def on_move(event):
    global start_point, end_point
    if not dragging:
        return
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    dx, dy = offset
    w = end_point[0] - start_point[0]
    h = end_point[1] - start_point[1]
    new_x1 = x - dx
    new_y1 = y - dy
    new_x1 = max(0, min(new_x1, current_img.shape[1] - w))
    new_y1 = max(0, min(new_y1, current_img.shape[0] - h))
    start_point = (new_x1, new_y1)
    end_point = (new_x1 + w, new_y1 + h)
    draw_box()


def on_release(event):
    global dragging
    dragging = False


def on_key(event):
    if event.key == 'c':  # tekan 'C' untuk menyimpan crop & preprocessing
        save_crop()

# Simpan hasil crop & preprocessing
def save_crop():
    "Simpan hasil crop dan preprocessing + OCR."
    x1, y1 = start_point
    x2, y2 = end_point
    crop = current_img[y1:y2, x1:x2]

    base_filename = os.path.splitext(os.path.basename(current_path))[0]

    crop_path = os.path.join(output_crop_folder, f"{base_filename}.jpg")
    cv2.imwrite(crop_path, crop)
    print(f"Crop disimpan ke: {crop_path}")

    processed = preprocess_image(crop, base_filename)
    pre_path = os.path.join(output_pre_folder, f"{base_filename}_pre.jpg")
    cv2.imwrite(pre_path, processed)
    print(f"Preprocessing disimpan ke: {pre_path}")

# Pilih gambar
def upload_image():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(
        title="Pilih Gambar",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    return file_path


# Tampilkan gambar dan ROI
def show_image(path):
    global current_img, current_path, fig, ax
    global start_point, end_point

    current_img = cv2.imread(path)
    current_path = path

    auto_box = auto_detect_box(current_img)
    if auto_box:
        start_point = (auto_box[0], auto_box[1])
        end_point = (auto_box[2], auto_box[3])
        print(f"Otomatis terdeteksi: x={auto_box[0]}, y={auto_box[1]}, "
              f"w={auto_box[2] - auto_box[0]}, h={auto_box[3] - auto_box[1]}")
    else:
        start_point = (50, 50)
        end_point = (300, 200)
        print("Tidak terdeteksi otomatis, gunakan posisi default.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Geser ROI lalu tekan 'C' untuk crop & segmentasi karakter")
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_box()
    plt.show()


# === Jalankan ===
if __name__ == "__main__":
    file_path = upload_image()
    if file_path:
        show_image(file_path)
    else:
        print("Tidak ada gambar yang dipilih.")
