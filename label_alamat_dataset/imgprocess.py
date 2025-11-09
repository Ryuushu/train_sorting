# library
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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

# Fungsi preprocessing citra setelah crop
def preprocess_image(img):
    "Preprocessing dilakukan setelah gambar di-crop."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh

# Deteksi ROI otomatis
def auto_detect_box(img):
    "Gunakan preprocessing sementara hanya untuk mendeteksi ROI otomatis."
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
    x = max(0, x)
    y = max(0, y + 360)
    w = min(img.shape[1] - x, w + 10)
    h = min(img.shape[0] - y, h + 22)
    return (x, y, x + w, y + h)


# Fungsi gambar bounding box
def draw_box():
    img_show = current_img.copy()
    if start_point and end_point:
        cv2.rectangle(img_show, start_point, end_point, (0, 255, 0), 3)
    ax.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    fig.canvas.draw()


# Fungsi kendali mouse
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


# Fungsi menyimpan hasil crop, preprocessing, dan dataset yolo
def save_crop():
    x1, y1 = start_point
    x2, y2 = end_point
    crop = current_img[y1:y2, x1:x2]

    # --- Simpan hasil crop ---
    crop_path = os.path.join(output_crop_folder, os.path.basename(current_path))
    cv2.imwrite(crop_path, crop)
    print(f"Crop disimpan ke: {crop_path}")

    # --- Simpan hasil preprocessing ---
    processed = preprocess_image(crop)
    pre_path = os.path.join(output_pre_folder, os.path.basename(current_path))
    cv2.imwrite(pre_path, processed)
    print(f"Preprocessing disimpan ke: {pre_path}")

    # --- Salin gambar asli ke dataset YOLO ---
    img_name = get_next_filename(images_train_folder, prefix="image", ext=".png")
    img_dest = os.path.join(images_train_folder, img_name)
    shutil.copy(current_path, img_dest)

    # === Simpan label YOLO ===
    img_h, img_w = current_img.shape[:2]

    # Hitung normalisasi koordinat
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w_norm = (x2 - x1) / img_w
    h_norm = (y2 - y1) / img_h

    class_id = 0  # misalnya kelas "alamat"
    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

    label_dest = os.path.join(labels_train_folder, os.path.splitext(img_name)[0] + ".txt")
    with open(label_dest, 'w') as f:
        f.write(label_line)

    print(f"Gambar asli disalin ke: {img_dest}")
    print(f"Label YOLO disimpan di: {label_dest}")
    print(f"Label: {label_line.strip()}")

# Mremilih gambar dari file
def upload_image():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(
        title="Pilih Gambar",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    return file_path


# Menampilkan gambar dan ROI
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
              f"w={auto_box[2]-auto_box[0]}, h={auto_box[3]-auto_box[1]}")
    else:
        start_point = (50, 50)
        end_point = (300, 200)
        print("Tidak terdeteksi otomatis, gunakan posisi default.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Geser ROI jika tidak sesuai, lalu tekan 'C' untuk crop & simpan label YOLO")
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
