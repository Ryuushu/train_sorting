# ================== LIBRARY ==================
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ================== KONFIG ==================
IMG_SIZE = 64
MODEL_PATH = r"char_model.h5"
DATASET_DIR = "../train_ocr"  

images_train_folder = r"images\train"
labels_train_folder = r"labels\train"
os.makedirs(images_train_folder, exist_ok=True)
os.makedirs(labels_train_folder, exist_ok=True)
os.makedirs("hasil_crop", exist_ok=True)
os.makedirs("hasil_preprocessing", exist_ok=True)
os.makedirs("hasil_csv", exist_ok=True)

# ================== VAR GLOBAL ==================
start_point = None
end_point = None
dragging = False
offset = (0, 0)
current_img = None
current_path = None
fig = None
ax = None

poly_patch = None
poly_coords = None

_model = None
_labels = None

# menyimpan hasil rotasi dan gambar 
def save_augmented_rotations(img, save_path, base_name):
    """
    Simpan rotasi huruf sesuai ukuran asli (tanpa mengecilkan)
    """

    os.makedirs(save_path, exist_ok=True)

    # ukuran asli character (misal 240x160)
    h, w = img.shape

    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    for i, angle in enumerate(angles):
        # rotasi mempertahankan ukuran asli
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)

        filename = f"{base_name}_rot{i}.png"
        cv2.imwrite(os.path.join(save_path, filename), rotated)

    print(f"[OK] Rotasi 8 arah disimpan untuk {base_name}")

# menyimpan karakter ke dataset
def save_training_chars_manual(chars):
    BASE_DIR = "train_ocr"
    os.makedirs(BASE_DIR, exist_ok=True)

    for idx, img in enumerate(chars):
        cv2.imshow("Label Karakter (tekan huruf)", img)
        key = cv2.waitKey(0)

        label = chr(key).upper()
        print("Label:", label)

        save_dir = os.path.join(BASE_DIR, label)
        os.makedirs(save_dir, exist_ok=True)

        count = len(os.listdir(save_dir))

        # ========= INI HARUS DI DALAM LOOP =========
        save_augmented_rotations(img, save_dir, f"{label}_{count}")
        # ===========================================

    cv2.destroyAllWindows()
    print("[OK] Semua karakter tersimpan.")

# ================== UTILITY ==================
def load_model_and_labels(model_path=MODEL_PATH, dataset_dir=DATASET_DIR):
    global _model, _labels
    if _model is not None and _labels is not None:
        return _model, _labels

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    print("[INFO] Loading model:", model_path)
    _model = tf.keras.models.load_model(model_path)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir tidak ditemukan: {dataset_dir}")

    labels = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    _labels = labels
    print("[INFO] Loaded labels:", _labels)
    return _model, _labels

def preprocess_image(img, img_size=IMG_SIZE):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype("float32") / 255.0
    img_final = np.expand_dims(img_norm, axis=(0, -1))
    return img_final

def preprocess_image1(img, base_filename="noname"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 12
    )
    return gray, enhanced, thresh

def crop_rotated(img, box):
    pts = np.float32(box)
    pts = pts[np.argsort(pts[:,1])]  # sort by Y
    top = pts[:2]
    bottom = pts[2:]
    top = top[np.argsort(top[:,0])]
    bottom = bottom[np.argsort(bottom[:,0])]
    pts = np.float32([top[0], top[1], bottom[1], bottom[0]])
    w = int(np.linalg.norm(pts[0] - pts[1]))
    h = int(np.linalg.norm(pts[0] - pts[3]))
    dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

def save_uploaded_image():
    base_filename = os.path.splitext(os.path.basename(current_path))[0]
    dst_path = os.path.join(images_train_folder, f"{base_filename}.jpg")
    cv2.imwrite(dst_path, current_img)
    print(f"Gambar asli disimpan ke: {dst_path}")

# ================== DETEKSI ROI ==================
def auto_detect_box(img, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Tidak ada kontur kertas terdeteksi.")
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    paper_cnt = contours[0]
    rect = cv2.minAreaRect(paper_cnt)
    paper_box = cv2.boxPoints(rect)
    paper_box = np.intp(paper_box)

    paper_crop = crop_rotated(gray, paper_box)
    _, text_thresh = cv2.threshold(paper_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_text, _ = cv2.findContours(text_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_boxes = []
    for c in contours_text:
        r = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = r
        if 20 < w < paper_crop.shape[1] * 0.9 and 10 < h < paper_crop.shape[0] * 0.5:
            text_boxes.append(r)

    if not text_boxes:
        print("Tidak ada area tulisan terdeteksi.")
        return paper_box  # fallback

    text_boxes = sorted(text_boxes, key=lambda r: r[1][0]*r[1][1], reverse=True)
    (cx, cy), (w_t, h_t), angle = text_boxes[0]
    text_box = cv2.boxPoints(text_boxes[0])
    text_box = np.intp(text_box)

    shrink = 0.1
    center = np.array([cx, cy])
    points = text_box - center
    points = points * (1-shrink)
    shrink_box = (points + center).astype(int)

    if debug:
        vis = img.copy()
        cv2.drawContours(vis, [paper_box], 0, (255,0,0), 3)
        cv2.drawContours(vis, [shrink_box], 0, (0,255,0), 3)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Deteksi Kertas & Tulisan")
        plt.axis("off")
        plt.show()

    return shrink_box

# ================== MOUSE HANDLER ==================
def on_press(event):
    global dragging, offset
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    if poly_patch is not None and poly_patch.get_path().contains_point((x,y)):
        dragging = True
        offset = (x, y)

def on_move(event):
    global poly_coords, poly_patch, offset
    if not dragging or poly_coords is None:
        return
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    dx = x - offset[0]
    dy = y - offset[1]
    offset = (x, y)
    poly_coords[:,0] += dx
    poly_coords[:,1] += dy
    poly_patch.set_xy(poly_coords)
    fig.canvas.draw_idle()

def on_release(event):
    global dragging
    dragging = False

def on_key(event):
    if event.key == 'c':
        save_crop()

# ================== SEGMENTASI KARAKTER ==================
def segment_characters(thresh_img, base_filename="noname"):
    if np.mean(thresh_img) > 127:
        thresh_img = cv2.bitwise_not(thresh_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    clean = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = clean.shape
    char_boxes = [(x,y,w,h) for cnt in contours for x,y,w,h in [cv2.boundingRect(cnt)] if 1<w<w_img*0.9 and 8<h<h_img*0.95]
    if not char_boxes:
        print("Tidak ada karakter terdeteksi.")
        return [], [], ""
    char_boxes = sorted(char_boxes, key=lambda b: b[0])
    target_size = (80,120)
    chars = []
    for x,y,w,h in char_boxes:
        char_crop = clean[y:y+h, x:x+w]
        pad_x = max(0,(h-w)//2)
        padded = cv2.copyMakeBorder(char_crop,5,5,pad_x+2,pad_x+2,cv2.BORDER_CONSTANT,value=(0,0,0))
        resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
        chars.append(resized)
    print(f"[{len(chars)} karakter tersegmentasi].")
    return chars, [], ""  # bisa tambahkan CNN prediction jika ingin

# ================== SAVE CROP ==================
def save_crop():
    global current_img, current_path, poly_coords
    save_uploaded_image()
    img_h, img_w = current_img.shape[:2]
    xs = poly_coords[:,0]; ys = poly_coords[:,1]
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    x_center = ((x1+x2)/2)/img_w
    y_center = ((y1+y2)/2)/img_h
    w = (x2-x1)/img_w
    h = (y2-y1)/img_h

    base_filename = os.path.splitext(os.path.basename(current_path))[0]
    label_path = os.path.join(labels_train_folder, f"{base_filename}.txt")
    with open(label_path,"w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    print(f"Label YOLO disimpan ke: {label_path}")

    crop_img = current_img[y1:y2, x1:x2]
    gray, enhanced, thresh = preprocess_image1(crop_img, base_filename)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(gray,cmap="gray"); plt.title("Gray"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(enhanced,cmap="gray"); plt.title("CLAHE"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(thresh,cmap="gray"); plt.title("Threshold"); plt.axis("off")
    plt.suptitle(f"Preprocessing ({base_filename})"); plt.tight_layout(); plt.show()

    chars, _, cnn_string = segment_characters(thresh, base_filename)

    # ==== TAMPILKAN HASIL SEGMENTASI KARAKTER ====
    if len(chars) > 0:
        cols = min(10, len(chars))        # kolom max 10
        rows = int(np.ceil(len(chars)/cols))

        plt.figure(figsize=(cols*1.5, rows*2))
        for i, c in enumerate(chars):
            plt.subplot(rows, cols, i+1)

            # ===== tampilkan lebih besar tanpa mengubah file =====
            c_display = cv2.resize(c, (160, 240), interpolation=cv2.INTER_NEAREST)
            plt.imshow(c_display, cmap='gray')
            # =====================================================

            plt.title(f"Char {i+1}")
            plt.axis("off")

        plt.suptitle("Hasil Segmentasi Karakter")
        plt.tight_layout()
        plt.show()

        save_training_chars_manual(chars)

    crop_path = f"hasil_crop/{base_filename}_crop.png"
    cv2.imwrite(crop_path, crop_img)
    thresh_path = f"hasil_preprocessing/{base_filename}_thresh.png"
    cv2.imwrite(thresh_path, thresh)
    csv_path = f"hasil_csv/{base_filename}.csv"
    with open(csv_path,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hasil_cnn"])
        writer.writerow([cnn_string])
    print("\n[SELESAI] Semua hasil tersimpan.")

# ================== PILIH GAMBAR ==================
def upload_image():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title="Pilih Gambar", filetypes=[("Image files","*.jpg *.jpeg *.png")])
    root.destroy()
    return file_path

# ================== SHOW IMAGE ==================
def show_image(path):
    global current_img, current_path, fig, ax, poly_coords, poly_patch
    current_img = cv2.imread(path)
    img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
    current_path = path
    auto_box = auto_detect_box(img_rgb, debug=False)

    if auto_box is None:
        auto_box = np.array([[150,100],[300,100],[300,200],[150,200]])

    poly_coords = auto_box.copy()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Geser ROI lalu tekan 'C' untuk crop & segmentasi karakter")

    poly_patch = plt.Polygon(poly_coords, closed=True, fill=False, linewidth=2, edgecolor='yellow')
    ax.add_patch(poly_patch)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

# ================== MAIN ==================
if __name__ == "__main__":
    file_path = upload_image()
    if file_path:
        show_image(file_path)
    else:
        print("Tidak ada gambar yang dipilih.")
