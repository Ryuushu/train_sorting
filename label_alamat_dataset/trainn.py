import os
import glob
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Ganti path ini sesuai lokasi dataset kamu
DATASET_PATH = r'C:/Users/Taufiqur Rahman/OneDrive/Documents/train_sorting/label_alamat_dataset/'
PROCESSED_PATH = os.path.join(DATASET_PATH, 'processed')

# Path ke file data.yaml
yaml_path = r"C:/Users/Taufiqur Rahman/OneDrive/Documents/train_sorting/label_alamat_dataset/processed/data.yaml"

# 1️⃣ Cek isi YAML


os.makedirs(PROCESSED_PATH + '\\images\\train', exist_ok=True)
os.makedirs(PROCESSED_PATH + '\\images\\val', exist_ok=True)
os.makedirs(PROCESSED_PATH + '\\labels\\train', exist_ok=True)
os.makedirs(PROCESSED_PATH + '\\labels\\val', exist_ok=True)

print("📁 Folder siap:", PROCESSED_PATH)

# ============================================================

def preprocess_image_for_text(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 15
    )

   

    cv2.imwrite(output_path, thresh)


print("⚙️ Memulai preprocessing gambar...")


# ============================================================
# 4️⃣ BUAT FILE data.yaml
# ============================================================

yaml_path = os.path.join(PROCESSED_PATH, 'data.yaml')

# Ubah path ke format forward slash dulu
PROCESSED_PATH_CLEAN = PROCESSED_PATH.replace("\\", "/")

yaml_content = f"""
train: {PROCESSED_PATH_CLEAN}/images/train
val: {PROCESSED_PATH_CLEAN}/images/val

nc: 1
names: ['alamatkiri']
"""


with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print("✅ data.yaml dibuat di:", yaml_path)
for subset in ['train', 'val']:
    input_folder = os.path.join(DATASET_PATH, 'images', subset)
    output_folder = os.path.join(PROCESSED_PATH, 'images', subset)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            preprocess_image_for_text(
                os.path.join(input_folder, filename),
                os.path.join(output_folder, filename)
            )

    # Salin label YOLO apa adanya
    os.system(f'copy "{DATASET_PATH}\\labels\\{subset}\\*" "{PROCESSED_PATH}\\labels\\{subset}\\"')

print("✅ Semua gambar sudah diproses dan disalin ke folder 'processed'.")
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)
print("📁 Isi YAML:", data)

# 2️⃣ Cek apakah semua folder eksis
for key in ["train", "val"]:
    path = data[key]
    if not os.path.exists(path):
        print(f"❌ Folder {key} tidak ditemukan:", path)
    else:
        print(f"✅ Folder {key} ditemukan:", path)

# 3️⃣ Cek apakah semua gambar punya label dengan nama yang sama
train_images = glob.glob(os.path.join(data["train"], "*"))
train_labels = glob.glob(os.path.join(data["train"].replace("images", "labels"), "*"))

print(f"\n🖼️ Jumlah gambar train: {len(train_images)}")
print(f"🏷️ Jumlah label train: {len(train_labels)}")

for img in train_images:
    base = os.path.splitext(os.path.basename(img))[0]
    label_path = os.path.join(data["train"].replace("images", "labels"), base + ".txt")
    if not os.path.exists(label_path):
        print("⚠️ Label hilang untuk:", img)
# ============================================================
# 5️⃣ TRAINING YOLOv8
# ============================================================

model = YOLO('yolov8n.pt')  # kamu bisa ganti ke yolov8s.pt kalau GPU kuat
model.train(
    data=yaml_path,
    epochs=25,
    imgsz=640,
    batch=1,
    project=os.path.join(DATASET_PATH, 'result'),
    name='train_alamat_preprocessed',
    exist_ok=True
)

print("🎯 Training selesai! Cek folder 'result/train_alamat_preprocessed' di dalam dataset kamu.")

# ============================================================
# 6️⃣ TESTING MODEL
# ============================================================

model = YOLO(os.path.join(DATASET_PATH, 'result', 'train_alamat_preprocessed', 'weights', 'best.pt'))
test_folder = os.path.join(PROCESSED_PATH, 'images', 'val')

results = model.predict(source=test_folder, save=True, show=True)

print("✅ Prediksi selesai. Lihat hasil di folder 'runs/detect/predict'")