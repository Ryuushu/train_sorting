import tensorflow as tf
import numpy as np
import cv2
import os

# CONFIG
IMG_SIZE = 64                  # ukuran gambar input
MODEL_PATH = r"C:\Users\Taufiqur Rahman\OneDrive\Documents\train_sorting\label_alamat_dataset\char_model.h5"

DATASET_DIR = "train_ocr"      # folder berisi A/, B/, C/, 0/, 1/, dst

# Load model CNN
print("[INFO] Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# Ambil label berdasarkan folder

labels = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

print("[INFO] Loaded Labels:", labels)

# Fungsi preprocess gambar
def preprocess_image(img):
    """C
    Input: img (numpy array, grayscale 0–255)
    Output: img siap masuk CNN (1, 64, 64, 1)
    """

    # pastikan grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalisasi
    img = img.astype("float32") / 255.0

    # reshape ke (1, 64, 64, 1)
    img = np.expand_dims(img, axis=(0, -1))

    return img
# Prediksi dari path file

def predict_char_from_path(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"[ERROR] Gambar tidak ditemukan: {img_path}")

    return predict_char_from_array(img)

# Prediksi dari array gambar
def predict_char_from_array(img):
    "Untuk dipakai jika gambar berasal dari upload/drag-drop atau hasil crop."
    processed = preprocess_image(img)
    pred = model.predict(processed)[0]

    idx = np.argmax(pred)
    confidence = float(pred[idx])
    char = labels[idx]

    return char, confidence

# Demo penggunaan
if __name__ == "__main__":
    TEST_IMG = r"C:\Users\Taufiqur Rahman\OneDrive\Documents\train_sorting\train_ocr\P\P_2.png"  # ganti dengan filename kamu

    print("[INFO] Predicting:", TEST_IMG)
    char, conf = predict_char_from_path(TEST_IMG)

    print("=== HASIL ===")
    print(f"Prediksi Karakter : {char}")
    print(f"Confidence        : {conf:.4f}")
