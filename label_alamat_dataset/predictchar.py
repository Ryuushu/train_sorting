import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# CONFIG
IMG_SIZE = 64                  
MODEL_PATH = r"char_model.h5"
DATASET_DIR = "train_ocr"      

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
    # Pastikan grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Inverse binary threshold (foreground putih, background hitam)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Normalisasi
    img = img.astype("float32") / 255.0
    
    # Tambahkan batch dimensi dan channel dimensi
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
    processed = preprocess_image(img)
    pred = model.predict(processed)[0]
    idx = np.argmax(pred)
    confidence = float(pred[idx])
    char = labels[idx]
    return char, confidence

# Demo penggunaan dengan subplot
if __name__ == "__main__":
    TEST_IMG = r"a.png"  # ganti dengan filename kamu

    print("[INFO] Predicting:", TEST_IMG)
    char, conf = predict_char_from_path(TEST_IMG)
    print("=== HASIL ===")
    print(f"Prediksi Karakter : {char}")
    print(f"Confidence        : {conf:.4f}")

    # Tampilkan di subplot
    img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap="gray")
    plt.title(f"Prediksi: {char}\nConfidence: {conf:.2f}")
    plt.axis('off')
    plt.show()
