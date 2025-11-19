import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Konfigurasi --- #
DATASET_DIR = "train_ocr"
BASE_DIR = "dataset_split"
IMG_SIZE = 64
BATCH = 32
EPOCHS = 10  # 10 epoch

# --- Buat folder split --- #
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(BASE_DIR, split), exist_ok=True)

# --- Split data per kelas --- #
for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    files = os.listdir(class_dir)
    
    # Split 70% train, 15% val, 15% test
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files  = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    for split_name, split_files in zip(["train","val","test"], [train_files, val_files, test_files]):
        split_class_dir = os.path.join(BASE_DIR, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(split_class_dir, f))

# --- ImageDataGenerator --- #
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)

# --- Model CNN --- #
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE,1)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# --- Checkpoint --- #
checkpoint = ModelCheckpoint("char_model.h5", monitor='val_accuracy', save_best_only=True)

# --- Training --- #
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# --- Plot grafik training --- #
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# --- Evaluasi Testing --- #
test_loss, test_acc = model.evaluate(test_gen)
print("Testing Accuracy:", test_acc)

# --- Prediksi & Confusion Matrix --- #
y_true = test_gen.classes
y_pred_prob = model.predict(test_gen)
y_pred = np.argmax(y_pred_prob, axis=1)
class_labels = list(test_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- Classification Report --- #
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)
