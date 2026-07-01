# ╔══════════════════════════════════════════════════════════════╗
# ║       MANGO DISEASE DETECTION - KAGGLE NOTEBOOK              ║
# ║       KLE Technological University, Belagavi                 ║
# ║       Team: Pavan Nana Badiger                               ║
# ╚══════════════════════════════════════════════════════════════╝
#
# HOW TO USE THIS ON KAGGLE:
# 1. Go to https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset
# 2. Click "New Notebook"
# 3. Paste this entire code
# 4. Settings (right sidebar) → Accelerator → GPU T4 x2
# 5. Click Run All
# 6. Download mango_disease_model.h5 from the output panel on the right

# ─────────────────────────────────────────────
# CELL 1 — Check GPU
# ─────────────────────────────────────────────
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ─────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────
import os, json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# CELL 3 — Download & locate dataset
# ─────────────────────────────────────────────
import kagglehub

path = kagglehub.dataset_download("warcoder/mangofruitdds")
print("Dataset downloaded to:", path)

# Find the actual folder containing class subfolders
DATASET_DIR = path
for root, dirs, files in os.walk(path):
    if dirs:  # first folder that has subfolders = class folders
        DATASET_DIR = root
        break

print("Using dataset directory:", DATASET_DIR)

# Verify dataset is visible
for folder in sorted(os.listdir(DATASET_DIR)):
    full_path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(full_path):
        count = len(os.listdir(full_path))
        print(f"  {folder}: {count} images")

# ─────────────────────────────────────────────
# CELL 4 — Configuration
# ─────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.0001
MODEL_PATH    = "/kaggle/working/mango_disease_model.h5"    # output folder

# ─────────────────────────────────────────────
# CELL 5 — Data Generators with Augmentation
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print(f"\n✅ Classes  : {train_generator.class_indices}")
print(f"✅ Train    : {train_generator.samples} images")
print(f"✅ Validate : {val_generator.samples} images")

NUM_CLASSES = len(train_generator.class_indices)

# ─────────────────────────────────────────────
# CELL 6 — Build VGG19 Model
# ─────────────────────────────────────────────
def build_model(num_classes):
    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    # Freeze early layers, fine-tune last block
    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=predictions)

model = build_model(NUM_CLASSES)
model.summary()

# ─────────────────────────────────────────────
# CELL 7 — Compile
# ─────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────────
# CELL 8 — Callbacks
# ─────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ─────────────────────────────────────────────
# CELL 9 — Train!
# ─────────────────────────────────────────────
print("\n🚀 Training started...\n")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# CELL 10 — Plot Accuracy & Loss
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="Train Accuracy",      color="#2ecc71", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#e74c3c", linewidth=2)
axes[0].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train Loss",      color="#2ecc71", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Validation Loss", color="#e74c3c", linewidth=2)
axes[1].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/kaggle/working/training_results.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# CELL 11 — Final Accuracy
# ─────────────────────────────────────────────
loss, acc = model.evaluate(val_generator, verbose=0)
print(f"\n✅ Final Validation Accuracy : {acc * 100:.2f}%")
print(f"✅ Final Validation Loss     : {loss:.4f}")

# ─────────────────────────────────────────────
# CELL 12 — Save class indices for web app
# ─────────────────────────────────────────────
inv_class_indices = {str(v): k for k, v in train_generator.class_indices.items()}
with open("/kaggle/working/class_indices.json", "w") as f:
    json.dump(inv_class_indices, f, indent=2)

print("\n📦 Files saved to /kaggle/working/:")
print("   ✅ mango_disease_model.h5")
print("   ✅ class_indices.json")
print("   ✅ training_results.png")
print("\n👉 Download them from the Output panel on the right sidebar!")
