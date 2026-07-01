"""
Mango Disease Detection - Model Training Script
Dataset: MangoLeafBD from Kaggle (aryashah2k/mango-leaf-disease-dataset)
Model: VGG19 Transfer Learning + Fine-tuning
Authors: Pavan Nana Badiger
Guide: Dr. Prema Akkasaligar
KLE Technological University, Belagavi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DATASET_DIR   = "dataset"          # Root folder after extracting Kaggle zip
IMG_SIZE      = (224, 224)         # VGG19 expects 224x224
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.0001
MODEL_PATH    = "mango_disease_model.h5"

# 8 classes in MangoLeafBD dataset
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# ─────────────────────────────────────────────
# 2. DATA AUGMENTATION & GENERATORS
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    validation_split=0.2          # 80% train, 20% validation
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

print(f"\n✅ Classes found: {train_generator.class_indices}")
print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}\n")

NUM_CLASSES = len(train_generator.class_indices)

# ─────────────────────────────────────────────
# 3. BUILD MODEL (VGG19 Transfer Learning)
# ─────────────────────────────────────────────
def build_model(num_classes):
    # Load VGG19 pre-trained on ImageNet, without top layers
    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    # Freeze first 15 layers, fine-tune the rest
    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
        layer.trainable = True

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model(NUM_CLASSES)
model.summary()

# ─────────────────────────────────────────────
# 4. COMPILE
# ─────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────────
# 5. CALLBACKS
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
# 6. TRAIN
# ─────────────────────────────────────────────
print("\n🚀 Starting training...\n")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 7. PLOT RESULTS
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy",      color="#2ecc71", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#e74c3c", linewidth=2)
    axes[0].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss",      color="#2ecc71", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation Loss", color="#e74c3c", linewidth=2)
    axes[1].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Plot saved as training_results.png")

plot_history(history)

# ─────────────────────────────────────────────
# 8. FINAL EVALUATION
# ─────────────────────────────────────────────
print("\n📊 Evaluating on validation set...")
loss, acc = model.evaluate(val_generator, verbose=0)
print(f"\n✅ Final Validation Accuracy : {acc * 100:.2f}%")
print(f"✅ Final Validation Loss     : {loss:.4f}")
print(f"✅ Model saved to           : {MODEL_PATH}")

# Save class indices for use in web app
import json
class_indices = train_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}
with open("class_indices.json", "w") as f:
    json.dump(inv_class_indices, f, indent=2)
print("✅ Class indices saved to class_indices.json")
