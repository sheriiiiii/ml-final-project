"""
Model training script.

Key improvements:
- Uses dedicated data/val when available (better validation signal)
- Uses MobileNetV2 transfer learning by default for stronger generalization
- Uses class weights and label smoothing for better class balance
- Stores labels in deterministic index order
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2

from config import (
    TRAIN_DIR,
    VAL_DIR,
    MODEL_DIR,
    MODEL_PATH,
    BEST_MODEL_PATH,
    LABELS_PATH,
    GESTURES,
    MODEL,
    TRAINING,
)
from preprocessing import strip_collection_overlays


SEED = 42


def _has_images(directory):
    if not os.path.isdir(directory):
        return False
    for file_name in os.listdir(directory):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            return True
    return False


def _has_full_validation_set():
    """Validation is considered valid when every gesture directory has images."""
    if not os.path.isdir(VAL_DIR):
        return False
    return all(_has_images(os.path.join(VAL_DIR, gesture)) for gesture in GESTURES)


def _build_generators(img_size, batch_size):
    aug = TRAINING["augmentation"]

    # Dedicated validation set is preferred when available.
    if _has_full_validation_set():
        print("\nUsing dedicated validation dataset from data/val")

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            preprocessing_function=strip_collection_overlays,
            rotation_range=aug["rotation_range"],
            width_shift_range=aug["width_shift_range"],
            height_shift_range=aug["height_shift_range"],
            shear_range=aug["shear_range"],
            zoom_range=aug["zoom_range"],
            horizontal_flip=aug["horizontal_flip"],
            fill_mode=aug["fill_mode"],
        )
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            preprocessing_function=strip_collection_overlays,
        )

        train_data = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=SEED,
        )
        val_data = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
        source = "explicit-val"
    else:
        print("\nNo full data/val found. Falling back to validation split from data/train")

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            preprocessing_function=strip_collection_overlays,
            rotation_range=aug["rotation_range"],
            width_shift_range=aug["width_shift_range"],
            height_shift_range=aug["height_shift_range"],
            shear_range=aug["shear_range"],
            zoom_range=aug["zoom_range"],
            horizontal_flip=aug["horizontal_flip"],
            fill_mode=aug["fill_mode"],
            validation_split=TRAINING["validation_split"],
        )
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            preprocessing_function=strip_collection_overlays,
            validation_split=TRAINING["validation_split"],
        )

        train_data = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=SEED,
        )
        val_data = val_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=SEED,
        )
        source = "split-from-train"

    if train_data.class_indices != val_data.class_indices:
        raise ValueError(
            "Class index mismatch between train and validation generators: "
            f"train={train_data.class_indices}, val={val_data.class_indices}"
        )

    return train_data, val_data, source


def _build_model(img_size, num_classes):
    architecture = MODEL.get("architecture", "mobilenetv2").lower()

    if architecture == "mobilenetv2":
        inputs = Input(shape=(img_size, img_size, 3))
        base_model = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )
        base_model.trainable = False

        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.35)(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs, name="gesture_mobilenetv2")
        print("Using MobileNetV2 transfer learning backbone")
    else:
        model = Sequential(
            [
                Input(shape=(img_size, img_size, 3)),
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.2),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.3),
                GlobalAveragePooling2D(),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.4),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ],
            name="gesture_custom_cnn",
        )
        print("Using custom CNN backbone")

    model.compile(
        optimizer=Adam(learning_rate=TRAINING["learning_rate"]),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def _plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    history_path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(history_path, dpi=150)
    print(f"✅ Training history plot saved to {history_path}")
    plt.show()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    img_size = MODEL["img_size"]
    batch_size = TRAINING["batch_size"]
    epochs = TRAINING["epochs"]

    train_data, val_data, validation_source = _build_generators(img_size, batch_size)

    print(f"\nClasses found: {train_data.class_indices}")
    print(f"Training samples: {train_data.samples}")
    print(f"Validation samples: {val_data.samples}")
    print(f"Validation source: {validation_source}")

    # Deterministic label order must match model output index order.
    class_labels = np.array(
        [label for label, idx in sorted(train_data.class_indices.items(), key=lambda item: item[1])]
    )
    np.save(LABELS_PATH, class_labels)
    print(f"✅ Class labels saved to {LABELS_PATH}: {class_labels.tolist()}")

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_data.classes),
        y=train_data.classes,
    )
    class_weights = {idx: float(weight) for idx, weight in enumerate(class_weights_array)}
    print(f"Class weights: {class_weights}")

    model = _build_model(img_size, train_data.num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=TRAINING["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=TRAINING["reduce_lr_factor"],
            patience=TRAINING["reduce_lr_patience"],
            min_lr=TRAINING["min_lr"],
            verbose=1,
        ),
    ]

    print("\nStarting training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"\n✅ Final model saved to {MODEL_PATH}")

    _plot_training_history(history)

    print("\nFinal model evaluation:")
    val_loss, val_accuracy = model.evaluate(val_data, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    if os.path.exists(BEST_MODEL_PATH):
        best_model = load_model(BEST_MODEL_PATH)
        best_loss, best_accuracy = best_model.evaluate(val_data, verbose=0)
        print("\nBest checkpoint evaluation:")
        print(f"Best Validation Loss: {best_loss:.4f}")
        print(f"Best Validation Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
