"""
Shared preprocessing utilities for gesture model training and inference.
"""

import os

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


def strip_collection_overlays(image):
    """
    Reduce ROI border/text artifacts introduced by the collection UI.

    This function is safe for both legacy data (with overlays) and new clean data.
    """
    if image is None or image.ndim != 3:
        return image

    processed = image.copy()
    h, w = processed.shape[:2]

    border = max(2, int(0.02 * min(h, w)))

    # Neutralize possible green ROI border artifacts.
    processed[:border, :, :] = 0
    processed[h - border :, :, :] = 0
    processed[:, :border, :] = 0
    processed[:, w - border :, :] = 0

    # Neutralize top-left progress text artifacts from collection mode.
    text_h = max(10, int(0.14 * h))
    text_w = max(40, int(0.36 * w))
    processed[:text_h, :text_w, :] = 0

    return processed


def prepare_image_for_model(image, target_size=128, input_color="rgb"):
    """
    Prepare an image for model inference with consistent color and scaling.

    input_color can be: rgb, bgr, gray, rgba.
    """
    if image is None:
        raise ValueError("Input image is None")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    color_mode = input_color.lower()
    if color_mode == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif color_mode == "rgba":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    image = strip_collection_overlays(image)

    return image.astype(np.float32) / 255.0


def load_image_paths_and_labels(data_dir, gestures):
    """
    Load image paths and labels from both flat and nested session structure.

    Supports:
    - legacy: data/{split}/{gesture}/*.jpg
    - nested: data/{split}/{gesture}/{session}/*.jpg
    """
    image_paths = []
    labels = []
    sorted_gestures = sorted(gestures)
    class_indices = {gesture: idx for idx, gesture in enumerate(sorted_gestures)}

    for gesture in sorted_gestures:
        gesture_dir = os.path.join(data_dir, gesture)
        if not os.path.isdir(gesture_dir):
            continue

        # Legacy flat files.
        for name in os.listdir(gesture_dir):
            file_path = os.path.join(gesture_dir, name)
            if os.path.isfile(file_path) and name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(file_path)
                labels.append(class_indices[gesture])

        # Nested session files.
        for session_name in os.listdir(gesture_dir):
            session_dir = os.path.join(gesture_dir, session_name)
            if not os.path.isdir(session_dir):
                continue
            for name in os.listdir(session_dir):
                file_path = os.path.join(session_dir, name)
                if os.path.isfile(file_path) and name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(file_path)
                    labels.append(class_indices[gesture])

    return image_paths, np.array(labels, dtype=np.int32), class_indices


class SessionDataSequence(Sequence):
    """Keras Sequence for nested gesture/session/image training data."""

    def __init__(
        self,
        data_dir,
        gestures,
        batch_size=32,
        target_size=128,
        augment=False,
        shuffle=True,
        augmentation_config=None,
        seed=42,
    ):
        self.data_dir = data_dir
        self.gestures = sorted(gestures)
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.image_paths, self._classes, self.class_indices = load_image_paths_and_labels(
            self.data_dir, self.gestures
        )
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")

        self._num_classes = len(self.class_indices)
        self.labels_one_hot = np.eye(self._num_classes, dtype=np.float32)[self._classes]

        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            self.rng.shuffle(self.indices)

        aug = augmentation_config or {}
        if self.augment:
            self.datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                preprocessing_function=strip_collection_overlays,
                rotation_range=aug.get("rotation_range", 12),
                width_shift_range=aug.get("width_shift_range", 0.1),
                height_shift_range=aug.get("height_shift_range", 0.1),
                shear_range=aug.get("shear_range", 0.1),
                zoom_range=aug.get("zoom_range", 0.1),
                horizontal_flip=aug.get("horizontal_flip", False),
                fill_mode=aug.get("fill_mode", "nearest"),
            )
        else:
            self.datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                preprocessing_function=strip_collection_overlays,
            )

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    @property
    def samples(self):
        return len(self.image_paths)

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indices]
        batch_labels = self.labels_one_hot[batch_indices]

        batch_images = []
        valid_labels = []
        for img_path, label in zip(batch_paths, batch_labels):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
            batch_images.append(img)
            valid_labels.append(label)

        if not batch_images:
            return (
                np.zeros((1, self.target_size, self.target_size, 3), dtype=np.float32),
                np.zeros((1, self._num_classes), dtype=np.float32),
            )

        x_batch = np.array(batch_images, dtype=np.float32)
        y_batch = np.array(valid_labels, dtype=np.float32)

        if self.augment:
            flow = self.datagen.flow(x_batch, y_batch, batch_size=len(x_batch), shuffle=False)
            x_batch, y_batch = next(flow)
        else:
            x_batch = self.datagen.standardize(x_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
