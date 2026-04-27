"""
Shared preprocessing utilities for gesture model training and inference.
"""

import cv2
import numpy as np


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
