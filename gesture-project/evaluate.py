"""
Model Evaluation Script
Generate comprehensive performance metrics and visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

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
    EVALUATION,
)
from preprocessing import strip_collection_overlays
from preprocessing import strip_collection_overlays, SessionDataSequence

def _has_images(directory):
    if not os.path.isdir(directory):
        return False
    return any(
        file_name.lower().endswith((".jpg", ".jpeg", ".png"))
        for file_name in os.listdir(directory)
    )


def _has_full_validation_set():
    if not os.path.isdir(VAL_DIR):
        return False
    for gesture in GESTURES:
        gesture_dir = os.path.join(VAL_DIR, gesture)
        has_images = _has_images(gesture_dir) or _has_nested_sessions(gesture_dir)
        if not has_images:
            return False
    return True


def _has_nested_sessions(gesture_dir):
    if not os.path.isdir(gesture_dir):
        return False
    for item in os.listdir(gesture_dir):
        item_path = os.path.join(gesture_dir, item)
        if os.path.isdir(item_path):
            for file_name in os.listdir(item_path):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    return True
    return False


def _load_labels(class_indices):
    labels_from_indices = [
        label for label, idx in sorted(class_indices.items(), key=lambda item: item[1])
    ]

    if os.path.exists(LABELS_PATH):
        labels = np.load(LABELS_PATH).tolist()
        if labels != labels_from_indices:
            print("Warning: label file does not match generator indices. Using generator labels.")
            labels = labels_from_indices
    else:
        labels = labels_from_indices

    return labels


def evaluate_model(model_path=None):
    """Evaluate model and generate performance report"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if model_path is None:
        model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    
    # Load model
    print("Loading model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)

    # Prepare validation data. Prefer explicit data/val when available.
    has_nested_val = any(_has_nested_sessions(os.path.join(VAL_DIR, gesture)) for gesture in GESTURES)
    if _has_full_validation_set() and has_nested_val:
        print("Using dedicated validation dataset from data/val (nested session structure)")
        val_data = SessionDataSequence(
            data_dir=VAL_DIR,
            gestures=GESTURES,
            batch_size=TRAINING["batch_size"],
            target_size=MODEL["img_size"],
            augment=False,
            shuffle=False,
            seed=42,
        )
    elif _has_full_validation_set():
        print("Using dedicated validation dataset from data/val")
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            preprocessing_function=strip_collection_overlays,
        )
        val_data = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(MODEL["img_size"], MODEL["img_size"]),
            batch_size=TRAINING["batch_size"],
            class_mode="categorical",
            shuffle=False,
        )
    else:
        print("Using validation split from data/train")
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            preprocessing_function=strip_collection_overlays,
            validation_split=TRAINING["validation_split"],
        )
        val_data = val_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(MODEL["img_size"], MODEL["img_size"]),
            batch_size=TRAINING["batch_size"],
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=42,
        )

    labels = _load_labels(val_data.class_indices)
    
    print(f"\nEvaluating on {val_data.samples} validation samples...")
    
    # Get predictions
    predictions = model.predict(val_data, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_data.classes
    
    # Calculate metrics
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(
        true_classes, 
        predicted_classes, 
        target_names=labels,
        digits=4,
        zero_division=0,
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=EVALUATION['confusion_matrix_cmap'],
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    if EVALUATION['save_plots']:
        plt.savefig(
            cm_path,
            dpi=150,
            bbox_inches='tight',
        )
    print(f"\n✅ Confusion matrix saved to {cm_path}")
    if EVALUATION['show_plots']:
        plt.show()
    
    # Calculate per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY")
    print("="*70)
    
    class_accuracies = np.divide(
        cm.diagonal(),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=float),
        where=cm.sum(axis=1) != 0,
    )
    for i, (label, acc) in enumerate(zip(labels, class_accuracies)):
        print(f"{label:15s}: {acc:.2%}")
    
    # Overall accuracy
    overall_acc = np.sum(cm.diagonal()) / np.sum(cm)
    print(f"\n{'Overall Accuracy':15s}: {overall_acc:.2%}")
    
    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, class_accuracies * 100, color='#4CAF50', alpha=0.8)
    plt.axhline(y=overall_acc * 100, color='r', linestyle='--', 
                label=f'Overall: {overall_acc:.1%}')
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Gesture', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    acc_path = os.path.join(MODEL_DIR, 'per_class_accuracy.png')
    if EVALUATION['save_plots']:
        plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    print(f"✅ Accuracy plot saved to {acc_path}")
    if EVALUATION['show_plots']:
        plt.show()
    
    # Confidence distribution
    confidence_scores = np.max(predictions, axis=1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidence_scores, bins=50, color='#2196F3', alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(confidence_scores), color='r', linestyle='--', 
                label=f'Mean: {np.mean(confidence_scores):.3f}')
    plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    correct_mask = predicted_classes == true_classes
    plt.hist([confidence_scores[correct_mask], confidence_scores[~correct_mask]], 
             bins=30, label=['Correct', 'Incorrect'], 
             color=['#4CAF50', '#F44336'], alpha=0.7)
    plt.title('Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    conf_path = os.path.join(MODEL_DIR, 'confidence_distribution.png')
    if EVALUATION['save_plots']:
        plt.savefig(conf_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confidence distribution saved to {conf_path}")
    if EVALUATION['show_plots']:
        plt.show()
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    evaluate_model()
