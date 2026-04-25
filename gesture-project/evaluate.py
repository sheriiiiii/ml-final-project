"""
Model Evaluation Script
Generate comprehensive performance metrics and visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model(model_path="model/gesture_model.h5", data_path="data/train"):
    """Evaluate model and generate performance report"""
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    
    # Load labels
    labels_path = "model/class_labels.npy"
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
    else:
        labels = ['l', 'peace', 'stop', 'thumbs_up']
    
    # Prepare validation data (using validation split)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    val_data = val_datagen.flow_from_directory(
        data_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
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
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = 'model/confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to {cm_path}")
    plt.show()
    
    # Calculate per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY")
    print("="*70)
    
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
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
    acc_path = 'model/per_class_accuracy.png'
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    print(f"✅ Accuracy plot saved to {acc_path}")
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
    conf_path = 'model/confidence_distribution.png'
    plt.savefig(conf_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confidence distribution saved to {conf_path}")
    plt.show()
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    evaluate_model()
