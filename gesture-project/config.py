"""
Configuration file for Hand Gesture Recognition System
Centralized settings for easy customization
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

def _resolve_project_dir(folder_name):
    """Resolve folder from workspace root first, then project root."""
    workspace_candidate = os.path.join(WORKSPACE_ROOT, folder_name)
    project_candidate = os.path.join(PROJECT_ROOT, folder_name)

    if os.path.isdir(workspace_candidate):
        return workspace_candidate
    if os.path.isdir(project_candidate):
        return project_candidate

    # Default to workspace root layout to match repository structure.
    return workspace_candidate

DATA_DIR = _resolve_project_dir('data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
MODEL_DIR = _resolve_project_dir('model')

# Model files
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_model.h5')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_gesture_model.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels.npy')

# ============================================================================
# GESTURE CLASSES
# ============================================================================
GESTURES = ['l', 'peace', 'stop', 'thumbs_up']

# ============================================================================
# DATA COLLECTION SETTINGS
# ============================================================================
DATA_COLLECTION = {
    'roi_size': 300,                # Size of region of interest box
    'target_count': 500,            # Target images per gesture
    'camera_width': 640,            # Camera resolution width
    'camera_height': 480,           # Camera resolution height
    'save_format': 'jpg',           # Image format (jpg, png)
}

# ============================================================================
# MODEL ARCHITECTURE SETTINGS
# ============================================================================
MODEL = {
    'img_size': 128,                # Input image size (128x128)
    'channels': 3,                  # RGB channels
    'architecture': 'mobilenetv2',  # Model backbone (mobilenetv2, custom_cnn)
    'dropout_rate': 0.5,            # Dropout rate for regularization
    'batch_norm': True,             # Use batch normalization
}

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
TRAINING = {
    'batch_size': 32,               # Batch size for training
    'epochs': 50,                   # Maximum epochs
    'learning_rate': 0.001,         # Initial learning rate
    'validation_split': 0.2,        # Validation data split
    
    # Data augmentation
    'augmentation': {
        'rotation_range': 12,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.1,
        'zoom_range': 0.1,
        'horizontal_flip': False,
        'fill_mode': 'nearest'
    },
    
    # Callbacks
    'early_stopping_patience': 10,  # Epochs to wait before stopping
    'reduce_lr_patience': 5,        # Epochs to wait before reducing LR
    'reduce_lr_factor': 0.5,        # Factor to reduce LR by
    'min_lr': 1e-7,                 # Minimum learning rate
}

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================
PREDICTION = {
    'confidence_threshold': 0.7,    # Minimum confidence for valid prediction
    'smoothing_buffer_size': 5,     # Number of predictions to average
    'fps_buffer_size': 30,          # Number of frames for FPS calculation
    'roi_size': 300,                # ROI size for prediction
    'camera_width': 640,
    'camera_height': 480,
}

# ============================================================================
# STREAMLIT APP SETTINGS
# ============================================================================
STREAMLIT = {
    'page_title': 'Hand Gesture Identifier',
    'page_icon': '🤚',
    'layout': 'wide',
    'default_confidence': 0.7,
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
VISUALIZATION = {
    'dpi': 150,                     # DPI for saved figures
    'figsize': (14, 5),            # Default figure size
    'color_scheme': {
        'primary': '#4CAF50',       # Green
        'secondary': '#2196F3',     # Blue
        'warning': '#FF9800',       # Orange
        'error': '#F44336',         # Red
        'success': '#00ff00',       # Bright green
    }
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVALUATION = {
    'confusion_matrix_cmap': 'Blues',
    'save_plots': True,
    'show_plots': True,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def ensure_directories():
    """Create all necessary directories"""
    directories = [
        MODEL_DIR,
        TRAIN_DIR,
        VAL_DIR,
    ]
    
    # Create gesture subdirectories
    for gesture in GESTURES:
        directories.append(os.path.join(TRAIN_DIR, gesture))
        directories.append(os.path.join(VAL_DIR, gesture))
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_gesture_count(dataset_type='train'):
    """Get count of images for each gesture"""
    counts = {}
    base_dir = TRAIN_DIR if dataset_type == 'train' else VAL_DIR
    
    for gesture in GESTURES:
        gesture_dir = os.path.join(base_dir, gesture)
        if os.path.exists(gesture_dir):
            counts[gesture] = len([
                f for f in os.listdir(gesture_dir) 
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ])
        else:
            counts[gesture] = 0
    
    return counts

def print_config_summary():
    """Print configuration summary"""
    print("="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\n📂 Project Root: {PROJECT_ROOT}")
    print(f"📊 Model Path: {MODEL_PATH}")
    print(f"🎯 Gestures: {', '.join(GESTURES)}")
    print(f"📐 Image Size: {MODEL['img_size']}x{MODEL['img_size']}")
    print(f"🔄 Batch Size: {TRAINING['batch_size']}")
    print(f"📈 Max Epochs: {TRAINING['epochs']}")
    print(f"💯 Confidence Threshold: {PREDICTION['confidence_threshold']}")
    
    print("\n📊 Dataset Status:")
    train_counts = get_gesture_count('train')
    for gesture, count in train_counts.items():
        print(f"  {gesture:15s}: {count:4d} images")
    
    print("="*70)

if __name__ == "__main__":
    ensure_directories()
    print_config_summary()
