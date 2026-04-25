# 🤚 Hand Gesture Recognition System

A complete machine learning project for real-time hand gesture recognition using Convolutional Neural Networks (CNN) and OpenCV.

## 📋 Features

- **Data Collection Tool** - Easy-to-use interface for collecting training data
- **Deep Learning Model** - CNN architecture with BatchNormalization and Dropout
- **Real-time Prediction** - Live webcam gesture recognition with FPS counter
- **Streamlit Web App** - Interactive web interface for testing
- **Model Evaluation** - Comprehensive performance metrics and visualizations
- **Confidence Tracking** - Shows prediction confidence and smoothed results

## 🗂️ Project Structure

```
hand-gesture-recognition/
├── data/
│   ├── train/          # Training images
│   │   ├── l/
│   │   ├── peace/
│   │   ├── stop/
│   │   └── thumbs_up/
│   └── val/            # Validation images (optional)
├── model/
│   ├── gesture_model.h5              # Trained model
│   ├── best_gesture_model.h5         # Best checkpoint
│   ├── class_labels.npy              # Class names
│   ├── training_history.png          # Training curves
│   ├── confusion_matrix.png          # Evaluation results
│   └── per_class_accuracy.png        # Per-class performance
├── collect_data.py     # Data collection script
├── train.py            # Model training script
├── predict.py          # Real-time prediction
├── app.py              # Streamlit web app
├── evaluate.py         # Model evaluation
└── requirements.txt    # Python dependencies
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Training Data

Run the data collection script to gather images for each gesture:

```bash
python collect_data.py
```

**Instructions:**
- Choose dataset type (training or validation)
- Choose capture mode:
  - **Manual Mode**: Press SPACE to capture each photo individually
  - **Auto-Capture Mode**: Photos are captured automatically at set intervals
- Position your hand in the green ROI box
- Press **A** to toggle auto-capture on/off during collection
- Press **+/-** to adjust capture interval (in auto mode)
- Press **Q** to move to the next gesture
- Press **ESC** to exit

**Alternative - Burst Mode (for rapid collection):**

```bash
python collect_data_burst.py
```

Captures multiple photos in quick succession per burst:
- Press **SPACE** to start a burst of photos
- Change hand position/angle between bursts for variety
- Perfect for quickly building diverse datasets

**Recommended:** Collect 500+ images per gesture for best results.

### 3. Train the Model

Train the CNN model on your collected data:

```bash
python train.py
```

**Features:**
- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing (saves best model)
- Training visualization graphs

**Training time:** ~10-30 minutes depending on dataset size and hardware.

### 4. Evaluate the Model

Generate comprehensive performance metrics:

```bash
python evaluate.py
```

**Outputs:**
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Per-class accuracy chart
- Confidence distribution analysis

## 🎯 Usage

### Option 1: Real-time Prediction (Webcam)

```bash
python predict.py
```

**Features:**
- Live gesture recognition
- FPS counter
- Confidence scores for all classes
- Prediction smoothing for stability
- ROI visualization

**Controls:**
- **Q** or **ESC** - Quit

### Option 2: Web App (Streamlit)

```bash
streamlit run app.py
```

**Features:**
- Camera input or image upload
- Interactive confidence visualization
- Adjustable confidence threshold
- Detailed prediction breakdown
- Model information display

Access the app at: `http://localhost:8501`

## 🎨 Supported Gestures

1. **L** - L-shape with thumb and index finger
2. **Peace** - ✌️ Peace sign (V-shape)
3. **Stop** - 🤚 Open palm (stop gesture)
4. **Thumbs Up** - 👍 Thumbs up

## 📊 Model Architecture

```
Input (128x128x3)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(4, softmax) [Output]
```

## 🔧 Key Improvements Over Original Code

### 1. **Data Collection (`collect_data.py`)**
- ✅ Visual ROI box for consistent framing
- ✅ Progress tracking and target count
- ✅ Better controls (SPACE to capture, Q to next)
- ✅ Support for both train/val datasets
- ✅ Auto-numbering and organization
- ✅ **Auto-capture mode** - captures photos automatically at intervals
- ✅ **Adjustable intervals** - change capture speed on the fly
- ✅ **Burst mode** (separate script) - rapid multi-photo capture

### 2. **Model Training (`train.py`)**
- ✅ Enhanced data augmentation
- ✅ BatchNormalization layers for faster convergence
- ✅ Early stopping to prevent overfitting
- ✅ Learning rate scheduling
- ✅ Model checkpointing (saves best model)
- ✅ Training history visualization
- ✅ Class label saving

### 3. **Real-time Prediction (`predict.py`)**
- ✅ FPS counter
- ✅ Prediction smoothing (moving average)
- ✅ Confidence threshold
- ✅ All class probabilities displayed
- ✅ Better visualization with ROI
- ✅ OOP design for cleaner code

### 4. **Streamlit App (`app.py`)**
- ✅ Professional UI with Plotly charts
- ✅ Adjustable confidence threshold
- ✅ Image upload support
- ✅ Detailed prediction breakdown
- ✅ Model information display
- ✅ Tips and instructions

### 5. **Model Evaluation (`evaluate.py`)**
- ✅ Confusion matrix visualization
- ✅ Per-class accuracy analysis
- ✅ Confidence score distribution
- ✅ Classification report
- ✅ Correct vs incorrect prediction analysis

## 💡 Tips for Best Results

### Data Collection
- 📍 Vary hand positions and angles
- 💡 Collect in different lighting conditions
- 👋 Include different backgrounds
- 🖐️ Use both left and right hands
- 📏 Vary distance from camera

### Training
- 🎯 Collect 500+ images per gesture
- ⚖️ Balance dataset (equal images per class)
- 🔄 Use data augmentation
- ⏱️ Monitor validation accuracy
- 💾 Use early stopping

### Prediction
- 📍 Center hand in ROI box
- 💡 Ensure good lighting
- 🖐️ Make clear, distinct gestures
- 🔄 Hold gesture steady for 1-2 seconds
- 📏 Keep hand at moderate distance

## 🐛 Troubleshooting

### Camera Not Opening
```bash
# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Model Not Found
- Ensure you've run `train.py` first
- Check that `model/gesture_model.h5` exists

### Low Accuracy
- Collect more training data (500+ per gesture)
- Ensure good variety in dataset
- Check data augmentation settings
- Try training for more epochs

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📈 Performance Metrics

Expected performance with well-collected data:
- **Training Accuracy:** 95-99%
- **Validation Accuracy:** 90-95%
- **Inference Speed:** 30+ FPS (CPU)
- **Confidence Scores:** 85-99% for clear gestures

## 🔮 Future Improvements

- [ ] Add more gesture classes
- [ ] Hand detection with MediaPipe
- [ ] Data augmentation preview
- [ ] Export to TensorFlow Lite for mobile
- [ ] Multi-hand gesture recognition
- [ ] Gesture sequence detection
- [ ] Background removal
- [ ] Transfer learning with pre-trained models

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Built with:
- TensorFlow/Keras for deep learning
- OpenCV for computer vision
- Streamlit for web interface
- Plotly for interactive visualizations

---

**Happy Gesture Recognition! 🤚✌️👍**
