"""
Enhanced Real-time Prediction Script
Improvements:
- FPS counter
- Confidence threshold
- Smoother predictions (moving average)
- Better visualization
- Hand detection ROI
- Prediction history
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
import os

class GesturePredictor:
    def __init__(self, model_path="model/gesture_model.h5", confidence_threshold=0.7):
        """Initialize the gesture predictor"""
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Load class labels if available
        labels_path = "model/class_labels.npy"
        if os.path.exists(labels_path):
            self.labels = np.load(labels_path)
        else:
            self.labels = ['l', 'peace', 'stop', 'thumbs_up']
        
        # Prediction smoothing (keep last N predictions)
        self.prediction_buffer = deque(maxlen=5)
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=30)
        self.prev_time = time.time()
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📋 Classes: {self.labels}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
    
    def preprocess_frame(self, frame, target_size=(128, 128)):
        """Preprocess frame for prediction"""
        img = cv2.resize(frame, target_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, frame):
        """Make prediction on frame"""
        preprocessed = self.preprocess_frame(frame)
        prediction = self.model.predict(preprocessed, verbose=0)[0]
        
        # Add to buffer for smoothing
        self.prediction_buffer.append(prediction)
        
        # Average predictions for stability
        smoothed_prediction = np.mean(self.prediction_buffer, axis=0)
        
        predicted_class = np.argmax(smoothed_prediction)
        confidence = smoothed_prediction[predicted_class]
        
        return predicted_class, confidence, smoothed_prediction
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)
    
    def draw_predictions(self, frame, predicted_class, confidence, all_predictions):
        """Draw prediction results on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Main prediction
        label = self.labels[predicted_class]
        color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255)
        
        cv2.putText(frame, f"Gesture: {label.upper()}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # FPS
        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # All class probabilities (right side)
        y_offset = 200
        cv2.putText(frame, "All Predictions:", (w - 250, y_offset - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, (prob, class_name) in enumerate(zip(all_predictions, self.labels)):
            y = y_offset + i * 35
            bar_width = int(prob * 200)
            
            # Draw probability bar
            cv2.rectangle(frame, (w - 250, y), (w - 250 + bar_width, y + 20),
                         (0, 255, 0), -1)
            cv2.rectangle(frame, (w - 250, y), (w - 50, y + 20),
                         (255, 255, 255), 1)
            
            # Label and percentage
            text = f"{class_name}: {prob:.1%}"
            cv2.putText(frame, text, (w - 240, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

def main():
    predictor = GesturePredictor(
        model_path="model/gesture_model.h5",
        confidence_threshold=0.7
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("\n" + "="*60)
    print("🎥 REAL-TIME GESTURE RECOGNITION")
    print("="*60)
    print("Controls:")
    print("  Q or ESC - Quit")
    print("="*60 + "\n")
    
    # ROI settings
    roi_size = 300
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # ROI position
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2
        
        # Extract ROI for prediction
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        # Make prediction
        predicted_class, confidence, all_predictions = predictor.predict(roi)
        
        # Draw ROI
        roi_color = (0, 255, 0) if confidence >= predictor.confidence_threshold else (0, 165, 255)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size),
                     roi_color, 2)
        
        # Draw predictions
        frame = predictor.draw_predictions(frame, predicted_class, confidence, all_predictions)
        
        # Instructions
        cv2.putText(frame, "Press Q or ESC to quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Gesture Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Program ended")

if __name__ == "__main__":
    main()
