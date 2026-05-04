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

from config import (
    MODEL_PATH,
    BEST_MODEL_PATH,
    LABELS_PATH,
    GESTURES,
    MODEL,
    PREDICTION,
)
from preprocessing import prepare_image_for_model

def get_prediction_display(label, confidence, low_threshold=None, high_threshold=None):
    if high_threshold is None:
        high_threshold = PREDICTION.get("confidence_threshold_high", PREDICTION.get("confidence_threshold", 0.7))
    if low_threshold is None:
        low_threshold = PREDICTION.get("confidence_threshold_low", 0.5)
    color_high = PREDICTION.get("color_high_bgr", (0, 255, 0))
    color_mid = PREDICTION.get("color_mid_bgr", (0, 220, 255))
    color_low = PREDICTION.get("color_low_bgr", (0, 165, 255))

    if confidence >= high_threshold:
        return label.upper(), color_high
    if confidence >= low_threshold:
        return f"{label.upper()} (not fully certain)", color_mid
    return "UNCERTAIN", color_low

class GesturePredictor:
    def __init__(self, model_path=None, confidence_threshold=0.7):
        """Initialize the gesture predictor"""
        if model_path is None:
            model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = PREDICTION.get("confidence_threshold_low", 0.5)
        self.img_size = MODEL["img_size"]
        
        # Load class labels if available
        if os.path.exists(LABELS_PATH):
            self.labels = np.load(LABELS_PATH).tolist()
        else:
            self.labels = GESTURES
        
        # Prediction smoothing (keep last N predictions)
        self.prediction_buffer = deque(maxlen=PREDICTION["smoothing_buffer_size"])
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=PREDICTION["fps_buffer_size"])
        self.prev_time = time.time()
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📋 Classes: {self.labels}")
        print(
            "🎯 Confidence thresholds: "
            f"low >= {self.low_confidence_threshold:.2f}, "
            f"high >= {self.confidence_threshold:.2f}"
        )
    
    def preprocess_frame(self, frame):
        """Preprocess frame for prediction using the same RGB convention as training."""
        img = prepare_image_for_model(
            frame,
            target_size=self.img_size,
            input_color="bgr",
        )
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
        display_label, color = get_prediction_display(
            label,
            confidence,
            low_threshold=self.low_confidence_threshold,
            high_threshold=self.confidence_threshold,
        )
        
        cv2.putText(frame, f"Gesture: {display_label}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Top Class: {label}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
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
        confidence_threshold=PREDICTION["confidence_threshold"]
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREDICTION["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREDICTION["camera_height"])
    
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
    roi_size = PREDICTION["roi_size"]
    
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
        _, roi_color = get_prediction_display(
            predictor.labels[predicted_class],
            confidence,
            low_threshold=predictor.low_confidence_threshold,
            high_threshold=predictor.confidence_threshold,
        )
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
