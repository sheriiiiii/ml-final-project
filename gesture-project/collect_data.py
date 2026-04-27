"""
Enhanced Data Collection Script
Improvements:
- Visual feedback with ROI box
- Better organization by gesture
- Counter display
- Instructions on screen
- Auto-save with spacebar
- Validation data collection option
- Auto-capture mode for rapid collection
- Burst mode for multiple captures
"""

import cv2
import os
import sys
import time

def collect_data(label, dataset_type='train', target_count=500, auto_capture=False, capture_interval=0.5):
    """
    Collect training or validation data for a specific gesture
    
    Args:
        label: Name of the gesture (e.g., 'peace', 'stop')
        dataset_type: 'train' or 'val'
        target_count: Number of images to collect
        auto_capture: Enable auto-capture mode (captures automatically)
        capture_interval: Seconds between auto-captures (default: 0.5)
    """
    save_path = f"data/{dataset_type}/{label}"
    os.makedirs(save_path, exist_ok=True)
    
    # Count existing images
    existing_images = len([f for f in os.listdir(save_path) if f.endswith('.jpg')])
    count = existing_images
    
    print(f"\n{'='*60}")
    print(f"📸 COLLECTING {dataset_type.upper()} DATA FOR: {label.upper()}")
    print(f"{'='*60}")
    print(f"Existing images: {existing_images}")
    print(f"Target: {target_count} images")
    print(f"Mode: {'AUTO-CAPTURE' if auto_capture else 'MANUAL'}")
    if auto_capture:
        print(f"Interval: {capture_interval}s between captures")
    print(f"\nControls:")
    print("  SPACE - Capture image (manual) / Toggle auto-capture")
    print("  A     - Toggle auto-capture mode")
    print("  +/-   - Adjust capture interval (auto mode)")
    print("  Q     - Quit and move to next gesture")
    print("  ESC   - Exit program")
    print(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    # ROI (Region of Interest) settings
    roi_size = 300
    
    # Auto-capture settings
    last_capture_time = time.time()
    is_auto_capturing = auto_capture
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the image
        raw_frame = frame.copy()     # Keep a clean frame for saving dataset images
        h, w = frame.shape[:2]
        
        # Calculate ROI position (center of frame)
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2
        
        # Auto-capture logic
        current_time = time.time()
        should_auto_capture = (is_auto_capturing and 
                              current_time - last_capture_time >= capture_interval and
                              count < target_count)
        
        # Draw ROI rectangle (color indicates auto-capture status)
        roi_color = (0, 255, 0) if is_auto_capturing else (255, 255, 0)
        cv2.rectangle(frame, 
                     (roi_x, roi_y), 
                     (roi_x + roi_size, roi_y + roi_size),
                     roi_color, 2)
        
        # Add text instructions
        mode_text = "AUTO" if is_auto_capturing else "MANUAL"
        mode_color = (0, 255, 0) if is_auto_capturing else (255, 255, 0)
        
        cv2.putText(frame, f"Gesture: {label.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {mode_text}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        progress = f"{count}/{target_count}"
        cv2.putText(frame, f"Progress: {progress}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show countdown for auto-capture
        if is_auto_capturing and count < target_count:
            time_until_next = capture_interval - (current_time - last_capture_time)
            if time_until_next > 0:
                countdown_text = f"Next in: {time_until_next:.1f}s"
                cv2.putText(frame, countdown_text, (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, "SPACE/A: Toggle Auto | +/-: Interval | Q: Next | ESC: Exit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show target completion
        if count >= target_count:
            cv2.putText(frame, "TARGET REACHED!", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            is_auto_capturing = False  # Stop auto-capture when target reached
        
        cv2.imshow("Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Auto-capture
        if should_auto_capture:
            roi = raw_frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            filename = f"{save_path}/{count:04d}.jpg"
            cv2.imwrite(filename, roi)
            count += 1
            last_capture_time = current_time
            print(f"✅ Auto-captured: {filename} ({count}/{target_count})")
        
        # Manual controls
        if key == 32:  # SPACE - Manual capture or toggle auto-capture
            if is_auto_capturing:
                is_auto_capturing = False
                print("⏸️  Auto-capture paused")
            else:
                # Manual capture
                roi = raw_frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                filename = f"{save_path}/{count:04d}.jpg"
                cv2.imwrite(filename, roi)
                count += 1
                print(f"✅ Saved: {filename} ({count}/{target_count})")
            
        elif key == ord('a') or key == ord('A'):  # A - Toggle auto-capture
            is_auto_capturing = not is_auto_capturing
            last_capture_time = time.time()  # Reset timer
            status = "enabled" if is_auto_capturing else "disabled"
            print(f"🔄 Auto-capture {status}")
            
        elif key == ord('+') or key == ord('='):  # + - Increase interval
            capture_interval = min(capture_interval + 0.1, 5.0)
            print(f"⏱️  Capture interval: {capture_interval:.1f}s")
            
        elif key == ord('-') or key == ord('_'):  # - - Decrease interval
            capture_interval = max(capture_interval - 0.1, 0.1)
            print(f"⏱️  Capture interval: {capture_interval:.1f}s")
            
        elif key == ord('q'):  # Q - Next gesture
            break
            
        elif key == 27:  # ESC - Exit program
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Completed {label}: {count} images collected\n")
    return True

if __name__ == "__main__":
    # Configuration
    gestures = ['l', 'peace', 'stop', 'thumbs_up']
    images_per_gesture = 500
    
    print("\n" + "="*60)
    print("🤖 HAND GESTURE DATA COLLECTION TOOL")
    print("="*60)
    
    # Choose dataset type
    print("\nSelect dataset type:")
    print("1. Training data")
    print("2. Validation data")
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    dataset_type = 'train' if choice == '1' else 'val'
    
    # Choose capture mode
    print("\nSelect capture mode:")
    print("1. Manual (press SPACE to capture each photo)")
    print("2. Auto-capture (captures automatically at intervals)")
    mode_choice = input("\nEnter choice (1 or 2): ").strip()
    
    auto_capture = (mode_choice == '2')
    capture_interval = 0.5  # Default interval
    
    if auto_capture:
        interval_input = input("\nCapture interval in seconds (default 0.5): ").strip()
        if interval_input:
            try:
                capture_interval = float(interval_input)
                capture_interval = max(0.1, min(capture_interval, 5.0))  # Clamp between 0.1 and 5
            except ValueError:
                print("Invalid input, using default 0.5s")
                capture_interval = 0.5
    
    # Collect data for each gesture
    for gesture in gestures:
        continue_collection = collect_data(
            label=gesture,
            dataset_type=dataset_type,
            target_count=images_per_gesture,
            auto_capture=auto_capture,
            capture_interval=capture_interval
        )
        
        if not continue_collection:
            print("\n⚠️  Collection stopped by user")
            break
    
    print("\n" + "="*60)
    print("✅ DATA COLLECTION COMPLETE!")
    print("="*60)
    
    # Print summary
    for dataset in ['train', 'val']:
        print(f"\n{dataset.upper()} Dataset:")
        for gesture in gestures:
            path = f"data/{dataset}/{gesture}"
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                print(f"  {gesture}: {count} images")
