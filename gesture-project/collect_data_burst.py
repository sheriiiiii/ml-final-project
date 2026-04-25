"""
Burst Mode Data Collection Script
Captures multiple photos rapidly in a single session with visual countdown
Perfect for quickly collecting diverse training data
"""

import cv2
import os
import sys
import time

def burst_capture(label, dataset_type='train', burst_size=10, burst_interval=3):
    """
    Capture multiple photos in burst mode
    
    Args:
        label: Name of the gesture
        dataset_type: 'train' or 'val'
        burst_size: Number of photos per burst
        burst_interval: Seconds between bursts
    """
    save_path = f"data/{dataset_type}/{label}"
    os.makedirs(save_path, exist_ok=True)
    
    # Count existing images
    existing_images = len([f for f in os.listdir(save_path) if f.endswith('.jpg')])
    count = existing_images
    
    print(f"\n{'='*60}")
    print(f"📸 BURST MODE: {label.upper()}")
    print(f"{'='*60}")
    print(f"Burst size: {burst_size} photos per burst")
    print(f"Interval: {burst_interval}s between bursts")
    print(f"Existing images: {existing_images}")
    print(f"\nControls:")
    print("  SPACE - Start burst capture")
    print("  Q     - Finish and move to next gesture")
    print("  ESC   - Exit program")
    print(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    roi_size = 300
    burst_active = False
    burst_start_time = 0
    burst_count = 0
    total_bursts = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2
        
        current_time = time.time()
        
        # Burst capture logic
        if burst_active:
            elapsed = current_time - burst_start_time
            photos_to_take = int(elapsed / (burst_interval / burst_size))
            
            if photos_to_take > burst_count and burst_count < burst_size:
                # Capture photo
                roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                filename = f"{save_path}/{count:04d}.jpg"
                cv2.imwrite(filename, roi)
                count += 1
                burst_count += 1
                print(f"📸 Burst {total_bursts + 1}: Photo {burst_count}/{burst_size} - {filename}")
            
            if burst_count >= burst_size:
                burst_active = False
                total_bursts += 1
                print(f"✅ Burst {total_bursts} complete! Total images: {count}")
        
        # Draw ROI
        roi_color = (0, 255, 0) if burst_active else (255, 255, 0)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size),
                     roi_color, 3)
        
        # Status display
        status = "CAPTURING!" if burst_active else "READY"
        status_color = (0, 255, 0) if burst_active else (255, 255, 255)
        
        cv2.putText(frame, f"Gesture: {label.upper()}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(frame, f"Total Images: {count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Bursts: {total_bursts}", (10, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if burst_active:
            progress = f"Burst Progress: {burst_count}/{burst_size}"
            cv2.putText(frame, progress, (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Progress bar
            bar_width = int((burst_count / burst_size) * 400)
            cv2.rectangle(frame, (10, 200), (410, 230), (100, 100, 100), 2)
            cv2.rectangle(frame, (10, 200), (10 + bar_width, 230), (0, 255, 0), -1)
        else:
            cv2.putText(frame, "Press SPACE to start burst", (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, "SPACE: Burst | Q: Next | ESC: Exit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Burst Mode Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32 and not burst_active:  # SPACE - Start burst
            burst_active = True
            burst_start_time = time.time()
            burst_count = 0
            print(f"\n🎬 Starting burst {total_bursts + 1}...")
            
        elif key == ord('q'):  # Q - Next gesture
            break
            
        elif key == 27:  # ESC - Exit
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Completed {label}: {count} total images, {total_bursts} bursts\n")
    return True

if __name__ == "__main__":
    gestures = ['l', 'peace', 'stop', 'thumbs_up']
    
    print("\n" + "="*60)
    print("🎬 BURST MODE DATA COLLECTION")
    print("="*60)
    
    # Configuration
    print("\nSelect dataset type:")
    print("1. Training data")
    print("2. Validation data")
    choice = input("\nEnter choice (1 or 2): ").strip()
    dataset_type = 'train' if choice == '1' else 'val'
    
    # Burst settings
    burst_size_input = input("\nPhotos per burst (default 10): ").strip()
    burst_size = int(burst_size_input) if burst_size_input else 10
    
    interval_input = input("Seconds for each burst (default 3): ").strip()
    burst_interval = float(interval_input) if interval_input else 3
    
    print(f"\n✅ Configuration:")
    print(f"   Dataset: {dataset_type}")
    print(f"   Burst size: {burst_size} photos")
    print(f"   Burst duration: {burst_interval}s")
    print(f"\n💡 Tip: Change hand position/angle slightly between bursts")
    print("=" * 60)
    
    # Collect for each gesture
    for gesture in gestures:
        continue_collection = burst_capture(
            label=gesture,
            dataset_type=dataset_type,
            burst_size=burst_size,
            burst_interval=burst_interval
        )
        
        if not continue_collection:
            print("\n⚠️  Collection stopped by user")
            break
    
    print("\n" + "="*60)
    print("✅ BURST MODE COLLECTION COMPLETE!")
    print("="*60)
    
    # Summary
    for dataset in ['train', 'val']:
        print(f"\n{dataset.upper()} Dataset:")
        for gesture in gestures:
            path = f"data/{dataset}/{gesture}"
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                print(f"  {gesture}: {count} images")
