"""
Enhanced Data Collection Script
Improvements:
- Visual feedback with ROI box
- Counter display
- Auto-capture mode
- Session-based data organization
"""

import cv2
import os
import time

from config import get_gesture_count_all_sessions, get_session_list, session_exists


def collect_data(label, session_name, dataset_type="train", target_count=500, auto_capture=False, capture_interval=0.5):
    """Collect training or validation data for a specific gesture and session."""
    save_path = f"data/{dataset_type}/{label}/{session_name}"
    os.makedirs(save_path, exist_ok=True)

    existing_images = len([f for f in os.listdir(save_path) if f.lower().endswith(".jpg")])
    count = existing_images

    print(f"\n{'='*60}")
    print(f"COLLECTING {dataset_type.upper()} DATA FOR: {label.upper()} [{session_name}]")
    print(f"{'='*60}")
    print(f"Existing images in this session: {existing_images}")
    print(f"Target: {target_count} images")
    print(f"Mode: {'AUTO-CAPTURE' if auto_capture else 'MANUAL'}")
    if auto_capture:
        print(f"Interval: {capture_interval}s between captures")
    print("\nControls:")
    print("  SPACE - Capture image (manual) / Toggle auto-capture")
    print("  A     - Toggle auto-capture mode")
    print("  +/-   - Adjust capture interval (auto mode)")
    print("  Q     - Quit and move to next gesture")
    print("  ESC   - Exit program")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return False

    roi_size = 300
    last_capture_time = time.time()
    is_auto_capturing = auto_capture

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        raw_frame = frame.copy()
        h, w = frame.shape[:2]

        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2

        current_time = time.time()
        should_auto_capture = (
            is_auto_capturing
            and current_time - last_capture_time >= capture_interval
            and count < target_count
        )

        roi_color = (0, 255, 0) if is_auto_capturing else (255, 255, 0)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), roi_color, 2)

        mode_text = "AUTO" if is_auto_capturing else "MANUAL"
        mode_color = (0, 255, 0) if is_auto_capturing else (255, 255, 0)

        cv2.putText(frame, f"Gesture: {label.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Session: {session_name}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {mode_text}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        cv2.putText(frame, f"Progress: {count}/{target_count}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if is_auto_capturing and count < target_count:
            time_until_next = capture_interval - (current_time - last_capture_time)
            if time_until_next > 0:
                cv2.putText(
                    frame,
                    f"Next in: {time_until_next:.1f}s",
                    (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

        cv2.putText(
            frame,
            "SPACE/A: Toggle Auto | +/-: Interval | Q: Next | ESC: Exit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if count >= target_count:
            cv2.putText(frame, "TARGET REACHED", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            is_auto_capturing = False

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if should_auto_capture:
            roi = raw_frame[roi_y : roi_y + roi_size, roi_x : roi_x + roi_size]
            filename = f"{save_path}/{count:04d}.jpg"
            cv2.imwrite(filename, roi)
            count += 1
            last_capture_time = current_time
            print(f"Auto-captured: {filename} ({count}/{target_count})")

        if key == 32:
            if is_auto_capturing:
                is_auto_capturing = False
                print("Auto-capture paused")
            else:
                roi = raw_frame[roi_y : roi_y + roi_size, roi_x : roi_x + roi_size]
                filename = f"{save_path}/{count:04d}.jpg"
                cv2.imwrite(filename, roi)
                count += 1
                print(f"Saved: {filename} ({count}/{target_count})")
        elif key in (ord("a"), ord("A")):
            is_auto_capturing = not is_auto_capturing
            last_capture_time = time.time()
            print(f"Auto-capture {'enabled' if is_auto_capturing else 'disabled'}")
        elif key in (ord("+"), ord("=")):
            capture_interval = min(capture_interval + 0.1, 5.0)
            print(f"Capture interval: {capture_interval:.1f}s")
        elif key in (ord("-"), ord("_")):
            capture_interval = max(capture_interval - 0.1, 0.1)
            print(f"Capture interval: {capture_interval:.1f}s")
        elif key == ord("q"):
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCompleted {label}: {count} images in session '{session_name}'\n")
    return True


if __name__ == "__main__":
    gestures = ["l", "peace", "stop", "thumbs_up"]
    images_per_gesture = 500

    print("\n" + "=" * 60)
    print("HAND GESTURE DATA COLLECTION TOOL")
    print("=" * 60)

    print("\nSelect dataset type:")
    print("1. Training data")
    print("2. Validation data")
    choice = input("\nEnter choice (1 or 2): ").strip()
    dataset_type = "train" if choice == "1" else "val"

    print("\nSelect capture mode:")
    print("1. Manual")
    print("2. Auto-capture")
    mode_choice = input("\nEnter choice (1 or 2): ").strip()
    auto_capture = mode_choice == "2"
    capture_interval = 0.5

    if auto_capture:
        interval_input = input("\nCapture interval in seconds (default 0.5): ").strip()
        if interval_input:
            try:
                capture_interval = max(0.1, min(float(interval_input), 5.0))
            except ValueError:
                print("Invalid input, using default 0.5s")

    print(f"\n{'='*60}")
    session_name = input("Enter session/person name (e.g., loona, jane): ").strip()
    if not session_name:
        print("Session name cannot be empty. Exiting.")
        raise SystemExit(1)
    print(f"{'='*60}")

    for gesture in gestures:
        if session_exists(gesture, dataset_type, session_name):
            print(f"\nSession '{session_name}' already exists for '{gesture}'")
            existing_sessions = get_session_list(gesture, dataset_type)
            print(f"Existing sessions: {', '.join(existing_sessions)}")
            proceed = input("Images will be appended. Continue? (y/n): ").strip().lower()
            if proceed != "y":
                print(f"Skipping {gesture}")
                continue

        continue_collection = collect_data(
            label=gesture,
            session_name=session_name,
            dataset_type=dataset_type,
            target_count=images_per_gesture,
            auto_capture=auto_capture,
            capture_interval=capture_interval,
        )
        if not continue_collection:
            print("\nCollection stopped by user")
            break

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)

    for dataset in ["train", "val"]:
        print(f"\n{dataset.upper()} Dataset (all sessions):")
        counts = get_gesture_count_all_sessions(dataset)
        for gesture in gestures:
            print(f"  {gesture}: {counts[gesture]} images")
