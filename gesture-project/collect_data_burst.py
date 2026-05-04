"""
Burst Mode Data Collection Script.
Captures multiple photos rapidly for each gesture/session.
"""

import cv2
import os
import time

from config import GESTURES, get_gesture_count_all_sessions, get_session_list, session_exists


def burst_capture(label, session_name, dataset_type="train", burst_size=10, burst_interval=3):
    """Capture multiple photos in burst mode for one gesture/session."""
    save_path = f"data/{dataset_type}/{label}/{session_name}"
    os.makedirs(save_path, exist_ok=True)

    existing_images = len([f for f in os.listdir(save_path) if f.lower().endswith(".jpg")])
    count = existing_images

    print(f"\n{'='*60}")
    print(f"BURST MODE: {label.upper()} [{session_name}]")
    print(f"{'='*60}")
    print(f"Burst size: {burst_size} photos")
    print(f"Burst duration: {burst_interval}s")
    print(f"Existing images in this session: {existing_images}")
    print("\nControls:")
    print("  SPACE - Start burst capture")
    print("  Q     - Finish and move to next gesture")
    print("  ESC   - Exit program")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return False

    roi_size = 300
    burst_active = False
    burst_start_time = 0
    burst_count = 0
    total_bursts = 0

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

        if burst_active:
            elapsed = current_time - burst_start_time
            photos_to_take = int(elapsed / (burst_interval / max(1, burst_size)))
            if photos_to_take > burst_count and burst_count < burst_size:
                roi = raw_frame[roi_y : roi_y + roi_size, roi_x : roi_x + roi_size]
                filename = f"{save_path}/{count:04d}.jpg"
                cv2.imwrite(filename, roi)
                count += 1
                burst_count += 1
                print(f"Burst {total_bursts + 1}: {burst_count}/{burst_size} - {filename}")

            if burst_count >= burst_size:
                burst_active = False
                total_bursts += 1
                print(f"Burst {total_bursts} complete. Total images: {count}")

        roi_color = (0, 255, 0) if burst_active else (255, 255, 0)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), roi_color, 3)

        status = "CAPTURING" if burst_active else "READY"
        status_color = (0, 255, 0) if burst_active else (255, 255, 255)

        cv2.putText(frame, f"Gesture: {label.upper()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Session: {session_name}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(frame, f"Total Images: {count}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Bursts: {total_bursts}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if burst_active:
            progress = f"Burst Progress: {burst_count}/{burst_size}"
            cv2.putText(frame, progress, (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to start burst", (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, "SPACE: Burst | Q: Next | ESC: Exit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Burst Mode Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32 and not burst_active:
            burst_active = True
            burst_start_time = time.time()
            burst_count = 0
            print(f"\nStarting burst {total_bursts + 1}...")
        elif key == ord("q"):
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCompleted {label}: {count} total images, {total_bursts} bursts\n")
    return True


if __name__ == "__main__":
    gestures = GESTURES

    print("\n" + "=" * 60)
    print("BURST MODE DATA COLLECTION")
    print("=" * 60)

    print("\nSelect dataset type:")
    print("1. Training data")
    print("2. Validation data")
    choice = input("\nEnter choice (1 or 2): ").strip()
    dataset_type = "train" if choice == "1" else "val"

    burst_size_input = input("\nPhotos per burst (default 10): ").strip()
    burst_size = int(burst_size_input) if burst_size_input else 10

    interval_input = input("Seconds for each burst (default 3): ").strip()
    burst_interval = float(interval_input) if interval_input else 3

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

        continue_collection = burst_capture(
            label=gesture,
            session_name=session_name,
            dataset_type=dataset_type,
            burst_size=burst_size,
            burst_interval=burst_interval,
        )
        if not continue_collection:
            print("\nCollection stopped by user")
            break

    print("\n" + "=" * 60)
    print("BURST MODE COLLECTION COMPLETE")
    print("=" * 60)

    for dataset in ["train", "val"]:
        print(f"\n{dataset.upper()} Dataset (all sessions):")
        counts = get_gesture_count_all_sessions(dataset)
        for gesture in gestures:
            print(f"  {gesture}: {counts[gesture]} images")
