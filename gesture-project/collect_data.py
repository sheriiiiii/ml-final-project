import cv2
import os

label = "stop"
save_path = f"data/train/{label}"

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):  # press S to save
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        count += 1
        print("Saved:", count)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()