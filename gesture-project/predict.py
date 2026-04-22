import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/gesture_model.h5")

labels = ['l', 'ok', 'peace', 'stop', 'thumbs up']  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    roi = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    img = cv2.resize(roi, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    prediction = model.predict(img)

    confidence = np.max(prediction)
    label = labels[np.argmax(prediction)]

    if confidence < 0.7:
        label = "Unknown"

    cv2.putText(frame, f"{label} ({confidence:.2f})", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()