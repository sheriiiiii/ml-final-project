import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/gesture_model.h5")

labels = ['peace', 'stop', 'thumbs_up']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    prediction = model.predict(img)
    label = labels[np.argmax(prediction)]

    cv2.putText(frame, label, (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()