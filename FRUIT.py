import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from playsound import playsound
import time

# Load the trained model
model = load_model("fruit_veg_rotten_detector.h5")

# The labels your model uses
class_labels = ['Apple_Fresh', 'Apple_Rotten', 'Banana_Fresh', 'Banana_Rotten', 'Tomato_Fresh', 'Tomato_Rotten']

# Open the camera
camera = cv2.VideoCapture(0)

def preprocess_frame(frame):
    img = cv2.resize(frame, (100, 100))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

print("Starting Fruit & Veggie Detector... Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    processed = preprocess_frame(frame)
    prediction = model.predict(processed)
    class_index = np.argmax(prediction)
    label = class_labels[class_index]
    confidence = prediction[0][class_index]

    # Decide if it’s rotten
    is_rotten = "Rotten" in label

    # Display label and confidence
    color = (0, 0, 255) if is_rotten else (0, 255, 0)
    cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Food Detector", frame)

    # Alarm for rotten food
    if is_rotten and confidence > 0.8:
        print(f"🚨 WARNING: {label} detected!")
        playsound("alarm.wav")

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
