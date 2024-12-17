import cv2
import numpy as np
from keras import models as model
from keras import preprocessing as pp

# https://github.com/dinuduke/Facial-Emotion-Recognition

# Load the pre-trained emotion recognition model
emotion_model_path = 'emotion_model.h5'
emotion_classifier = model.load_model(emotion_model_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess input image for emotion recognition
def preprocess_input(image, target_size):
    image = cv2.resize(image, target_size)
    image = pp.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face = gray[y:y+h, x:x+w]

        # Preprocess face image
        face_input = preprocess_input(face, target_size=(48, 48))

        # Predict emotion
        emotion_prediction = emotion_classifier.predict(face_input)
        max_index = np.argmax(emotion_prediction[0])
        emotion_label = emotion_labels[max_index]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
