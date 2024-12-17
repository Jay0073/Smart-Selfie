import cv2
from keras import models as mod
import numpy as np

# load emotion model
model = mod.load_model('emotion_model.h5')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# load haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open video file
cap = cv2.VideoCapture(0)


# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # detecting faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # detecting emotions
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        preds = model.predict(face)[0]
        emotion = emotions[np.argmax(preds)]
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


    # Display the frame
    cv2.imshow('Human Detection', frame)

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release video capture object and close display window
cap.release()
cv2.destroyAllWindows()
