import cv2

# Initialize YuNet
face_detector = cv2.FaceDetectorYN_create(
    model="face_detection_yunet_2022mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU
    # backend_id=3,
    # target_id=0
)

def detect_faces(frame):
    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(frame)
    
    if faces is not None:
        for face in faces:
            box = face[0:4].astype(int)
            # Draw rectangle around face
            cv2.rectangle(frame, box, (0, 255, 0), 2)
    
    return frame, faces

# Camera capture loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame, faces = detect_faces(frame)
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
