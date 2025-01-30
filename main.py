# just a sample one
import cv2
import numpy as np
from mediapipe import Image, solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

face_detector = cv2.FaceDetectorYN_create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(720, 720),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Initialize face landmarker detector
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

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

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image = annotated_image,
            landmark_list = face_landmarks_proto,
            connections = solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec = None,
            connection_drawing_spec = solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        
        solutions.drawing_utils.draw_landmarks(
            image = annotated_image,
            landmark_list = face_landmarks_proto,
            connections = solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = solutions.drawing_styles.get_default_face_mesh_contours_style())
        
        solutions.drawing_utils.draw_landmarks(
            image = annotated_image,
            landmark_list = face_landmarks_proto,
            connections = solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec = None,
            connection_drawing_spec = solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

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