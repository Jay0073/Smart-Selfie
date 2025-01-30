# from gemini combined yunet and facemesh

import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize face landmarker detector (MediaPipe Face Mesh)
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)  # You can increase num_faces if needed
detector = vision.FaceLandmarker.create_from_options(options)

# Initialize YuNet face detector
face_detector = cv2.FaceDetectorYN_create(
    model="face_detection_yunet_2023mar.onnx",  # Make sure this path is correct
    config="",
    input_size=(720, 720),  # Initial input size, will be adjusted
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU
)


def draw_landmarks_on_image(rgb_image, detection_result):
    if detection_result.face_landmarks: # Check if landmarks are detected
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        for face_landmarks in face_landmarks_list:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style())

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style())

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
        return annotated_image
    else:
        return rgb_image # Return original image if no landmarks are detected


def detect_faces_yunet(frame):
    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(frame)
    return faces


# Camera capture loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YuNet face detection
    faces = detect_faces_yunet(frame)

    if faces is not None:
        for face in faces:
            box = face[0:4].astype(int)
            x1, y1, x2, y2 = box
            # Crop the face region for MediaPipe
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0: # Check if face ROI is valid
                # MediaPipe Face Mesh
                rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB) # MediaPipe needs RGB
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_face_roi)
                detection_result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

                # Draw landmarks on the cropped face region
                annotated_face_roi = draw_landmarks_on_image(rgb_face_roi, detection_result)
                annotated_face_roi_bgr = cv2.cvtColor(annotated_face_roi, cv2.COLOR_RGB2BGR)

                # Put the annotated face ROI back into the original frame
                frame[y1:y2, x1:x2] = annotated_face_roi_bgr
            cv2.rectangle(frame, box, (0, 255, 0), 2)  # Draw YuNet's bounding box

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()  # Close the MediaPipe face landmarker