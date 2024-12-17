
import cv2
# Open video file
cap = cv2.VideoCapture(0)
print(cap)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Human Detection', frame)

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release video capture object and close display window
cap.release()
cv2.destroyAllWindows()
