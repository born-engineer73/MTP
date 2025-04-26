import cv2
import os

# Use 1 for external webcam (0 is default for internal)
camera_index = 0  # Change to 0 if external webcam is not detected

# Open the external webcam
cap = cv2.VideoCapture(camera_index)

# Set resolution (optional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_filename = '18.avi'
out = cv2.VideoWriter(output_filename, fourcc, 30.0, (640, 480))

print("Recording... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)  # Save the frame to file
    cv2.imshow('Recording (Press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as: {os.path.abspath(output_filename)}")
