import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("Eye Control Plugin")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# Checkbox to start/stop the webcam feed
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Function to check if the webcam is available
def is_webcam_available():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.release()
        return True
    return False

# Option to use video file if webcam is not available
use_video_file = False
video_file_path = 'path_to_video_file.mp4'  # Update this with your video file path

if not is_webcam_available():
    st.warning("Webcam is not available. Switching to video file.")
    use_video_file = True

# Initialize video capture (webcam or video file)
cap = cv2.VideoCapture(0 if not use_video_file else video_file_path)

# Check if the video capture is opened successfully
if not cap.isOpened():
    st.error("Failed to open video source. Please check your webcam or video file path.")
else:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read frame from video source")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB frame using MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        # Display the frame
        FRAME_WINDOW.image(frame)

    cap.release()
    st.stop()  # Stop the Streamlit script when done

# Clean up
cap.release()
cv2.destroyAllWindows()
