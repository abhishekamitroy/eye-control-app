import streamlit as st
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp

st.title("Eye Control Plugin")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Increment frame count
        self.frame_count += 1

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the RGB frame using MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )
            st.write(f"Frame {self.frame_count}: Face landmarks detected")
        else:
            st.write(f"Frame {self.frame_count}: No face landmarks detected")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the webcam stream
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Display an info message
st.info("Webcam stream started. Check the console for frame processing status.")
