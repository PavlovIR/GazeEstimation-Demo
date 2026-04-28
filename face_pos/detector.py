import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:
    def __init__(self, model_path: str):
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(self.options)

        # Specific MediaPipe landmark indices we need for PnP
        # Order: Nose tip, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

    def extract_key_landmarks(self, image: np.ndarray):
        """Extracts 6 specific 2D image points for PnP."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]
        h, w, _ = image.shape

        image_points = []
        for idx in self.landmark_indices:
            landmark = face_landmarks[idx]
            # Convert normalized coordinates [0, 1] to pixel coordinates
            px, py = int(landmark.x * w), int(landmark.y * h)
            image_points.append([px, py])

        return np.array(image_points, dtype=np.float64)
