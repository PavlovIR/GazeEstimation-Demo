import cv2
import numpy as np


class DistanceEstimator:
    def __init__(self, camera):
        self.camera = camera

        # Canonical 3D face model points in millimeters
        self.model_points_3d = np.array(
            [
                [0.0, 0.0, 0.0],  # Nose tip
                [0.0, -87.6, -15.2],  # Chin
                [-43.3, 32.7, -26.0],  # Left eye
                [43.3, 32.7, -26.0],  # Right eye
                [-28.9, -28.9, -24.1],  # Left mouth
                [28.9, -28.9, -24.1],  # Right mouth
            ],
            dtype=np.float64,
        )

    def estimate_3d_position(self, image_points_2d: np.ndarray):
        """Calculates the 3D position of the face relative to the camera in mm."""
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3d,
            image_points_2d,
            self.camera.intrinsics,
            self.camera.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success:
            # translation_vector is a 3x1 array: [[x], [y], [z]] in mm
            x_mm = translation_vector[0][0]
            y_mm = translation_vector[1][0]
            z_mm = translation_vector[2][0]

            return np.array([x_mm, y_mm, z_mm])

        return None
