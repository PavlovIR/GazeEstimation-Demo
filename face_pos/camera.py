import numpy as np
import math

class Camera:
    def __init__(self, width: int, height: int, fov_degrees: float = 60.0):
        self.width = width
        self.height = height
        self.fov_degrees = fov_degrees
        self.intrinsics = self._approximate_intrinsics()
        # Assuming no lens distortion for uncalibrated webcams
        self.dist_coeffs = np.zeros((4, 1))

    def _approximate_intrinsics(self) -> np.ndarray:
        """Approximates the intrinsic matrix K based on resolution and FOV."""
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        fov_radians = math.radians(self.fov_degrees)
        focal_length = self.width / (2.0 * math.tan(fov_radians / 2.0))
        
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return K