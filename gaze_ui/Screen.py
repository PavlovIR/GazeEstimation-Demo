import sys
from math import cos, tan

import numpy as np
from numpy._core.multiarray import ndarray

# Handle TOML parsing depending on Python version
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        print("Please install tomli: pip install tomli")
        sys.exit(1)


class Screen:
    def __init__(self, config_path: str = "config.toml"):
        self.config = self._load_config(config_path)

        # Extract Camera Offsets into a Numpy Vector
        cam = self.config["camera_offset"]
        self.camera_offset = np.array(
            [cam["x_mm"], cam["y_mm"], cam["z_mm"]], dtype=np.float64
        )

        # Extract Screen Properties
        screen = self.config["screen"]
        self.width_px = screen["res_width"]
        self.height_px = screen["res_height"]
        self.width_mm = screen["phys_width_mm"]
        self.height_mm = screen["phys_height_mm"]

        # The screen normal always points directly at the user along the Z-axis
        self.screen_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _load_config(self, path: str) -> dict:
        """Loads and parses the TOML configuration file."""
        with open(path, "rb") as f:
            return tomllib.load(f)

    def _mm_to_px(self, predicted_mm):
        x_mm, y_mm = predicted_mm
        # 4. Map Physical Coordinates (mm) to Pixel Coordinates
        # Convert physical hit point to normalized space [0.0 to 1.0]
        # X: 0 is left edge, 1 is right edge
        norm_x = (x_mm / self.width_mm) + 0.5

        # Y: 0 is top edge, 1 is bottom edge
        # Note: Physical Y is positive UP, but Pixel Y is positive DOWN, hence the subtraction.
        norm_y = 0.5 - (y_mm / self.height_mm)

        # Scale by resolution
        pixel_x = int(norm_x * self.width_px)
        pixel_y = int(norm_y * self.height_px)

        return (pixel_x, pixel_y)

    def _px_to_mm(self, x: float, y: float):
        x_mm = (x / self.width_px - 0.5) * self.width_mm
        y_mm = (y / self.height_px - 0.5) * self.height_mm

        return x_mm, y_mm

    def _pith_yaw_to_mm(self, face_pos_cam: ndarray, pitch: float, yaw: float):
        face_pos_screen = face_pos_cam + self.camera_offset
        xf_mm, yf_mm, zf_mm = face_pos_screen[0], face_pos_screen[1], face_pos_screen[2]
        xs_mm = xf_mm + zf_mm * tan(-yaw)
        ys_mm = yf_mm + zf_mm * (tan(-pitch) / cos(-yaw))
        return xs_mm, ys_mm
