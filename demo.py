from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_IRIS_DATA_DIR,
    DEFAULT_MEDIAPIPE_MODEL,
    DistanceAwareCalibration,
    FaceDistanceTracker,
    build_gaze_estimator,
)
from gaze_ui.DemoUi import GazeDemoUI
from gaze_ui.Screen import Screen
from GazeEstimation import EstimationResult


class DemoScreenAdapter:
    def __init__(self, screen: Screen) -> None:
        self.screen = screen
        self.width_px = screen.width_px
        self.height_px = screen.height_px

    def _gaze_to_point(self, predicted_px) -> tuple[int, int]:
        xy = np.asarray(predicted_px, dtype=np.float32)
        return (
            int(np.clip(round(float(xy[0])), 0, self.width_px - 1)),
            int(np.clip(round(float(xy[1])), 0, self.height_px - 1)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the calibrated gaze demo grid.")
    parser.add_argument("--screen-config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.toml.")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH), help="Path to calibration JSON.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--fov-degrees", type=float, default=60.0, help="Approximate webcam horizontal FOV.")
    parser.add_argument("--device", default="auto", help="ANN model device: auto, cpu, cuda, etc.")
    parser.add_argument("--iris-device", default="cpu", help="Iris detector device.")
    parser.add_argument("--block-num", type=int, default=4, help="Number of grid blocks per row/column.")
    parser.add_argument("--block-ratio", type=float, default=0.15, help="Block size relative to screen.")
    parser.add_argument("--margin-ratio", type=float, default=0.08, help="Grid margin relative to screen.")
    parser.add_argument("--gap-ratio", type=float, default=0.08, help="Grid gap relative to screen.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found at {calibration_path}. Run calibration.py first."
        )

    screen = Screen(args.screen_config)
    calibration = DistanceAwareCalibration.load(calibration_path)
    gaze_estimator = build_gaze_estimator(
        screen_size=(screen.width_px, screen.height_px),
        mediapipe_model_path=DEFAULT_MEDIAPIPE_MODEL,
        iris_data_dir=DEFAULT_IRIS_DATA_DIR,
        model_device=args.device,
        iris_device=args.iris_device,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Failed to access webcam.")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read from webcam.")

    height, width = frame.shape[:2]
    distance_tracker = FaceDistanceTracker(
        width=width,
        height=height,
        mediapipe_model_path=DEFAULT_MEDIAPIPE_MODEL,
        fov_degrees=args.fov_degrees,
    )
    ui = GazeDemoUI(
        DemoScreenAdapter(screen),
        block_ratio=args.block_ratio,
        margin_ratio=args.margin_ratio,
        gap_ratio=args.gap_ratio,
        block_num=args.block_num,
    )

    print(f"Loaded calibration from {calibration_path.resolve()}. Press ESC in the demo window to quit.")

    try:
        while ui.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            predicted_px = None
            try:
                face_distance_mm = distance_tracker.estimate(frame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = gaze_estimator.predict(rgb_frame, return_details=True)
                if not isinstance(result, EstimationResult):
                    raise RuntimeError("Estimator did not return detailed output.")
                predicted_px = calibration.apply(result.raw_xy, face_distance_mm)
            except (RuntimeError, ValueError) as exc:
                print(f"Gaze demo skipped: {exc}", end="\r")

            _, still_running = ui.update(predicted_px)
            if not still_running:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ui.close()
        cap.release()


if __name__ == "__main__":
    main()
