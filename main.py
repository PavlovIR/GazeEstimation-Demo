from __future__ import annotations

from pathlib import Path

import cv2

from calibration import (
    DEFAULT_CALIBRATION_PATH,
    DistanceAwareCalibration,
    FaceDistanceTracker,
)
from gaze_ui.AccuracyUi import GazeAccuracyUI
from gaze_ui.Screen import Screen
from GazeEstimation import Estimator as GazeEstimator
from IrisDetection import Detector as IrisDetector

ROOT = Path(__file__).resolve().parent
DEFAULT_MEDIAPIPE_MODEL = ROOT / "models" / "face_landmarker_v2_with_blendshapes.task"
DEFAULT_IRIS_DATA_DIR = ROOT / "IrisDetection" / "data"
DEFAULT_CONFIG = ROOT / "config.toml"
DEFAULT_POINTS = ROOT / "points.csv"
DEFAULT_CALIBRATION = DEFAULT_CALIBRATION_PATH


def _optional_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _opencv_has_gui() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line.upper()
    return False


def _console_quit_requested() -> bool:
    try:
        import msvcrt
    except ImportError:
        return False

    while msvcrt.kbhit():
        if msvcrt.getwch().lower() == "q":
            return True
    return False


def _build_gaze_estimator(screen_size: tuple[int, int] | None) -> GazeEstimator:
    iris_detector = IrisDetector(
        mediapipe_model_path=_optional_path(DEFAULT_MEDIAPIPE_MODEL),
        iris_data_dir=_optional_path(DEFAULT_IRIS_DATA_DIR),
        device="cuda",
    )
    return GazeEstimator(
        IrisDetector=iris_detector,
        screen_size=screen_size,
        device="auto",
    )


def _build_accuracy_ui() -> tuple[Screen | None, GazeAccuracyUI | None]:
    if not DEFAULT_CONFIG.exists():
        print("config.toml not found; running without accuracy UI.")
        return None, None

    screen = Screen(str(DEFAULT_CONFIG))
    points_path = str(DEFAULT_POINTS) if DEFAULT_POINTS.exists() else None
    return screen, GazeAccuracyUI(screen, points_path=points_path)


def _load_calibration() -> DistanceAwareCalibration | None:
    if not DEFAULT_CALIBRATION.exists():
        return None
    calibration = DistanceAwareCalibration.load(DEFAULT_CALIBRATION)
    print(f"Loaded distance-aware calibration from {DEFAULT_CALIBRATION}.")
    return calibration


def main() -> None:
    screen, accuracy_ui = _build_accuracy_ui()
    screen_size = (screen.width_px, screen.height_px) if screen is not None else None
    calibration = _load_calibration()
    if screen_size is None and calibration is not None:
        screen_size = calibration.screen_size
    gaze_estimator = _build_gaze_estimator(screen_size=screen_size)
    show_camera_window = accuracy_ui is None and _opencv_has_gui()
    distance_tracker = None
    distance_tracker_failed = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to access webcam.")

    quit_hint = (
        "Press Q to quit."
        if accuracy_ui is not None or show_camera_window
        else "Press Q or Ctrl+C to quit."
    )
    print(f"ANN gaze estimation running. {quit_hint}")

    try:
        while cap.isOpened():
            if (
                accuracy_ui is None
                and not show_camera_window
                and _console_quit_requested()
            ):
                break

            ret, frame = cap.read()
            if not ret:
                break

            if (
                calibration is not None
                and distance_tracker is None
                and not distance_tracker_failed
            ):
                height, width = frame.shape[:2]
                try:
                    distance_tracker = FaceDistanceTracker(
                        width=width,
                        height=height,
                        mediapipe_model_path=DEFAULT_MEDIAPIPE_MODEL,
                    )
                except Exception as exc:
                    distance_tracker_failed = True
                    print(f"Face distance unavailable: {exc}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predicted_px = None
            status = "No gaze prediction yet."

            try:
                result = gaze_estimator.predict(rgb_frame, return_details=True)
                if calibration is not None:
                    face_distance_mm = (
                        distance_tracker.estimate(frame)
                        if distance_tracker is not None
                        else None
                    )
                    predicted_px = calibration.apply(result.raw_xy, face_distance_mm)
                else:
                    predicted_px = result.screen_xy
                status = None
            except (RuntimeError, ValueError) as exc:
                status = f"Gaze estimation skipped: {exc}"
                print(status, end="\r")

            if accuracy_ui is not None:
                _, _, still_running = accuracy_ui.update_px(predicted_px, status=status)
                if not still_running:
                    break
            else:
                if predicted_px is not None:
                    x, y = float(predicted_px[0]), float(predicted_px[1])
                    cv2.putText(
                        frame,
                        f"gaze: {x:.1f}, {y:.1f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    print(f"Gaze px: {x:.1f}, {y:.1f}", end="\r")

                if show_camera_window:
                    cv2.imshow("ANN gaze estimation", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if accuracy_ui is not None:
            accuracy_ui.close()
        cap.release()
        if show_camera_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
