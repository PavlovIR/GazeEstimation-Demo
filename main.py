from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from calibration import (
    DEFAULT_CALIBRATION_PATH,
    DistanceAwareCalibration,
    FaceDistanceTracker,
)
from gaze_ui.AccuracyUi import GazeAccuracyUI
from gaze_ui.accuracy_logger import AccuracyLogger
from gaze_ui.Screen import Screen
from GazeEstimation import EstimationResult, Estimator as GazeEstimator
from IrisDetection import Detector as IrisDetector

ROOT = Path(__file__).resolve().parent
DEFAULT_MEDIAPIPE_MODEL = ROOT / "models" / "face_landmarker_v2_with_blendshapes.task"
DEFAULT_IRIS_DATA_DIR = ROOT / "IrisDetection" / "data"
DEFAULT_CONFIG = ROOT / "config.toml"
DEFAULT_POINTS = ROOT / "points.csv"
DEFAULT_CALIBRATION = DEFAULT_CALIBRATION_PATH
DEFAULT_LOG_DIR = ROOT / "accuracy_runs"
DEFAULT_ACTIVATION_FUNCTION = "leaky_relu"


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


def _build_gaze_estimator(
    screen_size: tuple[int, int] | None,
    weights_path: str | Path | None,
    activation_function: str,
    model_device: str,
    iris_device: str,
) -> GazeEstimator:
    iris_detector = IrisDetector(
        mediapipe_model_path=_optional_path(DEFAULT_MEDIAPIPE_MODEL),
        iris_data_dir=_optional_path(DEFAULT_IRIS_DATA_DIR),
        device=iris_device,
    )
    return GazeEstimator(
        IrisDetector=iris_detector,
        weights_path=weights_path,
        screen_size=screen_size,
        device=model_device,
        activation_function=activation_function,
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


def _check_calibration_model_match(
    calibration: DistanceAwareCalibration | None,
    gaze_estimator: GazeEstimator,
) -> None:
    if calibration is None:
        return

    activation = calibration.metadata.get("activation_function")
    if activation is None:
        print(
            "Warning: calibration.json has no activation metadata. "
            "Re-run calibration.py for reliable Leaky ReLU calibration."
        )
        return
    if str(activation) != gaze_estimator.activation_function:
        raise RuntimeError(
            "Calibration/model mismatch: "
            f"calibration was collected with activation={activation!r}, "
            f"but runtime loaded activation={gaze_estimator.activation_function!r}. "
            "Re-run calibration.py with the same --activation/--weights settings."
        )

    weights_path = calibration.metadata.get("weights_path")
    if weights_path is not None and Path(weights_path) != Path(gaze_estimator.weights_path):
        print(
            "Warning: calibration.json was collected with a different weights path: "
            f"{weights_path!r}; runtime uses {str(gaze_estimator.weights_path)!r}."
        )


def _face_bbox_xyxy(result: EstimationResult | None) -> tuple[float, float, float, float] | None:
    if result is None:
        return None
    box = result.detection.face_box
    return box.x, box.y, box.x2, box.y2


def _px_to_cm(screen: Screen, predicted_px) -> tuple[float, float] | None:
    if predicted_px is None:
        return None
    x_mm, y_mm = screen._px_to_mm(float(predicted_px[0]), float(predicted_px[1]))
    return float(x_mm) / 10.0, float(y_mm) / 10.0


def _target_cm(accuracy_ui: GazeAccuracyUI | None) -> tuple[float, float] | None:
    if accuracy_ui is None or accuracy_ui.target_mm is None:
        return None
    return float(accuracy_ui.target_mm[0]) / 10.0, float(accuracy_ui.target_mm[1]) / 10.0


def _error_deg(error_mm: float | None, face_distance_mm: float | None) -> float | None:
    if error_mm is None or face_distance_mm is None or face_distance_mm <= 0.0:
        return None
    import math

    return math.degrees(math.atan(float(error_mm) / float(face_distance_mm)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANN gaze accuracy UI.")
    parser.add_argument(
        "--weights",
        help=(
            "Path to ANN checkpoint. If omitted, Estimator picks the default checkpoint "
            "for --activation."
        ),
    )
    parser.add_argument(
        "--activation",
        default=DEFAULT_ACTIVATION_FUNCTION,
        choices=("relu", "leaky_relu"),
        help="Model activation variant to load.",
    )
    parser.add_argument("--device", default="auto", help="ANN model device: auto, cpu, cuda, etc.")
    parser.add_argument("--iris-device", default="cpu", help="Iris detector device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screen, accuracy_ui = _build_accuracy_ui()
    screen_size = (screen.width_px, screen.height_px) if screen is not None else None
    calibration = _load_calibration()
    if screen_size is None and calibration is not None:
        screen_size = calibration.screen_size
    gaze_estimator = _build_gaze_estimator(
        screen_size=screen_size,
        weights_path=args.weights,
        activation_function=args.activation,
        model_device=args.device,
        iris_device=args.iris_device,
    )
    print(
        "Loaded gaze model "
        f"activation={gaze_estimator.activation_function}, "
        f"weights={gaze_estimator.weights_path}."
    )
    _check_calibration_model_match(calibration, gaze_estimator)
    show_camera_window = accuracy_ui is None and _opencv_has_gui()
    distance_tracker = None
    distance_tracker_failed = False
    logger = (
        AccuracyLogger(
            str(DEFAULT_LOG_DIR),
            screen,
            calibration_path=str(DEFAULT_CALIBRATION) if DEFAULT_CALIBRATION.exists() else None,
        )
        if screen is not None and accuracy_ui is not None
        else None
    )
    if logger is not None:
        print(f"Logging accuracy run to {logger.run_dir}.")

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
            result = None
            face_distance_mm = None
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
                target_px = accuracy_ui.target_px
                target_cm = _target_cm(accuracy_ui)
                error_mm, _, still_running = accuracy_ui.update_px(predicted_px, status=status)
                if logger is not None:
                    logger.log(
                        frame=frame,
                        face_bbox=_face_bbox_xyxy(result),
                        gaze_cm=_px_to_cm(screen, predicted_px),
                        target_cm=target_cm,
                        target_px=target_px,
                        error_cm=(float(error_mm) / 10.0) if error_mm is not None else None,
                        error_deg=_error_deg(error_mm, face_distance_mm),
                    )
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
        if logger is not None:
            logger.close()
        cap.release()
        if show_camera_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
