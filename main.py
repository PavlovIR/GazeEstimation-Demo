from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from calibration import (
    DistanceAwareCalibration,
    FaceDistanceTracker,
)
from gaze_ui.accuracy_logger import AccuracyLogger
from gaze_ui.AccuracyUi import GazeAccuracyUI
from gaze_ui.Screen import Screen
from Env.run_parameters import (
    DEFAULT_ACTIVATION_FUNCTION,
    DEFAULT_RUN_PARAMETERS_PATH,
    RunParameters,
    load_run_parameters,
    normalize_estimator_lib,
)
from Env.runtime_estimator import (
    RuntimeEstimationResult,
    build_runtime_gaze_estimator,
    calibration_metadata_matches_estimator,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MEDIAPIPE_MODEL = ROOT / "models" / "face_landmarker_v2_with_blendshapes.task"
DEFAULT_IRIS_DATA_DIR = ROOT / "IrisDetection" / "data"
DEFAULT_POINTS = ROOT / "csv_points" / "points.csv"
DEFAULT_LOG_DIR = ROOT / "accuracy_runs"


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


def _build_accuracy_ui(
    config_path: str | Path,
) -> tuple[Screen | None, GazeAccuracyUI | None]:
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"{config_path} not found; running without accuracy UI.")
        return None, None

    screen = Screen(str(config_path))
    points_path = str(DEFAULT_POINTS) if DEFAULT_POINTS.exists() else None
    return screen, GazeAccuracyUI(screen, points_path=points_path)


def _load_calibration(
    calibration_path: str | Path,
) -> DistanceAwareCalibration | None:
    calibration_path = Path(calibration_path)
    if not calibration_path.exists():
        print(f"Calibration file not found at {calibration_path}; running uncalibrated.")
        return None
    calibration = DistanceAwareCalibration.load(calibration_path)
    print(f"Loaded distance-aware calibration from {calibration_path}.")
    return calibration


def _resolve_model_options(
    args: argparse.Namespace,
    calibration: DistanceAwareCalibration | None,
    run_parameters: RunParameters,
    estimator_lib: str,
) -> tuple[str, str | Path | None]:
    calibration_activation = None
    if calibration is not None:
        calibration_activation = calibration.metadata.get("activation_function")

    activation_function = (
        args.activation
        or run_parameters.activation_function
        or (str(calibration_activation) if calibration_activation is not None else None)
        or DEFAULT_ACTIVATION_FUNCTION
    )
    weights_path = (
        args.weights
        or run_parameters.weights
        or _calibration_weights_for_estimator(calibration, estimator_lib)
    )
    return activation_function, weights_path


def _calibration_weights_for_estimator(
    calibration: DistanceAwareCalibration | None,
    estimator_lib: str,
) -> str | None:
    if calibration is None:
        return None

    weights_path = calibration.metadata.get("weights_path")
    if weights_path is None:
        return None

    metadata_match = calibration_metadata_matches_estimator(
        calibration.metadata,
        estimator_lib,
    )
    if metadata_match is False:
        return None
    if metadata_match is None and estimator_lib != "GazeEstimation":
        return None
    return str(weights_path)


def _check_calibration_model_match(
    calibration: DistanceAwareCalibration | None,
    gaze_estimator,
) -> None:
    if calibration is None:
        return

    estimator_lib = gaze_estimator.estimator_lib
    metadata_match = calibration_metadata_matches_estimator(
        calibration.metadata,
        estimator_lib,
    )
    if metadata_match is False:
        raise RuntimeError(
            "Calibration/model mismatch: "
            f"calibration was collected with estimator_lib={calibration.metadata.get('estimator_lib')!r}, "
            f"but runtime loaded estimator_lib={estimator_lib!r}."
        )
    if metadata_match is None and estimator_lib == "GazeCaptureEstimator":
        print(
            "Warning: calibration file has no estimator_lib metadata. "
            "Make sure it was collected with GazeCaptureEstimator; ANN calibration "
            "will not map GazeCapture raw outputs correctly."
        )

    if estimator_lib != "GazeEstimation":
        weights_path = calibration.metadata.get("weights_path")
        if (
            metadata_match is True
            and weights_path is not None
            and Path(weights_path) != Path(gaze_estimator.weights_path)
        ):
            print(
                "Warning: calibration file was collected with a different weights path: "
                f"{weights_path!r}; runtime uses {str(gaze_estimator.weights_path)!r}."
            )
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
    if weights_path is not None and Path(weights_path) != Path(
        gaze_estimator.weights_path
    ):
        print(
            "Warning: calibration.json was collected with a different weights path: "
            f"{weights_path!r}; runtime uses {str(gaze_estimator.weights_path)!r}."
        )


def _face_bbox_xyxy(
    result: RuntimeEstimationResult | None,
) -> tuple[float, float, float, float] | None:
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
    return float(accuracy_ui.target_mm[0]) / 10.0, float(
        accuracy_ui.target_mm[1]
    ) / 10.0


def _error_deg(error_mm: float | None, face_distance_mm: float | None) -> float | None:
    if error_mm is None or face_distance_mm is None or face_distance_mm <= 0.0:
        return None
    import math

    return math.degrees(math.atan(float(error_mm) / float(face_distance_mm)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gaze accuracy UI.")
    parser.add_argument(
        "--run-parameters",
        default=str(DEFAULT_RUN_PARAMETERS_PATH),
        help="Path to shared run_parameters.toml.",
    )
    parser.add_argument(
        "--estimator-lib",
        help="Estimator library backend. Overrides run_parameters.toml.",
    )
    parser.add_argument(
        "--screen-config",
        help="Path to config.toml. Overrides run_parameters.toml.",
    )
    parser.add_argument(
        "--weights",
        help=(
            "Path to estimator checkpoint. If omitted, the selected backend picks "
            "its default checkpoint."
        ),
    )
    parser.add_argument(
        "--activation",
        default=None,
        choices=("relu", "leaky_relu"),
        help=(
            "Model activation variant to load. Defaults to run_parameters.toml, "
            "then calibration metadata."
        ),
    )
    parser.add_argument(
        "--calibration-file",
        help="Path to calibration JSON. Overrides run_parameters.toml.",
    )
    parser.add_argument(
        "--device", default="auto", help="Model device: auto, cpu, cuda, etc."
    )
    parser.add_argument(
        "--iris-device",
        default="auto",
        help="Iris detector device: auto, cpu, cuda, mps, etc.",
    )
    parser.add_argument(
        "--fov-degrees",
        type=float,
        default=60.0,
        help="Approximate webcam horizontal FOV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_parameters = load_run_parameters(args.run_parameters)
    estimator_lib = normalize_estimator_lib(
        args.estimator_lib or run_parameters.estimator_lib
    )
    screen_config = (
        Path(args.screen_config)
        if args.screen_config is not None
        else run_parameters.screen_config
    )
    calibration_path = (
        Path(args.calibration_file)
        if args.calibration_file is not None
        else run_parameters.calibration_file
    )

    screen, accuracy_ui = _build_accuracy_ui(screen_config)
    screen_size = (screen.width_px, screen.height_px) if screen is not None else None
    calibration = _load_calibration(calibration_path)
    if screen_size is None and calibration is not None:
        screen_size = calibration.screen_size
    activation_function, weights_path = _resolve_model_options(
        args, calibration, run_parameters, estimator_lib
    )
    gaze_estimator = build_runtime_gaze_estimator(
        estimator_lib=estimator_lib,
        screen=screen,
        screen_size=screen_size,
        weights_path=weights_path,
        activation_function=activation_function,
        model_device=args.device,
        iris_device=args.iris_device,
        mediapipe_model_path=DEFAULT_MEDIAPIPE_MODEL,
        iris_data_dir=DEFAULT_IRIS_DATA_DIR,
        fov_degrees=args.fov_degrees,
    )
    print(
        "Loaded gaze model "
        f"estimator_lib={estimator_lib}, "
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
            calibration_path=str(calibration_path)
            if calibration_path.exists()
            else None,
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
                        fov_degrees=args.fov_degrees,
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
                if not isinstance(result, RuntimeEstimationResult):
                    raise RuntimeError("Estimator did not return detailed output.")
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
                error_mm, _, still_running = accuracy_ui.update_px(
                    predicted_px, status=status
                )
                if logger is not None:
                    logger.log(
                        frame=frame,
                        face_bbox=_face_bbox_xyxy(result),
                        gaze_cm=_px_to_cm(screen, predicted_px),
                        target_cm=target_cm,
                        target_px=target_px,
                        error_cm=(float(error_mm) / 10.0)
                        if error_mm is not None
                        else None,
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
