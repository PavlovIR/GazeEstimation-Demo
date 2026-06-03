from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration import (
    DEFAULT_ACTIVATION_FUNCTION,
    DEFAULT_IRIS_DATA_DIR,
    DEFAULT_MEDIAPIPE_MODEL,
    DistanceAwareCalibration,
    FaceDistanceTracker,
)
from gaze_ui.DemoUi import GazeDemoUI
from gaze_ui.Screen import Screen
from Env.run_parameters import (
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
DEFAULT_RELU_WEIGHTS_PATH = ROOT / "GazeEstimation" / "weights" / "best.pt"
DEFAULT_RELU_CALIBRATION_PATH = ROOT / "calibration" / "calibration-relu.json"


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
        "--calibration",
        help=(
            "Path to calibration JSON. Defaults to calibration.json, or "
            "calibration-relu.json in --standard-relu mode when that file exists."
        ),
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--video-output",
        type=Path,
        help="Save the rendered pygame scene to a video file. Use .mp4 for best compatibility.",
    )
    parser.add_argument("--video-fps", type=float, default=30.0, help="Frame rate for --video-output.")
    parser.add_argument("--fov-degrees", type=float, default=60.0, help="Approximate webcam horizontal FOV.")
    parser.add_argument("--device", default="auto", help="Model device: auto, cpu, cuda, etc.")
    parser.add_argument(
        "--iris-device",
        default="auto",
        help="Iris detector device: auto, cpu, cuda, mps, etc.",
    )
    parser.add_argument(
        "--standard-relu",
        action="store_true",
        help=(
            "Launch with the standard ReLU checkpoint. Shortcut for "
            "--activation relu --weights GazeEstimation/weights/best.pt."
        ),
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
        help="Model activation variant to load. Overrides run_parameters.toml.",
    )
    parser.add_argument("--block-num", type=int, default=4, help="Number of grid blocks per row/column.")
    parser.add_argument("--block-ratio", type=float, default=0.15, help="Block size relative to screen.")
    parser.add_argument("--margin-ratio", type=float, default=0.08, help="Grid margin relative to screen.")
    parser.add_argument("--gap-ratio", type=float, default=0.08, help="Grid gap relative to screen.")
    return parser.parse_args()


def resolve_model_options(
    args: argparse.Namespace,
    run_parameters: RunParameters,
) -> tuple[str, str | Path | None]:
    if not args.standard_relu:
        return (
            args.activation
            or run_parameters.activation_function
            or DEFAULT_ACTIVATION_FUNCTION,
            args.weights or run_parameters.weights,
        )

    weights = args.weights if args.weights is not None else str(DEFAULT_RELU_WEIGHTS_PATH)
    return "relu", weights


def resolve_calibration_path(
    args: argparse.Namespace, run_parameters: RunParameters
) -> Path:
    if args.calibration is not None:
        return Path(args.calibration)
    if args.standard_relu and DEFAULT_RELU_CALIBRATION_PATH.exists():
        return DEFAULT_RELU_CALIBRATION_PATH
    return run_parameters.calibration_file


def open_scene_video_writer(
    output_path: Path,
    *,
    width_px: int,
    height_px: int,
    fps: float,
) -> cv2.VideoWriter:
    if fps <= 0:
        raise ValueError("--video-fps must be greater than 0.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width_px, height_px),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open video writer for {output_path}. "
            "Use an .mp4 output path and make sure OpenCV has video codec support."
        )
    return writer


def check_calibration_model_match(
    calibration: DistanceAwareCalibration,
    gaze_estimator,
) -> None:
    estimator_lib = gaze_estimator.estimator_lib
    metadata_match = calibration_metadata_matches_estimator(
        calibration.metadata,
        estimator_lib,
    )
    if metadata_match is False:
        raise RuntimeError(
            "Calibration/model mismatch: "
            f"calibration was collected with estimator_lib={calibration.metadata.get('estimator_lib')!r}, "
            f"but demo loaded estimator_lib={estimator_lib!r}."
        )
    if metadata_match is None and estimator_lib == "GazeCaptureEstimator":
        print(
            "Warning: calibration file has no estimator_lib metadata. "
            "Make sure it was collected with GazeCaptureEstimator; ANN calibration "
            "will not map GazeCapture raw outputs correctly."
        )

    if estimator_lib != "GazeEstimation":
        calibration_weights = calibration.metadata.get("weights_path")
        if (
            metadata_match is True
            and calibration_weights is not None
            and Path(calibration_weights) != Path(gaze_estimator.weights_path)
        ):
            print(
                "Warning: calibration file was collected with a different weights path: "
                f"{calibration_weights!r}; demo uses {str(gaze_estimator.weights_path)!r}."
            )
        return

    calibration_activation = calibration.metadata.get("activation_function")
    if calibration_activation is None:
        print(
            "Warning: calibration file has no activation metadata. "
            "Re-run calibration.py for reliable Leaky ReLU calibration."
        )
    elif str(calibration_activation) != gaze_estimator.activation_function:
        raise RuntimeError(
            "Calibration/model mismatch: "
            f"calibration was collected with activation={calibration_activation!r}, "
            f"but demo loaded activation={gaze_estimator.activation_function!r}. "
            "Re-run calibration.py with the same --activation/--weights settings. "
            "For standard ReLU, use: calibration.py --activation relu "
            "--weights GazeEstimation/weights/best.pt --output calibration-relu.json"
        )

    calibration_weights = calibration.metadata.get("weights_path")
    if calibration_weights is not None and Path(calibration_weights) != Path(
        gaze_estimator.weights_path
    ):
        print(
            "Warning: calibration file was collected with a different weights path: "
            f"{calibration_weights!r}; demo uses {str(gaze_estimator.weights_path)!r}."
        )


def main() -> None:
    args = parse_args()
    run_parameters = load_run_parameters(args.run_parameters)
    estimator_lib = normalize_estimator_lib(
        args.estimator_lib or run_parameters.estimator_lib
    )
    if args.standard_relu:
        estimator_lib = "GazeEstimation"
    screen_config = (
        Path(args.screen_config)
        if args.screen_config is not None
        else run_parameters.screen_config
    )
    activation_function, weights_path = resolve_model_options(args, run_parameters)
    calibration_path = resolve_calibration_path(args, run_parameters)
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found at {calibration_path}. Run calibration.py first."
        )

    screen = Screen(str(screen_config))
    calibration = DistanceAwareCalibration.load(calibration_path)
    gaze_estimator = build_runtime_gaze_estimator(
        estimator_lib=estimator_lib,
        screen=screen,
        screen_size=(screen.width_px, screen.height_px),
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
    check_calibration_model_match(calibration, gaze_estimator)

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
    scene_writer = None
    if args.video_output is not None:
        scene_writer = open_scene_video_writer(
            args.video_output,
            width_px=screen.width_px,
            height_px=screen.height_px,
            fps=args.video_fps,
        )
        print(f"Recording rendered scene to {args.video_output.resolve()}.")

    print(f"Loaded calibration from {calibration_path.resolve()}. Press ESC in the demo window to quit.")

    try:
        while ui.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            predicted_px = None
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                face_distance_mm = distance_tracker.estimate(frame)
                result = gaze_estimator.predict(rgb_frame, return_details=True)
                if not isinstance(result, RuntimeEstimationResult):
                    raise RuntimeError("Estimator did not return detailed output.")
                predicted_px = calibration.apply(result.raw_xy, face_distance_mm)
            except (RuntimeError, ValueError) as exc:
                print(f"Gaze demo skipped: {exc}", end="\r")

            _, still_running = ui.update(predicted_px, camera_frame=rgb_frame)
            if scene_writer is not None:
                scene_frame_rgb = ui.capture_frame_rgb()
                scene_frame_bgr = cv2.cvtColor(scene_frame_rgb, cv2.COLOR_RGB2BGR)
                scene_writer.write(scene_frame_bgr)
            if not still_running:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if scene_writer is not None:
            scene_writer.release()
        ui.close()
        cap.release()


if __name__ == "__main__":
    main()
