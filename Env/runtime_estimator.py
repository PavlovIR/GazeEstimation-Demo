from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from GazeCaptureEstimator import GC_Estimator
from GazeEstimation import EstimationResult as AnnEstimationResult
from GazeEstimation import Estimator as AnnEstimator
from IrisDetection import BoundingBox, DetectionResult
from IrisDetection import Detector as IrisDetector
from IrisDetection import _bounding_box_from_points
from IrisDetection import load_rgb_image
from gaze_ui.Screen import Screen
from Env.run_parameters import normalize_estimator_lib


ENV_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENV_DIR.parent
DEFAULT_GAZE_CAPTURE_CHECKPOINT = (
    PROJECT_ROOT / "GazeCaptureEstimator" / "checkpoint.pth.tar"
)


@dataclass(slots=True)
class RuntimeEstimationResult:
    raw_xy: np.ndarray
    screen_xy: np.ndarray
    detection: DetectionResult | "GazeCaptureDetectionResult"


@dataclass(slots=True)
class GazeCaptureDetectionResult:
    landmarks: np.ndarray
    face_box: BoundingBox
    left_eye_box: BoundingBox
    right_eye_box: BoundingBox


class AnnRuntimeEstimator:
    estimator_lib = "GazeEstimation"

    def __init__(
        self,
        *,
        screen_size: tuple[int, int] | None,
        weights_path: str | Path | None,
        activation_function: str,
        model_device: str,
        iris_device: str,
        mediapipe_model_path: str | Path,
        iris_data_dir: str | Path | None,
    ) -> None:
        iris_detector = IrisDetector(
            mediapipe_model_path=_optional_path(Path(mediapipe_model_path)),
            iris_data_dir=_optional_path(Path(iris_data_dir))
            if iris_data_dir is not None
            else None,
            device=iris_device,
        )
        self._estimator = AnnEstimator(
            IrisDetector=iris_detector,
            weights_path=weights_path,
            screen_size=screen_size,
            device=model_device,
            activation_function=activation_function,
        )

    @property
    def activation_function(self) -> str:
        return self._estimator.activation_function

    @property
    def weights_path(self) -> Path:
        return self._estimator.weights_path

    def predict(
        self, image: str | Path | np.ndarray, return_details: bool = False
    ) -> np.ndarray | RuntimeEstimationResult:
        result = self._estimator.predict(image, return_details=return_details)
        if not return_details:
            return result
        if not isinstance(result, AnnEstimationResult):
            raise RuntimeError("Estimator did not return detailed output.")
        return RuntimeEstimationResult(
            raw_xy=result.raw_xy,
            screen_xy=result.screen_xy,
            detection=result.detection,
        )


class GazeCaptureRuntimeEstimator:
    estimator_lib = "GazeCaptureEstimator"
    activation_function = "n/a"

    def __init__(
        self,
        *,
        screen: Screen | None,
        weights_path: str | Path | None,
        model_device: str,
        iris_device: str,
        mediapipe_model_path: str | Path,
        iris_data_dir: str | Path | None,
    ) -> None:
        self.screen = screen
        self.weights_path = _resolve_gaze_capture_weights(weights_path)
        self._detector = IrisDetector(
            mediapipe_model_path=_optional_path(Path(mediapipe_model_path)),
            iris_data_dir=_optional_path(Path(iris_data_dir))
            if iris_data_dir is not None
            else None,
            device=iris_device,
        )
        self._estimator = GC_Estimator(self.weights_path, device=model_device)

    def predict(
        self, image: str | Path | np.ndarray, return_details: bool = False
    ) -> np.ndarray | RuntimeEstimationResult:
        rgb_image = _as_rgb_image(image)
        detection = self._detect_regions(rgb_image)
        raw_xy = self._estimator.predict_from_bboxes(
            rgb_image,
            face_box=_box_to_xywh(detection.face_box),
            left_eye_box=_box_to_xywh(detection.left_eye_box),
            right_eye_box=_box_to_xywh(detection.right_eye_box),
            frame_color="RGB",
            as_numpy=True,
        )
        if raw_xy is None:
            raise RuntimeError("GazeCaptureEstimator could not crop face/eye inputs.")

        raw_xy = np.asarray(raw_xy, dtype=np.float32).reshape(-1, 2)[0]
        screen_xy = self._raw_camera_cm_to_screen_px(raw_xy)
        if return_details:
            return RuntimeEstimationResult(
                raw_xy=raw_xy,
                screen_xy=screen_xy,
                detection=detection,
            )
        return screen_xy

    def _detect_regions(self, rgb_image: np.ndarray) -> GazeCaptureDetectionResult:
        face_result = self._detector.detect_face(rgb_image)
        if face_result is None:
            raise RuntimeError("No face landmarks detected in the input image.")

        left_eye_points, right_eye_points = IrisDetector._split_eye_regions(face_result)
        face_box = _bounding_box_from_points(
            face_result.landmarks[:, :2],
            margin=self._detector.face_box_margin,
            image_width=face_result.image_width,
            image_height=face_result.image_height,
        )
        left_eye_box = _bounding_box_from_points(
            left_eye_points[:, :2],
            margin=self._detector.eye_box_margin,
            image_width=face_result.image_width,
            image_height=face_result.image_height,
        )
        right_eye_box = _bounding_box_from_points(
            right_eye_points[:, :2],
            margin=self._detector.eye_box_margin,
            image_width=face_result.image_width,
            image_height=face_result.image_height,
        )
        return GazeCaptureDetectionResult(
            landmarks=face_result.landmarks,
            face_box=face_box,
            left_eye_box=left_eye_box,
            right_eye_box=right_eye_box,
        )

    def _raw_camera_cm_to_screen_px(self, raw_xy: np.ndarray) -> np.ndarray:
        if self.screen is None:
            return np.asarray(raw_xy, dtype=np.float32)

        screen_mm = np.asarray(raw_xy, dtype=np.float32) * 10.0
        screen_mm[0] += float(self.screen.camera_offset[0])
        screen_mm[1] += float(self.screen.camera_offset[1])
        screen_px = self.screen._mm_to_px(screen_mm)
        return np.asarray(screen_px, dtype=np.float32)


def build_runtime_gaze_estimator(
    *,
    estimator_lib: str,
    screen: Screen | None,
    screen_size: tuple[int, int] | None,
    weights_path: str | Path | None,
    activation_function: str,
    model_device: str,
    iris_device: str,
    mediapipe_model_path: str | Path,
    iris_data_dir: str | Path | None,
) -> AnnRuntimeEstimator | GazeCaptureRuntimeEstimator:
    estimator_lib = normalize_estimator_lib(estimator_lib)
    if estimator_lib == "GazeCaptureEstimator":
        return GazeCaptureRuntimeEstimator(
            screen=screen,
            weights_path=weights_path,
            model_device=model_device,
            iris_device=iris_device,
            mediapipe_model_path=mediapipe_model_path,
            iris_data_dir=iris_data_dir,
        )

    return AnnRuntimeEstimator(
        screen_size=screen_size,
        weights_path=weights_path,
        activation_function=activation_function,
        model_device=model_device,
        iris_device=iris_device,
        mediapipe_model_path=mediapipe_model_path,
        iris_data_dir=iris_data_dir,
    )


def _resolve_gaze_capture_weights(weights_path: str | Path | None) -> Path:
    resolved = Path(weights_path) if weights_path is not None else DEFAULT_GAZE_CAPTURE_CHECKPOINT
    if not resolved.exists():
        raise FileNotFoundError(
            "No GazeCaptureEstimator checkpoint found. Set `weights` in "
            "run_parameters.toml or place a checkpoint at "
            f"{DEFAULT_GAZE_CAPTURE_CHECKPOINT}."
        )
    return resolved


def _box_to_xywh(box: BoundingBox) -> tuple[float, float, float, float]:
    return box.x, box.y, box.width, box.height


def _optional_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _as_rgb_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return load_rgb_image(image)


def calibration_metadata_matches_estimator(
    metadata: dict[str, Any], estimator_lib: str
) -> bool | None:
    metadata_lib = metadata.get("estimator_lib")
    if metadata_lib is None:
        return None
    return normalize_estimator_lib(str(metadata_lib)) == normalize_estimator_lib(
        estimator_lib
    )
