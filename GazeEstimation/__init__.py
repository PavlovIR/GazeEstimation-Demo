from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50

from face_pos.camera import Camera
from gaze_normalization import estimate_head_pose, normalize_face_data_with_geometry
from IrisDetection import DetectionResult, Detector, FaceLandmarksResult, load_rgb_image
from map_to_display import maptodisplay


_SUPPORTED_ACTIVATION_FUNCTIONS = ("relu", "leaky_relu")
_BACKBONE_BUILDERS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}
_PARKS_WEIGHTS_STEM = "parks"
_HEAD_POSE_LANDMARK_INDICES = np.asarray([1, 152, 33, 263, 61, 291])
_HEAD_POSE_FACE_MODEL_MM = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.0, -87.6, -15.2],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ],
    dtype=np.float64,
)
_CAMERA_TO_SCREEN_AXIS_FLIP = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def _as_size_tuple(value: int | tuple[int, int] | list[int], default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError("Size values must be an int or a two-item [height, width] pair.")
    return (int(value[0]), int(value[1]))


def _grid_input_dim(size: tuple[int, int]) -> int:
    return int(size[0]) * int(size[1])


def _normalize_activation_function(activation_function: str) -> str:
    normalized = activation_function.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "relu": "relu",
        "leakyrelu": "leaky_relu",
        "leaky_relu": "leaky_relu",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(
        f"Unsupported activation_function={activation_function!r}. "
        f"Use one of: {', '.join(_SUPPORTED_ACTIVATION_FUNCTIONS)}."
    )


def _normalize_backbone_arch(backbone_arch: str) -> str:
    normalized = backbone_arch.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resnet18": "resnet18",
        "resnet_18": "resnet18",
        "resnet34": "resnet34",
        "resnet_34": "resnet34",
        "resnet50": "resnet50",
        "resnet_50": "resnet50",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError("backbone_arch must be one of: resnet18, resnet34, resnet50.")


def _make_activation(activation_function: str) -> nn.Module:
    normalized = _normalize_activation_function(activation_function)
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    raise AssertionError(f"Unexpected activation function: {normalized}")


def _replace_relu_with_activation(module: nn.Module, activation_function: str) -> None:
    normalized = _normalize_activation_function(activation_function)
    if normalized == "relu":
        return
    for child_name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, child_name, _make_activation(normalized))
        else:
            _replace_relu_with_activation(child, normalized)


@dataclass(slots=True)
class ModelConfig:
    use_spatial_weighting: bool = True
    backbone_arch: str = "resnet18"
    face_feature_dim: int = 256
    eye_feature_dim: int = 128
    face_grid_hidden_dim: int = 128
    iris_grid_hidden_dim: int = 16
    fusion_hidden_dim: int = 512
    dropout: float = 0.2
    activation_function: str = "relu"
    face_crop_size: tuple[int, int] = (224, 224)
    eye_crop_size: tuple[int, int] = (96, 96)
    face_grid_size: tuple[int, int] = (50, 50)
    iris_grid_size: tuple[int, int] = (10, 5)

    def __post_init__(self) -> None:
        self.backbone_arch = _normalize_backbone_arch(self.backbone_arch)
        self.activation_function = _normalize_activation_function(self.activation_function)
        self.face_crop_size = _as_size_tuple(self.face_crop_size, (224, 224))
        self.eye_crop_size = _as_size_tuple(self.eye_crop_size, (96, 96))
        self.face_grid_size = _as_size_tuple(self.face_grid_size, (50, 50))
        self.iris_grid_size = _as_size_tuple(self.iris_grid_size, (10, 5))


@dataclass(slots=True)
class EstimationResult:
    raw_xy: np.ndarray
    screen_xy: np.ndarray
    detection: DetectionResult
    model_output: np.ndarray | None = None

    def tolist(self) -> list[float]:
        return self.screen_xy.astype(float).tolist()


class SpatialWeightingBlock(nn.Module):
    def __init__(self, channels: int, activation_function: str) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            _make_activation(activation_function),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.projection(x)


class EyeEncoder(nn.Module):
    def __init__(self, output_dim: int, activation_function: str) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            _make_activation(activation_function),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            _make_activation(activation_function),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            _make_activation(activation_function),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            _make_activation(activation_function),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class MlpBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU(inplace=True)])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.extend([nn.Linear(previous_dim, output_dim), nn.ReLU(inplace=True)])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FaceEncoder(nn.Module):
    def __init__(self, feature_dim: int, use_spatial_weighting: bool, activation_function: str, backbone_arch: str) -> None:
        super().__init__()
        backbone = _BACKBONE_BUILDERS[backbone_arch](weights=None)
        backbone_out_channels = int(backbone.fc.in_features)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.spatial_weighting = (
            SpatialWeightingBlock(backbone_out_channels, activation_function) if use_spatial_weighting else nn.Identity()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels, feature_dim),
            _make_activation(activation_function),
        )
        _replace_relu_with_activation(self, activation_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spatial_weighting(x)
        x = self.pool(x)
        return self.projection(x)


class GazeEstimationAnn(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.face_encoder = FaceEncoder(
            feature_dim=config.face_feature_dim,
            use_spatial_weighting=config.use_spatial_weighting,
            activation_function=config.activation_function,
            backbone_arch=config.backbone_arch,
        )
        self.eye_encoder = EyeEncoder(
            output_dim=config.eye_feature_dim,
            activation_function=config.activation_function,
        )
        self.face_grid_branch = MlpBranch(
            input_dim=_grid_input_dim(config.face_grid_size),
            hidden_dims=[256],
            output_dim=config.face_grid_hidden_dim,
            dropout=config.dropout,
        )
        self.iris_grid_branch = MlpBranch(
            input_dim=_grid_input_dim(config.iris_grid_size),
            hidden_dims=[32],
            output_dim=config.iris_grid_hidden_dim,
            dropout=config.dropout,
        )

        fused_dim = (
            config.face_feature_dim
            + (2 * config.eye_feature_dim)
            + config.face_grid_hidden_dim
            + (2 * config.iris_grid_hidden_dim)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        face: torch.Tensor,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        face_grid: torch.Tensor,
        left_iris_grid: torch.Tensor,
        right_iris_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        face_features = self.face_encoder(face)
        left_eye_features = self.eye_encoder(left_eye)
        right_eye_features = self.eye_encoder(right_eye)
        face_grid_features = self.face_grid_branch(face_grid)
        left_iris_features = self.iris_grid_branch(left_iris_grid)
        right_iris_features = self.iris_grid_branch(right_iris_grid)
        fused = torch.cat(
            [
                face_features,
                left_eye_features,
                right_eye_features,
                face_grid_features,
                left_iris_features,
                right_iris_features,
            ],
            dim=1,
        )
        return {
            "prediction": self.fusion_head(fused),
            "aux_prediction": self.auxiliary_head(fused),
        }


def _load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint payload at {path}.")


def _activation_function_from_weights_path(path: Path) -> str | None:
    normalized_parts = {
        part.lower().replace("-", "_").replace(" ", "_")
        for part in path.parts
    }
    normalized_stem = path.stem.lower().replace("-", "_").replace(" ", "_")
    if "leaky_relu" in normalized_parts or "leaky_relu" in normalized_stem:
        return "leaky_relu"
    if "relu" in normalized_parts or normalized_stem == "relu":
        return "relu"
    return None


def _default_weights_path(activation_function: str = "relu") -> Path | None:
    normalized = _normalize_activation_function(activation_function)
    package_path = Path(__file__).resolve().parent
    if normalized == "relu":
        candidates = [
            package_path / "weights" / "best.pt",
            package_path / "weights" / "relu" / "best.pt",
            package_path.parent / "outputs" / "train" / "fold_p00" / "best.pt",
            package_path.parent / "outputs" / "activation_comparison" / "relu" / "fold_p00" / "best.pt",
        ]
    else:
        candidates = [
            package_path / "weights" / "leaky_relu" / "best.pt",
            package_path / "weights" / "best_leaky_relu.pt",
            package_path / "weights" / "leaky_relu_best.pt",
            package_path.parent / "outputs" / "activation_comparison" / "leaky_relu" / "fold_p00" / "best.pt",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _missing_weights_message(activation_function: str) -> str:
    normalized = _normalize_activation_function(activation_function)
    if normalized == "relu":
        location = "`GazeEstimation/weights/best.pt`"
    else:
        location = "`GazeEstimation/weights/leaky_relu/best.pt`"
    return (
        f"No pretrained weights were found for activation_function={normalized!r}. "
        f"Pass `weights_path`, or place a checkpoint at {location}."
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _weights_require_gaze_normalization(path: Path) -> bool:
    return path.stem.strip().lower().replace("-", "_").replace(" ", "_") == _PARKS_WEIGHTS_STEM


def _as_rgb_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return load_rgb_image(image)


def _perspective_transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate(
        [xy[:, :2], np.ones((xy.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float32).T
    z = transformed[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, np.where(z < 0, -1e-6, 1e-6), z)
    return transformed[:, :2] / z


def _crop_points_region(
    image: np.ndarray,
    points: np.ndarray,
    size: tuple[int, int],
    margin: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    target_height, target_width = size
    min_x = float(np.min(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_x = float(np.max(points[:, 0]))
    max_y = float(np.max(points[:, 1]))
    box_width = max(max_x - min_x, 1.0)
    box_height = max(max_y - min_y, 1.0)
    pad_x = box_width * margin
    pad_y = box_height * margin

    x1 = int(np.floor(np.clip(min_x - pad_x, 0.0, float(width - 1))))
    y1 = int(np.floor(np.clip(min_y - pad_y, 0.0, float(height - 1))))
    x2 = int(np.ceil(np.clip(max_x + pad_x, float(x1 + 1), float(width))))
    y2 = int(np.ceil(np.clip(max_y + pad_y, float(y1 + 1), float(height))))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Normalized eye crop is empty.")
    return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _screen_to_camera_extrinsics(screen: Any) -> tuple[np.ndarray, np.ndarray]:
    rotation = _CAMERA_TO_SCREEN_AXIS_FLIP
    camera_offset = np.asarray(screen.camera_offset, dtype=np.float64).reshape(3, 1)
    translation = -rotation @ camera_offset
    return rotation, translation


class Estimator:
    def __init__(
        self,
        IrisDetector: Detector | None = None,
        weights_path: str | Path | None = None,
        device: str = "auto",
        model_config: ModelConfig | None = None,
        screen_size: tuple[int, int] | None = None,
        screen: Any | None = None,
        fov_degrees: float = 60.0,
        calibration_matrix: np.ndarray | None = None,
        calibration_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        clamp_to_screen: bool = True,
        activation_function: str | None = None,
        use_gaze_normalization: bool | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.screen = screen
        self.fov_degrees = float(fov_degrees)
        self.screen_size = (
            screen_size
            if screen_size is not None
            else (
                (int(screen.width_px), int(screen.height_px))
                if screen is not None
                else None
            )
        )
        self.calibration_matrix = None if calibration_matrix is None else np.asarray(calibration_matrix, dtype=np.float32)
        self.calibration_fn = calibration_fn
        self.clamp_to_screen = clamp_to_screen
        self._camera_cache: dict[tuple[int, int], Camera] = {}

        requested_activation = _normalize_activation_function(
            activation_function
            or (model_config.activation_function if model_config is not None else "relu")
        )
        self.weights_path = (
            Path(weights_path)
            if weights_path is not None
            else _default_weights_path(requested_activation)
        )
        if self.weights_path is None:
            raise FileNotFoundError(_missing_weights_message(requested_activation))

        checkpoint = _load_torch_checkpoint(self.weights_path, self.device)
        checkpoint_config = self._config_from_checkpoint(checkpoint)
        inferred_activation = _activation_function_from_weights_path(self.weights_path)
        selected_activation = _normalize_activation_function(
            activation_function
            or (model_config.activation_function if model_config is not None else "")
            or self._activation_from_checkpoint(checkpoint)
            or inferred_activation
            or "relu"
        )
        self.activation_function = selected_activation
        self.model_config = replace(
            model_config or checkpoint_config or ModelConfig(),
            activation_function=selected_activation,
        )
        self.use_gaze_normalization = (
            _weights_require_gaze_normalization(self.weights_path)
            if use_gaze_normalization is None
            else bool(use_gaze_normalization)
        )
        if IrisDetector is not None:
            self.iris_detector = IrisDetector
            self.iris_detector.face_crop_size = self.model_config.face_crop_size
            self.iris_detector.eye_crop_size = self.model_config.eye_crop_size
            self.iris_detector.face_grid_size = self.model_config.face_grid_size
            self.iris_detector.iris_grid_size = self.model_config.iris_grid_size
            eye_height, eye_width = self.model_config.eye_crop_size
            self.iris_detector._use_mediapipe_iris = (
                getattr(self.iris_detector, "_use_mediapipe_iris", False)
                or eye_height != eye_width
            )
        else:
            self.iris_detector = Detector(
                device=str(self.device),
                face_crop_size=self.model_config.face_crop_size,
                eye_crop_size=self.model_config.eye_crop_size,
                face_grid_size=self.model_config.face_grid_size,
                iris_grid_size=self.model_config.iris_grid_size,
            )
        self.model = GazeEstimationAnn(self.model_config).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _config_from_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig | None:
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            return None
        config = metadata.get("model_config")
        if not isinstance(config, dict):
            return None
        allowed = set(ModelConfig.__dataclass_fields__)
        return ModelConfig(**{key: value for key, value in config.items() if key in allowed})

    @staticmethod
    def _activation_from_checkpoint(checkpoint: dict[str, Any]) -> str | None:
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            return None
        config = metadata.get("model_config")
        if not isinstance(config, dict):
            return None
        activation_function = config.get("activation_function")
        if not isinstance(activation_function, str):
            return None
        return _normalize_activation_function(activation_function)

    @staticmethod
    def fit_affine_calibration(predicted_points: np.ndarray, screen_points: np.ndarray) -> np.ndarray:
        predicted = np.asarray(predicted_points, dtype=np.float32)
        target = np.asarray(screen_points, dtype=np.float32)
        if predicted.ndim != 2 or predicted.shape[1] != 2:
            raise ValueError("predicted_points must have shape Nx2.")
        if target.shape != predicted.shape:
            raise ValueError("screen_points must have the same shape as predicted_points.")
        if predicted.shape[0] < 3:
            raise ValueError("At least three point pairs are required for affine calibration.")

        design = np.concatenate([predicted, np.ones((predicted.shape[0], 1), dtype=np.float32)], axis=1)
        matrix_t, *_ = np.linalg.lstsq(design, target, rcond=None)
        return matrix_t.T.astype(np.float32)

    def set_affine_calibration(self, predicted_points: np.ndarray, screen_points: np.ndarray) -> np.ndarray:
        self.calibration_matrix = self.fit_affine_calibration(predicted_points, screen_points)
        return self.calibration_matrix

    def _apply_calibration(self, xy: np.ndarray) -> np.ndarray:
        calibrated = np.asarray(xy, dtype=np.float32)
        if self.calibration_fn is not None:
            calibrated = np.asarray(self.calibration_fn(calibrated), dtype=np.float32)
        if self.calibration_matrix is not None:
            if self.calibration_matrix.shape != (2, 3):
                raise ValueError("calibration_matrix must have shape 2x3.")
            calibrated = self.calibration_matrix @ np.asarray([calibrated[0], calibrated[1], 1.0], dtype=np.float32)
        if self.screen_size is not None and self.clamp_to_screen:
            width, height = self.screen_size
            calibrated = np.asarray(
                [
                    np.clip(calibrated[0], 0.0, float(width - 1)),
                    np.clip(calibrated[1], 0.0, float(height - 1)),
                ],
                dtype=np.float32,
            )
        return calibrated

    def _tensor_from_image(self, image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image.astype("float32").transpose(2, 0, 1)).unsqueeze(0).to(self.device) / 255.0

    def _tensor_from_grid(self, grid: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(grid.astype("float32").reshape(1, -1)).to(self.device)

    def _camera_for_frame(self, width: int, height: int) -> Camera:
        key = (int(width), int(height))
        camera = self._camera_cache.get(key)
        if camera is None:
            camera = Camera(width=key[0], height=key[1], fov_degrees=self.fov_degrees)
            self._camera_cache[key] = camera
        return camera

    def _normalize_detection(
        self,
        rgb_image: np.ndarray,
        detection: DetectionResult,
    ):
        image_height, image_width = rgb_image.shape[:2]
        camera = self._camera_for_frame(width=image_width, height=image_height)

        image_points = detection.landmarks[_HEAD_POSE_LANDMARK_INDICES, :2].astype(
            np.float64
        )
        rvec, tvec = estimate_head_pose(
            image_points.reshape(-1, 1, 2),
            _HEAD_POSE_FACE_MODEL_MM.reshape(-1, 1, 3),
            camera.intrinsics,
            camera.dist_coeffs,
            iterate=True,
        )

        face_height, face_width = self.model_config.face_crop_size
        normalized = normalize_face_data_with_geometry(
            rgb_image,
            _HEAD_POSE_FACE_MODEL_MM.T,
            rvec,
            tvec,
            camera.intrinsics,
            roiSize=(face_width, face_height),
            focal_norm=700,
            distance_norm=600,
            grayscale=False,
            equalize=False,
        )

        normalized_landmarks = _perspective_transform_points(
            detection.landmarks[:, :2],
            normalized.warp,
        )
        normalized_landmark_result = FaceLandmarksResult(
            landmarks=normalized_landmarks,
            image_width=face_width,
            image_height=face_height,
        )
        left_eye_points, right_eye_points = Detector._split_eye_regions(
            normalized_landmark_result
        )
        eye_size = self.model_config.eye_crop_size
        left_eye = _crop_points_region(
            normalized.image,
            left_eye_points[:, :2],
            size=eye_size,
            margin=0.45,
        )
        right_eye = _crop_points_region(
            normalized.image,
            right_eye_points[:, :2],
            size=eye_size,
            margin=0.45,
        )
        canonical_right_eye = np.ascontiguousarray(np.flip(right_eye, axis=1))

        return (
            replace(
                detection,
                face=np.ascontiguousarray(normalized.image),
                left_eye=np.ascontiguousarray(left_eye),
                right_eye=canonical_right_eye,
            ),
            normalized,
        )

    def _project_normalized_prediction(
        self,
        pitchyaw: np.ndarray,
        normalization,
    ) -> np.ndarray:
        if self.screen is None:
            return np.asarray(pitchyaw, dtype=np.float32)

        screen_rotation, screen_translation = _screen_to_camera_extrinsics(self.screen)
        screen_points_mm = maptodisplay(
            np.asarray(pitchyaw, dtype=np.float32).reshape(1, 2),
            [normalization.rotation_matrix],
            [normalization.face_center],
            screen_rotation,
            screen_translation,
        )
        screen_mm = np.asarray(screen_points_mm[0], dtype=np.float32).reshape(2)
        if not np.all(np.isfinite(screen_mm)):
            raise RuntimeError("Projected gaze point is not finite.")
        return np.asarray(self.screen._mm_to_px(screen_mm), dtype=np.float32)

    @torch.inference_mode()
    def predict(self, image: str | Path | np.ndarray, return_details: bool = False) -> np.ndarray | EstimationResult:
        rgb_image = _as_rgb_image(image)
        detection = self.iris_detector.detect(rgb_image)
        normalization = None
        if self.use_gaze_normalization:
            detection, normalization = self._normalize_detection(rgb_image, detection)

        outputs = self.model(
            face=self._tensor_from_image(detection.face),
            left_eye=self._tensor_from_image(detection.left_eye),
            right_eye=self._tensor_from_image(detection.right_eye),
            face_grid=self._tensor_from_grid(detection.face_grid),
            left_iris_grid=self._tensor_from_grid(detection.left_iris_grid),
            right_iris_grid=self._tensor_from_grid(detection.right_iris_grid),
        )
        model_output = outputs["prediction"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        raw_xy = (
            self._project_normalized_prediction(model_output, normalization)
            if normalization is not None
            else model_output
        )
        screen_xy = self._apply_calibration(raw_xy)
        if return_details:
            return EstimationResult(
                raw_xy=raw_xy,
                screen_xy=screen_xy,
                detection=detection,
                model_output=model_output if normalization is not None else None,
            )
        return screen_xy

    def estimate(self, image: str | Path | np.ndarray, return_details: bool = False) -> np.ndarray | EstimationResult:
        return self.predict(image, return_details=return_details)

    def __call__(self, image: str | Path | np.ndarray, return_details: bool = False) -> np.ndarray | EstimationResult:
        return self.predict(image, return_details=return_details)


__all__ = [
    "EstimationResult",
    "Estimator",
    "GazeEstimationAnn",
    "ModelConfig",
]
