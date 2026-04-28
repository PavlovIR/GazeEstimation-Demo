from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from IrisDetection import DetectionResult, Detector


@dataclass(slots=True)
class ModelConfig:
    use_spatial_weighting: bool = True
    face_feature_dim: int = 256
    eye_feature_dim: int = 128
    face_grid_hidden_dim: int = 128
    iris_grid_hidden_dim: int = 16
    fusion_hidden_dim: int = 512
    dropout: float = 0.2


@dataclass(slots=True)
class EstimationResult:
    raw_xy: np.ndarray
    screen_xy: np.ndarray
    detection: DetectionResult

    def tolist(self) -> list[float]:
        return self.screen_xy.astype(float).tolist()


class SpatialWeightingBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.projection(x)


class EyeEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class MlpBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0) -> None:
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
    def __init__(self, feature_dim: int, use_spatial_weighting: bool) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.spatial_weighting = SpatialWeightingBlock(512) if use_spatial_weighting else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

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
        )
        self.eye_encoder = EyeEncoder(output_dim=config.eye_feature_dim)
        self.face_grid_branch = MlpBranch(
            input_dim=2500,
            hidden_dims=[256],
            output_dim=config.face_grid_hidden_dim,
            dropout=config.dropout,
        )
        self.iris_grid_branch = MlpBranch(
            input_dim=50,
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


def _default_weights_path() -> Path | None:
    package_path = Path(__file__).resolve().parent
    candidates = [
        package_path / "weights" / "best.pt",
        package_path.parent / "outputs" / "train" / "fold_p00" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


class Estimator:
    def __init__(
        self,
        IrisDetector: Detector | None = None,
        weights_path: str | Path | None = None,
        device: str = "auto",
        model_config: ModelConfig | None = None,
        screen_size: tuple[int, int] | None = None,
        calibration_matrix: np.ndarray | None = None,
        calibration_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        clamp_to_screen: bool = True,
    ) -> None:
        self.iris_detector = IrisDetector if IrisDetector is not None else Detector(device="cpu")
        self.device = _resolve_device(device)
        self.screen_size = screen_size
        self.calibration_matrix = None if calibration_matrix is None else np.asarray(calibration_matrix, dtype=np.float32)
        self.calibration_fn = calibration_fn
        self.clamp_to_screen = clamp_to_screen

        self.weights_path = Path(weights_path) if weights_path is not None else _default_weights_path()
        if self.weights_path is None:
            raise FileNotFoundError(
                "No pretrained weights were found. Pass `weights_path`, or place `best.pt` at "
                "`GazeEstimation/weights/best.pt`."
            )

        checkpoint = _load_torch_checkpoint(self.weights_path, self.device)
        self.model_config = model_config or self._config_from_checkpoint(checkpoint) or ModelConfig()
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

    @torch.inference_mode()
    def predict(self, image: str | Path | np.ndarray, return_details: bool = False) -> np.ndarray | EstimationResult:
        detection = self.iris_detector.detect(image)
        outputs = self.model(
            face=self._tensor_from_image(detection.face),
            left_eye=self._tensor_from_image(detection.left_eye),
            right_eye=self._tensor_from_image(detection.right_eye),
            face_grid=self._tensor_from_grid(detection.face_grid),
            left_iris_grid=self._tensor_from_grid(detection.left_iris_grid),
            right_iris_grid=self._tensor_from_grid(detection.right_iris_grid),
        )
        raw_xy = outputs["prediction"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        screen_xy = self._apply_calibration(raw_xy)
        if return_details:
            return EstimationResult(raw_xy=raw_xy, screen_xy=screen_xy, detection=detection)
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
