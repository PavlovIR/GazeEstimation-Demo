from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

CALIBRATION_DIR = Path(__file__).resolve().parent
ROOT = CALIBRATION_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from face_pos.camera import Camera
from face_pos.detector import FaceDetector
from face_pos.estimator import DistanceEstimator
from gaze_ui.AccuracyUi import set_dpi_awareness
from gaze_ui.Screen import Screen
from GazeEstimation import Estimator as GazeEstimator
from IrisDetection import Detector as IrisDetector
from Env.run_parameters import (
    DEFAULT_RUN_PARAMETERS_PATH,
    load_run_parameters,
    normalize_estimator_lib,
)
from Env.runtime_estimator import RuntimeEstimationResult, build_runtime_gaze_estimator

DEFAULT_CALIBRATION_PATH = CALIBRATION_DIR / "calibration.json"
DEFAULT_CALIBRATION_PARAMETERS_PATH = CALIBRATION_DIR / "calibration_parameters.toml"
DEFAULT_CONFIG_PATH = CALIBRATION_DIR / "config.toml"
DEFAULT_POINTS_PATH = ROOT / "csv_points" / "calibration-points.csv"
DEFAULT_MEDIAPIPE_MODEL = ROOT / "models" / "face_landmarker_v2_with_blendshapes.task"
DEFAULT_IRIS_DATA_DIR = ROOT / "IrisDetection" / "data"
DEFAULT_ACTIVATION_FUNCTION = "leaky_relu"

DEFAULT_RELATIVE_POINTS: tuple[tuple[float, float], ...] = (
    (0.25, 0.25),
    (0.50, 0.25),
    (0.75, 0.25),
    (0.25, 0.50),
    (0.50, 0.50),
    (0.75, 0.50),
    (0.25, 0.75),
    (0.50, 0.75),
    (0.75, 0.75),
)


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    estimator_lib: str = "GazeEstimation"
    activation_function: str = DEFAULT_ACTIVATION_FUNCTION
    screen_config: Path = DEFAULT_CONFIG_PATH
    calibration_file: Path = DEFAULT_CALIBRATION_PATH
    points: Path = DEFAULT_POINTS_PATH
    weights: Path | None = None
    camera: int = 0
    fov_degrees: float = 60.0
    settle_seconds: float = 0.6
    sample_seconds: float = 1.2
    min_frames: int = 4
    device: str = "auto"
    iris_device: str = "auto"
    mediapipe_model: Path = DEFAULT_MEDIAPIPE_MODEL
    iris_data_dir: Path | None = DEFAULT_IRIS_DATA_DIR


@dataclass(frozen=True, slots=True)
class CalibrationStage:
    name: str
    instruction: str


DEFAULT_STAGES: tuple[CalibrationStage, ...] = (
    CalibrationStage("near", "Move closer to the camera, then press SPACE."),
    CalibrationStage(
        "normal", "Move to your normal working distance, then press SPACE."
    ),
    CalibrationStage("far", "Move farther from the camera, then press SPACE."),
)


@dataclass(slots=True)
class CalibrationSample:
    stage: str
    point_index: int
    target_xy: np.ndarray
    raw_xy: np.ndarray
    face_distance_mm: float
    frame_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "point_index": self.point_index,
            "target_xy": self.target_xy.astype(float).tolist(),
            "raw_xy": self.raw_xy.astype(float).tolist(),
            "face_distance_mm": float(self.face_distance_mm),
            "frame_count": int(self.frame_count),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CalibrationSample":
        return cls(
            stage=str(payload["stage"]),
            point_index=int(payload["point_index"]),
            target_xy=np.asarray(payload["target_xy"], dtype=np.float32),
            raw_xy=np.asarray(payload["raw_xy"], dtype=np.float32),
            face_distance_mm=float(payload["face_distance_mm"]),
            frame_count=int(payload.get("frame_count", 1)),
        )


@dataclass(slots=True)
class DistanceCalibrationBin:
    stage: str
    distance_mm: float
    matrix: np.ndarray
    sample_count: int

    def apply(self, raw_xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(raw_xy, dtype=np.float32)
        return self.matrix @ np.asarray([xy[0], xy[1], 1.0], dtype=np.float32)

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "distance_mm": float(self.distance_mm),
            "matrix": self.matrix.astype(float).tolist(),
            "sample_count": int(self.sample_count),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "DistanceCalibrationBin":
        matrix = np.asarray(payload["matrix"], dtype=np.float32)
        if matrix.shape != (2, 3):
            raise ValueError("Calibration matrix must have shape 2x3.")
        return cls(
            stage=str(payload["stage"]),
            distance_mm=float(payload["distance_mm"]),
            matrix=matrix,
            sample_count=int(payload.get("sample_count", 0)),
        )


@dataclass(slots=True)
class DistanceAwareCalibration:
    screen_size: tuple[int, int]
    bins: list[DistanceCalibrationBin]
    global_matrix: np.ndarray
    samples: list[CalibrationSample]
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        samples: Sequence[CalibrationSample],
        screen_size: tuple[int, int],
        metadata: dict[str, Any] | None = None,
    ) -> "DistanceAwareCalibration":
        if len(samples) < 3:
            raise ValueError("At least three calibration samples are required.")

        global_matrix = _fit_affine(
            predicted_points=np.asarray(
                [sample.raw_xy for sample in samples], dtype=np.float32
            ),
            screen_points=np.asarray(
                [sample.target_xy for sample in samples], dtype=np.float32
            ),
        )

        bins: list[DistanceCalibrationBin] = []
        for stage in dict.fromkeys(sample.stage for sample in samples):
            stage_samples = [sample for sample in samples if sample.stage == stage]
            if len(stage_samples) < 3:
                continue

            matrix = _fit_affine(
                predicted_points=np.asarray(
                    [sample.raw_xy for sample in stage_samples], dtype=np.float32
                ),
                screen_points=np.asarray(
                    [sample.target_xy for sample in stage_samples], dtype=np.float32
                ),
            )
            bins.append(
                DistanceCalibrationBin(
                    stage=stage,
                    distance_mm=float(
                        np.median([sample.face_distance_mm for sample in stage_samples])
                    ),
                    matrix=matrix,
                    sample_count=len(stage_samples),
                )
            )

        if not bins:
            bins.append(
                DistanceCalibrationBin(
                    stage="global",
                    distance_mm=float(
                        np.median([sample.face_distance_mm for sample in samples])
                    ),
                    matrix=global_matrix,
                    sample_count=len(samples),
                )
            )

        bins.sort(key=lambda item: item.distance_mm)
        return cls(
            screen_size=(int(screen_size[0]), int(screen_size[1])),
            bins=bins,
            global_matrix=global_matrix,
            samples=list(samples),
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={} if metadata is None else dict(metadata),
        )

    def apply(
        self,
        raw_xy: np.ndarray,
        face_distance_mm: float | None,
        clamp_to_screen: bool = True,
    ) -> np.ndarray:
        if (
            not self.bins
            or face_distance_mm is None
            or not np.isfinite(face_distance_mm)
        ):
            calibrated = self._apply_matrix(self.global_matrix, raw_xy)
        else:
            calibrated = self._apply_distance_bins(raw_xy, float(face_distance_mm))

        if clamp_to_screen:
            calibrated = _clamp_xy(calibrated, self.screen_size)
        return calibrated.astype(np.float32)

    def _apply_distance_bins(
        self, raw_xy: np.ndarray, face_distance_mm: float
    ) -> np.ndarray:
        if len(self.bins) == 1:
            return self.bins[0].apply(raw_xy)

        if face_distance_mm <= self.bins[0].distance_mm:
            return self.bins[0].apply(raw_xy)
        if face_distance_mm >= self.bins[-1].distance_mm:
            return self.bins[-1].apply(raw_xy)

        for left, right in zip(self.bins, self.bins[1:]):
            if left.distance_mm <= face_distance_mm <= right.distance_mm:
                span = max(right.distance_mm - left.distance_mm, 1e-6)
                t = (face_distance_mm - left.distance_mm) / span
                return ((1.0 - t) * left.apply(raw_xy)) + (t * right.apply(raw_xy))

        return self._apply_matrix(self.global_matrix, raw_xy)

    @staticmethod
    def _apply_matrix(matrix: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(raw_xy, dtype=np.float32)
        return matrix @ np.asarray([xy[0], xy[1], 1.0], dtype=np.float32)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "created_at": self.created_at,
            "screen_size": [int(self.screen_size[0]), int(self.screen_size[1])],
            "global_matrix": self.global_matrix.astype(float).tolist(),
            "distance_bins": [item.to_json() for item in self.bins],
            "samples": [sample.to_json() for sample in self.samples],
            "metadata": self.metadata,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "DistanceAwareCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != 1:
            raise ValueError("Unsupported calibration file version.")

        global_matrix = np.asarray(payload["global_matrix"], dtype=np.float32)
        if global_matrix.shape != (2, 3):
            raise ValueError("global_matrix must have shape 2x3.")

        bins = [
            DistanceCalibrationBin.from_json(item) for item in payload["distance_bins"]
        ]
        bins.sort(key=lambda item: item.distance_mm)
        return cls(
            screen_size=(
                int(payload["screen_size"][0]),
                int(payload["screen_size"][1]),
            ),
            bins=bins,
            global_matrix=global_matrix,
            samples=[
                CalibrationSample.from_json(item) for item in payload.get("samples", [])
            ],
            created_at=str(payload.get("created_at", "")),
            metadata=dict(payload.get("metadata", {})),
        )


class FaceDistanceTracker:
    def __init__(
        self,
        width: int,
        height: int,
        mediapipe_model_path: str | Path = DEFAULT_MEDIAPIPE_MODEL,
        fov_degrees: float = 60.0,
    ) -> None:
        self.camera = Camera(width=width, height=height, fov_degrees=fov_degrees)
        self.detector = FaceDetector(str(mediapipe_model_path))
        self.estimator = DistanceEstimator(self.camera)

    def estimate(self, frame_bgr: np.ndarray) -> float | None:
        image_points = self.detector.extract_key_landmarks(frame_bgr)
        if image_points is None:
            return None
        face_pos_cam_mm = self.estimator.estimate_3d_position(image_points)
        if face_pos_cam_mm is None:
            return None
        return abs(float(face_pos_cam_mm[2]))


class CalibrationWindow:
    def __init__(self, screen: Screen) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise SystemExit(
                "pygame is required for calibration. Install with `pip install pygame`."
            ) from exc

        self.pygame = pygame
        self.screen_cfg = screen
        set_dpi_awareness()
        pygame.init()
        self.surface = pygame.display.set_mode(
            (screen.width_px, screen.height_px), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Gaze calibration")
        self.title_font = pygame.font.SysFont("monospace", 34)
        self.font = pygame.font.SysFont("monospace", 22)
        self.small_font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()
        self.running = True
        self._display_target_xy: np.ndarray | None = None
        self._last_tick = time.monotonic()

    def poll_action(self) -> str | None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
                return "quit"
            if event.type == self.pygame.KEYDOWN:
                if event.key in (self.pygame.K_ESCAPE, self.pygame.K_q):
                    self.running = False
                    return "quit"
                if event.key in (self.pygame.K_SPACE, self.pygame.K_RETURN):
                    return "continue"
        return None

    def draw(
        self,
        title: str,
        lines: Sequence[str],
        target_xy: tuple[int, int] | None = None,
        progress: float | None = None,
    ) -> None:
        pygame = self.pygame
        now = time.monotonic()
        dt = min(max(now - self._last_tick, 0.0), 0.05)
        self._last_tick = now
        self.surface.fill((12, 12, 12))

        if target_xy is not None:
            target = np.asarray(target_xy, dtype=np.float32)
            if self._display_target_xy is None:
                self._display_target_xy = target
            else:
                step = min(1.0, dt * 8.0)
                self._display_target_xy = self._display_target_xy + (
                    (target - self._display_target_xy) * step
                )

            draw_target = tuple(int(round(v)) for v in self._display_target_xy)
            pulse = (np.sin(now * 7.5) + 1.0) * 0.5
            radius = int(round(19 + (pulse * 8)))
            pygame.draw.circle(
                self.surface, (40, 95, 130), draw_target, radius + 10, width=2
            )
            pygame.draw.circle(self.surface, (80, 180, 255), draw_target, radius)
            pygame.draw.circle(self.surface, (230, 245, 255), draw_target, 5)
        else:
            self._display_target_xy = None

        y = 28
        self._draw_line(title, x=32, y=y, font=self.title_font, color=(245, 245, 245))
        y += 48
        for line in lines:
            self._draw_line(line, x=32, y=y, font=self.font, color=(230, 230, 230))
            y += 30

        if progress is not None:
            bar_x = 32
            bar_y = self.screen_cfg.height_px - 52
            bar_w = self.screen_cfg.width_px - 64
            pygame.draw.rect(
                self.surface, (50, 50, 50), (bar_x, bar_y, bar_w, 18), border_radius=9
            )
            pygame.draw.rect(
                self.surface,
                (80, 180, 255),
                (bar_x, bar_y, int(bar_w * float(np.clip(progress, 0.0, 1.0))), 18),
                border_radius=9,
            )

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_line(
        self, text: str, x: int, y: int, font: Any, color: tuple[int, int, int]
    ) -> None:
        surf = font.render(text, True, color)
        self.surface.blit(surf, (x, y))

    def close(self) -> None:
        self.pygame.quit()


class AdvancedCalibrationPipeline:
    def __init__(
        self,
        screen: Screen,
        output_path: str | Path = DEFAULT_CALIBRATION_PATH,
        points_path: str | Path | None = DEFAULT_POINTS_PATH,
        camera_index: int = 0,
        fov_degrees: float = 60.0,
        settle_seconds: float = 0.6,
        sample_seconds: float = 1.2,
        min_frames_per_point: int = 4,
        stages: Sequence[CalibrationStage] = DEFAULT_STAGES,
        mediapipe_model_path: str | Path = DEFAULT_MEDIAPIPE_MODEL,
        iris_data_dir: str | Path | None = DEFAULT_IRIS_DATA_DIR,
        model_device: str = "auto",
        iris_device: str = "auto",
        weights_path: str | Path | None = None,
        activation_function: str = DEFAULT_ACTIVATION_FUNCTION,
        estimator_lib: str = "GazeEstimation",
    ) -> None:
        self.screen = screen
        self.output_path = Path(output_path)
        self.target_points = load_target_points(screen, points_path)
        self.camera_index = int(camera_index)
        self.fov_degrees = float(fov_degrees)
        self.settle_seconds = float(settle_seconds)
        self.sample_seconds = float(sample_seconds)
        self.min_frames_per_point = int(min_frames_per_point)
        self.stages = list(stages)
        self.mediapipe_model_path = Path(mediapipe_model_path)
        self.iris_data_dir = Path(iris_data_dir) if iris_data_dir is not None else None
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.activation_function = activation_function
        self.gaze_estimator = build_runtime_gaze_estimator(
            estimator_lib=estimator_lib,
            screen=screen,
            screen_size=(screen.width_px, screen.height_px),
            mediapipe_model_path=self.mediapipe_model_path,
            iris_data_dir=self.iris_data_dir,
            model_device=model_device,
            iris_device=iris_device,
            weights_path=self.weights_path,
            activation_function=self.activation_function,
        )
        self.estimator_lib = self.gaze_estimator.estimator_lib
        self.activation_function = self.gaze_estimator.activation_function
        self.weights_path = self.gaze_estimator.weights_path

    def run(self) -> DistanceAwareCalibration:
        cap = cv2.VideoCapture(self.camera_index)
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
            mediapipe_model_path=self.mediapipe_model_path,
            fov_degrees=self.fov_degrees,
        )
        ui = CalibrationWindow(self.screen)
        samples: list[CalibrationSample] = []

        try:
            for stage_index, stage in enumerate(self.stages, start=1):
                self._wait_for_stage_start(
                    cap, ui, distance_tracker, stage, stage_index
                )
                for point_index, target_xy in enumerate(self.target_points, start=1):
                    sample = None
                    while sample is None:
                        sample = self._collect_point_sample(
                            cap=cap,
                            ui=ui,
                            distance_tracker=distance_tracker,
                            stage=stage,
                            stage_index=stage_index,
                            point_index=point_index,
                            target_xy=target_xy,
                        )
                        if sample is None:
                            self._show_message(
                                ui,
                                "Calibration sample failed",
                                [
                                    "Not enough valid face/gaze frames were collected.",
                                    "Keep your face visible and press SPACE to retry.",
                                    "Press Q or ESC to abort.",
                                ],
                            )
                    samples.append(sample)
        finally:
            ui.close()
            cap.release()

        calibration = DistanceAwareCalibration.fit(
            samples=samples,
            screen_size=(self.screen.width_px, self.screen.height_px),
            metadata={
                "estimator_lib": self.gaze_estimator.estimator_lib,
                "activation_function": self.gaze_estimator.activation_function,
                "weights_path": str(self.gaze_estimator.weights_path),
            },
        )
        calibration.save(self.output_path)
        return calibration

    def _wait_for_stage_start(
        self,
        cap: cv2.VideoCapture,
        ui: CalibrationWindow,
        distance_tracker: FaceDistanceTracker,
        stage: CalibrationStage,
        stage_index: int,
    ) -> None:
        while ui.running:
            ret, frame = cap.read()
            distance_mm = distance_tracker.estimate(frame) if ret else None
            action = ui.poll_action()
            if action == "quit":
                raise KeyboardInterrupt("Calibration aborted.")
            if action == "continue" and distance_mm is not None:
                return

            distance_line = _format_distance(distance_mm)
            ui.draw(
                title=f"Distance {stage_index}/{len(self.stages)}: {stage.name}",
                lines=[
                    stage.instruction,
                    distance_line,
                    "Press SPACE when ready. Press Q or ESC to abort.",
                ],
            )

    def _collect_point_sample(
        self,
        cap: cv2.VideoCapture,
        ui: CalibrationWindow,
        distance_tracker: FaceDistanceTracker,
        stage: CalibrationStage,
        stage_index: int,
        point_index: int,
        target_xy: tuple[int, int],
    ) -> CalibrationSample | None:
        started_at = time.monotonic()
        raw_samples: list[np.ndarray] = []
        distance_samples: list[float] = []
        last_distance: float | None = None
        last_status = "Hold your gaze on the blue dot."

        total_seconds = self.settle_seconds + self.sample_seconds
        while ui.running:
            elapsed = time.monotonic() - started_at
            if elapsed >= total_seconds:
                break

            action = ui.poll_action()
            if action == "quit":
                raise KeyboardInterrupt("Calibration aborted.")

            ret, frame = cap.read()
            if not ret:
                last_status = "Camera frame unavailable."
                continue

            last_distance = distance_tracker.estimate(frame)
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.gaze_estimator.predict(rgb_frame, return_details=True)
                if not isinstance(result, RuntimeEstimationResult):
                    raise RuntimeError("Estimator did not return detailed output.")
                if elapsed >= self.settle_seconds and last_distance is not None:
                    raw_samples.append(result.raw_xy.astype(np.float32))
                    distance_samples.append(float(last_distance))
                    last_status = f"Collecting sample frames: {len(raw_samples)}"
                elif elapsed < self.settle_seconds:
                    last_status = "Settling. Keep looking at the target."
                else:
                    last_status = "Face distance unavailable."
            except (RuntimeError, ValueError) as exc:
                last_status = f"Gaze unavailable: {exc}"

            ui.draw(
                title=f"{stage.name} {stage_index}/{len(self.stages)} - point {point_index}/{len(self.target_points)}",
                lines=[
                    "Hold your gaze on the blue dot.",
                    _format_distance(last_distance),
                    last_status,
                ],
                target_xy=target_xy,
                progress=elapsed / total_seconds,
            )

        if len(raw_samples) < self.min_frames_per_point or not distance_samples:
            return None

        return CalibrationSample(
            stage=stage.name,
            point_index=point_index,
            target_xy=np.asarray(target_xy, dtype=np.float32),
            raw_xy=np.median(np.asarray(raw_samples, dtype=np.float32), axis=0).astype(
                np.float32
            ),
            face_distance_mm=float(np.median(distance_samples)),
            frame_count=len(raw_samples),
        )

    def _show_message(
        self, ui: CalibrationWindow, title: str, lines: Sequence[str]
    ) -> None:
        while ui.running:
            action = ui.poll_action()
            if action == "quit":
                raise KeyboardInterrupt("Calibration aborted.")
            if action == "continue":
                return
            ui.draw(title=title, lines=lines)


def build_gaze_estimator(
    screen_size: tuple[int, int],
    mediapipe_model_path: str | Path = DEFAULT_MEDIAPIPE_MODEL,
    iris_data_dir: str | Path | None = DEFAULT_IRIS_DATA_DIR,
    model_device: str = "auto",
    iris_device: str = "auto",
    weights_path: str | Path | None = None,
    activation_function: str = DEFAULT_ACTIVATION_FUNCTION,
) -> GazeEstimator:
    iris_detector = IrisDetector(
        mediapipe_model_path=_optional_path(Path(mediapipe_model_path)),
        iris_data_dir=_optional_path(Path(iris_data_dir))
        if iris_data_dir is not None
        else None,
        device=iris_device,
    )
    return GazeEstimator(
        IrisDetector=iris_detector,
        weights_path=weights_path,
        screen_size=screen_size,
        device=model_device,
        activation_function=activation_function,
    )


def load_target_points(
    screen: Screen, points_path: str | Path | None = DEFAULT_POINTS_PATH
) -> list[tuple[int, int]]:
    points: list[tuple[float, float]] = []
    if points_path is not None and Path(points_path).exists():
        with Path(points_path).open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("x") and row.get("y"):
                    points.append((float(row["x"]), float(row["y"])))

    if not points:
        points = list(DEFAULT_RELATIVE_POINTS)

    return [_to_screen_point(point, screen) for point in points]


def _to_screen_point(point: tuple[float, float], screen: Screen) -> tuple[int, int]:
    x, y = point
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= screen.width_px
        y *= screen.height_px
    return (
        int(np.clip(round(x), 0, screen.width_px - 1)),
        int(np.clip(round(y), 0, screen.height_px - 1)),
    )


def _fit_affine(predicted_points: np.ndarray, screen_points: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_points, dtype=np.float32)
    target = np.asarray(screen_points, dtype=np.float32)
    if predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted_points must have shape Nx2.")
    if target.shape != predicted.shape:
        raise ValueError("screen_points must have the same shape as predicted_points.")
    if predicted.shape[0] < 3:
        raise ValueError(
            "At least three point pairs are required for affine calibration."
        )

    design = np.concatenate(
        [predicted, np.ones((predicted.shape[0], 1), dtype=np.float32)], axis=1
    )
    matrix_t, *_ = np.linalg.lstsq(design, target, rcond=None)
    return matrix_t.T.astype(np.float32)


def _clamp_xy(xy: np.ndarray, screen_size: tuple[int, int]) -> np.ndarray:
    width, height = screen_size
    return np.asarray(
        [
            np.clip(float(xy[0]), 0.0, float(width - 1)),
            np.clip(float(xy[1]), 0.0, float(height - 1)),
        ],
        dtype=np.float32,
    )


def _optional_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _format_distance(distance_mm: float | None) -> str:
    if distance_mm is None:
        return "Face distance: unavailable"
    return f"Face distance: {distance_mm:.0f} mm"


def load_calibration_parameters(
    path: str | Path = DEFAULT_CALIBRATION_PARAMETERS_PATH,
) -> CalibrationParameters:
    parameters_path = Path(path)
    if not parameters_path.exists():
        return CalibrationParameters()

    with parameters_path.open("rb") as file:
        payload = tomllib.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{parameters_path} must contain TOML settings.")

    payload = _flatten_calibration_parameters(payload)
    return CalibrationParameters(
        estimator_lib=_string_config(
            payload,
            "estimator_lib",
            aliases=("gaze_estimation_lib", "gaze_estimation_library", "lib"),
            default="GazeEstimation",
        ),
        activation_function=_string_config(
            payload,
            "activation_function",
            aliases=("activation",),
            default=DEFAULT_ACTIVATION_FUNCTION,
        ),
        screen_config=_path_config(
            payload,
            "screen_config",
            aliases=("screen_config_toml", "config", "config_file"),
            default=DEFAULT_CONFIG_PATH,
        ),
        calibration_file=_path_config(
            payload,
            "calibration_file",
            aliases=("output", "output_path", "calibration"),
            default=DEFAULT_CALIBRATION_PATH,
        ),
        points=_path_config(
            payload,
            "points",
            aliases=("calibration_points", "points_file", "points_csv"),
            default=DEFAULT_POINTS_PATH,
        ),
        weights=_optional_path_config(
            payload,
            "weights",
            aliases=("weights_path", "checkpoint"),
        ),
        camera=_int_config(payload, "camera", aliases=("camera_index",), default=0),
        fov_degrees=_float_config(payload, "fov_degrees", aliases=("fov",), default=60.0),
        settle_seconds=_float_config(payload, "settle_seconds", default=0.6),
        sample_seconds=_float_config(payload, "sample_seconds", default=1.2),
        min_frames=_int_config(payload, "min_frames", aliases=("min_frames_per_point",), default=4),
        device=_string_config(payload, "device", aliases=("model_device",), default="auto"),
        iris_device=_string_config(payload, "iris_device", default="auto"),
        mediapipe_model=_path_config(
            payload,
            "mediapipe_model",
            aliases=("mediapipe_model_path", "face_landmarker_model"),
            default=DEFAULT_MEDIAPIPE_MODEL,
        ),
        iris_data_dir=_optional_path_config(
            payload,
            "iris_data_dir",
            default=DEFAULT_IRIS_DATA_DIR,
        ),
    )


def _flatten_calibration_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    flattened = {
        key: value for key, value in payload.items() if not isinstance(value, dict)
    }
    for section in (
        "paths",
        "estimator",
        "model",
        "gaze_estimation",
        "calibration",
        "capture",
        "timing",
        "devices",
        "assets",
    ):
        value = payload.get(section)
        if isinstance(value, dict):
            flattened.update(value)
    return flattened


def _string_config(
    payload: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    default: str,
) -> str:
    value = _first_config_value(payload, (key, *aliases), default)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string.")
    value = value.strip()
    return default if value == "" else value


def _path_config(
    payload: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    default: Path,
) -> Path:
    value = _first_config_value(payload, (key, *aliases), str(default))
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a path string.")
    return _resolve_config_path(value)


def _optional_path_config(
    payload: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    default: Path | None = None,
) -> Path | None:
    value = _first_config_value(
        payload,
        (key, *aliases),
        None if default is None else str(default),
    )
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a path string.")
    value = value.strip()
    if value == "":
        return None
    return _resolve_config_path(value)


def _int_config(
    payload: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    default: int,
) -> int:
    value = _first_config_value(payload, (key, *aliases), default)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer.")
    return value


def _float_config(
    payload: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    default: float,
) -> float:
    value = _first_config_value(payload, (key, *aliases), default)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number.")
    return float(value)


def _first_config_value(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _resolve_config_path(value: str) -> Path:
    path = Path(value.strip()).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect distance-aware gaze calibration."
    )
    parser.add_argument(
        "--calibration-parameters",
        default=str(DEFAULT_CALIBRATION_PARAMETERS_PATH),
        help="Path to calibration_parameters.toml.",
    )
    parser.add_argument(
        "--run-parameters",
        help="Optional shared run_parameters.toml fallback.",
    )
    parser.add_argument(
        "--estimator-lib",
        help="Estimator library backend. Overrides calibration_parameters.toml.",
    )
    parser.add_argument(
        "--screen-config", help="Path to config.toml. Overrides calibration_parameters.toml."
    )
    parser.add_argument(
        "--output",
        help="Calibration JSON output path. Overrides calibration_parameters.toml.",
    )
    parser.add_argument(
        "--points",
        help="CSV containing x,y target points. Overrides calibration_parameters.toml.",
    )
    parser.add_argument("--camera", type=int, help="Webcam index.")
    parser.add_argument(
        "--fov-degrees",
        type=float,
        help="Approximate webcam horizontal FOV.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        help="Delay before sampling each target.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        help="Sampling duration per target.",
    )
    parser.add_argument(
        "--min-frames", type=int, help="Minimum valid frames per target."
    )
    parser.add_argument(
        "--device", help="Model device: auto, cpu, cuda, etc."
    )
    parser.add_argument(
        "--iris-device",
        help="Iris detector device: auto, cpu, cuda, mps, etc.",
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
            "Model activation variant to use for collecting calibration samples. "
            "Used by GazeEstimation and ignored by GazeCaptureEstimator."
        ),
    )
    parser.add_argument(
        "--mediapipe-model",
        help="Face landmarker model path.",
    )
    parser.add_argument(
        "--iris-data-dir",
        help="Iris data directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_parameters = load_calibration_parameters(args.calibration_parameters)
    run_parameters = (
        load_run_parameters(args.run_parameters) if args.run_parameters is not None else None
    )
    estimator_lib = normalize_estimator_lib(
        args.estimator_lib
        or calibration_parameters.estimator_lib
        or (run_parameters.estimator_lib if run_parameters is not None else "GazeEstimation")
    )
    screen_config = (
        Path(args.screen_config)
        if args.screen_config is not None
        else calibration_parameters.screen_config
    )
    output_path = (
        Path(args.output)
        if args.output is not None
        else calibration_parameters.calibration_file
    )
    points_path = (
        Path(args.points) if args.points is not None else calibration_parameters.points
    )
    weights_path = (
        args.weights
        or calibration_parameters.weights
        or (run_parameters.weights if run_parameters is not None else None)
    )
    activation_function = (
        args.activation
        or calibration_parameters.activation_function
        or (run_parameters.activation_function if run_parameters is not None else None)
        or DEFAULT_ACTIVATION_FUNCTION
    )
    mediapipe_model = (
        Path(args.mediapipe_model)
        if args.mediapipe_model is not None
        else calibration_parameters.mediapipe_model
    )
    iris_data_dir = (
        Path(args.iris_data_dir)
        if args.iris_data_dir is not None
        else calibration_parameters.iris_data_dir
    )

    screen = Screen(str(screen_config))
    pipeline = AdvancedCalibrationPipeline(
        screen=screen,
        output_path=output_path,
        points_path=points_path,
        camera_index=args.camera if args.camera is not None else calibration_parameters.camera,
        fov_degrees=(
            args.fov_degrees
            if args.fov_degrees is not None
            else calibration_parameters.fov_degrees
        ),
        settle_seconds=(
            args.settle_seconds
            if args.settle_seconds is not None
            else calibration_parameters.settle_seconds
        ),
        sample_seconds=(
            args.sample_seconds
            if args.sample_seconds is not None
            else calibration_parameters.sample_seconds
        ),
        min_frames_per_point=(
            args.min_frames
            if args.min_frames is not None
            else calibration_parameters.min_frames
        ),
        mediapipe_model_path=mediapipe_model,
        iris_data_dir=iris_data_dir,
        model_device=args.device or calibration_parameters.device,
        iris_device=args.iris_device or calibration_parameters.iris_device,
        weights_path=weights_path,
        activation_function=activation_function,
        estimator_lib=estimator_lib,
    )
    try:
        calibration = pipeline.run()
    except KeyboardInterrupt:
        print("Calibration aborted.")
        return

    print(f"Saved calibration to {output_path.resolve()}")
    for item in calibration.bins:
        print(f"{item.stage}: {item.distance_mm:.0f} mm, {item.sample_count} samples")


if __name__ == "__main__":
    main()
