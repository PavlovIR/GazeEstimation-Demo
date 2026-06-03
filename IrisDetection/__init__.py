from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
from PIL import Image, ImageOps
from Env.runtime_device import resolve_iris_device_name


MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

EYE_REGION_A = np.asarray([33, 133, 160, 159, 158, 157, 173, 246, 161, 144, 145, 153, 154, 155])
EYE_REGION_B = np.asarray([263, 362, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382])
LEFT_IRIS_REGION = np.asarray([468, 469, 470, 471, 472])
RIGHT_IRIS_REGION = np.asarray([473, 474, 475, 476, 477])


@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    def clamp(self, image_width: int, image_height: int) -> "BoundingBox":
        x1 = min(max(self.x, 0.0), float(image_width - 1))
        y1 = min(max(self.y, 0.0), float(image_height - 1))
        x2 = min(max(self.x2, x1 + 1.0), float(image_width))
        y2 = min(max(self.y2, y1 + 1.0), float(image_height))
        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def to_int_xyxy(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.x2)),
            int(round(self.y2)),
        )


@dataclass(slots=True)
class FaceLandmarksResult:
    landmarks: np.ndarray
    image_width: int
    image_height: int

    @property
    def eye_regions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.landmarks[EYE_REGION_A], self.landmarks[EYE_REGION_B]


@dataclass(slots=True)
class DetectionResult:
    face: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    face_grid: np.ndarray
    left_iris_grid: np.ndarray
    right_iris_grid: np.ndarray
    landmarks: np.ndarray
    face_box: BoundingBox
    left_eye_box: BoundingBox
    right_eye_box: BoundingBox
    left_iris_box_eye: BoundingBox
    right_iris_box_eye: BoundingBox
    left_iris_box_image: BoundingBox
    right_iris_box_image: BoundingBox


def load_rgb_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _image_size_hw(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("Image size must be an int or a two-item [height, width] pair.")
    return int(size[0]), int(size[1])


def _resize_image(image: np.ndarray, size: int | tuple[int, int] | list[int]) -> np.ndarray:
    height, width = _image_size_hw(size)
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))


def _crop_image(image: np.ndarray, box: BoundingBox, size: int | tuple[int, int] | list[int]) -> np.ndarray:
    x1, y1, x2, y2 = box.to_int_xyxy()
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("Crop is empty.")
    return _resize_image(cropped, size)


def _flip_horizontal(image: np.ndarray) -> np.ndarray:
    return np.asarray(ImageOps.mirror(Image.fromarray(image)))


def _bounding_box_from_points(points: np.ndarray, margin: float, image_width: int, image_height: int) -> BoundingBox:
    min_x = float(np.min(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_x = float(np.max(points[:, 0]))
    max_y = float(np.max(points[:, 1]))
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    pad_x = width * margin
    pad_y = height * margin
    return BoundingBox(
        x=min_x - pad_x,
        y=min_y - pad_y,
        width=width + (2.0 * pad_x),
        height=height + (2.0 * pad_y),
    ).clamp(image_width=image_width, image_height=image_height)


def _box_from_landmarks(points: np.ndarray) -> BoundingBox:
    min_x = float(np.min(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_x = float(np.max(points[:, 0]))
    max_y = float(np.max(points[:, 1]))
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max(max_x - min_x, 1.0),
        height=max(max_y - min_y, 1.0),
    )


def _horizontal_flip_box(box: BoundingBox, canvas_width: int) -> BoundingBox:
    return BoundingBox(
        x=float(canvas_width) - (box.x + box.width),
        y=box.y,
        width=box.width,
        height=box.height,
    )


def _project_box_from_crop(box: BoundingBox, crop_box: BoundingBox, crop_size: int | tuple[int, int] | list[int]) -> BoundingBox:
    crop_height, crop_width = _image_size_hw(crop_size)
    scale_x = crop_box.width / max(crop_width, 1)
    scale_y = crop_box.height / max(crop_height, 1)
    return BoundingBox(
        x=crop_box.x + (box.x * scale_x),
        y=crop_box.y + (box.y * scale_y),
        width=box.width * scale_x,
        height=box.height * scale_y,
    )


def _project_points_to_crop(points: np.ndarray, crop_box: BoundingBox, crop_size: int | tuple[int, int] | list[int]) -> np.ndarray:
    crop_height, crop_width = _image_size_hw(crop_size)
    scale_x = crop_width / max(crop_box.width, 1.0)
    scale_y = crop_height / max(crop_box.height, 1.0)
    projected = np.empty_like(points, dtype=np.float32)
    projected[:, 0] = (points[:, 0] - crop_box.x) * scale_x
    projected[:, 1] = (points[:, 1] - crop_box.y) * scale_y
    if points.shape[1] > 2:
        projected[:, 2:] = points[:, 2:]
    projected[:, 0] = np.clip(projected[:, 0], 0.0, float(crop_width - 1))
    projected[:, 1] = np.clip(projected[:, 1], 0.0, float(crop_height - 1))
    return projected


def _rasterize_box_to_grid(
    box: BoundingBox,
    canvas_width: int,
    canvas_height: int,
    grid_width: int,
    grid_height: int,
) -> np.ndarray:
    grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    x_scale = grid_width / max(canvas_width, 1)
    y_scale = grid_height / max(canvas_height, 1)

    x1 = int(np.floor(box.x * x_scale))
    y1 = int(np.floor(box.y * y_scale))
    x2 = int(np.ceil((box.x + box.width) * x_scale))
    y2 = int(np.ceil((box.y + box.height) * y_scale))

    x1 = max(min(x1, grid_width - 1), 0)
    y1 = max(min(y1, grid_height - 1), 0)
    x2 = max(min(x2, grid_width), x1 + 1)
    y2 = max(min(y2, grid_height), y1 + 1)
    grid[y1:y2, x1:x2] = 1.0
    return grid


def _candidate_data_dirs(iris_data_dir: str | Path | None) -> Iterable[Path]:
    if iris_data_dir:
        yield Path(iris_data_dir)
    env_path = os.getenv("IRISLIBS_DATA_DIR")
    if env_path:
        yield Path(env_path)
    yield Path("IrisLibs") / "data"
    yield Path("data")


class Detector:
    def __init__(
        self,
        mediapipe_model_path: str | Path | None = None,
        iris_data_dir: str | Path | None = None,
        device: str = "auto",
        download_mediapipe_model: bool = True,
        face_crop_size: int | tuple[int, int] | list[int] = 224,
        eye_crop_size: int | tuple[int, int] | list[int] = 96,
        face_grid_size: tuple[int, int] = (50, 50),
        iris_grid_size: tuple[int, int] = (10, 5),
        face_box_margin: float = 0.15,
        eye_box_margin: float = 0.45,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mediapipe_model_path = Path(mediapipe_model_path) if mediapipe_model_path else Path("assets/mediapipe/face_landmarker.task")
        self.iris_data_dir = self._resolve_iris_data_dir(iris_data_dir)
        self.device = resolve_iris_device_name(device)
        self.download_mediapipe_model = download_mediapipe_model
        self.face_crop_size = face_crop_size
        self.eye_crop_size = eye_crop_size
        self.face_grid_size = face_grid_size
        self.iris_grid_size = iris_grid_size
        self.face_box_margin = face_box_margin
        self.eye_box_margin = eye_box_margin
        self.num_faces = num_faces
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._face_detector = None
        self._mp = None
        self._iris_model = None
        eye_height, eye_width = _image_size_hw(self.eye_crop_size)
        self._use_mediapipe_iris = eye_height != eye_width

        if self.iris_data_dir is not None:
            os.environ["IRISLIBS_DATA_DIR"] = str(self.iris_data_dir)

    @staticmethod
    def _resolve_iris_data_dir(iris_data_dir: str | Path | None) -> Path | None:
        for candidate in _candidate_data_dirs(iris_data_dir):
            if (candidate / "iris_landmark.pth").exists() and (candidate / "weights.pkl").exists():
                return candidate.resolve()
        return Path(iris_data_dir).resolve() if iris_data_dir else None

    def _ensure_mediapipe_model(self) -> Path:
        if self.mediapipe_model_path.exists():
            return self.mediapipe_model_path
        if not self.download_mediapipe_model:
            raise FileNotFoundError(f"Missing MediaPipe face landmarker asset at {self.mediapipe_model_path}.")

        self.mediapipe_model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(MEDIAPIPE_MODEL_URL, self.mediapipe_model_path)
        except URLError as exc:
            raise RuntimeError(
                "Unable to download the MediaPipe face landmarker model. "
                "Pass `mediapipe_model_path` or place the asset manually."
            ) from exc
        return self.mediapipe_model_path

    def _load_face_detector(self):
        if self._face_detector is not None:
            return self._face_detector

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for face landmark detection.") from exc

        model_path = self._ensure_mediapipe_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_faces=self.num_faces,
            min_face_detection_confidence=self.min_face_detection_confidence,
            min_face_presence_confidence=self.min_face_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_detector = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        return self._face_detector

    def _load_iris_model(self):
        if self._iris_model is not None:
            return self._iris_model

        try:
            from IrisLibs.iris import load_iris_model
        except ImportError as exc:
            raise RuntimeError("IrisLibs must be importable to use IrisDetection.Detector.") from exc

        load_kwargs: dict[str, object] = {"device": self.device}
        if self.iris_data_dir is not None:
            load_kwargs["ckpt_path"] = self.iris_data_dir / "iris_landmark.pth"
            load_kwargs["weights_path"] = self.iris_data_dir / "weights.pkl"
        self._iris_model = load_iris_model(**load_kwargs)
        return self._iris_model

    def detect_face(self, image: np.ndarray) -> FaceLandmarksResult | None:
        detector = self._load_face_detector()
        image_height, image_width = image.shape[:2]
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image),
        )
        result = detector.detect(mp_image)
        if not result.face_landmarks:
            return None

        points = [
            [landmark.x * image_width, landmark.y * image_height, landmark.z * image_width]
            for landmark in result.face_landmarks[0]
        ]
        return FaceLandmarksResult(
            landmarks=np.asarray(points, dtype=np.float32),
            image_width=image_width,
            image_height=image_height,
        )

    def predict_iris(self, eye_crop: np.ndarray, is_left: bool) -> np.ndarray:
        try:
            from IrisLibs.iris import predict_iris_landmarks
        except ImportError as exc:
            raise RuntimeError("IrisLibs must be importable to use IrisDetection.Detector.") from exc

        return predict_iris_landmarks(
            eye_crop,
            model=self._load_iris_model(),
            is_left=is_left,
        )

    def _predict_iris_or_use_mediapipe(
        self,
        eye_crop: np.ndarray,
        eye_box: BoundingBox,
        landmarks_result: FaceLandmarksResult,
        is_left: bool,
    ) -> np.ndarray:
        if not self._use_mediapipe_iris:
            try:
                return self.predict_iris(eye_crop, is_left=is_left)
            except RuntimeError:
                self._use_mediapipe_iris = True

        iris_indices = LEFT_IRIS_REGION if is_left else RIGHT_IRIS_REGION
        if landmarks_result.landmarks.shape[0] <= int(np.max(iris_indices)):
            try:
                return self.predict_iris(eye_crop, is_left=is_left)
            except RuntimeError:
                raise
        return _project_points_to_crop(
            landmarks_result.landmarks[iris_indices],
            crop_box=eye_box,
            crop_size=self.eye_crop_size,
        )

    @staticmethod
    def _split_eye_regions(landmarks_result: FaceLandmarksResult) -> tuple[np.ndarray, np.ndarray]:
        eye_region_a, eye_region_b = landmarks_result.eye_regions
        center_a_x = float(np.mean(eye_region_a[:, 0]))
        center_b_x = float(np.mean(eye_region_b[:, 0]))
        subject_right_eye = eye_region_a if center_a_x < center_b_x else eye_region_b
        subject_left_eye = eye_region_b if center_a_x < center_b_x else eye_region_a
        return subject_left_eye, subject_right_eye

    def detect(self, image: str | Path | np.ndarray) -> DetectionResult:
        if not isinstance(image, np.ndarray):
            image = load_rgb_image(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape HxWx3.")

        detection = self.detect_face(image)
        if detection is None:
            raise RuntimeError("No face landmarks detected in the input image.")

        image_height, image_width = image.shape[:2]
        left_eye_points, right_eye_points = self._split_eye_regions(detection)

        face_box = _bounding_box_from_points(
            detection.landmarks[:, :2],
            margin=self.face_box_margin,
            image_width=image_width,
            image_height=image_height,
        )
        left_eye_box = _bounding_box_from_points(
            left_eye_points[:, :2],
            margin=self.eye_box_margin,
            image_width=image_width,
            image_height=image_height,
        )
        right_eye_box = _bounding_box_from_points(
            right_eye_points[:, :2],
            margin=self.eye_box_margin,
            image_width=image_width,
            image_height=image_height,
        )

        face_crop = _crop_image(image, face_box, size=self.face_crop_size)
        left_eye_crop = _crop_image(image, left_eye_box, size=self.eye_crop_size)
        right_eye_crop = _crop_image(image, right_eye_box, size=self.eye_crop_size)

        left_iris_landmarks = self._predict_iris_or_use_mediapipe(
            left_eye_crop,
            eye_box=left_eye_box,
            landmarks_result=detection,
            is_left=True,
        )
        right_iris_landmarks = self._predict_iris_or_use_mediapipe(
            right_eye_crop,
            eye_box=right_eye_box,
            landmarks_result=detection,
            is_left=False,
        )

        left_iris_box_eye = _box_from_landmarks(left_iris_landmarks[:, :2])
        right_iris_box_eye = _box_from_landmarks(right_iris_landmarks[:, :2])
        left_iris_box_image = _project_box_from_crop(left_iris_box_eye, crop_box=left_eye_box, crop_size=self.eye_crop_size)
        right_iris_box_image = _project_box_from_crop(right_iris_box_eye, crop_box=right_eye_box, crop_size=self.eye_crop_size)

        eye_crop_height, eye_crop_width = _image_size_hw(self.eye_crop_size)
        canonical_right_eye = _flip_horizontal(right_eye_crop)
        canonical_right_iris_box = _horizontal_flip_box(right_iris_box_eye, canvas_width=eye_crop_width)

        face_grid = _rasterize_box_to_grid(
            face_box,
            canvas_width=image_width,
            canvas_height=image_height,
            grid_width=self.face_grid_size[0],
            grid_height=self.face_grid_size[1],
        )
        left_iris_grid = _rasterize_box_to_grid(
            left_iris_box_eye,
            canvas_width=eye_crop_width,
            canvas_height=eye_crop_height,
            grid_width=self.iris_grid_size[0],
            grid_height=self.iris_grid_size[1],
        )
        right_iris_grid = _rasterize_box_to_grid(
            canonical_right_iris_box,
            canvas_width=eye_crop_width,
            canvas_height=eye_crop_height,
            grid_width=self.iris_grid_size[0],
            grid_height=self.iris_grid_size[1],
        )

        return DetectionResult(
            face=face_crop,
            left_eye=left_eye_crop,
            right_eye=canonical_right_eye,
            face_grid=face_grid,
            left_iris_grid=left_iris_grid,
            right_iris_grid=right_iris_grid,
            landmarks=detection.landmarks,
            face_box=face_box,
            left_eye_box=left_eye_box,
            right_eye_box=right_eye_box,
            left_iris_box_eye=left_iris_box_eye,
            right_iris_box_eye=canonical_right_iris_box,
            left_iris_box_image=left_iris_box_image,
            right_iris_box_image=right_iris_box_image,
        )


__all__ = [
    "BoundingBox",
    "DetectionResult",
    "Detector",
    "FaceLandmarksResult",
    "load_rgb_image",
]
