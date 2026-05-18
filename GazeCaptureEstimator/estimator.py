from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .model import ITrackerModel
from .preprocessing import (
    DEFAULT_GRID_SIZE,
    DEFAULT_IMAGE_SIZE,
    build_transforms,
    crop_with_bbox,
    face_grid_from_bbox,
    face_grid_to_tensor,
    image_to_tensor,
    load_mean_images,
)


class GC_Estimator:
    """Load a GazeCapture iTracker checkpoint and run gaze prediction.

    Parameters
    ----------
    path_to_checkpoint:
        Path to a ``.pth``, ``.pt``, or ``.pth.tar`` checkpoint. Both raw
        state dicts and trainer dictionaries containing ``state_dict`` are
        supported.
    device:
        ``"auto"``, ``"cpu"``, or a CUDA device string such as ``"cuda:0"``.
    mean_dir:
        Directory containing ``mean_face_224.mat``, ``mean_left_224.mat``, and
        ``mean_right_224.mat``. Defaults to the package's bundled mean images.
    input_color:
        Color order for NumPy image inputs passed to ``predict``. Use ``"BGR"``
        for OpenCV crops and ``"RGB"`` for PIL or standard image arrays.
    """

    def __init__(
        self,
        path_to_checkpoint,
        *,
        device="auto",
        mean_dir=None,
        image_size=DEFAULT_IMAGE_SIZE,
        grid_size=DEFAULT_GRID_SIZE,
        input_color="RGB",
        data_parallel=None,
        strict=True,
    ):
        self.image_size = tuple(image_size)
        if self.image_size != DEFAULT_IMAGE_SIZE:
            raise ValueError(
                "This checkpoint architecture expects 224x224 crops; "
                f"got image_size={self.image_size}."
            )
        self.grid_size = int(grid_size)
        self.input_color = input_color
        self.device = self._resolve_device(device)

        face_mean, eye_left_mean, eye_right_mean = load_mean_images(mean_dir)
        self.face_transform, self.eye_left_transform, self.eye_right_transform = (
            build_transforms(
                face_mean,
                eye_left_mean,
                eye_right_mean,
                image_size=self.image_size,
            )
        )

        self.model = self._build_model(data_parallel=data_parallel)
        self.load_checkpoint(path_to_checkpoint, strict=strict)
        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            cudnn.benchmark = True

    def _resolve_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(device)

    def _build_model(self, data_parallel=None):
        model = ITrackerModel(grid_size=self.grid_size)
        if data_parallel is None:
            data_parallel = self.device.type == "cuda" and torch.cuda.device_count() > 1
        if data_parallel:
            model = torch.nn.DataParallel(model)
        return model

    @staticmethod
    def _state_dict_from_checkpoint(checkpoint):
        if isinstance(checkpoint, Mapping):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, Mapping):
                    return value
        return checkpoint

    @staticmethod
    def _match_data_parallel_prefix(state_dict, model):
        model_state = model.state_dict()
        model_has_module = any(k.startswith("module.") for k in model_state)
        checkpoint_has_module = any(k.startswith("module.") for k in state_dict)

        if model_has_module and not checkpoint_has_module:
            return {f"module.{k}": v for k, v in state_dict.items()}
        if checkpoint_has_module and not model_has_module:
            return {
                k.replace("module.", "", 1) if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }
        return state_dict

    def load_checkpoint(self, path_to_checkpoint, *, strict=True):
        checkpoint = torch.load(path_to_checkpoint, map_location=self.device)
        state_dict = self._state_dict_from_checkpoint(checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError("Checkpoint must be a state dict or contain a state dict.")
        state_dict = self._match_data_parallel_prefix(state_dict, self.model)
        self.model.load_state_dict(state_dict, strict=strict)

    def _split_eye_crops(self, eye_crops, left_eye_crop, right_eye_crop):
        if eye_crops is not None:
            if isinstance(eye_crops, Mapping):
                if "left" in eye_crops:
                    left_eye_crop = eye_crops["left"]
                elif "left_eye" in eye_crops:
                    left_eye_crop = eye_crops["left_eye"]

                if "right" in eye_crops:
                    right_eye_crop = eye_crops["right"]
                elif "right_eye" in eye_crops:
                    right_eye_crop = eye_crops["right_eye"]
            elif isinstance(eye_crops, Sequence) and len(eye_crops) == 2:
                left_eye_crop, right_eye_crop = eye_crops
            else:
                raise ValueError(
                    "eye_crops must be a (left_eye, right_eye) sequence or mapping."
                )

        if left_eye_crop is None or right_eye_crop is None:
            raise ValueError("Provide both left and right eye crops.")
        return left_eye_crop, right_eye_crop

    def _prepare_inputs(
        self,
        face_crop,
        eye_crops=None,
        face_grid=None,
        *,
        left_eye_crop=None,
        right_eye_crop=None,
        input_color=None,
    ):
        if face_grid is None:
            raise ValueError("face_grid is required.")

        input_color = input_color or self.input_color
        left_eye_crop, right_eye_crop = self._split_eye_crops(
            eye_crops, left_eye_crop, right_eye_crop
        )

        face_tensor = image_to_tensor(
            face_crop, self.face_transform, self.image_size, color_order=input_color
        )
        left_tensor = image_to_tensor(
            left_eye_crop,
            self.eye_left_transform,
            self.image_size,
            color_order=input_color,
        )
        right_tensor = image_to_tensor(
            right_eye_crop,
            self.eye_right_transform,
            self.image_size,
            color_order=input_color,
        )
        grid_tensor = face_grid_to_tensor(face_grid, grid_size=self.grid_size)

        batch_sizes = {
            face_tensor.shape[0],
            left_tensor.shape[0],
            right_tensor.shape[0],
            grid_tensor.shape[0],
        }
        if len(batch_sizes) != 1:
            raise ValueError(
                "face_crop, eye crops, and face_grid must have the same batch size."
            )

        return (
            face_tensor.to(self.device, non_blocking=True),
            left_tensor.to(self.device, non_blocking=True),
            right_tensor.to(self.device, non_blocking=True),
            grid_tensor.to(self.device, non_blocking=True),
        )

    @torch.inference_mode()
    def predict(
        self,
        face_crop,
        eye_crops=None,
        face_grid=None,
        *,
        left_eye_crop=None,
        right_eye_crop=None,
        input_color=None,
        as_numpy=False,
    ):
        """Predict gaze from prepared crops and face grid.

        For a single sample, this returns ``(x, y)`` floats. For a batched input,
        it returns a ``B x 2`` NumPy array unless ``as_numpy=False`` and the
        output is a single sample.
        """

        inputs = self._prepare_inputs(
            face_crop,
            eye_crops=eye_crops,
            face_grid=face_grid,
            left_eye_crop=left_eye_crop,
            right_eye_crop=right_eye_crop,
            input_color=input_color,
        )
        output = self.model(*inputs).detach().cpu()
        values = output.tolist()

        if output.shape[0] == 1 and not as_numpy:
            x, y = values[0]
            return float(x), float(y)
        return np.asarray(values, dtype=np.float32)

    @torch.inference_mode()
    def predict_from_bboxes(
        self,
        frame,
        face_box,
        left_eye_box,
        right_eye_box,
        *,
        frame_color="BGR",
        as_numpy=False,
    ):
        """Predict from a full frame and ``(x, y, width, height)`` boxes."""

        face_crop = crop_with_bbox(frame, face_box, color_order=frame_color)
        left_crop = crop_with_bbox(frame, left_eye_box, color_order=frame_color)
        right_crop = crop_with_bbox(frame, right_eye_box, color_order=frame_color)
        if face_crop is None or left_crop is None or right_crop is None:
            return None

        face_grid = face_grid_from_bbox(face_box, frame.shape, grid_size=self.grid_size)
        return self.predict(
            face_crop,
            (left_crop, right_crop),
            face_grid,
            input_color="RGB",
            as_numpy=as_numpy,
        )
