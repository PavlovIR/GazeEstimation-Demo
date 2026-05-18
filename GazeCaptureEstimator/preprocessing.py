from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as torch_f
from PIL import Image


DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_GRID_SIZE = 25
_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


class SubtractMean:
    def __init__(self, mean_image):
        self.mean_tensor = mean_image_to_tensor(mean_image)

    def __call__(self, tensor):
        return tensor.sub(self.mean_tensor)


class ImageTransform:
    def __init__(self, mean_image, image_size=DEFAULT_IMAGE_SIZE):
        self.image_size = tuple(image_size)
        self.mean_tensor = mean_image_to_tensor(mean_image)

    def __call__(self, image):
        image = image.resize((self.image_size[1], self.image_size[0]), _BILINEAR)
        return pil_to_tensor(image).sub(self.mean_tensor)


def mean_image_to_tensor(mean_image):
    # Avoid torch.from_numpy so inference still works when PyTorch was built
    # against a different NumPy ABI than the runtime NumPy version.
    mean_chw = (mean_image.transpose((2, 0, 1)) / 255.0).tolist()
    return torch.tensor(mean_chw, dtype=torch.float32)


def pil_to_tensor(image):
    image = image.convert("RGB")
    width, height = image.size
    if hasattr(torch, "frombuffer"):
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).clone()
    else:
        storage = torch.ByteStorage.from_buffer(image.tobytes())
        tensor = torch.ByteTensor(storage)
    tensor = tensor.view(height, width, 3)
    return tensor.permute(2, 0, 1).float().div(255.0)


def package_mean_dir():
    return Path(__file__).resolve().parent / "mean_images"


def load_mean_images(mean_dir=None):
    mean_dir = Path(mean_dir) if mean_dir is not None else package_mean_dir()
    return (
        sio.loadmat(mean_dir / "mean_face_224.mat")["image_mean"],
        sio.loadmat(mean_dir / "mean_left_224.mat")["image_mean"],
        sio.loadmat(mean_dir / "mean_right_224.mat")["image_mean"],
    )


def build_transforms(face_mean, eye_left_mean, eye_right_mean, image_size=DEFAULT_IMAGE_SIZE):
    return (
        ImageTransform(face_mean, image_size=image_size),
        ImageTransform(eye_left_mean, image_size=image_size),
        ImageTransform(eye_right_mean, image_size=image_size),
    )


def _as_batched_chw_tensor(value, image_size, mean_tensor=None):
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError(
            "Image tensors must have shape CxHxW, BxCxHxW, HxWxC, or BxHxWxC."
        )

    if value.shape[1] not in (1, 3) and value.shape[-1] in (1, 3):
        value = value.permute(0, 3, 1, 2)
    if value.shape[1] != 3:
        raise ValueError("Image tensors must have exactly 3 channels.")

    value = value.float()
    if value.max().item() > 2:
        value = value / 255.0
    if tuple(value.shape[-2:]) != tuple(image_size):
        value = torch_f.interpolate(
            value,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
    if mean_tensor is not None:
        value = value - mean_tensor.to(value.device).unsqueeze(0)
    return value


def image_to_tensor(image, transform, image_size=DEFAULT_IMAGE_SIZE, color_order="RGB"):
    """Convert PIL, ndarray, tensor, or path image input to a BCHW float tensor."""

    if isinstance(image, torch.Tensor):
        mean_tensor = getattr(transform, "mean_tensor", None)
        return _as_batched_chw_tensor(image, image_size=image_size, mean_tensor=mean_tensor)

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("NumPy image inputs must have shape HxWx3 or HxWx4.")
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if color_order.upper() == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_order.upper() != "RGB":
            raise ValueError('color_order must be either "RGB" or "BGR".')
        if np.issubdtype(image.dtype, np.floating):
            scale = 255.0 if image.max() <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255)
        image = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError(
            "Image inputs must be PIL images, NumPy arrays, torch tensors, or file paths."
        )

    return transform(image).unsqueeze(0)


def face_grid_to_tensor(face_grid, grid_size=DEFAULT_GRID_SIZE):
    if isinstance(face_grid, torch.Tensor):
        tensor = face_grid.float()
    else:
        if isinstance(face_grid, np.ndarray):
            face_grid = face_grid.tolist()
        tensor = torch.tensor(face_grid, dtype=torch.float32)

    if tensor.ndim == 1:
        expected = grid_size * grid_size
        if tensor.numel() != expected:
            raise ValueError(f"Flat face_grid must have {expected} values.")
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        if tuple(tensor.shape) == (grid_size, grid_size):
            tensor = tensor.reshape(1, -1)
        elif tensor.shape[1] == grid_size * grid_size:
            pass
        else:
            raise ValueError(
                "2D face_grid must have shape grid_size x grid_size or B x grid_size^2."
            )
    elif tensor.ndim == 3:
        if tuple(tensor.shape[1:]) != (grid_size, grid_size):
            raise ValueError("3D face_grid must have shape B x grid_size x grid_size.")
        tensor = tensor.reshape(tensor.shape[0], -1)
    else:
        raise ValueError("face_grid must be a flat, 2D, or batched 3D array/tensor.")

    return tensor


def face_grid_from_bbox(face_box, frame_shape, grid_size=DEFAULT_GRID_SIZE):
    """Create the iTracker face grid from ``(x, y, width, height)`` pixels."""

    height, width = frame_shape[:2]
    x, y, w, h = face_box
    gx = int(np.clip(round(x / width * grid_size), 0, grid_size - 1))
    gy = int(np.clip(round(y / height * grid_size), 0, grid_size - 1))
    gw = max(1, int(np.clip(round(w / width * grid_size), 1, grid_size)))
    gh = max(1, int(np.clip(round(h / height * grid_size), 1, grid_size)))

    gx = min(gx, grid_size - gw)
    gy = min(gy, grid_size - gh)

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid[gy : gy + gh, gx : gx + gw] = 1.0
    return grid.flatten()


def crop_with_bbox(frame, bbox, color_order="BGR"):
    """Crop ``(x, y, width, height)`` from a frame, padding outside-frame areas."""

    x, y, w, h = [int(v) for v in bbox]
    image_height, image_width = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + w)
    y2 = min(image_height, y + h)

    if x1 >= x2 or y1 >= y2:
        return None

    patch = np.zeros((h, w, 3), dtype=frame.dtype)
    patch_y1 = y1 - y
    patch_x1 = x1 - x
    patch[patch_y1 : patch_y1 + (y2 - y1), patch_x1 : patch_x1 + (x2 - x1)] = frame[
        y1:y2, x1:x2
    ]

    if color_order.upper() == "BGR":
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    elif color_order.upper() != "RGB":
        raise ValueError('color_order must be either "RGB" or "BGR".')
    return patch
