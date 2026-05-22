from __future__ import annotations

import platform

import torch


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def resolve_torch_device_name(device: str | None = "auto") -> str:
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not _mps_is_available():
        return "cpu"
    return requested


def resolve_iris_device_name(device: str | None = "auto") -> str:
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if platform.system() == "Darwin" and _mps_is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not _mps_is_available():
        return "cpu"
    return requested
