from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENV_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENV_DIR.parent
DEFAULT_RUN_PARAMETERS_PATH = ENV_DIR / "run_parameters.toml"
DEFAULT_SCREEN_CONFIG_PATH = PROJECT_ROOT / "calibration" / "config.toml"
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "calibration" / "calibration.json"
DEFAULT_ESTIMATOR_LIB = "GazeEstimation"
DEFAULT_ACTIVATION_FUNCTION = "leaky_relu"

_ESTIMATOR_LIB_ALIASES = {
    "gazeestimation": "GazeEstimation",
    "gaze_estimation": "GazeEstimation",
    "gazeestimation.estimator": "GazeEstimation",
    "ann": "GazeEstimation",
    "gazecaptureestimator": "GazeCaptureEstimator",
    "gaze_capture_estimator": "GazeCaptureEstimator",
    "gaze_capture": "GazeCaptureEstimator",
    "gazecapture": "GazeCaptureEstimator",
    "gc": "GazeCaptureEstimator",
}


@dataclass(frozen=True, slots=True)
class RunParameters:
    estimator_lib: str = DEFAULT_ESTIMATOR_LIB
    activation_function: str = DEFAULT_ACTIVATION_FUNCTION
    screen_config: Path = DEFAULT_SCREEN_CONFIG_PATH
    calibration_file: Path = DEFAULT_CALIBRATION_PATH
    weights: Path | None = None


def load_run_parameters(path: str | Path = DEFAULT_RUN_PARAMETERS_PATH) -> RunParameters:
    parameters_path = Path(path)
    if not parameters_path.exists():
        return RunParameters()

    with parameters_path.open("rb") as file:
        payload = tomllib.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"{parameters_path} must contain TOML key/value settings.")

    base_dir = PROJECT_ROOT
    return RunParameters(
        estimator_lib=_string_value(
            payload,
            "estimator_lib",
            default=DEFAULT_ESTIMATOR_LIB,
        ),
        activation_function=_string_value(
            payload,
            "activation_function",
            aliases=("activation",),
            default=DEFAULT_ACTIVATION_FUNCTION,
        ),
        screen_config=_path_value(
            payload,
            "screen_config",
            aliases=("config", "config_file"),
            default=DEFAULT_SCREEN_CONFIG_PATH,
            base_dir=base_dir,
        ),
        calibration_file=_path_value(
            payload,
            "calibration_file",
            aliases=("calibration",),
            default=DEFAULT_CALIBRATION_PATH,
            base_dir=base_dir,
        ),
        weights=_optional_path_value(
            payload,
            "weights",
            aliases=("weights_path",),
            base_dir=base_dir,
        ),
    )


def normalize_estimator_lib(estimator_lib: str) -> str:
    normalized = estimator_lib.strip().replace("-", "_").replace(" ", "_")
    canonical = _ESTIMATOR_LIB_ALIASES.get(normalized.lower())
    if canonical is None:
        supported = ", ".join(sorted(set(_ESTIMATOR_LIB_ALIASES.values())))
        raise ValueError(
            f"Unsupported estimator_lib={estimator_lib!r}. "
            f"Supported runtime estimator libraries: {supported}."
        )
    return canonical


def _string_value(
    payload: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    default: str,
) -> str:
    value = _first_present(payload, (key, *aliases), default)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string.")
    value = value.strip()
    return default if value == "" else value


def _path_value(
    payload: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    default: Path,
    base_dir: Path,
) -> Path:
    value = _first_present(payload, (key, *aliases), str(default))
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a path string.")
    return _resolve_path(value, base_dir=base_dir)


def _optional_path_value(
    payload: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    base_dir: Path,
) -> Path | None:
    value = _first_present(payload, (key, *aliases), None)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a path string.")
    value = value.strip()
    if value == "":
        return None
    return _resolve_path(value, base_dir=base_dir)


def _first_present(
    payload: dict[str, Any], keys: tuple[str, ...], default: Any
) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value.strip()).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path
