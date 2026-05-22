_EXPORTS = {
    "DEFAULT_ACTIVATION_FUNCTION",
    "DEFAULT_CALIBRATION_PATH",
    "DEFAULT_CALIBRATION_PARAMETERS_PATH",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_IRIS_DATA_DIR",
    "DEFAULT_MEDIAPIPE_MODEL",
    "DEFAULT_POINTS_PATH",
    "AdvancedCalibrationPipeline",
    "CalibrationParameters",
    "DistanceAwareCalibration",
    "FaceDistanceTracker",
    "build_gaze_estimator",
    "load_calibration_parameters",
    "load_target_points",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import calibration as calibration_module

    value = getattr(calibration_module, name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
