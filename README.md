# ANN UI

Real-time gaze estimation UI built around the ANN estimator in `GazeEstimation`.
The project provides:

- Distance-aware calibration using three face-distance stages.
- Accuracy UI with target dots, error tracking, and run logging.
- Demo grid UI that highlights the block containing the calibrated gaze point.

## Requirements

- Python 3.11+
- Webcam
- Windows is the primary tested environment
- Project dependencies from `pyproject.toml`
- ANN weights at `GazeEstimation/weights/best.pt`
- MediaPipe face landmarker at `models/face_landmarker_v2_with_blendshapes.task`

Install dependencies:

```powershell
uv sync
```

Run commands with the project virtualenv:

```powershell
.\.venv\Scripts\python.exe <script>.py
```

## Configuration

Screen and camera offset settings live in:

```text
calibration/config.toml
```

Important fields:

- `screen.res_width` / `screen.res_height`: active display resolution in pixels.
- `screen.phys_width_mm` / `screen.phys_height_mm`: physical display size in millimeters.
- `camera_offset`: camera position relative to the monitor center in millimeters.

Shared runtime settings for `main.py` and `demo.py` live in:

```text
Env/run_parameters.toml
```

Important fields:

- `estimator_lib`: `GazeEstimation` or `GazeCaptureEstimator`.
- `activation_function`: used by `GazeEstimation`; ignored by `GazeCaptureEstimator`.
- `screen_config`: path to the screen/camera TOML file.
- `calibration_file`: path to the distance-aware calibration JSON.
- `weights`: optional checkpoint override. `GazeCaptureEstimator` defaults to `GazeCaptureEstimator/checkpoint.pth.tar`.

Calibration-specific settings live in:

```text
calibration/calibration_parameters.toml
```

Important fields:

- `screen_config`: path to the screen/camera TOML file.
- `calibration_file`: output path for the calibration JSON.
- `points`: CSV containing calibration target points.
- `estimator_lib`: `GazeEstimation` or `GazeCaptureEstimator`.
- `activation_function`: used by `GazeEstimation`; ignored by `GazeCaptureEstimator`.
- `weights`: optional checkpoint override.

Calibration target points are declared by `calibration/calibration_parameters.toml`; the default is:

```text
csv_points/calibration-points.csv
```

The default set contains near-edge, corner, center, and intermediate points. With default calibration timing, each distance stage takes about 38 seconds.

## Calibration

Run distance-aware calibration:

```powershell
.\.venv\Scripts\python.exe -m calibration.calibration
```

The calibration flow has three stages:

- `near`: move closer to the camera.
- `normal`: move to your normal working distance.
- `far`: move farther from the camera.

For each stage, press `SPACE` when ready, then look at each pulsing blue target. Press `Q` or `ESC` to abort.

The output is written to:

```text
calibration/calibration.json
```

This file stores per-distance affine calibration matrices, raw samples, and estimator metadata. Runtime code reuses it automatically when present. If you switch `estimator_lib` between `GazeEstimation` and `GazeCaptureEstimator`, collect a new calibration for that backend.

Useful options:

```powershell
.\.venv\Scripts\python.exe -m calibration.calibration --calibration-parameters calibration/calibration_parameters.toml
.\.venv\Scripts\python.exe -m calibration.calibration --camera 0 --device auto --iris-device cpu
.\.venv\Scripts\python.exe -m calibration.calibration --settle-seconds 0.6 --sample-seconds 1.2
.\.venv\Scripts\python.exe -m calibration.calibration --output calibration/calibration.json
```

To use the GazeCapture backend for calibration, set this in `calibration/calibration_parameters.toml`:

```toml
estimator_lib = "GazeCaptureEstimator"
weights = "GazeCaptureEstimator/checkpoint.pth.tar"
```

Then point `Env/run_parameters.toml` at the same estimator/checkpoint/calibration file for `main.py` and `demo.py`.

## Accuracy Run

Run the main accuracy UI:

```powershell
.\.venv\Scripts\python.exe main.py
```

Behavior:

- Loads `calibration/calibration.json` when present.
- Estimates face distance using `face_pos`.
- Gets raw ANN gaze from `GazeEstimation.Estimator`.
- Applies distance-aware calibration.
- Shows the target and predicted gaze in the accuracy UI.
- Logs successful samples under `accuracy_runs/<timestamp>/`.

Logs include:

- `accuracy.json`
- Cropped face frames under `frames/`
- A copy of `calibration/calibration.json` when available
- Target/prediction coordinates and error metrics

## Demo Grid

Run the calibrated demo grid:

```powershell
.\.venv\Scripts\python.exe demo.py
```

The demo loads `calibration/calibration.json`, applies distance-aware gaze correction, and highlights the grid cell containing the predicted gaze point.

Useful options:

```powershell
.\.venv\Scripts\python.exe demo.py --block-num 4
.\.venv\Scripts\python.exe demo.py --calibration calibration/calibration.json
```

## Project Layout

```text
GazeEstimation/        ANN gaze estimator and model definition
IrisDetection/         Face/eye/iris preprocessing for the ANN
face_pos/              Face pose and distance estimation
gaze_ui/               Pygame UIs and accuracy logger
calibration/          Distance-aware calibration pipeline and local calibration files
main.py                Accuracy UI runtime
demo.py                Calibrated demo grid runtime
calibration/config.toml Screen and camera configuration
calibration/calibration_parameters.toml Calibration collection profile
Env/run_parameters.toml Shared runtime estimator/config/calibration selection
csv_points/           Calibration and accuracy target point sets
```

## Troubleshooting

If no predicted dot appears:

- Make sure your face is visible and well lit.
- Confirm `GazeEstimation/weights/best.pt` exists.
- Confirm the MediaPipe model exists under `models/`.
- Run calibration first so `calibration/calibration.json` exists.
- Watch the terminal for `Gaze estimation skipped` messages.

If calibration feels too slow or too fast:

- Per-point duration is `settle_seconds + sample_seconds`.
- Default is `0.6 + 1.2 = 1.8s` per point.
- With 21 points, one distance stage takes about `37.8s`.

If the UI opens on the wrong display or scale:

- Check `calibration/config.toml` resolution and physical dimensions.
- On Windows, DPI awareness is enabled by the UI helpers, but OS scaling can still affect perceived placement.

## Generated Files

These are runtime outputs and should not be committed:

```text
accuracy_runs/
calibration/calibration*.json
```
