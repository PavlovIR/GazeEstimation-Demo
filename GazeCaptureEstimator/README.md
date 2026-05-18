# GazeCaptureEstimator

Small inference package for the GazeCapture iTracker neural network.

```python
from GazeCaptureEstimator import GC_Estimator

estimator = GC_Estimator("pytorch/checkpoint.pth.tar", device="auto")

x, y = estimator.predict(
    face_crop,
    (left_eye_crop, right_eye_crop),
    face_grid,
)
```

`face_crop`, `left_eye_crop`, and `right_eye_crop` may be PIL images, NumPy
arrays, torch tensors, or image paths. NumPy inputs are interpreted as RGB by
default; pass `input_color="BGR"` to `GC_Estimator(...)` or `predict(...)` for
OpenCV crops.

`face_grid` may be a flat 625-value vector, a `25 x 25` grid, or a batched
array/tensor. The prediction output is `(x_cm, y_cm)` in the same
camera-centimeter coordinate space used by GazeCapture labels.

For full-frame inference with boxes:

```python
from GazeCaptureEstimator import GC_Estimator, face_grid_from_bbox

estimator = GC_Estimator("pytorch/checkpoint.pth.tar")
gaze = estimator.predict_from_bboxes(
    frame,
    face_box=(x, y, w, h),
    left_eye_box=(lx, ly, lw, lh),
    right_eye_box=(rx, ry, rw, rh),
)
```

The package bundles the standard `mean_*_224.mat` files under
`GazeCaptureEstimator/mean_images`. To use custom mean files, pass
`mean_dir="/path/to/mean_images"` during initialization.
