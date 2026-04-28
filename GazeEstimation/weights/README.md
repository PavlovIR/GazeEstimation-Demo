Place a trained ANN checkpoint named `best.pt` in this folder to make
`Estimator()` load it by default.

This package also searches `outputs/train/fold_p00/best.pt` when used from
the original repository root. Pass `weights_path=...` to use any other
checkpoint explicitly.
