Place a trained ANN checkpoint named `best.pt` in this folder to make
`Estimator()` load the regular ReLU model by default.

Place a Leaky ReLU visual encoder checkpoint at `leaky_relu/best.pt` to make
`Estimator(activation_function="leaky_relu")` load it automatically.

This package also searches `outputs/train/fold_p00/best.pt` when used from
the original repository root for ReLU, and
`outputs/activation_comparison/leaky_relu/fold_p00/best.pt` for Leaky ReLU.
Pass `weights_path=...` to use any other checkpoint explicitly.
