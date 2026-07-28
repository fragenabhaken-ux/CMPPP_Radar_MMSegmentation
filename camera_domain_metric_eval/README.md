# Camera-domain marked-PPP metric evaluation

This codebase evaluates four-channel image-space PPP models from HDF5 exports.
It intentionally contains **no RadarGen point-cloud metrics**. Predictions stay
in the camera output domain.

## Metrics

- Expected-count bias, MAE, RMSE, and count ratio
- Optimally Gaussianized spatial PPP log-likelihood score
- Depth MAE, RMSE, bias, and absolute-relative error
- Circular radial-velocity MAE, RMSE, and bias
- RCS MAE, RMSE, and bias
- Frame-weighted and point-weighted global mark summaries

Higher is better only for `likelihood_optimal_score`. Error metrics are better
when lower, and signed biases are better when closer to zero.

## Export hook

Copy:

```text
mmseg_hooks/ppp_finals_CamDom_metric_hook.py
```

into:

```text
mmseg/engine/hooks/ppp_finals_CamDom_metric_hook.py
```

The recommended hook block is shown in:

```text
config_snippets/camera_metric_export_config_snippet.py
```

The hook exports `pred_lambda` as expected-count mass per full-image cell:

```text
exp(log_lambda) / (H0 * W0)
```

The updated hook checks that prediction and GT grids have the full image shape.
This prevents a cropped or padded output from being silently evaluated with the
full-image cell measure.

The uploaded model config must also load the intended checkpoint. It currently
does not contain a `load_from` entry, so provide one in the config or through:

```bash
python tools/test.py CONFIG.py CHECKPOINT.pth --launcher slurm
```

or the equivalent command used by your test script.

## Configure model comparison

Edit `metric_eval_config.py`:

```python
H5_DIRS = {
    "UNet": "/path/to/unet/metric_eval_data/test",
    "UPerNet": "/path/to/upernet/metric_eval_data/test",
    "Transformer": "/path/to/transformer/metric_eval_data/test",
}
```

All exports must contain the same physical frames. The evaluator verifies this
using `video_name` and `video_frame_idx` and raises an error when model sample
sets differ.

For signed velocity, the evaluator uses the full Doppler period stored as
`doppler_period` in each sample. For a period of 20 m/s, predictions 9 m/s and
GT -9 m/s differ by 2 m/s. Missing periods are treated as an error by default.

## Run

```bash
cd camera_domain_metric_eval
python run_metric_eval.py metric_eval_config.py
```

Outputs:

```text
camera_metric_results/sample_metrics.csv
camera_metric_results/model_comparison_summary.csv
camera_metric_results/sample_manifest.csv
```

The main thesis-level columns are:

```text
count_rmse                              lower is better
likelihood_optimal_score_mean          higher is better
depth_mae_global                       lower is better
velocity_mae_global                    lower is better
rcs_mae_global                         lower is better
```

## Plotting

```python
import pandas as pd
from ppp_plotting import plot_expected_count_scatter, plot_metric_boxplots

samples = pd.read_csv("camera_metric_results/sample_metrics.csv")
plot_expected_count_scatter(samples, output_dir="camera_metric_results/plots")
plot_metric_boxplots(
    samples,
    [
        "likelihood_optimal_score",
        "depth_mae",
        "velocity_mae",
        "rcs_mae",
    ],
    output_dir="camera_metric_results/plots",
)
```

## Validate an export before full evaluation

```bash
python check_camera_export.py /path/to/metric_eval_data/test
```

This reports sample count, duplicate identifiers, grid shapes, expected-count
range, and GT-count range.

## Tests

```bash
pytest -q
```
