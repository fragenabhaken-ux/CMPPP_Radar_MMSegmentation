# Cartesian-BEV marked-PPP evaluation

This is a new offline evaluation codebase for the Cartesian-BEV HDF5 exports
produced by the three metric hooks:

- dense polar-grid models,
- camera-domain models projected to BEV with predicted depth,
- polar LIIF encoder-decoder models with continuous GT points.

The evaluator assumes a shared fixed Cartesian raster with array convention
`array[row=x_forward, column=y_lateral]` and reads the geometry from HDF5
attributes.

## Metrics

### Expected count

`pred_lambda` is expected-count mass per Cartesian cell. The evaluator reports
expected count, GT count, signed error, absolute error, squared error, ratio,
and model-level count RMSE.

### Optimally Gaussianized Cartesian PPP log-likelihood

The prediction mass is blurred with a circular Gaussian in **metres**. Because
`dx` and `dy` can differ, the implementation passes
`(sigma_m / dx, sigma_m / dy)` to the raster Gaussian filter. Total mass is
renormalized after filtering.

By default, the score uses physical Cartesian intensity:

```text
lambda_density[cell] = pred_lambda[cell] / (dx * dy)
score = (sum_cell gt_support[cell] * log(lambda_density[cell])
         - sum_cell pred_lambda[cell]) / gt_count
```

The positive sigma interval is optimized with log-space golden-section search,
and sigma zero is checked explicitly.

### Velocity and RCS marks

When exact GT arrays are available, every continuous GT point is mapped to its
Cartesian cell and compared with the predicted mark in that cell. Velocity uses
the frame-specific circular Doppler distance by default. RCS uses ordinary
signed error. MAE, RMSE, bias, and point counts are reported.

### RadarGen point-cloud metrics

RadarGen evaluates explicit point clouds, while a PPP predicts a distribution.
For every configured Monte Carlo draw, this code uses the same two sampling
procedures as `PPPMarkVisHookLight`, adapted to the Cartesian BEV mass grid:

- **thinning**: draw a homogeneous proposal PPP on the continuous raster domain
  at the maximum cell mass and accept each proposal with the local mass ratio;
- **block**: activate each block with probability `1 - exp(-block_mass)` and
  place one point in every active block, weighted by cell mass by default.

The resulting continuous grid coordinates are converted affinely to metric BEV
coordinates. Raster row is forward `x`; raster column is lateral `y`. Velocity
and RCS are taken from the Cartesian cell containing each sampled point. A fixed
seed combined with the stable sample ID, sampling method, and draw index makes
results reproducible and uses common random numbers across models. Metrics are
reported separately for thinning and block sampling rather than averaged across
sampling methods.

Implemented whole-frame metrics:

- location Chamfer distance,
- normalized full-vector Chamfer distance,
- point-cloud IoU, precision, recall, and F1 at 1 m,
- distance-attribute Hungarian precision, recall, and F1,
- biased squared RBF MMD for location, RCS, and velocity.

The distance-attribute velocity threshold uses circular Doppler distance, as in
the supplied thesis description. Velocity MMD remains ordinary Euclidean MMD,
matching the RadarGen definition.

The IoU implementation follows the explicit Euclidean-distance formula in the
supplied Overleaf text. It does not reproduce the extra square-root operation
present in the current RadarGen helper after nearest-neighbour distances have
already been returned.

Per-box density, hit/miss, and invariant class metrics are not implemented,
because the HDF5 exports contain no object boxes or class labels.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Configuration

Edit `bev_metric_eval_config.py` and set `H5_DIRS` to the test export directory
for every model. Keep the expected BEV geometry identical to the export hooks.

The default alignment mode is `strict`: all model runs must contain exactly the
same stable `(video_name, video_frame_idx)` sample IDs.

## Run

```bash
python run_bev_metric_eval.py bev_metric_eval_config.py
```

Or from a notebook:

```python
from bev_eval import run_from_config

df_samples, df_summary = run_from_config("bev_metric_eval_config.py")
```

Outputs:

```text
sample_metrics.csv
model_summary.csv
sample_manifest.csv
config_snapshot.json
```

## Important configuration decisions

- `likelihood.sigma_max_m`: maximum allowed spatial tolerance in metres.
- `radargen.num_draws`: Monte Carlo point-cloud draws per frame.
- `radargen.sampling_methods`: any ordered subset of `thinning` and `block`.
- `radargen.block_size`: block side length in Cartesian raster cells.
- `radargen.block_placement`: `weighted` or `uniform` inside active blocks.
- RCS and velocity bounds for normalized full-vector Chamfer.
- strict versus intersection sample alignment.

These values must remain fixed across all compared models.
