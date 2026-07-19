# PPP EDA RCS + Spatial-Domain Extension

Copy the files into the matching repository paths.

```text
tools/ppp_eda/*
mmseg/datasets/transforms/sys100_ppp_transforms.py
configs/ppp_eda/ppp_eda_spatial_config_snippet.py
```

Then copy the `ppp_eda_spatial` dictionary from the snippet into the existing
`configs/ppp_eda/ppp_eda.py` file. The complete current EDA config was not part
of the supplied files, so it is intentionally not replaced by this package.

Run the scan again. Existing `point_values.npz` files do not contain the new
Cartesian and polar arrays.

```bash
sbatch ppp_eda_scan.sbatch train 5000
sbatch ppp_eda_scan.sbatch val 5000
```

Then rerun selection/rendering, summaries, and comparison.

## New point arrays

```text
lateral
forward
range
azimuth_deg
```

## New selected-example files

```text
*_cartesian_bev.png
*_polar.png
```

## New summary density files

```text
cartesian_bev_density.png
polar_density.png
```

## New train-validation density files

```text
cartesian_bev_density_train.png
cartesian_bev_density_val.png
cartesian_bev_density_difference_val_minus_train.png
polar_density_train.png
polar_density_val.png
polar_density_difference_val_minus_train.png
```

## `None` semantics

Every spatial bound can be `None`.

- Camera: `None` uses the complete corresponding image extent.
- Cartesian BEV and polar: `None` applies no point filter and auto-scales that plot edge from finite data.
- Invalid geometric back-projections remain `NaN` and are never plotted.


## Included full config

Use `configs/ppp_eda/ppp_eda.py`. Every camera, Cartesian BEV, and polar bound may be set to `None`.
