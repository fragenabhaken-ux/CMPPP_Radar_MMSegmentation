# PPP EDA velocity-stage extension

This package extends the existing `rangeGateLength` and `dopplerRange` EDA with
the complete radial-velocity target-construction chain.

## Quantities exported

Frame-level values in `frames.csv`:

```text
ego_longitudinal_velocity
ego_yaw_rate
doppler_range
velocity_wrap_count
velocity_wrap_fraction
velocity_mean_abs_wrap_index
velocity_max_abs_wrap_index
```

Point-level values in `point_values.npz`:

```text
velocity_measured_relative
velocity_expected_static
velocity_compensated_pre_wrap
velocity                         # final wrapped compensated training target
velocity_compensated_wrapped     # explicit alias of velocity
doppler_wrap_index
```

The construction is

```python
velocity_compensated_pre_wrap = (
    velocity_measured_relative - velocity_expected_static
)
doppler_wrap_index = floor(
    velocity_compensated_pre_wrap / doppler_range + 0.5
)
velocity = (
    velocity_compensated_pre_wrap
    - doppler_range * doppler_wrap_index
)
```

`ego_longitudinal_velocity` is the longitudinal vehicle-dynamics signal from
SYS100. It is deliberately not labelled generically as an IMU velocity.

## Required EDA config

The supplied `ppp_eda.py` already contains:

```python
dict(
    type='BuildPPPHeatmap',
    thres=0.05,
    rel_x_centre=0.491,
    rel_y_centre=0.4370,
    return_auxiliary_attributes=True,
)
```

The argument defaults to `False`, so normal training pipelines are unchanged.
The supplied spatial domain uses `170 m` for Cartesian forward range and polar
range.

## New EDA plots

Single-split summary plots include:

```text
ego_longitudinal_velocity_hist.png
ego_yaw_rate_hist.png
velocity_wrap_fraction_hist.png
wrap_fraction_vs_ego_velocity.png
velocity_measured_relative_hist.png
velocity_expected_static_hist.png
velocity_compensated_pre_wrap_hist.png
doppler_wrap_index_bar.png
velocity_measured_vs_expected_static.png
velocity_pre_wrap_vs_wrapped.png
velocity_expected_static_vs_azimuth.png
velocity_pre_wrap_over_doppler_range.png
velocity_wrapped_over_doppler_range.png
```

The three velocity-stage histograms also receive log-y variants. Train–validation
comparison creates corresponding comparison histograms, a wrap-index comparison,
and a normalized wrapped-velocity comparison.

Intermediate velocity stages, ego speed, yaw rate, `rangeGateLength`, and
`dopplerRange` use one fixed display interval derived from the complete plotted
sample's empirical 1% and 99% quantiles. The final wrapped velocity retains the
established fixed display interval `[-7, 7] m/s`. Display clipping never removes
values from summary statistics.

## New selected-example categories and overlays

```text
high_ego_speed
high_abs_yaw_rate
high_wrap_fraction
high_mean_abs_expected_static_velocity
```

Every selected example now additionally renders:

```text
*_velocity_measured_relative.png
*_velocity_expected_static.png
*_velocity_compensated_pre_wrap.png
```

Their colour scales are fixed across all selected examples using the scan-wide
1% and 99% quantiles from `point_values.npz`.

## Installation

Copy each file into the matching repository path. See `INSTALLATION.md` and
`MANIFEST.txt`.

Existing scans are incompatible. Rerun train and validation scanning before
selection, rendering, summaries, or split comparison.
