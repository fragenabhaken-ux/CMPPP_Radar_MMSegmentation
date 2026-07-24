# Validation

The package was validated as follows:

1. Every Python file was compiled with `python -m py_compile`.
2. `ppp_eda_geometry.points_from_gt` was tested with aligned synthetic
   auxiliary maps.
3. Synthetic train and validation scan directories were generated with:
   - frame-level ego velocity, yaw rate, Doppler range, and wrap diagnostics;
   - all point-level velocity stages and wrap indices;
   - the existing spatial, depth, RCS, and range-gate arrays.
4. The following scripts completed end to end:

```text
plot_ppp_eda_summary.py
select_ppp_examples.py
compare_ppp_eda_splits.py
```

5. The normalized wrapped target check reported zero observations outside
   `[-0.5, 0.5)` in the synthetic test.

The actual SYS100 dataloader could not be executed in this isolated artifact
environment because the proprietary data and repository runtime are not
available here. The scan script contains a runtime alignment assertion that
reconstructs the wrapped velocity from the exported pre-wrap velocity, frame
Doppler range, and wrap index and stops if it differs from `gt_seg_map`.
