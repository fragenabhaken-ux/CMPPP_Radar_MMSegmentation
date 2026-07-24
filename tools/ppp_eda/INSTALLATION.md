# Installation

From the extracted package root, copy:

```text
mmseg/datasets/transforms/sys100_ppp_transforms.py
    -> mmseg/datasets/transforms/sys100_ppp_transforms.py

tools/ppp_eda/*
    -> tools/ppp_eda/*

ppp_eda.py
    -> configs/ppp_eda/ppp_eda.py

ppp_eda_spatial_config_snippet.py
    -> configs/ppp_eda/ppp_eda_spatial_config_snippet.py
```

No modification of `Sys100Dataset` is required. It already returns:

```text
velocity
yaw_rate
dopplerRange
```

Then rerun the workflow:

```bash
sbatch ppp_eda_scan.sbatch train 5000
sbatch ppp_eda_scan.sbatch val 5000

sbatch ppp_eda_select_render.sbatch train 5000
sbatch ppp_eda_select_render.sbatch val 5000

sbatch ppp_eda_plot_summary.sbatch train 5000
sbatch ppp_eda_plot_summary.sbatch val 5000

sbatch ppp_eda_compare_splits.sbatch 5000
```

Do not reuse old `frames.csv` or `point_values.npz` files. The updated scripts
intentionally fail with a clear message when the velocity-stage fields are absent.
