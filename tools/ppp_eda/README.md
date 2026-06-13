# PPP EDA Tools

Standalone tools for exploratory data analysis of the SYS100 PPP ground-truth targets.

The goal is to inspect the dataset independently of model training, losses, hooks, and evaluators.

## Quick execution order

Run from inside `jsc_jupiter/`:

```bash
cd /e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/jsc_jupiter
mkdir -p logs
```

All EDA output is written to scratch:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/
```

Scan 5000 train and validation samples:

```bash
sbatch ppp_eda_scan.sbatch train 5000
sbatch ppp_eda_scan.sbatch val 5000
```

5000 images take about 15 minutes to scan.

This creates:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/
/e/scratch/nxtaim-1/huber7/ppp_eda/val_seed43_n5000/
```

with:

```text
frames.csv
point_values.npz
scan_summary.json
used_config.py
```

Select examples and render overlays:

```bash
sbatch ppp_eda_select_render.sbatch train 5000
sbatch ppp_eda_select_render.sbatch val 5000
```

This adds:

```text
selected_examples.json
selected_examples.csv
examples/
```

Create summary plots:

```bash
sbatch ppp_eda_plot_summary.sbatch train 5000
sbatch ppp_eda_plot_summary.sbatch val 5000
```

This adds:

```text
plots/
```

Compare train and validation:

```bash
sbatch ppp_eda_compare_splits.sbatch 5000
```

This creates:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/compare_train_seed42_n5000_vs_val_seed43_n5000/
```

Render temporal radar accumulation for one demo frame:

```bash
sbatch ppp_eda_temporal_accumulation.sbatch 160 20 1 demo_temporal_accumulation
```

This creates:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/demo_temporal_accumulation/
```

## Workflow

```text
EDA dataloader
    ↓
scan_ppp_eda.py
    ↓
frames.csv + point_values.npz
    ↓
select_ppp_examples.py
    ↓
selected_examples.json + selected_examples.csv
    ↓
render_ppp_examples.py
    ↓
example image folders
    ↓
plot_ppp_eda_summary.py
    ↓
summary plots and summary CSV files
```

Train and validation scans can additionally be compared with:

```text
compare_ppp_eda_splits.py
```

The demo sequence can additionally be rendered with:

```text
render_ppp_demo_sequence.py
render_ppp_demo_temporal_accumulation.py
```

## Files

```text
tools/ppp_eda/
├── scan_ppp_eda.py
├── select_ppp_examples.py
├── render_ppp_examples.py
├── plot_ppp_eda_summary.py
├── compare_ppp_eda_splits.py
├── render_ppp_demo_sequence.py
├── render_ppp_demo_temporal_accumulation.py
└── README.md
```

Slurm wrappers:

```text
jsc_jupiter/
├── ppp_eda_scan.sbatch
├── ppp_eda_select_render.sbatch
├── ppp_eda_plot_summary.sbatch
├── ppp_eda_compare_splits.sbatch
├── ppp_eda_render_demo.sbatch
└── ppp_eda_temporal_accumulation.sbatch
```

EDA config:

```text
configs/ppp_eda/ppp_eda.py
```

## Config

The config defines EDA dataloaders only.

Relevant dataloaders:

```text
eda_train_dataloader
eda_val_dataloader
eda_demo_dataloader
```

The train and validation dataloaders use deterministic random subsampling:

```text
target_dataset_size
random_subsampling=True
seed
sampler=DefaultSampler(shuffle=False)
```

The EDA pipeline contains only:

```text
LoadSys100Sample
BuildPPPHeatmap
```

It intentionally does not use:

```text
PPPRandomCrop
PPPRandomFlip
PPPPackSegInputs
```

The scripts operate directly on:

```text
sample["img"]
sample["gt_seg_map"]
sample["metainfo"]
```

## Output directories

All outputs are written below:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/
```

Typical output directories:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/
/e/scratch/nxtaim-1/huber7/ppp_eda/val_seed43_n5000/
/e/scratch/nxtaim-1/huber7/ppp_eda/compare_train_seed42_n5000_vs_val_seed43_n5000/
/e/scratch/nxtaim-1/huber7/ppp_eda/demo_sequence/
/e/scratch/nxtaim-1/huber7/ppp_eda/demo_temporal_accumulation/
```

A completed train EDA directory looks like this:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/
├── frames.csv
├── point_values.npz
├── scan_summary.json
├── used_config.py
├── selected_examples.json
├── selected_examples.csv
├── examples/
└── plots/
```

Avoid listing the whole EDA root if it becomes large. Use:

```bash
ls -ld /e/scratch/nxtaim-1/huber7/ppp_eda
```

or inspect specific subfolders:

```bash
ls -lh /e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000
```

## Scan output

### `frames.csv`

One row per scanned frame.

Important columns:

```text
eda_idx
idx
demo_idx
global_idx
sample_id
video_name
video_path
video_frame_idx

point_count

brightness_mean
brightness_std
brightness_q01
brightness_q05
brightness_q50
brightness_q95
brightness_q99

dark_fraction_0_05
dark_fraction_0_10
dark_fraction_0_15
dark_fraction_0_20

bright_fraction_0_80
bright_fraction_0_85
bright_fraction_0_90
bright_fraction_0_95

contrast_q95_minus_q05
contrast_q99_minus_q01

laplacian_var
```

Meaning:

```text
eda_idx
    Local index inside the deterministic EDA scan.

idx
    Dataset index passed into the SYS100 loader.

point_count
    Number of projected GT radar points in the PPP target.

brightness_*
    Whole-image brightness statistics after converting the image from [-1, 1] to [0, 1].

dark_fraction_*
    Fraction of pixels below the brightness threshold.

bright_fraction_*
    Fraction of pixels above the brightness threshold.

contrast_*
    Simple whole-image contrast proxies.

laplacian_var
    Blur or sharpness proxy.
```

### `point_values.npz`

One row per projected GT radar point.

Arrays:

```text
eda_idx
x_pix
y_pix
depth
velocity
rgb
```

Meaning:

```text
eda_idx
    Local frame index inside the EDA scan.

x_pix, y_pix
    Image-space pixel position of the projected GT radar point.

depth
    GT depth mark.

velocity
    GT radial velocity mark.

rgb
    Camera RGB value at the projected GT radar point.
```

Recommended dtypes:

```text
eda_idx   int32
x_pix     int16
y_pix     int16
depth     float32
velocity  float32
rgb       uint8, shape (N, 3)
```

## Example selection

The selection script reads:

```text
frames.csv
point_values.npz
```

and writes:

```text
selected_examples.json
selected_examples.csv
```

Current categories:

```text
empty
single_point
sparse
dark
bright
high_positive_velocity
high_negative_velocity
high_mean_abs_velocity
mostly_static
```

Meaning:

```text
empty
    Frames with point_count == 0.

single_point
    Frames with point_count == 1.

sparse
    Frames with 2 <= point_count <= sparse_max_points.

dark
    Frames with smallest brightness_mean.

bright
    Frames with largest brightness_mean.

high_positive_velocity
    Frames with largest mean signed GT velocity.

high_negative_velocity
    Frames with smallest mean signed GT velocity.

high_mean_abs_velocity
    Frames with largest mean absolute GT velocity.

mostly_static
    Frames with smallest mean absolute GT velocity.
```

Depth is not used as a selection category, because most frames contain both near and far radar points. Depth is still rendered as an overlay.

## Rendered examples

The renderer rebuilds the same EDA dataloader, loads selected `eda_idx` frames, verifies that rendered samples match the scan, and saves overlays.

Each selected frame gets three overlays:

```text
points
depth
velocity
```

Example:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/examples/dark/
├── dark_rank001_eda000123_points.png
├── dark_rank001_eda000123_depth.png
└── dark_rank001_eda000123_velocity.png
```

Meaning:

```text
points.png
    Camera image with GT radar points in white.

depth.png
    Camera image with GT radar points colored by GT depth.

velocity.png
    Camera image with GT radar points colored by signed GT radial velocity.
```

## Summary plots

Run with Slurm from inside `jsc_jupiter/`:

```bash
sbatch ppp_eda_plot_summary.sbatch train 5000
sbatch ppp_eda_plot_summary.sbatch val 5000
```

This creates:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/plots/
/e/scratch/nxtaim-1/huber7/ppp_eda/val_seed43_n5000/plots/
```

Important plots:

```text
point_count_hist.png
point_count_bucket_bar.png
brightness_hist.png
contrast_hist.png
laplacian_var_hist.png
depth_hist.png
velocity_signed_hist.png
velocity_signed_hist_log_y.png
velocity_abs_hist.png
velocity_abs_hist_log_y.png
velocity_bucket_bar.png
image_space_density.png
depth_vs_y_pix.png
velocity_vs_depth.png
rgb_at_radar_points_hist.png
```

Also written:

```text
summary_frames.csv
summary_points.csv
plot_summary.json
```

The point-count bucket plot uses fine buckets in the main region and a short high-count tail:

```text
0
1
2-10
11-25
26-50
51-75
76-100
101-125
126-150
151-175
176-200
201-225
226-250
251-275
276-300
301-325
326-350
>350
```

The velocity bucket plot stays coarse:

```text
v < -0.5
|v| <= 0.5
v > 0.5
```

The additional log-y velocity histograms are used because radial velocity often has a very large spike around zero and much smaller tails.

## Train-validation comparison

Run from inside `jsc_jupiter/` after both scans exist:

```bash
sbatch ppp_eda_compare_splits.sbatch 5000
```

This compares:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/
/e/scratch/nxtaim-1/huber7/ppp_eda/val_seed43_n5000/
```

and writes:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/compare_train_seed42_n5000_vs_val_seed43_n5000/
```

Important comparison outputs:

```text
point_count_train_vs_val.png
point_count_bucket_train_vs_val.png
brightness_train_vs_val.png
contrast_train_vs_val.png
laplacian_var_train_vs_val.png
depth_train_vs_val.png
velocity_signed_train_vs_val.png
velocity_signed_train_vs_val_log_y.png
velocity_abs_train_vs_val.png
velocity_abs_train_vs_val_log_y.png
velocity_bucket_train_vs_val.png
radar_point_brightness_train_vs_val.png
image_space_density_train.png
image_space_density_val.png
image_space_density_difference_val_minus_train.png
```

Also written:

```text
frame_summary_train_val.csv
point_summary_train_val.csv
point_count_buckets_train_val.csv
velocity_buckets_train_val.csv
compare_summary.json
```

## Demo sequence rendering

Run from inside `jsc_jupiter/`:

```bash
cd /e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/jsc_jupiter
mkdir -p logs
```

Render full demo sequence:

```bash
sbatch ppp_eda_render_demo.sbatch demo_sequence 1 none
```

Render every fourth frame:

```bash
sbatch ppp_eda_render_demo.sbatch demo_sequence_every4 4 none
```

Render only five frames for debugging:

```bash
sbatch ppp_eda_render_demo.sbatch demo_sequence_debug_5 1 5
```

Output goes to:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/demo_sequence/
├── points/
├── depth/
├── velocity/
├── render_summary.json
└── rendered_frames.json
```

## Demo temporal accumulation

This renders one center camera frame and overlays projected radar GT points from neighboring demo frames.

Run from inside `jsc_jupiter/`:

```bash
sbatch ppp_eda_temporal_accumulation.sbatch 160 20 1 demo_temporal_accumulation
```

Arguments:

```text
160
    center eda_idx inside the demo dataloader

20
    temporal radius

1
    stride inside the temporal window

demo_temporal_accumulation
    output folder name below the EDA root
```

The command uses frames:

```text
center_eda_idx - radius
...
center_eda_idx
...
center_eda_idx + radius
```

For the default command:

```text
140, 141, ..., 160, ..., 179, 180
```

Output:

```text
/e/scratch/nxtaim-1/huber7/ppp_eda/demo_temporal_accumulation/
├── center_frame_points.png
├── accumulated_points_time_colored.png
├── accumulated_points_alpha_fade.png
├── accumulated_depth.png
├── accumulated_velocity.png
├── accumulated_points.csv
└── temporal_accumulation_summary.json
```

Meaning:

```text
center_frame_points.png
    Center camera frame with only the center-frame GT radar points.

accumulated_points_time_colored.png
    Neighboring-frame GT radar points overlaid on the center image, colored by relative frame index.

accumulated_points_alpha_fade.png
    Neighboring-frame GT radar points overlaid on the center image, with older/future frames faded.

accumulated_depth.png
    Accumulated projected GT radar points colored by GT depth.

accumulated_velocity.png
    Accumulated projected GT radar points colored by GT radial velocity.

accumulated_points.csv
    One row per accumulated projected radar point.

temporal_accumulation_summary.json
    Metadata and point counts for the temporal window.
```

Important: this is image-space accumulation. It does not use ego-motion compensation.

So it is correct to describe it as:

```text
Temporal accumulation of projected radar detections around one demo frame.
```

Do not describe it as:

```text
Ego-motion compensated radar accumulation.
```

## Adding evaluations to the architecture

There are two cases.

### 1. Evaluation based on already extracted data

Use this route if the needed data already exists on disk.

Examples:

```text
frames.csv
point_values.npz
selected_examples.csv
HDF5 metric export files
saved predictions
saved expected counts
saved GT point tables
```

Then do not touch the model, dataloader, or export hook. Add the evaluation to the script that already reads the relevant extracted data.

Use these files:

```text
tools/ppp_eda/plot_ppp_eda_summary.py
    Add functions here for single-split EDA plots based on frames.csv and point_values.npz.

tools/ppp_eda/compare_ppp_eda_splits.py
    Add functions here for train-vs-validation comparison plots.

tools/ppp_eda/select_ppp_examples.py
    Add functions here if the evaluation selects special frames or categories.

tools/ppp_eda/render_ppp_examples.py
    Add functions here if the evaluation creates image overlays for selected examples.

tools/ppp_offline_metric_eval.py
    Add functions here for model-output metrics based on existing HDF5 exports.
```

Typical change pattern for adding a new plot:

```text
1. Find where the needed arrays are already loaded.
2. Add a small helper function only if the plotting logic is reused.
3. Add the new plot call in main().
4. Save the output into the existing output directory.
5. Add the new filename to the README.
6. If there is a train-vs-val version, add the analogous plot to compare_ppp_eda_splits.py.
```

Example: adding a new single-split plot in `plot_ppp_eda_summary.py`.

If the data is already available in `main()`:

```python
velocity = points["velocity"].astype(float)
abs_velocity = np.abs(velocity)
```

then add another plot call next to the existing related plot:

```python
plot_hist(
    values=abs_velocity,
    path=out_dir / "velocity_abs_hist_log_y.png",
    title="Absolute GT radial velocity distribution, log-y",
    xlabel="abs(GT radial velocity)",
    bins=100,
    log_y=True,
)
```

If the helper does not support the needed option yet, extend the helper first. Example:

```python
def plot_hist(..., log_y: bool = False):
    ...
    if log_y:
        plt.yscale("log")
```

For train-vs-val comparison, add the same idea to `compare_ppp_eda_splits.py`.

If `plot_hist_compare()` needs a new option, extend the function signature:

```python
def plot_hist_compare(..., log_y: bool = False):
    ...
    if log_y:
        plt.yscale("log")
```

Then add the comparison plot in `main()` next to the existing related comparison:

```python
plot_hist_compare(
    train_values=np.abs(train_velocity),
    val_values=np.abs(val_velocity),
    path=out_dir / "velocity_abs_train_vs_val_log_y.png",
    title="Absolute GT radial velocity, train vs val, log-y",
    xlabel="abs(GT radial velocity)",
    bins=args.hist_bins,
    clip_quantile=args.clip_quantile,
    log_y=True,
)
```

Typical change pattern for changing buckets:

```text
1. Find the bucket helper function.
2. Change the bin edges and labels together.
3. Keep the same function name if downstream code already uses it.
4. Apply the same bucket definition in both single-split and train-vs-val scripts.
```

Relevant files:

```text
tools/ppp_eda/plot_ppp_eda_summary.py
    point_count_buckets()

tools/ppp_eda/compare_ppp_eda_splits.py
    point_count_buckets()
```

Important: if a bucket definition exists in both files, keep them identical.

Rule:

```text
Data already exists:
    add or extend a function in the existing post-processing script.
    do not create a new model hook.
```

### 2. Evaluation based on data that is not yet extracted

Use this route if the needed data is only available during model inference.

Examples:

```text
raw model logits
predicted PPP intensity maps
sampled predicted point clouds
predicted depth maps
predicted velocity maps
per-sample loss terms
backbone features
decoder features
attention maps
intermediate tensors
```

Then first extend the existing export hook.

Find the hook file with:

```bash
grep -R "class PPPMarkMetricEvalExportHook" -n mmseg tools
grep -R "class PPPMarkMetricEvalExportHookTwoDepth" -n mmseg tools
```

Add the extraction logic to the file that contains:

```text
PPPMarkMetricEvalExportHook
PPPMarkMetricEvalExportHookTwoDepth
```

The hook should only export missing data. It should not contain final plotting logic.

Typical exported fields:

```text
sample_id
video_name
frame_idx
gt_point_count
pred_expected_count
gt_points
pred_points
pred_intensity
pred_depth
pred_velocity
loss_terms
```

Recommended formats:

```text
HDF5
    large arrays, dense maps, predictions

CSV
    per-sample scalar values

NPZ
    compact numpy arrays

JSON
    metadata and config summaries
```

After the missing data is exported, add the actual evaluation to the existing offline analysis script:

```text
tools/ppp_offline_metric_eval.py
```

Typical pattern:

```text
1. Extend export hook to write the missing values.
2. Rerun the export.
3. Add a metric or plot function to tools/ppp_offline_metric_eval.py.
4. Register the metric in the existing METRICS structure if needed.
5. Save CSV, JSON, and plots to the existing metric output directory.
```

Rule:

```text
Data does not exist yet:
    first add extraction to the existing export hook.
    then add the evaluation function to the existing offline metric script.
```

### Decision rule

```text
If the data already exists:
    add a function or plot call to the existing analysis script.

If the data does not exist:
    extend the export hook first.
    then add the analysis function to the offline metric script.
```

Keep the architecture clean:

```text
model / dataloader / hooks
    produce and export data

EDA / offline evaluation scripts
    analyze exported data
```


## Consistency checks

`render_ppp_examples.py` checks that scan and render refer to the same samples.

For every selected `eda_idx`, it verifies:

```text
1. point_count in frames.csv
   equals
   recomputed point_count from rendered gt_seg_map

2. metadata in frames.csv
   matches
   metadata from the rendered sample
```

Checked metadata keys:

```text
idx
demo_idx
global_idx
sample_id
video_name
video_path
video_frame_idx
```

If scan and render do not match, the renderer stops with:

```text
EDA SAMPLE MISMATCH: point_count differs.
```

or:

```text
EDA SAMPLE MISMATCH: metadata differs.
```

## Determinism requirements

The following must match between scan and render:

```text
same CSV path
same target_dataset_size
same random_subsampling setting
same seed
same dataloader
same sampler order
```

For train:

```text
eda_train_dataloader
seed=42
PPP_EDA_TRAIN_SAMPLES=<N>
```

For validation:

```text
eda_val_dataloader
seed=43
PPP_EDA_VAL_SAMPLES=<N>
```

If the scan was run with:

```bash
sbatch ppp_eda_scan.sbatch train 5000
```

then select-render must be run with:

```bash
sbatch ppp_eda_select_render.sbatch train 5000
```

Do not scan with 500 samples and render with 5000 samples.

## Quick inspection commands

Inspect scan output:

```bash
ls -lh /e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000
```

Inspect selected examples:

```bash
cat /e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/selected_examples.json
```

Inspect rendered example images:

```bash
find /e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/examples -maxdepth 2 -type f | sort | head -100
```

Inspect plots:

```bash
find /e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/plots -maxdepth 1 -type f | sort
```

Inspect train-validation comparison:

```bash
find /e/scratch/nxtaim-1/huber7/ppp_eda/compare_train_seed42_n5000_vs_val_seed43_n5000 -maxdepth 1 -type f | sort
```

Inspect temporal accumulation:

```bash
find /e/scratch/nxtaim-1/huber7/ppp_eda/demo_temporal_accumulation -maxdepth 1 -type f | sort
```

Inspect selected examples as table:

```bash
cd /e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp
source jsc_jupiter/activate_jupiter.sh

python - <<'PY'
import pandas as pd

df = pd.read_csv("/e/scratch/nxtaim-1/huber7/ppp_eda/train_seed42_n5000/selected_examples.csv")
cols = [
    "category",
    "rank",
    "eda_idx",
    "point_count",
    "brightness_mean",
    "velocity_mean",
    "velocity_abs_mean",
]
print(df[cols].head(80))
PY
```

## Notes

The EDA tools are intentionally independent of model training.

They should not import model backbones, heads, necks, losses, evaluators, or visualization hooks.

They only require:

```text
mmseg.datasets.sys100_ppp_heatmap_dataset
mmseg.datasets.sys100_ppp_heatmap_dataset_demo_vid
mmseg.datasets.transforms.sys100_ppp_transforms
```

The scan is CPU and I/O heavy. On JUPITER it still runs inside a Booster allocation. The scripts disable accidental CUDA usage with:

```bash
CUDA_VISIBLE_DEVICES=""
```
