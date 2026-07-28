"""Changes to apply to each camera-domain metric-export config."""

# Full-image evaluation. This avoids silently exporting a cropped/padded grid.
out_img_size = (544, 1792)

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[0.0, 0.0, 0.0],
    std=[1.0, 1.0, 1.0],
    pad_val=0,
    seg_pad_val=0,
    size=out_img_size,
)

custom_hooks = [
    # Keep PrefetchStatsHook too when desired.
    dict(
        type="PPPMarkMetricEvalExportHook",
        max_samples_test=10000,
        max_samples_val=0,
        img_mean=[0.0, 0.0, 0.0],
        img_std=[1.0, 1.0, 1.0],
        depth_link="exp",
        velocity_link="cube",
        rcs_link="cube",
        require_four_channels=True,
        require_full_image_shape=True,
    ),
]

# The uploaded config does not define load_from. Supply the intended checkpoint
# either here or through --cfg-options load_from=/path/to/checkpoint.pth.
# load_from = "/path/to/best_checkpoint.pth"
