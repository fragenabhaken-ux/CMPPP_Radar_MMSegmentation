_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        # Dataset and transforms only.
        # No model, no losses, no hooks needed for EDA.
        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.sys100_ppp_heatmap_dataset_demo_vid',
        'mmseg.datasets.transforms.sys100_ppp_transforms',
    ],
    allow_failed_imports=False,
)

import os

# -----------------------------------------------------------------------------
# Basic geometry / target settings
# -----------------------------------------------------------------------------

out_img_size = (544, 1792)
fovs_wo_fisheyeart = (105, 48.4)

dataset_type = 'Sys100PPPHeatmapDataset'
demo_dataset_type = 'Sys100PPPHeatmapDatasetDemoVid'

# -----------------------------------------------------------------------------
# EDA subset settings
# -----------------------------------------------------------------------------

# Keep these small for first tests, then increase.
# Example override:
#   PPP_EDA_TRAIN_SAMPLES=5000 PPP_EDA_VAL_SAMPLES=5000
num_samples_train_eda = int(os.environ.get('PPP_EDA_TRAIN_SAMPLES', 5000))
num_samples_val_eda = int(os.environ.get('PPP_EDA_VAL_SAMPLES', 5000))

eda_train_seed = int(os.environ.get('PPP_EDA_TRAIN_SEED', 42))
eda_val_seed = int(os.environ.get('PPP_EDA_VAL_SEED', 43))

# For initial debugging, use PPP_EDA_NUM_WORKERS=0.
eda_num_workers = int(os.environ.get('PPP_EDA_NUM_WORKERS', 4))
eda_batch_size = int(os.environ.get('PPP_EDA_BATCH_SIZE', 1))

# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------

DATA_PATH_TRAIN = os.environ.get(
    'SYS100_TRAIN_DATA_PATH',
    os.environ.get(
        'SYS100_DATA_PATH',
        '/e/scratch/nxtaim-1/propietary/aumovio/SYS100_Dataset.csv',
    ),
)

DATA_PATH_VAL = os.environ.get(
    'SYS100_VAL_DATA_PATH',
    '/e/scratch/nxtaim-1/propietary/aumovio/SYS100_Validation_Dataset.csv',
)

# -----------------------------------------------------------------------------
# Fixed demo sequence
# -----------------------------------------------------------------------------

DEMO_H5_PATH = os.environ.get(
    'SYS100_DEMO_H5_PATH',
    (
        '/e/scratch/nxtaim-1/propietary/aumovio/sys100_validation/'
        '2023.03.02_at_14.16.46/'
        '2023.03.02_at_14.16.46_camera-radar-mi_5374_radargrids_64.h5'
    ),
)

DEMO_CENTER_FRAME = 13678
DEMO_HALF_WINDOW = 640
DEMO_FRAME_START = DEMO_CENTER_FRAME - DEMO_HALF_WINDOW  # 13038
DEMO_FRAME_END = DEMO_CENTER_FRAME + DEMO_HALF_WINDOW    # 14318
DEMO_FRAME_STRIDE = 4

# Use only one of the two underlying video streams.
# If the output looks wrong, switch this to "ODD".
DEMO_VIDEO_NAME_CONTAINS = os.environ.get('SYS100_DEMO_VIDEO_STREAM', 'EVEN')

DEMO_NUM_FRAMES = (DEMO_FRAME_END - DEMO_FRAME_START) // DEMO_FRAME_STRIDE + 1

# -----------------------------------------------------------------------------
# Common EDA pipeline
# -----------------------------------------------------------------------------
#
# Important:
# - no PPPRandomCrop
# - no PPPRandomFlip
# - no PPPPackSegInputs
#
# The EDA scripts should receive plain:
#   sample["img"]         H, W, 3 float32 in [-1, 1]
#   sample["gt_seg_map"]  H, W, 3 float32
#   sample["metainfo"]    metadata
# -----------------------------------------------------------------------------

def make_eda_pipeline(
    sys100_root,
    source_is_summary_csv,
    target_dataset_size=None,
    random_subsampling=True,
    seed=None,
    prefetch=False,
):
    return [
        dict(
            type='LoadSys100Sample',
            sys100_root=sys100_root,
            source_is_summary_csv=source_is_summary_csv,
            target_dataset_size=target_dataset_size,
            random_subsampling=random_subsampling,
            seed=seed,

            out_img_size=out_img_size,
            undistort=True,
            fovs=fovs_wo_fisheyeart,
            strict=False,

            # Keep prefetch off for EDA first.
            # DataLoader workers are easier to reason about than nested prefetching.
            prefetch=prefetch,
        ),
        dict(
            type='BuildPPPHeatmap',
            thres=0.05,
            rel_x_centre=0.491,
            rel_y_centre=0.4370,
            clip_depth=(0.0, 150.0),
            clip_vel=(-10.0, 10.0),
        ),
    ]


# -----------------------------------------------------------------------------
# Train EDA dataloader
# -----------------------------------------------------------------------------

eda_train_dataloader = dict(
    batch_size=eda_batch_size,
    num_workers=eda_num_workers,
    persistent_workers=(eda_num_workers > 0),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_TRAIN,
        source_is_summary_csv=True,

        target_dataset_size=num_samples_train_eda,
        random_subsampling=True,
        seed=eda_train_seed,

        out_img_size=out_img_size,
        undistort=True,
        fovs=fovs_wo_fisheyeart,
        strict=False,

        pipeline=make_eda_pipeline(
            sys100_root=DATA_PATH_TRAIN,
            source_is_summary_csv=True,
            target_dataset_size=num_samples_train_eda,
            random_subsampling=True,
            seed=eda_train_seed,
            prefetch=False,
        ),
    ),
)

# -----------------------------------------------------------------------------
# Validation EDA dataloader
# -----------------------------------------------------------------------------

eda_val_dataloader = dict(
    batch_size=eda_batch_size,
    num_workers=eda_num_workers,
    persistent_workers=(eda_num_workers > 0),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_VAL,
        source_is_summary_csv=True,

        target_dataset_size=num_samples_val_eda,
        random_subsampling=True,
        seed=eda_val_seed,

        out_img_size=out_img_size,
        undistort=True,
        fovs=fovs_wo_fisheyeart,
        strict=False,

        pipeline=make_eda_pipeline(
            sys100_root=DATA_PATH_VAL,
            source_is_summary_csv=True,
            target_dataset_size=num_samples_val_eda,
            random_subsampling=True,
            seed=eda_val_seed,
            prefetch=False,
        ),
    ),
)

# -----------------------------------------------------------------------------
# Fixed demo sequence EDA dataloader
# -----------------------------------------------------------------------------
#
# This is for rendering all demo frames, or every nth frame.
# It does not use random subsampling.
# -----------------------------------------------------------------------------

eda_demo_dataloader = dict(
    batch_size=1,
    num_workers=eda_num_workers,
    persistent_workers=(eda_num_workers > 0),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=demo_dataset_type,
        sys100_root=DEMO_H5_PATH,
        source_is_summary_csv=False,

        demo_h5_path=DEMO_H5_PATH,
        demo_frame_start=DEMO_FRAME_START,
        demo_frame_end=DEMO_FRAME_END,
        demo_frame_stride=DEMO_FRAME_STRIDE,
        demo_use_video_frame_counter=True,
        demo_video_name_contains=DEMO_VIDEO_NAME_CONTAINS,

        out_img_size=out_img_size,
        undistort=True,
        fovs=fovs_wo_fisheyeart,
        strict=False,

        pipeline=make_eda_pipeline(
            sys100_root=DEMO_H5_PATH,
            source_is_summary_csv=False,
            target_dataset_size=None,
            random_subsampling=False,
            seed=None,
            prefetch=False,
        ),
    ),
)

# -----------------------------------------------------------------------------
# Convenience metadata for scripts
# -----------------------------------------------------------------------------

ppp_eda_meta = dict(
    out_img_size=out_img_size,
    fovs=fovs_wo_fisheyeart,

    train_samples=num_samples_train_eda,
    val_samples=num_samples_val_eda,
    train_seed=eda_train_seed,
    val_seed=eda_val_seed,

    data_path_train=DATA_PATH_TRAIN,
    data_path_val=DATA_PATH_VAL,

    demo_h5_path=DEMO_H5_PATH,
    demo_frame_start=DEMO_FRAME_START,
    demo_frame_end=DEMO_FRAME_END,
    demo_frame_stride=DEMO_FRAME_STRIDE,
    demo_video_name_contains=DEMO_VIDEO_NAME_CONTAINS,
    demo_num_frames=DEMO_NUM_FRAMES,
)