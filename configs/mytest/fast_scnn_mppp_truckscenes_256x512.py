# configs/ppp_mark/fast_scnn_mppp_truckscenes_256x512.py

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.ppp_mark_head',

        # TruckScenes dataset + transforms (you implement these files)
        'mmseg.datasets.truckscenes_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.truckscenes_ppp_transforms',

        'mmseg.evaluation.metrics.ppp_metrics',
        'mmseg.engine.hooks.ppp_mark_visualization_hook'
    ],
    allow_failed_imports=False
)

import os

# ----------------------------
# Paths / Dataset
# ----------------------------
TRUCKSCENES_ROOT = os.environ.get(
    "TRUCKSCENES",
    "/home/uik04723/data/man-truckscenes-mini/man-truckscenes"
)

TRUCKSCENES_VERSION = os.environ.get(
    "TRUCKSCENES_VERSION",
    "v1.1-mini"
)

# normally set by runner; override here for convenience
work_dir = os.environ.get(
    "MMSEG_WORK_DIR",
    "/p/scratch/nxtaim-1/huber7/work_dirs/ppp_mark_truckscenes"
)

# ----------------------------
# Image sizes / normalization
# ----------------------------
crop_size = (256, 512)

# For your PPPLoss/PPPMetric normalization, safest: use the *network input size*
out_img_size = crop_size

# start with ImageNet stats; compute dataset-specific later if needed
data_rbg_mean = [123.675, 116.28, 103.53]
data_rbg_std  = [58.395, 57.12, 57.375]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=data_rbg_mean,
    std=data_rbg_std,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size,     # Note: preprocessor may pad to this size
)

# ----------------------------
# Model (unchanged)
# ----------------------------
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
    ),
    decode_head=dict(
        type='PPPMarkHead',
        in_channels=512,
        channels=32,
        num_convs=2,

        # marked PPP targets:
        num_classes=3,

        ppp_weight=1,
        depth_weight=1,
        velocity_weight=1,

        loss_ppp=dict(
            type='PPPLoss',
            out_img_size=out_img_size,
            loss_weight=1,
        ),

        in_index=-1,
        align_corners=False,
    ),
    auxiliary_head=None,
    test_cfg=dict(mode='whole'),
)

# ----------------------------
# TruckScenes pipeline
# ----------------------------
train_pipeline = [
    dict(
        type='LoadTruckScenesSample',
        dataroot=TRUCKSCENES_ROOT,
        version=TRUCKSCENES_VERSION,
        cam_channel='CAMERA_LEFT_FRONT',
        radar_channel='RADAR_LEFT_FRONT',
        verbose=False,
    ),
    dict(
        type='BuildTruckScenesPPPHeatmap',
        cam_channel='CAMERA_LEFT_FRONT',
        radar_channel='RADAR_LEFT_FRONT',
        dot_size=2,               # match render_pointcloud_in_image(... dot_size=2)
        clip_depth=(0.0, 150.0),
        clip_vel=(0.0, 30.0),     # abs velocity range (adjust later)
    ),
    # dict(type='PPPRandomCrop', crop_size=crop_size),
    # dict(type='PPPRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='TruckScenesPackSegInputs'),
]

test_pipeline = [
    dict(
        type='LoadTruckScenesSample',
        dataroot=TRUCKSCENES_ROOT,
        version=TRUCKSCENES_VERSION,
        cam_channel='CAMERA_LEFT_FRONT',
        radar_channel='RADAR_LEFT_FRONT',
        verbose=False,
    ),
    dict(
        type='BuildTruckScenesPPPHeatmap',
        cam_channel='CAMERA_LEFT_FRONT',
        radar_channel='RADAR_LEFT_FRONT',
        dot_size=2,
        clip_depth=(0.0, 150.0),
        clip_vel=(0.0, 30.0),
    ),
    dict(type='TruckScenesPackSegInputs'),
]

dataset_type = 'TruckScenesPPPHeatmapDataset'

train_dataloader = dict(
    batch_size=16,     # start small; projection is heavier than SYS100 grid
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        dataroot=TRUCKSCENES_ROOT,
        version=TRUCKSCENES_VERSION,
        frame_stride=1,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        dataroot=TRUCKSCENES_ROOT,
        version=TRUCKSCENES_VERSION,
        frame_stride=10,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

# ----------------------------
# Optim
# ----------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(
        _delete_=True,
        max_norm=1.0,
        norm_type=2
    ),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.5)})
)

# ----------------------------
# Train / Val loops
# ----------------------------
# Use your develbooster style short run (adjust as needed)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='PPPMetric', out_img_size=out_img_size)
test_evaluator = dict(type='PPPMetric', out_img_size=out_img_size)

# ----------------------------
# Hooks
# ----------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=5,
        save_best='ppp/count_abs_error',
        rule='less'
    ),
)

custom_hooks = [
    dict(type='PPPMarkVisualizationHook', interval=200, img_mean=data_rbg_mean, img_std=data_rbg_std),
    dict(type='VRAMHook', interval=5, reset_peak=True, log_all_ranks=True),
    dict(type='LossPlotHook', out_file='loss.png', smooth_alpha=0.05),
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)