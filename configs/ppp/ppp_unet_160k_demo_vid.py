_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.ppp_mark_unet_head',
        'mmseg.datasets.sys100_ppp_heatmap_dataset_demo_vid',
        'mmseg.datasets.transforms.sys100_ppp_transforms',
        'mmseg.evaluation.metrics.ppp_metrics_loss_mmd',
        'mmseg.engine.hooks.prefetch_stats_hook',
        'mmseg.engine.hooks.ppp_demo_vis_hook',
    ],
    allow_failed_imports=False
)

import os

crop_size = (256, 512)
out_img_size = (544, 1792)
data_rbg_mean = [0.0, 0.0, 0.0]
data_rbg_std = [1.0, 1.0, 1.0]
fovs_wo_fisheyeart = (105, 48.4)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=data_rbg_mean,
    std=data_rbg_std,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False,
    ),
    decode_head=dict(
        type='PPPMarkUNetHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        dropout_ratio=0.1,
        num_classes=3,
        ppp_weight=1,
        depth_weight=0,
        velocity_weight=0,
        loss_ppp=dict(
            type='PPPLoss',
            out_img_size=out_img_size,
            normalize_by_gt_count=True,
        ),
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ----------------------------
# Data paths
# ----------------------------

DATA_PATH_TRAIN = os.environ.get(
   "SYS100_DATA_PATH",
   "/p/data1/nxtaim/proprietary/continental/sys100/SYS100_Dataset.csv"
)
DATA_PATH_VAL = os.environ.get(
   "SYS100_DATA_PATH",
   "/p/data1/nxtaim/proprietary/continental/sys100_validation/SYS100_Validation_Dataset.csv"
)

# Demo sequence
DEMO_H5_PATH = (
    "/p/data1/nxtaim/proprietary/continental/sys100_validation/"
    "2023.03.02_at_14.16.46/"
    "2023.03.02_at_14.16.46_camera-radar-mi_5374_radargrids_64.h5"
)

DEMO_CENTER_FRAME = 13678
DEMO_HALF_WINDOW = 640
DEMO_FRAME_START = DEMO_CENTER_FRAME - DEMO_HALF_WINDOW  # 13038
DEMO_FRAME_END = DEMO_CENTER_FRAME + DEMO_HALF_WINDOW    # 14318
DEMO_FRAME_STRIDE = 4

# Use only one of the two underlying video streams.
# If the output looks wrong, switch this to "ODD".
DEMO_VIDEO_NAME_CONTAINS = "EVEN"

# Expected number after choosing one stream.
DEMO_NUM_FRAMES = (DEMO_FRAME_END - DEMO_FRAME_START) // DEMO_FRAME_STRIDE + 1

dataset_type = 'Sys100PPPHeatmapDatasetDemoVid'

batch_size_train = 25
batch_size_val = 10
batch_size_test = 10

# ----------------------------
# Train dataloader
# ----------------------------

train_dataloader = dict(
    batch_size=batch_size_train,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_TRAIN,
        source_is_summary_csv=True,
        fovs=fovs_wo_fisheyeart,
        pipeline=[
            dict(
                type='LoadSys100Sample',
                sys100_root=DATA_PATH_TRAIN,
                source_is_summary_csv=True,
                out_img_size=out_img_size,
                undistort=True,
                fovs=fovs_wo_fisheyeart,
                strict=False,
                prefetch=True,
                prefetch_dict={
                    "buffer_size": 1024,
                    "queue_size": 120,
                    "priority": 10,
                    "num_prefetch_workers": 4,
                    "wait_till_buffer_has": batch_size_train,
                    "buffer_fill_pause_threshold": 0.9,
                    "default_sleep_time": 60,
                }
            ),
            dict(
                type='BuildPPPHeatmap',
                thres=0.05,
                rel_x_centre=0.491,
                rel_y_centre=0.4370,
            ),
            dict(type='PPPRandomCrop', crop_size=crop_size),
            dict(type='PPPRandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PPPPackSegInputs'),
        ]
    )
)

# ----------------------------
# Validation dataloader
# ----------------------------

num_samples_val = 80

val_dataloader = dict(
    batch_size=batch_size_val,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_VAL,
        source_is_summary_csv=True,
        target_dataset_size=num_samples_val,
        fovs=fovs_wo_fisheyeart,
        pipeline=[
            dict(
                type='LoadSys100Sample',
                sys100_root=DATA_PATH_VAL,
                source_is_summary_csv=True,
                target_dataset_size=num_samples_val,
                out_img_size=out_img_size,
                undistort=True,
                fovs=fovs_wo_fisheyeart,
                strict=False,
                prefetch=True,
                prefetch_dict={
                    "buffer_size": 500,
                    "queue_size": 120,
                    "priority": 19,
                    "num_prefetch_workers": 4,
                    "wait_till_buffer_has": 100,
                    "buffer_fill_pause_threshold": 0.9,
                    "default_sleep_time": 60,
                }
            ),
            dict(
                type='BuildPPPHeatmap',
                thres=0.05,
                rel_x_centre=0.491,
                rel_y_centre=0.4370,
            ),
            dict(type='PPPPackSegInputs'),
        ]
    )
)

# ----------------------------
# Demo test dataloader
# ----------------------------

test_dataloader = dict(
    batch_size=batch_size_test,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        sys100_root=DEMO_H5_PATH,
        source_is_summary_csv=False,

        demo_h5_path=DEMO_H5_PATH,
        demo_frame_start=DEMO_FRAME_START,
        demo_frame_end=DEMO_FRAME_END,
        demo_frame_stride=DEMO_FRAME_STRIDE,
        demo_use_video_frame_counter=True,
        demo_video_name_contains=DEMO_VIDEO_NAME_CONTAINS,

        fovs=fovs_wo_fisheyeart,
        pipeline=[
            dict(
                type='LoadSys100Sample',
                sys100_root=DEMO_H5_PATH,
                source_is_summary_csv=False,
                target_dataset_size=None,
                out_img_size=out_img_size,
                undistort=True,
                fovs=fovs_wo_fisheyeart,
                strict=False,
                prefetch=False,
            ),
            dict(
                type='BuildPPPHeatmap',
                thres=0.05,
                rel_x_centre=0.491,
                rel_y_centre=0.4370,
            ),
            dict(type='PPPPackSegInputs'),
        ]
    )
)

# ----------------------------
# Optimizer / schedule
# ----------------------------

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-2,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(
        max_norm=1.0,
        norm_type=2
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        end_factor=1.0,
        begin=0,
        end=7000,
        by_epoch=False
    ),
    dict(
        type='LinearLR',
        start_factor=1.0,
        end_factor=1e-1,
        begin=7001,
        end=70000,
        by_epoch=False
    ),
    dict(
        type='LinearLR',
        start_factor=1.0,
        end_factor=1e-1,
        begin=70001,
        end=140000,
        by_epoch=False
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=140000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='PPPLossMMD', out_img_size=out_img_size)
test_evaluator = dict(type='PPPLossMMD', out_img_size=out_img_size)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=5,
        save_best='ppp/optimal_score',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(
        type='PPPDemoVisHook',
        max_samples_test=DEMO_NUM_FRAMES,
        out_dir='demo_frames',
        gamma=0.2,
        sim_block_size=8,
        sim_block_placement='weighted',
        point_size=28,
        figsize=(14, 20),
        dpi=90,
        write_all_ranks=True,
    ),
    dict(type='PrefetchStatsHook')
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)