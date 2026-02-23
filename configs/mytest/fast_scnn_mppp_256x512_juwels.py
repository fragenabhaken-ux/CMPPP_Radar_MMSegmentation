_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.ppp_mark_head',
        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.sys100_ppp_transforms',
        'mmseg.evaluation.metrics.ppp_metrics',
        'mmseg.engine.hooks.ppp_mark_visualization_hook' # 'mmseg.engine.hooks.ppp_visualization_hook'
    ],
    allow_failed_imports=False
)

import os
DATA_PATH = os.environ.get(
    "SYS100_DATA_PATH",
    "/p/data1/nxtaim/proprietary_samples/continental_sys100/"
)
#DATA_PATH = os.environ.get(
#    "SYS100_DATA_PATH",
#    "/p/data1/nxtaim/proprietary/continental/"
#)
# wird normalerweise vom runner automatisch auf work_dirs im repo gesetzt, hier überschreiben
work_dir = os.environ.get(
    "MMSEG_WORK_DIR",
    "/p/scratch/nxtaim-1/huber7/work_dirs/ppp_mark"
)

crop_size = (256, 512)
out_img_size = (544, 1792)
data_rbg_mean = [80.182, 87.428, 79.65] # -> marked_ppp_gt_exploration.ipynb
data_rbg_std = [25.751, 25.934, 27.112]
fovs_wo_fisheyeart=(105, 48.4) # (110, 48.4) still had the fisheye artefact on the right

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=data_rbg_mean,
    std=data_rbg_std,
    pad_val=0,
    seg_pad_val=0, # for seg label data, 255 is the "ignore" label (nonissue, since no padding necessary)
    size=crop_size # Note: No resizing is applied during validation.
)

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

        # for marked PPP targets:
        num_classes=3,

        ppp_weight=1,
        depth_weight=1,
        velocity_weight=1,

        loss_ppp=dict(
            type='PPPLoss',
            out_img_size=out_img_size,
            loss_weight=1,
        ),
        # loss_depth=dict(
        #     type='SmoothL1Loss',
        #     loss_weight=1.0,
        # ),
        # loss_velocity=dict(
        #     type='SmoothL1Loss',
        #     loss_weight=1.0,
        # ),

        in_index=-1,
        align_corners=False,
    ),
    auxiliary_head=None,
    test_cfg=dict(mode='whole'),
)

# ----------------------------
# SYS100 pipeline
# ----------------------------

# Note: This is my sys100_ppp_transforms
train_pipeline = [
    dict(
        type='LoadSys100Sample',
        sys100_root=DATA_PATH,
        out_img_size=out_img_size,
        undistort=True,
        fovs=fovs_wo_fisheyeart,
        strict=False
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

test_pipeline = [
    dict(
        type='LoadSys100Sample',
        sys100_root=DATA_PATH,
        out_img_size=out_img_size,
        undistort=True,
        fovs=fovs_wo_fisheyeart,
        strict=False
    ),
    dict(
        type='BuildPPPHeatmap',
        thres=0.05,
        rel_x_centre=0.491,
        rel_y_centre=0.4370,
    ),
    dict(type='PPPPackSegInputs'),
]

dataset_type = 'Sys100PPPHeatmapDataset'

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH,
        fovs=fovs_wo_fisheyeart,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=20,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH,
        fovs=fovs_wo_fisheyeart,
        frame_stride=10,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

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

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=200) # for develbooster run
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default back to 40k scheduler...
# # redefine the scheduler, because I inherit the 40k one above.
# param_scheduler = [
#     dict(
#         type='CosineAnnealingLR',
#         T_max=80000,
#         eta_min=1e-6,
#         begin=0,
#         end=80000,
#         by_epoch=False
#     )
# ]

val_evaluator = dict(type='PPPMetric', out_img_size=out_img_size)
test_evaluator = dict(type='PPPMetric', out_img_size=out_img_size,)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=5,
        save_best='ppp/count_abs_error', # after distance/velocity normalization save_best='ppp/selection_score', -> ppp_metrics
        rule='less'
    ),
)

custom_hooks = [
    dict(type='PPPMarkVisualizationHook', interval=200, img_mean=data_rbg_mean, img_std=data_rbg_std),
    dict(type='LossPlotHook', out_file='loss.png', smooth_alpha=0.05),
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)
