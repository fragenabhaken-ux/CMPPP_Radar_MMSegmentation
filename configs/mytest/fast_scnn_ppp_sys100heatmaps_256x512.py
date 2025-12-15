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

crop_size = (256, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
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

        loss_ppp=dict(
            type='PPPLoss',
            loss_weight=1.0,
            test_resolution=crop_size,
            # if your PPPLoss supports this:
            # intensity_channel=0,
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
DATA_PATH = '/home/lorenz-linux/repositories/aumovio_data/aumovio_sys100_data'

train_pipeline = [
    dict(
        type='LoadSys100Sample',
        sys100_root=DATA_PATH,
        out_img_size=(544, 1792),
        undistort=True,
        fovs=(110, 48.4),
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
        out_img_size=(544, 1792),
        undistort=True,
        fovs=(110, 48.4),
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
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0003,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.5)})
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='PPPMetric')
test_evaluator = dict(type='PPPMetric')

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=1000))

custom_hooks = [
    dict(type='PPPMarkVisualizationHook', interval=200, out_dir='work_dirs/ppp_vis')
]
