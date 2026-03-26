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
        'mmseg.engine.hooks.ppp_mark_gen_data_meric_eval_hook',
        'mmseg.engine.hooks.prefetch_stats_hook',
    ],
    allow_failed_imports=False
)

import os

######################################################################

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

######################################################################

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
        num_classes=3,
        ppp_weight=1,
        depth_weight=0,
        velocity_weight=0,
        loss_ppp=dict(
            type='PPPLoss',
            out_img_size=out_img_size,
        ),
        in_index=-1,
        align_corners=False,
    ),
    auxiliary_head=None,
    test_cfg=dict(mode='whole'),
)

# ----------------------------
# Data pipeline
# ----------------------------

DATA_PATH_TRAIN = os.environ.get(
   "SYS100_DATA_PATH",
   "/p/data1/nxtaim/proprietary/continental/sys100/SYS100_Dataset.csv"
)
DATA_PATH_VAL = os.environ.get(
   "SYS100_DATA_PATH",
   "/p/data1/nxtaim/proprietary/continental/sys100_validation/SYS100_Validation_Dataset.csv"
)

dataset_type = 'Sys100PPPHeatmapDataset'

batch_size_train = 80
batch_size_test = 10
batch_size_val = 10

#### Train dataloader #####

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

#### val dataloader ####

num_samples_val = 500

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

#### Test dataloader – 500 samples for offline metric evaluation ####

num_samples_test = 500

test_dataloader = dict(
    batch_size=batch_size_test,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_VAL,
        source_is_summary_csv=True,
        target_dataset_size=num_samples_test,
        fovs=fovs_wo_fisheyeart,
        pipeline=[
            dict(
                type='LoadSys100Sample',
                sys100_root=DATA_PATH_VAL,
                source_is_summary_csv=True,
                target_dataset_size=num_samples_test,
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

#########################################################

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

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='PPPMetric', out_img_size=out_img_size)
test_evaluator = dict(type='PPPMetric', out_img_size=out_img_size)

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
    dict(
        type='PPPMarkGenDataMetricEvalHook',
        max_samples_test=125,     # 500 total / 4 GPUs = 125 samples per rank
        max_samples_val=0,        # disabled during val; only generates data on test run
        img_mean=data_rbg_mean,
        img_std=data_rbg_std,
    ),
    dict(
        type='PrefetchStatsHook',
    ),
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)
