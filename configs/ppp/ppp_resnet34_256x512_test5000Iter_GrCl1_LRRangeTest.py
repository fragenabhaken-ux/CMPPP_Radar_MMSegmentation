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
        'mmseg.engine.hooks.ppp_mark_vis_hook_light', # 'mmseg.engine.hooks.ppp_visualization_hook'
        'mmseg.engine.hooks.prefetch_stats_hook',
    ],
    allow_failed_imports=False
)

import os
# DATA_PATH = os.environ.get(
#     "SYS100_DATA_PATH",
#     "/p/data1/nxtaim/proprietary_samples/continental_sys100/"
# )
#DATA_PATH = os.environ.get(
#    "SYS100_DATA_PATH",
#    "/p/data1/nxtaim/proprietary/continental/"
#)

# # wird normalerweise vom runner automatisch auf work_dirs im repo gesetzt, hier überschreiben
# work_dir = os.environ.get(
#     "MMSEG_WORK_DIR",
#     "/p/scratch/nxtaim-1/huber7/work_dirs/ppp_mark"
# )

######################################################################

crop_size = (256, 512) # (384, 768)
out_img_size = (544, 1792)
data_rbg_mean = [0.0, 0.0, 0.0] # [80.182, 87.428, 79.65] # -> marked_ppp_gt_exploration.ipynb
data_rbg_std = [1.0, 1.0, 1.0] # [25.751, 25.934, 27.112]
fovs_wo_fisheyeart=(105, 48.4) # (110, 48.4) still had the fisheye artefact on the right

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=data_rbg_mean,
    std=data_rbg_std,
    pad_val=0,
    seg_pad_val=0, # for seg label data, 255 is the "ignore" label (nonissue, since no padding necessary)
    size=crop_size # Note: No resizing is applied during validation.
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
    # backbone=dict(
    #     type='ResNeXt',
    #     depth=101,
    #     groups=32,
    #     base_width=4,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     strides=(1, 2, 1, 1),
    #     dilations=(1, 1, 2, 4),
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=False,
    #     style='pytorch',
    # ),
    decode_head=dict(
        type='PPPMarkHead',
        # in_channels=2048, # This works with a depth of 50. For the ResNet 34 architecture the last layer has 512 channels. 
        in_channels=512, # This works with a depth of 50. For the ResNet 34 architecture the last layer has 512 channels. 
        channels=32,
        num_convs=2,

        # for marked PPP targets:
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
# decrease batch size in val/test, because the GPU memory constraints on full sized images
batch_size_test = 10
batch_size_val = 10

#### Train dataloader #####

train_dataloader = dict(
    batch_size=batch_size_train,
    num_workers=4, # we parallelize with Jonas' aumovio/prefetcheddataset.py
    # persistent_workers=True, # for torch Dataloader
    pin_memory=True, # for torch Dataloader
    prefetch_factor=4, # for torch Dataloader
    sampler=dict(type='InfiniteSampler', shuffle=True), # for torch Dataloader
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_TRAIN,
        source_is_summary_csv=True,
        # target_dataset_size=1000000,
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
                    "buffer_size": 1024, # 4096,
                    "queue_size": 120,
                    "priority": 10,
                    "num_prefetch_workers": 4,   # dann gegen 8 vergleichen
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

num_samples_val = 1000

val_dataloader = dict(
    batch_size=batch_size_val,
    num_workers=4, # we parallelize with Jonas' aumovio/prefetcheddataset.py
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_VAL, # Frage: Existiert ein Test-Datensatz? Ist VAL hier ok?
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
                    "buffer_size": 500, # 4096,
                    "queue_size": 120,
                    "priority": 19,
                    "num_prefetch_workers": 4,   # dann gegen 8 vergleichen
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

#### Test data loader ####

num_samples_test = 1000

test_dataloader = dict(
    batch_size=batch_size_test,
    num_workers=4, # we parallelize with Jonas' aumovio/prefetcheddataset.py
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
                    "buffer_size": 500, # 4096,
                    "queue_size": 120,
                    "priority": 19,
                    "num_prefetch_workers": 4,   # dann gegen 8 vergleichen
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
        lr=1e-9,              # maximale LR
        weight_decay=0.01,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(
        _delete_=True,
        max_norm=1.0,
        norm_type=2
    ),
)

param_scheduler = [
    dict(
        type='ExponentialLR',
        gamma=(2e-3 / 1e-9) ** (1 / 5000),  # (lr_max / lr_min)^(1/T)
        begin=0,
        end=5000,
        by_epoch=False
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=6000)
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
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),  # match VRAMHook interval
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=6000,
        max_keep_ckpts=5,
        save_best='ppp/count_abs_error', # after distance/velocity normalization save_best='ppp/selection_score', -> ppp_metrics
        rule='less'
    ),
)

custom_hooks = [
    # dict(
    #     type='PPPMarkVisHookLight',
    #     train_interval=100,
    #     max_samples_train=1,
    #     max_samples_val=5,
    #     max_samples_test=10,
    #     img_mean=data_rbg_mean,
    #     img_std=data_rbg_std,
    #     gamma=0.2,
    #     sim_block_size=8,
    #     sim_block_placement='weighted',
    #     point_size=28,
    # ),
    # dict(type='VRAMHook', interval=5, reset_peak=True, log_all_ranks=True),
    #dict(type='LossPlotHook', out_file='loss.png', smooth_alpha=0.05),
    dict(
        type='PrefetchStatsHook',
    )
]



visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)
