_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        # New DINOv3 architecture components
        'mmseg.models.backbones.lora_timm_backbone',
        'mmseg.models.necks.dino_rope_transformer_neck',
        'mmseg.models.decode_heads.ppp_mark_vit_patch_rearrange_head',

        # Existing project components
        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.sys100_ppp_transforms',
        'mmseg.evaluation.metrics.ppp_metrics_loss_mmd',
        'mmseg.engine.hooks.prefetch_stats_hook',
        'mmseg.engine.hooks.ppp_mark_metric_eval_export_hook',
    ],
    allow_failed_imports=False
)

import os

crop_size = (544, 1792)
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

# DINOv3 ViT-S/16:
# patch grid for crop_size=(544,1792) is approximately 34 x 112.
# embed_dims for vit_small_patch16_dinov3.lvd1689m is 384.
dinov3_embed_dims = 384

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='LoRATIMMBackbone',
        model_name='vit_small_patch16_dinov3',
        pretrained=False,
        checkpoint_path='/p/project1/nxtaim-1/huber7/pretrained/vit_small_patch16_dinov3.lvd1689m/model.safetensors',
        in_channels=3,
        patch_size=16,

        frozen_base=True,

        # Conservative first LoRA setting
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.0,

        # Start only with attention projections
        target_modules=('attn.qkv', 'attn.proj'), # aggessiver target_modules=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    ),

    neck=dict(
        type='DINORoPETransformerNeck',
        in_channels=384,
        embed_dims=384,
        in_index=0,
        num_layers=6,
        num_heads=6,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        norm_cfg=dict(type='LN'),
    ),

    decode_head=dict(
        type='PPPMarkViTPatchRearrangeHead',
        in_channels=384,
        in_index=0,
        channels=32,      # required by BaseDecodeHead
        num_classes=3,

        patch_size=(16, 16),
        refine_channels=64,
        num_refine_convs=3,
        pre_factor=4,

        ppp_weight=1.0,
        depth_weight=1.0,
        velocity_weight=0.0,

        loss_ppp=dict(
            type='PPPLoss',
            out_img_size=out_img_size,
            normalize_by_gt_count=True,
        ),
    ),

    auxiliary_head=None,
    test_cfg=dict(mode='whole'),
)

# ----------------------------
# Data pipeline
# ----------------------------

# DATA_PATH_TRAIN = os.environ.get(
#    "SYS100_DATA_PATH",
#    "/p/data1/nxtaim/proprietary/continental/sys100/SYS100_Dataset.csv"
# )
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

batch_size_train = 8
batch_size_test = 8
batch_size_val = 8

# ----------------------------
# Train dataloader
# ----------------------------

# train_dataloader = dict(
#     batch_size=batch_size_train,
#     num_workers=4,
#     pin_memory=True,
#     prefetch_factor=4,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         sys100_root=DATA_PATH_TRAIN,
#         source_is_summary_csv=True,
#         fovs=fovs_wo_fisheyeart,
#         pipeline=[
#             dict(
#                 type='LoadSys100Sample',
#                 sys100_root=DATA_PATH_TRAIN,
#                 source_is_summary_csv=True,
#                 out_img_size=out_img_size,
#                 undistort=True,
#                 fovs=fovs_wo_fisheyeart,
#                 strict=False,
#                 prefetch=True,
#                 prefetch_dict={
#                     "buffer_size": 1024,
#                     "queue_size": 120,
#                     "priority": 10,
#                     "num_prefetch_workers": 4,
#                     "wait_till_buffer_has": batch_size_train,
#                     "buffer_fill_pause_threshold": 0.9,
#                     "default_sleep_time": 60,
#                 }
#             ),
#             dict(
#                 type='BuildPPPHeatmap',
#                 thres=0.05,
#                 rel_x_centre=0.491,
#                 rel_y_centre=0.4370,
#             ),
#             dict(type='PPPRandomCrop', crop_size=crop_size),
#             dict(type='PPPRandomFlip', prob=0.5, direction='horizontal'),
#             dict(type='PPPPackSegInputs'),
#         ]
#     )
# )

#### val dataloader ####

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
                    "queue_size": 1000,
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

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=1e-7,              # maximale LR
#         weight_decay=0.01,
#         betas=(0.9, 0.999)
#     ),
#     clip_grad=dict(
#         max_norm=1.0,
#         norm_type=2
#     ),
# )

# param_scheduler = [
#     dict(
#         type='ExponentialLR',
#         gamma=(1 / 1e-7) ** (1 / 5000),  # (lr_max / lr_min)^(1/T)
#         begin=0,
#         end=5000,
#         by_epoch=False
#     )
# ]

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=140000, val_interval=5000)
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
        interval=1000,
        max_keep_ckpts=5,
        save_best='ppp/optimal_score',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [

    dict(
        type='PrefetchStatsHook',
    ),
    dict(
        type='PPPMarkMetricEvalExportHook',
        max_samples_test=10000,
        max_samples_val=0,
        img_mean=data_rbg_mean,
        img_std=data_rbg_std,
    )
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)