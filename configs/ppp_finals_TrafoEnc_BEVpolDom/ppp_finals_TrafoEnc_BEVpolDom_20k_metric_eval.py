_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.polar_encoder_decoder',
        'mmseg.models.ppp_polar_data_preprocessor',

        'mmseg.models.backbones.lora_timm_multilayer_backbone',
        'mmseg.models.necks.dino_rope_transformer_neck',
        'mmseg.models.decode_heads.ppp_finals_vit_patch_rearrange_head_polarBEV',
        'mmseg.models.losses.polar_point_ppp_loss',

        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.sys100_polar_transforms',
        'mmseg.datasets.transforms.sys100_polar_grid_transforms',
        'mmseg.evaluation.metrics.ppp_metrics_loss_mmd',
        'mmseg.engine.hooks.ppp_finals_denseBEVDom_metric_hook',
        'mmseg.engine.hooks.prefetch_stats_hook',
    ],
    allow_failed_imports=False
)

import math
import os

crop_size = (544, 1792)
out_img_size = (544, 1792)
polar_grid_shape = (256, 256)

data_rbg_mean = [0.0, 0.0, 0.0]
data_rbg_std = [1.0, 1.0, 1.0]
fovs_wo_fisheyeart = (105, 48.4)

r_min = 0.0
r_max = 170.0
theta_half_fov = math.radians(fovs_wo_fisheyeart[0] / 2.0)
theta_min = -theta_half_fov
theta_max = theta_half_fov

data_preprocessor = dict(
    type='PolarPPPDataPreProcessor',
    mean=data_rbg_mean,
    std=data_rbg_std,
    pad_val=0,
    size=crop_size
)

# DINOv3 ViT-S/16:
# patch grid for crop_size=(544,1792) is approximately 34 x 112.
# embed_dims for vit_small_patch16_dinov3.lvd1689m is 384.
dinov3_embed_dims = 384

model = dict(
    type='PolarEncoderDecoder',
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='LoRAMultiLayerTIMMBackbone',
        model_name='vit_small_patch16_dinov3',
        pretrained=False,
        checkpoint_path='/e/project1/nxtaim-1/huber7/pretrained/vit_small_patch16_dinov3.lvd1689m/model.safetensors',
        in_channels=3,
        patch_size=16,

        frozen_base=True,
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.0,
        target_modules=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),

        out_indices=(2, 5, 8, 11),
        in_feature_channels=384,
        per_layer_out_channels=128,
        out_channels=384,
        apply_output_norm=True,
        use_fuse_conv=True,
    ),

    neck=dict(
        type='DINORoPETransformerNeck',
        in_channels=384,
        embed_dims=384,
        in_index=0,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        norm_cfg=dict(type='LN'),
    ),

    decode_head=dict(
        type='PPPPolarViTPatchRearrangeHead',
        in_channels=384,
        in_index=0,
        channels=64,
        num_classes=3,

        output_grid_shape=polar_grid_shape,

        rearrange_factor=8,
        dense_channels=64,
        token_refine_blocks=2,
        dense_refine_blocks=2,
        downsample_strides=((1, 2),),
        post_downsample_refine_blocks=2,
        post_pool_refine_blocks=2,
        norm='gn',

        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,

        ppp_weight=100.0,
        velocity_weight=10.0,
        rcs_weight=1.0,
        velocity_mode='signed',
        velocity_link='cube',
        rcs_link='cube',

        loss_ppp=dict(
            type='PolarPointPPPLoss',
            grid_shape=polar_grid_shape,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            normalize_by_gt_count=True,
        ),
    ),

    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ----------------------------
# Data pipeline
# ----------------------------

DATA_PATH_VAL = os.environ.get(
    "SYS100_DATA_PATH",
    "/e/scratch/nxtaim-1/propietary/aumovio/"
    "SYS100_Validation_Dataset.csv"
)

# Fixed pseudo-random subsets used for model comparison.
# Keep these values identical in every model evaluation config.
VAL_SUBSET_SEED = 20260725
TEST_SUBSET_SEED = 20260726

dataset_type = 'Sys100PPPHeatmapDataset'

batch_size_train = 20
batch_size_test = 20
batch_size_val = 20

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
#                 type='LoadSys100PolarSample',
#                 sys100_root=DATA_PATH_TRAIN,
#                 source_is_summary_csv=True,
#                 out_img_size=out_img_size,
#                 undistort=True,
#                 fovs=fovs_wo_fisheyeart,
#                 strict=False,
#                 prefetch=True,
#                 prefetch_dict={
#                     "buffer_size": 1024,
#                     "queue_size": 1000,
#                     "priority": 10,
#                     "num_prefetch_workers": 4,
#                     "wait_till_buffer_has": batch_size_train,
#                     "buffer_fill_pause_threshold": 0.9,
#                     "default_sleep_time": 60,
#                 }
#             ),
#             dict(
#                 type='PPPPolarRandomFlip',
#                 prob=0.5,
#                 direction='horizontal',
#             ),
#             dict(
#                 type='BuildPPPPolarGrid',
#                 grid_shape=polar_grid_shape,
#                 r_min=r_min,
#                 r_max=r_max,
#                 theta_min=theta_min,
#                 theta_max=theta_max,
#             ),
#             dict(type='PPPPackPolarGridInputs'),
#         ]
#     )
# )

num_samples_val = 2000

val_dataloader = dict(
    batch_size=batch_size_val,
    num_workers=4,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_VAL,
        source_is_summary_csv=True,

        target_dataset_size=num_samples_val,
        random_subsampling=True,
        seed=VAL_SUBSET_SEED,

        fovs=fovs_wo_fisheyeart,
        pipeline=[
            dict(
                type='LoadSys100PolarSample',
                sys100_root=DATA_PATH_VAL,
                source_is_summary_csv=True,

                target_dataset_size=num_samples_val,
                random_subsampling=True,
                seed=VAL_SUBSET_SEED,

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
                type='BuildPPPPolarGrid',
                grid_shape=polar_grid_shape,
                r_min=r_min,
                r_max=r_max,
                theta_min=theta_min,
                theta_max=theta_max,
            ),
            dict(type='PPPPackPolarGridInputs'),
        ]
    )
)

num_samples_test = 2000

test_dataloader = dict(
    batch_size=batch_size_test,
    num_workers=4,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_VAL,
        source_is_summary_csv=True,
        target_dataset_size=num_samples_test,
        random_subsampling=True,
        seed=TEST_SUBSET_SEED,

        fovs=fovs_wo_fisheyeart,
        pipeline=[
            dict(
                type='LoadSys100PolarSample',
                sys100_root=DATA_PATH_VAL,
                source_is_summary_csv=True,
                target_dataset_size=num_samples_test,
                random_subsampling=True,
                seed=TEST_SUBSET_SEED,

                out_img_size=out_img_size,
                undistort=True,
                fovs=fovs_wo_fisheyeart,
                strict=False,
                prefetch=False,
            ),
            dict(
                type='BuildPPPPolarGrid',
                grid_shape=polar_grid_shape,
                r_min=r_min,
                r_max=r_max,
                theta_min=theta_min,
                theta_max=theta_max,
            ),
            dict(type='PPPPackPolarGridInputs'),
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
#         lr=1e-2,
#         weight_decay=0.01,
#         betas=(0.9, 0.999)
#     ),
#     clip_grad=dict(
#         max_norm=1.0,
#         norm_type=2
#     )
# )

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1e-4,
#         end_factor=1.0,
#         begin=0,
#         end=7000,
#         by_epoch=False
#     ),
#     dict(
#         type='LinearLR',
#         start_factor=1.0,
#         end_factor=1e-1,
#         begin=7001,
#         end=70000,
#         by_epoch=False
#     ),
#     dict(
#         type='LinearLR',
#         start_factor=1.0,
#         end_factor=1e-1,
#         begin=70001,
#         end=140000,
#         by_epoch=False
#     )
# ]

# train_cfg = dict(
#     type='IterBasedTrainLoop',
#     max_iters=140000,
#     val_interval=5000,
# )

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='PPPLossMMD', out_img_size=polar_grid_shape)
test_evaluator = dict(type='PPPLossMMD', out_img_size=polar_grid_shape)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
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
        type='PPPPolarDenseMetricEvalExportHook',

        max_samples_test=10000,
        max_samples_val=0,

        img_mean=data_rbg_mean,
        img_std=data_rbg_std,

        # Polar prediction domain
        r_min=0.0,
        r_max=170.0,
        theta_min=theta_min,
        theta_max=theta_max,

        # Cartesian evaluation domain
        bev_x_min=0.0,
        bev_x_max=170.0,
        bev_y_min=-85.0,
        bev_y_max=85.0,
        bev_height=256,
        bev_width=256,

        velocity_mode='signed',
        velocity_link='cube',
        rcs_link='cube',
    ),
]
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)