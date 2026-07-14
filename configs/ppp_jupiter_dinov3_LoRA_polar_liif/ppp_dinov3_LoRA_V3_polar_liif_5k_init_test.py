_base_ = [
    '../_base_/default_runtime.py',
]

import os


custom_imports = dict(
    imports=[
        'mmseg.models.ppp_polar_data_preprocessor',

        'mmseg.models.backbones.lora_timm_multilayer_backbone',
        'mmseg.models.necks.dino_rope_transformer_neck',

        'mmseg.models.decode_heads.ppp_polar_liif_ppp_head',
        'mmseg.models.losses.polar_point_ppp_loss',

        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.sys100_polar_transforms',

        'mmseg.engine.hooks.ppp_polar_liif_vis_hook',
        'mmseg.engine.hooks.prefetch_stats_hook',
    ],
    allow_failed_imports=False,
)


work_dir = (
    '/e/scratch/nxtaim-1/huber7/work_dirs/'
    'ppp_dinov3_LoRA_V3_polar_liif_5k_init_test'
)

crop_size = (544, 1792)
out_img_size = (544, 1792)

data_rbg_mean = [0.0, 0.0, 0.0]
data_rbg_std = [1.0, 1.0, 1.0]
fovs_wo_fisheyeart = (105, 48.4)

r_min = 0.0
r_max = 150.0

import math

camera_hfov_deg = float(fovs_wo_fisheyeart[0])

theta_half_fov = math.radians(camera_hfov_deg / 2.0)
theta_min = -theta_half_fov
theta_max = +theta_half_fov

# theta_min = -1.5707963267948966
# theta_max = 1.5707963267948966

polar_grid_shape = (64, 64)

data_preprocessor = dict(
    type='PolarPPPDataPreProcessor',
    mean=data_rbg_mean,
    std=data_rbg_std,
    pad_val=0,
    size=crop_size,
)


model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='LoRAMultiLayerTIMMBackbone',
        model_name='vit_small_patch16_dinov3',
        pretrained=False,
        checkpoint_path=(
            '/e/project1/nxtaim-1/huber7/pretrained/'
            'vit_small_patch16_dinov3.lvd1689m/model.safetensors'
        ),
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
        type='PolarLIIFPPPHead',
        in_channels=384,
        in_index=0,
        channels=64,
        num_classes=1,

        polar_grid_shape=polar_grid_shape,
        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,

        num_latent_convs=2,
        liif_hidden_channels=128,
        liif_num_layers=3,

        ppp_weight=1.0,
        velocity_weight=0.0,
        velocity_mode='signed',

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


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

DATA_PATH_TRAIN = os.environ.get(
    'SYS100_DATA_PATH',
    '/e/scratch/nxtaim-1/propietary/aumovio/SYS100_Dataset.csv',
)

dataset_type = 'Sys100PPPHeatmapDataset'

batch_size_train = 28

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
                type='LoadSys100PolarSample',
                sys100_root=DATA_PATH_TRAIN,
                source_is_summary_csv=True,
                out_img_size=out_img_size,
                undistort=True,
                fovs=fovs_wo_fisheyeart,
                strict=False,

                prefetch=True,
                prefetch_dict=dict(
                    buffer_size=1024,
                    queue_size=1000,
                    priority=10,
                    num_prefetch_workers=4,
                    wait_till_buffer_has=batch_size_train,
                    buffer_fill_pause_threshold=0.9,
                    default_sleep_time=60,
                ),
            ),
            dict(
                type='PPPPolarRandomFlip',
                prob=0.5,
                direction='horizontal',
            ),
            dict(
                type='PPPPolarFilterDomain',
                r_min=r_min,
                r_max=r_max,
                theta_min=theta_min,
                theta_max=theta_max,
            ),
            dict(type='PPPPackPolarInputs'),
        ],
    ),
)


# No validation before the polar evaluator exists.
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None


# ---------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    ),
    clip_grad=dict(
        max_norm=1.0,
        norm_type=2,
    ),
)

param_scheduler = [
    # 1e-7 -> 3e-4
    dict(
        type='LinearLR',
        start_factor=1.0 / 3000.0,
        end_factor=1.0,
        begin=0,
        end=800,
        by_epoch=False,
    ),

    # 3e-4 -> 3e-5
    dict(
        type='LinearLR',
        start_factor=1.0,
        end_factor=0.1,
        begin=800,
        end=5000,
        by_epoch=False,
    ),
]

# ---------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=5000,
    val_interval=1000000,
)

val_cfg = None
test_cfg = None

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=20,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=5,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(
        type='PPPPolarLIIFVisHook',
        train_interval=100,
        max_samples_train=1,
        max_samples_val=0,
        max_samples_test=0,
        img_mean=data_rbg_mean,
        img_std=data_rbg_std,
        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,
        gamma=0.3,
        point_size=12,
    ),
    dict(type='PrefetchStatsHook'),
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

log_processor = dict(by_epoch=False)

default_scope = 'mmseg'