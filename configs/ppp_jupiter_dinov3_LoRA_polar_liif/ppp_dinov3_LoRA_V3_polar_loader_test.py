_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.ppp_polar_data_preprocessor',

        # Backbone / neck
        'mmseg.models.backbones.lora_timm_multilayer_backbone',
        'mmseg.models.necks.dino_rope_transformer_neck',

        # New polar model parts
        'mmseg.models.decode_heads.ppp_polar_liif_ppp_head',
        'mmseg.models.losses.polar_point_ppp_loss',

        # Dataset / transforms
        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.sys100_polar_transforms',
        'mmseg.engine.hooks.ppp_polar_liif_vis_hook',
    ],
    allow_failed_imports=False,
)

import os

work_dir = '/e/scratch/nxtaim-1/huber7/work_dirs/polar_liif_loader_test'

crop_size = (544, 1792)
out_img_size = (544, 1792)

data_rbg_mean = [0.0, 0.0, 0.0]
data_rbg_std = [1.0, 1.0, 1.0]
fovs_wo_fisheyeart = (105, 48.4)

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
        type='PolarLIIFPPPHead',
        in_channels=384,
        in_index=0,
        channels=64,
        num_classes=1,

        polar_grid_shape=(64, 64),
        r_min=0.0,
        r_max=104.0,
        theta_min=-1.5707963267948966,
        theta_max=1.5707963267948966,

        num_latent_convs=2,
        liif_hidden_channels=128,
        liif_num_layers=3,

        ppp_weight=1.0,
        velocity_weight=0.0,
        velocity_mode='signed',

        loss_ppp=dict(
            type='PolarPointPPPLoss',
            grid_shape=(64, 64),
            r_min=0.0,
            r_max=104.0,
            theta_min=-1.5707963267948966,
            theta_max=1.5707963267948966,
            normalize_by_gt_count=True,
        ),
    ),

    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ----------------------------
# Data
# ----------------------------

DATA_PATH_TRAIN = os.environ.get(
    "SYS100_DATA_PATH",
    "/e/scratch/nxtaim-1/propietary/aumovio/SYS100_Dataset.csv",
)

dataset_type = 'Sys100PPPHeatmapDataset'

batch_size_train = 2

train_dataloader = dict(
    batch_size=batch_size_train,
    num_workers=0,
    pin_memory=False,
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
                prefetch=False,
            ),
            dict(
                type='PPPPolarRandomFlip',
                prob=0.5,
                direction='horizontal',
            ),
            dict(type='PPPPackPolarInputs'),
        ],
    ),
)

# ----------------------------
# Optimizer
# ----------------------------

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    ),
    clip_grad=dict(
        max_norm=1.0,
        norm_type=2,
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        end_factor=1.0,
        begin=0,
        end=1,
        by_epoch=False,
    ),
]

# ----------------------------
# Runtime
# ----------------------------

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1,
    val_interval=1000000,
)

val_cfg = None
test_cfg = None

val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1,
        max_keep_ckpts=1,
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
        r_min=0.0,
        r_max=104.0,
        theta_min=-1.5707963267948966,
        theta_max=1.5707963267948966,
        gamma=0.3,
        point_size=12,
    ),
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
)

log_processor = dict(
    by_epoch=False,
)

default_scope = 'mmseg'