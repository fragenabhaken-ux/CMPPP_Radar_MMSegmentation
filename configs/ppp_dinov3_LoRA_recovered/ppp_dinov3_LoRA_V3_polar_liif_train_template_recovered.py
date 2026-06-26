_base_ = [
    '../_base_/default_runtime.py',
]

import math
import os

custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.polar_encoder_decoder',
        'mmseg.models.ppp_polar_data_preprocessor',
        'mmseg.models.backbones.lora_timm_multilayer_backbone',
        'mmseg.models.necks.dino_rope_transformer_neck',
        'mmseg.models.decode_heads.ppp_polar_liif_ppp_head',
        'mmseg.models.losses.polar_point_ppp_loss',
        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.sys100_polar_transforms',
        'mmseg.engine.hooks.ppp_polar_liif_vis_hook',
        'mmseg.engine.hooks.prefetch_stats_hook',
        'mmseg.evaluation.metrics.polar_ppp_metric',
    ],
    allow_failed_imports=False,
)

crop_size = (544, 1792)
out_img_size = crop_size

data_rbg_mean = [0.0, 0.0, 0.0]
data_rbg_std = [1.0, 1.0, 1.0]
fovs_wo_fisheyeart = (105.0, 48.4)

r_min = 0.0
r_max = 104.0
theta_half_fov = math.radians(fovs_wo_fisheyeart[0] / 2.0)
theta_min = -theta_half_fov
theta_max = theta_half_fov

latent_grid_shape = (64, 64)
quadrature_grid_shape = (256, 256)

# Last known working depth was one block. Increase only after a dedicated
# memory/time smoke test; no unverified depth is baked into this recovery.
decoder_num_layers = 1

batch_size_train = 25
batch_size_val = 25
batch_size_test = 25
num_samples_val = 100
num_samples_test = 100

data_preprocessor = dict(
    type='PolarPPPDataPreProcessor',
    mean=data_rbg_mean,
    std=data_rbg_std,
    pad_val=0,
    size=crop_size,
)

model = dict(
    type='PolarEncoderDecoder',
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
        latent_grid_shape=latent_grid_shape,
        quadrature_grid_shape=quadrature_grid_shape,
        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,
        decoder_embed_dims=384,
        decoder_num_layers=decoder_num_layers,
        decoder_num_heads=6,
        decoder_mlp_ratio=4.0,
        decoder_attn_dropout=0.0,
        decoder_ffn_dropout=0.0,
        num_latent_convs=2,
        liif_hidden_channels=128,
        liif_num_layers=3,
        use_feature_unfold=False,
        query_chunk_size=32768,
        ppp_weight=1.0,
        velocity_weight=0.0,
        velocity_mode='signed',
        loss_ppp=dict(
            type='PolarPointPPPLoss',
            grid_shape=quadrature_grid_shape,
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

DATA_PATH_TRAIN = os.environ.get(
    'SYS100_DATA_PATH',
    '/e/scratch/nxtaim-1/propietary/aumovio/SYS100_Dataset.csv',
)
DATA_PATH_VAL = os.environ.get('SYS100_VAL_DATA_PATH', DATA_PATH_TRAIN)
DATA_PATH_TEST = os.environ.get('SYS100_TEST_DATA_PATH', DATA_PATH_VAL)
dataset_type = 'Sys100PPPHeatmapDataset'

train_dataloader = dict(
    batch_size=batch_size_train,
    num_workers=4,
    persistent_workers=True,
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
            dict(type='PPPPolarRandomFlip', prob=0.5, direction='horizontal'),
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

val_dataloader = dict(
    batch_size=batch_size_val,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_VAL,
        source_is_summary_csv=True,
        target_dataset_size=num_samples_val,
        fovs=fovs_wo_fisheyeart,
        pipeline=[
            dict(
                type='LoadSys100PolarSample',
                sys100_root=DATA_PATH_VAL,
                source_is_summary_csv=True,
                target_dataset_size=num_samples_val,
                random_subsampling=False,
                out_img_size=out_img_size,
                undistort=True,
                fovs=fovs_wo_fisheyeart,
                strict=False,
                prefetch=False,
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

test_dataloader = dict(
    batch_size=batch_size_test,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        sys100_root=DATA_PATH_TEST,
        source_is_summary_csv=True,
        target_dataset_size=num_samples_test,
        fovs=fovs_wo_fisheyeart,
        pipeline=[
            dict(
                type='LoadSys100PolarSample',
                sys100_root=DATA_PATH_TEST,
                source_is_summary_csv=True,
                target_dataset_size=num_samples_test,
                random_subsampling=False,
                out_img_size=out_img_size,
                undistort=True,
                fovs=fovs_wo_fisheyeart,
                strict=False,
                prefetch=False,
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

val_evaluator = dict(
    type='PolarPPPMetric',
    grid_shape=quadrature_grid_shape,
    r_min=r_min,
    r_max=r_max,
    theta_min=theta_min,
    theta_max=theta_max,
    normalize_nll_by_gt_count=True,
    prefix='val',
)

test_evaluator = dict(
    type='PolarPPPMetric',
    grid_shape=quadrature_grid_shape,
    r_min=r_min,
    r_max=r_max,
    theta_min=theta_min,
    theta_max=theta_max,
    normalize_nll_by_gt_count=True,
    prefix='test',
)

custom_hooks = [
    dict(
        type='PPPPolarLIIFVisHook',
        train_interval=100,
        max_samples_train=1,
        max_samples_val=10,
        max_samples_test=1,
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
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)
log_processor = dict(by_epoch=False)
default_scope = 'mmseg'

work_dir = '/e/scratch/nxtaim-1/huber7/work_dirs/ppp_polar_liif_recovered_train'

# This is a template only. The learning rate is intentionally not presented as
# tuned: run the architecture smoke test first, then an LR-range test.
base_lr = 1e-4
max_iters = 2400

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=400,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.1,
        by_epoch=False,
        begin=400,
        end=max_iters,
    ),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=max_iters)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # No save_best: the currently recovered metric is only a grid diagnostic.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1200, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
