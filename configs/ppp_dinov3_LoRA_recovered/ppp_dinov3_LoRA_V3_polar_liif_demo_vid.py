_base_ = [
    '../_base_/default_runtime.py',
]

import math
import os


custom_imports = dict(
    imports=[
        # Polar BEV model
        'mmseg.models.segmentors.polar_encoder_decoder',
        'mmseg.models.ppp_polar_data_preprocessor',
        'mmseg.models.backbones.lora_timm_multilayer_backbone',
        'mmseg.models.necks.dino_rope_transformer_neck',
        'mmseg.models.decode_heads.ppp_polar_liif_ppp_head',
        'mmseg.models.losses.polar_point_ppp_loss',

        # Demo-video dataset and polar transforms
        'mmseg.datasets.sys100_ppp_heatmap_dataset_demo_vid',
        'mmseg.datasets.transforms.sys100_polar_transforms',

        # Demo-frame export hook
        'mmseg.engine.hooks.ppp_polar_liif_demo_vis_hook',

        # Test metric
        'mmseg.evaluation.metrics.polar_ppp_metric',
    ],
    allow_failed_imports=False,
)


# -------------------------------------------------------------------------
# Image and polar BEV domain
# -------------------------------------------------------------------------

crop_size = (544, 1792)
out_img_size = crop_size

data_rbg_mean = [0.0, 0.0, 0.0]
data_rbg_std = [1.0, 1.0, 1.0]

fovs_wo_fisheyeart = (105.0, 48.4)

r_min = 0.0
r_max = 150.0

theta_half_fov = math.radians(fovs_wo_fisheyeart[0] / 2.0)
theta_min = -theta_half_fov
theta_max = theta_half_fov

latent_grid_shape = (64, 64)
quadrature_grid_shape = (256, 256)

batch_size_test = 6


# -------------------------------------------------------------------------
# Model
# Must match the checkpoint architecture exactly.
# -------------------------------------------------------------------------

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
            '/p/project1/nxtaim-1/huber7/pretrained/'
            'vit_small_patch16_dinov3.lvd1689m/model.safetensors'
        ),
        in_channels=3,
        patch_size=16,
        frozen_base=True,

        lora_r=32,
        lora_alpha=64.0,
        lora_dropout=0.0,
        target_modules=(
            'attn.qkv',
            'attn.proj',
            'mlp.fc1',
            'mlp.fc2',
        ),

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
        decoder_num_layers=4,
        decoder_num_heads=6,
        decoder_mlp_ratio=4.0,
        decoder_attn_dropout=0.0,
        decoder_ffn_dropout=0.0,

        num_latent_convs=4,
        liif_hidden_channels=256,
        liif_num_layers=5,
        use_feature_unfold=True,
        query_chunk_size=32768,

        ppp_weight=1.0,

        # This is the non-velocity architecture from the uploaded 12k config.
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


# -------------------------------------------------------------------------
# Demo video data
# -------------------------------------------------------------------------

DATA_PATH_VAL = os.environ.get(
    'SYS100_DATA_PATH',
    '/p/data1/nxtaim/proprietary/continental/sys100_validation/'
    'SYS100_Validation_Dataset.csv',
)

DEMO_H5_PATH = (
    '/p/data1/nxtaim/proprietary/continental/sys100_validation/'
    '2023.03.02_at_14.16.46/'
    '2023.03.02_at_14.16.46_camera-radar-mi_5374_radargrids_64.h5'
)

DEMO_CENTER_FRAME = 13678
DEMO_HALF_WINDOW = 640

DEMO_FRAME_START = DEMO_CENTER_FRAME - DEMO_HALF_WINDOW
DEMO_FRAME_END = DEMO_CENTER_FRAME + DEMO_HALF_WINDOW
DEMO_FRAME_STRIDE = 4

# Use one of the two alternating camera streams.
DEMO_VIDEO_NAME_CONTAINS = 'EVEN'

DEMO_NUM_FRAMES = (
    (DEMO_FRAME_END - DEMO_FRAME_START) // DEMO_FRAME_STRIDE + 1
)

dataset_type = 'Sys100PPPHeatmapDatasetDemoVid'


# -------------------------------------------------------------------------
# Demo test dataloader
# -------------------------------------------------------------------------

test_dataloader = dict(
    batch_size=batch_size_test,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,

    # Required: preserve temporal order.
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
        round_up=False,
    ),

    dataset=dict(
        type=dataset_type,

        # The demo dataset reads directly from this H5 file.
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
                type='LoadSys100PolarSample',
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


# -------------------------------------------------------------------------
# Test evaluation
# This runs alongside frame export.
# -------------------------------------------------------------------------

test_evaluator = dict(
    type='PolarPPPMetric',

    grid_shape=quadrature_grid_shape,
    r_min=r_min,
    r_max=r_max,
    theta_min=theta_min,
    theta_max=theta_max,

    bev_x_min=0.0,
    bev_x_max=150.0,
    bev_y_min=-85.0,
    bev_y_max=85.0,
    bev_resolution=0.5,

    score_type='loglik',
    sigma_min_m=0.0,
    sigma_max_m=20.0,
    max_iter=30,
    tol_sigma_m=0.25,
    normalize_by_gt=True,
    check_zero=True,

    verbose_logging=False,
    prefix='test',
)

test_cfg = dict(type='TestLoop')


# -------------------------------------------------------------------------
# Hooks
# -------------------------------------------------------------------------

default_hooks = dict(
    timer=dict(type='IterTimerHook'),

    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=False,
    ),

    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(
        type='PPPPolarLIIFDemoVisHook',

        # No training or validation visualization in this config.
        train_interval=0,
        max_samples_train=0,
        max_samples_val=0,

        # Enough to write every selected demo frame.
        max_samples_test=DEMO_NUM_FRAMES,

        out_dir='demo_frames_polar',

        img_mean=data_rbg_mean,
        img_std=data_rbg_std,

        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,

        gamma=0.3,
        point_size=12,

        dpi=100,
        figsize=(18, 18),

        # Essential in distributed inference:
        # every rank writes the subset assigned by its sampler shard.
        write_all_ranks=True,
    ),
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