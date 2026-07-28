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
        'mmseg.models.decode_heads.ppp_finals_polar_liif_head',
        'mmseg.models.losses.polar_point_ppp_loss',
        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.sys100_polar_transforms',
        'mmseg.engine.hooks.ppp_polar_liif_wvel_wrcs_demo_vis_hook',
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
r_max = 170.0
theta_half_fov = math.radians(fovs_wo_fisheyeart[0] / 2.0)
theta_min = -theta_half_fov
theta_max = theta_half_fov

latent_grid_shape = (64, 64)
quadrature_grid_shape = (256, 256)

batch_size_train = 14
batch_size_val = 14
batch_size_test = 14
num_samples_val = 112
num_samples_test = 112

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
        checkpoint_path='/e/project1/nxtaim-1/huber7/pretrained/vit_small_patch16_dinov3.lvd1689m/model.safetensors',
        in_channels=3,
        patch_size=16,
        frozen_base=True,
        lora_r=32, # 16
        lora_alpha=64.0, # 32.0
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
        type='PPPfinalsPolarLIIFPPPHead',
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
        ppp_weight=100.0,
        velocity_weight=10.0,
        rcs_weight=1.0,
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
   "SYS100_DATA_PATH",
   "/e/scratch/nxtaim-1/propietary/aumovio/SYS100_Dataset.csv"
)
DATA_PATH_VAL = os.environ.get(
   "SYS100_DATA_PATH",
   "/e/scratch/nxtaim-1/propietary/aumovio/SYS100_Validation_Dataset.csv"
)

# Demo sequence
DEMO_H5_PATH = (
    "/e/scratch/nxtaim-1/propietary/aumovio/sys100_validation/"
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

# train_dataloader = dict(
#     batch_size=batch_size_train,
#     num_workers=4,
#     persistent_workers=True,
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
#                 prefetch_dict=dict(
#                     buffer_size=1024,
#                     queue_size=1000,
#                     priority=10,
#                     num_prefetch_workers=4,
#                     wait_till_buffer_has=batch_size_train,
#                     buffer_fill_pause_threshold=0.9,
#                     default_sleep_time=60,
#                 ),
#             ),
#             dict(type='PPPPolarRandomFlip', prob=0.5, direction='horizontal'),
#             dict(
#                 type='PPPPolarFilterDomain',
#                 r_min=r_min,
#                 r_max=r_max,
#                 theta_min=theta_min,
#                 theta_max=theta_max,
#             ),
#             dict(type='PPPPackPolarInputs'),
#         ],
#     ),
# )

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
                type='LoadSys100PolarSample',
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

num_samples_test = 300

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

val_evaluator = dict(
    type='PolarPPPMetric',
    grid_shape=quadrature_grid_shape,
    r_min=r_min,
    r_max=r_max,
    theta_min=theta_min,
    theta_max=theta_max,

    # Cartesian BEV raster used only by the evaluation metric.
    bev_x_min=0.0,
    bev_x_max=170.0,
    bev_y_min=-150.0,
    bev_y_max=150.0,
    bev_resolution=0.5,

    # Same optimal-sigma score type as the old PPPLossMMD.
    score_type='loglik',
    sigma_min_m=0.0,
    sigma_max_m=20.0,
    max_iter=30,
    tol_sigma_m=0.25,
    normalize_by_gt=True,
    check_zero=True,

    verbose_logging=False,

    # This gives the logged key: ppp/optimal_score
    prefix='ppp',
)

test_evaluator = dict(
    type='PolarPPPMetric',
    grid_shape=quadrature_grid_shape,
    r_min=r_min,
    r_max=r_max,
    theta_min=theta_min,
    theta_max=theta_max,

    bev_x_min=0.0,
    bev_x_max=170.0,
    bev_y_min=-150.0,
    bev_y_max=150.0,
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

visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)
log_processor = dict(by_epoch=False)
default_scope = 'mmseg'

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=2e-4,
#         weight_decay=0.01,
#         betas=(0.9, 0.999)
#     ),
#     clip_grad=dict(
#         max_norm=1.0, # t 2
#         norm_type=2
#     )
# )

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1e-4,
#         end_factor=1,
#         begin=0,
#         end=2200,
#         by_epoch=False
#     ),
#     dict(
#         type='LinearLR',
#         start_factor=1,
#         end_factor=1e-1,
#         begin=2201,
#         end=22000,
#         by_epoch=False
#     )
# ]

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=22000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        save_best='ppp/optimal_score',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(
        type='PPPPolarLIIFWVelWRcsDemoVisHook',

        max_samples_test=DEMO_NUM_FRAMES,
        out_dir='demo_frames_polar_wvel_rcs',
        write_all_ranks=True,

        # Must match the model prediction domain
        r_min=r_min,
        r_max=r_max,
        theta_min=theta_min,
        theta_max=theta_max,

        # Square Cartesian BEV display:
        # forward interval = 170 m
        # lateral interval = 170 m
        bev_y_min=-85.0,
        bev_y_max=85.0,

        # Second-row intensity heatmap
        intensity_gamma=0.3,
        heatmap_cmap='hot',

        # GT and sampled points
        point_size=20.0,
        sample_point_size=20.0,
        point_color='white',
        sample_point_color='white',
        point_edge_color='black',
        point_edge_width=0.45,

        # Velocity
        velocity_mode='signed',
        velocity_grid_is_raw=True,
        velocity_vmin=-11.0,
        velocity_vmax=11.0,
        velocity_cmap='coolwarm',

        # RCS
        rcs_grid_is_raw=True,
        rcs_link='cube',
        rcs_vmin=-60.0,
        rcs_vmax=10.0,
        rcs_cmap='viridis',

        # Reproducible PPP samples
        sampling_seed_offset=1000,

        # Figure
        figsize=(21, 18),
        dpi=100,
    ),
    dict(type='PrefetchStatsHook'),
]