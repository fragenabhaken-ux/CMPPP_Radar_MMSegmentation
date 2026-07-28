_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        # New DINOv3 architecture components
        'mmseg.models.backbones.lora_timm_multilayer_backbone',
        'mmseg.models.necks.dino_rope_transformer_neck',
        'mmseg.models.decode_heads.ppp_finals_vit_patch_rearrange_head',

        # Existing project components
        'mmseg.datasets.sys100_ppp_heatmap_dataset',
        'mmseg.datasets.transforms.sys100_ppp_transforms',
        'mmseg.evaluation.metrics.ppp_metrics_loss_mmd',
        'mmseg.engine.hooks.prefetch_stats_hook',
        'mmseg.engine.hooks.ppp_finals_CamDom_to_BEVDom_metric_hook',
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
        type='PPPfinalsViTPatchRearrangeHead',
        in_channels=384,
        in_index=0,
        channels=64,
        num_classes=4,

        patch_size=(16, 16),
        refine_channels=128,
        num_refine_convs=5,
        pre_factor=4,

        local_refine_blocks=2,
        local_refine_norm='gn',

        ppp_weight=1.0,
        depth_weight=10.0,
        velocity_weight=10.0,
        rcs_weight=1.0,
        velocity_mode='signed',
        depth_loss_type='log',
        loss_ppp=dict(
            type='PPPLoss',
            out_img_size=out_img_size,
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

batch_size_train = 14
batch_size_test = 14
batch_size_val = 14


# ----------------------------

num_samples_val = 300

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
                type='LoadSys100Sample',
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
                type='BuildPPPHeatmap',
                thres=0.05,
                rel_x_centre=0.491,
                rel_y_centre=0.4370,
            ),
            dict(type='PPPPackSegInputs'),
        ]
    )
)

num_samples_test = 300

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
                type='LoadSys100Sample',
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
                type='BuildPPPHeatmap',
                thres=0.05,
                rel_x_centre=0.491,
                rel_y_centre=0.4370,
            ),
            dict(type='PPPPackSegInputs'),
        ]
    )
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='PPPLossMMD', out_img_size=out_img_size)
test_evaluator = dict(type='PPPLossMMD', out_img_size=out_img_size)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
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
        type='PPPCameraToCartesianBEVMetricEvalExportHook',

        max_samples_test=10000,
        max_samples_val=0,

        img_mean=data_rbg_mean,
        img_std=data_rbg_std,

        # Used only as fallback. The per-sample metainfo["fovs"]
        # value is preferred.
        h_fov_deg=105.0,
        principal_x_rel=0.491,
        lateral_sign=1.0,

        # Must be identical to the dense-BEV export hook.
        bev_x_min=0.0,
        bev_x_max=170.0,
        bev_y_min=-85.0,
        bev_y_max=85.0,
        bev_height=256,
        bev_width=256,

        rcs_link='cube',

        output_subdir='metric_eval_data_dense_bev_cartesian',
    ),
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)