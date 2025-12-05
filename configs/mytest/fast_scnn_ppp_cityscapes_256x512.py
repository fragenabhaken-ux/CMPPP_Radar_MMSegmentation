_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

custom_imports = dict(
    imports=[
        'mmseg.datasets.transforms.load_instance_annotations',
        'mmseg.evaluation.metrics.ppp_metrics',
        'mmseg.engine.hooks.ppp_visualization_hook'
    ],
    allow_failed_imports=False
)

crop_size = (256, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # d8-Setup: kein Downsampling in den letzten beiden Stages,
        # stattdessen Dilated Convs
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
    ),
    decode_head=dict(
        type='PPPHead',
        in_channels=512,
        channels=32,
        num_convs=2,
        num_classes=1,          # 1-Kanal-Intensitätsmap
        loss_ppp=dict(
            type='PPPLoss',
            loss_weight=1.0,
            test_resolution=crop_size
        ),
        in_index=-1,            # letztes Feature (x[3])
        align_corners=False,    # wie in den meisten mmseg-Heads
    ),
    auxiliary_head=None,
    test_cfg=dict(mode='whole'),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadInstanceAnnotations'),
    dict(type='RandomResize', scale=(512, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 1024)),
    dict(type='LoadInstanceAnnotations'),
    dict(type='PackSegInputs')
]

dataset_type = 'CityscapesInstancesDataset'
data_root = '/home/lorenz-linux/repositories/mmsegmentation/data/cityscapes'

train_dataloader = dict(
    batch_size=2,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        ),
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005
    )
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=50000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='PPPMetric')
test_evaluator = dict(type='PPPMetric')

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=1000))


custom_hooks = [
    dict(
        type='PPPVisualizationHook',
        interval=200,
        out_dir='work_dirs/ppp_vis'
    )
]

# custom_hooks = [
#     dict(type='PPPVisualizationHook', interval=500, out_dir='work_dirs/ppp_vis'),
#     dict(type='PPPDebugHook', interval=200, out_dir='work_dirs/debug_ppp')
# ]
