# FCN + ResNet18 – simple config for 4GB GPU
# Cityscapes 256×512

_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

crop_size = (256, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
)

# ---------------- MODEL ----------------
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 2, 1),
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=128,
        num_convs=1,
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),
    auxiliary_head=None,
    test_cfg=dict(mode='whole')    # <<< IMPORTANT FIX
)

# ---------------- DATALOADER ----------------
train_dataloader = dict(batch_size=1, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = dict(batch_size=1, num_workers=2)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# ---------------- OPTIMIZER ----------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
)

# ---------------- SCHEDULE ----------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')    # <<< IMPORTANT FIX

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000))


