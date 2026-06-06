#!/bin/bash
#SBATCH --job-name=dinov3_head_test
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --export=ALL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.." || exit 1

source jsc/modules.sh
source jsc/activate.sh

mkdir -p logs

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-unset}"
echo "SLURM_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
import torch

# Important: imports register modules in MMSeg registry
import mmseg.models.backbones.frozen_timm_backbone
import mmseg.models.necks.dino_transformer_neck
import mmseg.models.decode_heads.ppp_mark_vit_upsample_head

from mmseg.registry import MODELS

ckpt = "/p/project1/nxtaim-1/huber7/pretrained/vit_small_patch16_dinov3.lvd1689m/model.safetensors"

backbone_cfg = dict(
    type="FrozenTIMMBackbone",
    model_name="vit_small_patch16_dinov3",
    pretrained=False,
    checkpoint_path=ckpt,
    in_channels=3,
    frozen=True,
    patch_size=16,
)

neck_cfg = dict(
    type="DINOTransformerNeck",
    in_channels=384,
    embed_dims=384,
    in_index=0,
    num_layers=6,
    num_heads=6,
    mlp_ratio=4.0,
    dropout=0.0,
    attn_dropout=0.0,
    norm_cfg=dict(type="LN"),
)

head_cfg = dict(
    type="PPPMarkViTUpsampleHead",
    in_channels=384,
    in_index=0,
    channels=256,
    num_convs=1,

    # 16x32 -> 32x64 -> 64x128 -> 128x256 -> 256x512
    num_upsample_layers=4,
    upsample_channels=(256, 128, 64, 64),

    dropout_ratio=0.1,
    num_classes=3,

    ppp_weight=1,
    depth_weight=0,
    velocity_weight=0,

    loss_ppp=dict(
        type="PPPLoss",
        out_img_size=(544, 1792),
        normalize_by_gt_count=True,
    ),

    norm_cfg=dict(type="BN", requires_grad=True),
    align_corners=False,
)

print("Building backbone...")
backbone = MODELS.build(backbone_cfg)

print("Building neck...")
neck = MODELS.build(neck_cfg)

print("Building head...")
head = MODELS.build(head_cfg)

device = "cuda" if torch.cuda.is_available() else "cpu"

backbone = backbone.to(device)
neck = neck.to(device)
head = head.to(device)

backbone.eval()
neck.train()
head.train()

print("Backbone, neck and head built successfully.")
print("Device:", device)

n_backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
n_neck_trainable = sum(p.numel() for p in neck.parameters() if p.requires_grad)
n_head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)

print("Backbone trainable parameters:", n_backbone_trainable)
print("Neck trainable parameters:", n_neck_trainable)
print("Head trainable parameters:", n_head_trainable)

x = torch.randn(1, 3, 256, 512, device=device)

with torch.no_grad():
    feats = backbone(x)

neck_out = neck(feats)
pred = head(neck_out)

print("Backbone output:")
for i, feat in enumerate(feats):
    print(
        f"  {i}: shape={tuple(feat.shape)}, "
        f"requires_grad={feat.requires_grad}, "
        f"dtype={feat.dtype}, device={feat.device}"
    )

print("Neck output:")
for i, feat in enumerate(neck_out):
    print(
        f"  {i}: shape={tuple(feat.shape)}, "
        f"requires_grad={feat.requires_grad}, "
        f"dtype={feat.dtype}, device={feat.device}"
    )

print(
    "Head prediction:",
    tuple(pred.shape),
    "requires_grad=", pred.requires_grad,
    "dtype=", pred.dtype,
    "device=", pred.device,
)

expected_backbone_shape = (1, 384, 16, 32)
expected_neck_shape = (1, 384, 16, 32)
expected_pred_shape = (1, 3, 256, 512)

assert tuple(feats[0].shape) == expected_backbone_shape, (
    f"Expected backbone output {expected_backbone_shape}, got {tuple(feats[0].shape)}"
)
assert tuple(neck_out[0].shape) == expected_neck_shape, (
    f"Expected neck output {expected_neck_shape}, got {tuple(neck_out[0].shape)}"
)
assert tuple(pred.shape) == expected_pred_shape, (
    f"Expected prediction {expected_pred_shape}, got {tuple(pred.shape)}"
)

assert n_backbone_trainable == 0, (
    f"Backbone is not frozen, trainable params={n_backbone_trainable}"
)
assert n_neck_trainable > 0, "Neck has no trainable parameters."
assert n_head_trainable > 0, "Head has no trainable parameters."

loss = pred.mean()
loss.backward()

neck_grad_count = sum(
    1 for p in neck.parameters()
    if p.requires_grad and p.grad is not None
)

head_grad_count = sum(
    1 for p in head.parameters()
    if p.requires_grad and p.grad is not None
)

backbone_grad_count = sum(
    1 for p in backbone.parameters()
    if p.grad is not None
)

print("Neck parameters with grad:", neck_grad_count)
print("Head parameters with grad:", head_grad_count)
print("Backbone parameters with grad:", backbone_grad_count)

assert neck_grad_count > 0, "No gradients in neck."
assert head_grad_count > 0, "No gradients in head."
assert backbone_grad_count == 0, "Backbone received gradients although it should be frozen."

print("Backbone + Neck + Head test finished successfully.")
PY