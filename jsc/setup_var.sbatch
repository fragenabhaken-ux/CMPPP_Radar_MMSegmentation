#!/bin/bash
#SBATCH --job-name=ppp_setup
#SBATCH --account=nxtaim-1
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#set -euo pipefail

cd $SLURM_SUBMIT_DIR
#source activate.sh

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*' )"
export MASTER_PORT=7010

echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"

export CUDA_VISIBLE_DEVICES=0,1,2,3

source setup.sh

python - <<'EOF'

import os
import traceback
import torch
from torch.autograd import Function

_orig_apply = Function.apply

def debug_apply(*args, **kwargs):
    if kwargs:
        print("\n🔥 FOUND kwargs passed into Function.apply 🔥", flush=True)
        print("kwargs =", kwargs, flush=True)
        print("\n=== FULL STACK TRACE ===", flush=True)

        stack = traceback.format_stack()
        for line in stack:
            print(line.rstrip(), flush=True)

        # also write to file (HPC-safe)
        out_path = os.path.abspath("jsc/logs/apply_kwargs_trace_full.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("kwargs = " + str(kwargs) + "\n\n")
            f.write("".join(stack))

        print(f"\nWrote full stack to: {out_path}\n", flush=True)

        raise RuntimeError("STOP HERE")

    return _orig_apply(*args)

Function.apply = debug_apply



print('Smoke-Test 0 – Import-Kette (minimal)')
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())

import mmengine
print("mmengine:", mmengine.__version__)

import mmcv
print("mmcv:", mmcv.__version__)

import mmcv.ops
print("mmcv.ops OK")

# from mmcv.ops import _ext
# print("_ext OK")

import mmseg
print("mmseg:", mmseg.__version__)

print('Smoke-Test 1 – Registry & Custom Repo')
from mmseg.registry import MODELS, DATASETS
print("Registered models:", len(MODELS))
print("Registered datasets:", len(DATASETS))

# optional: check your custom classes
print("PPPMarkHead in registry:",
      "PPPMarkHead" in MODELS)

print("Smoke-Test 2 – CUDA execution")

x = torch.randn(1024, 1024, device="cuda")
y = torch.matmul(x, x)
torch.cuda.synchronize()

print("CUDA matmul OK, mean:", y.mean().item())

print("Smoke-Test 3 – mmcv.ops kernel")

from mmcv.ops import roi_align

feat = torch.randn(1, 256, 32, 32, device="cuda")
rois = torch.tensor([[0, 4, 4, 20, 20]], dtype=torch.float32, device="cuda")

out = roi_align(feat, rois, output_size=7, spatial_scale=1.0)
torch.cuda.synchronize()

print("roi_align OK:", out.shape)

print("Smoke-Test 4 – Build model from config")

from mmengine.config import Config
from mmseg.registry import MODELS

cfg = Config(dict(
    type="PPPMarkHead",   # your custom head
    in_channels=64,
    num_classes=1,
))

model = MODELS.build(cfg)
print("Model built:", type(model).__name__)

print("Smoke-Test 6 – Forward pass (CUDA)")

model = model.cuda()
dummy_feat = dummy_feat.cuda()

with torch.no_grad():
    out = model(dummy_feat)
torch.cuda.synchronize()

print("CUDA forward OK")

print("Smoke-Test 7 – Loss & backward")

model.train()
dummy_feat = torch.randn(2, 64, 32, 32, device="cuda", requires_grad=True)

out = model(dummy_feat)

# adapt this to your actual output format
loss = out.mean()
loss.backward()

print("Backward OK, grad mean:", dummy_feat.grad.abs().mean().item())

EOF

