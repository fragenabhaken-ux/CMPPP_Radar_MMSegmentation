#!/bin/bash
#SBATCH --job-name=debug_config_import
#SBATCH --account=nxtaim-1
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.." || exit 1
mkdir -p jsc/logs

source jsc/activate_jupiter.sh

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1

CONFIG="configs/ppp/ppp_dinov3_rope_big_neck_40k_dbwa1.py"

echo "=== Environment ==="
which python
python -V

python - <<'PY'
import torch, torchvision, timm, mmcv, mmengine, mmseg
print("torch:", torch.__version__, torch.version.cuda, torch.__file__)
print("torchvision:", torchvision.__version__, torchvision.__file__)
print("timm:", timm.__version__, timm.__file__)
print("mmcv:", mmcv.__version__, mmcv.__file__)
print("mmengine:", mmengine.__version__, mmengine.__file__)
print("mmseg:", mmseg.__version__, mmseg.__file__)

from mmcv.ops import sigmoid_focal_loss, point_sample
print("mmcv ops OK")
PY

echo
echo "=== Config.fromfile signature ==="
python - <<'PY'
from mmengine.config import Config
import inspect
print(inspect.signature(Config.fromfile))
PY

echo
echo "=== Parse config WITHOUT custom imports if supported ==="
python - <<'PY'
from mmengine.config import Config
import inspect

cfg_path = "configs/ppp/ppp_dinov3_rope_big_neck_40k_dbwa1.py"
sig = inspect.signature(Config.fromfile)

if "import_custom_modules" in sig.parameters:
    cfg = Config.fromfile(cfg_path, import_custom_modules=False)
else:
    cfg = Config.fromfile(cfg_path)

print("config parsed")
print("custom_imports:", cfg.get("custom_imports", None))
PY

echo
echo "=== Isolate custom imports in subprocesses ==="
python - <<'PY'
import subprocess
import sys

mods = [
    "mmseg.models.backbones.frozen_timm_backbone",
    "mmseg.models.necks.dino_rope_transformer_neck",
    "mmseg.models.decode_heads.ppp_mark_vit_patch_rearrange_head",
    "mmseg.datasets.sys100_ppp_heatmap_dataset",
    "mmseg.datasets.transforms.sys100_ppp_transforms",
    "mmseg.evaluation.metrics.ppp_metrics_loss_mmd",
    "mmseg.engine.hooks.ppp_mark_vis_hook_light",
    "mmseg.engine.hooks.prefetch_stats_hook",
]

for m in mods:
    print("\n--- testing import:", m, "---", flush=True)
    code = f"""
import faulthandler
faulthandler.enable()
print("START {m}", flush=True)
import {m}
print("OK {m}", flush=True)
"""
    r = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    print("returncode:", r.returncode)
    print("stdout:")
    print(r.stdout)
    print("stderr:")
    print(r.stderr)

    if r.returncode != 0:
        print("FAILED IMPORT:", m)
        raise SystemExit(r.returncode)

print("\nAll custom imports OK.")
PY

echo
echo "=== Full Config.fromfile ==="
python -X faulthandler - <<'PY'
from mmengine.config import Config

cfg = Config.fromfile("configs/ppp/ppp_dinov3_rope_big_neck_40k_dbwa1.py")
print("config loaded")
print("model type:", cfg.model.get("type", None))
print("custom imports OK")
PY

echo
echo "Debug config import finished."