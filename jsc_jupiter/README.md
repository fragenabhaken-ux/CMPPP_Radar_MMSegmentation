Setup MMSeg Framework on JUPITER 
(Achtung: Nacheinander ausführen, wegen den sbatch!) 

cd /e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/jsc_jupiter
mkdir -p logs packages tmp 
bash setup_jupiter.sh 
sbatch build_torchvision_jupiter.sbatch
watch -n 1 'squeue -p booster -u "$USER"' 
source activate_jupiter.sh 
python -m pip install --no-deps "timm>=1.0.20" 
sbatch build_mmcv_full_jupiter.sbatch
watch -n 1 'squeue -p booster -u "$USER"' 
source activate_jupiter.sh 


TEST SUCCESS:

 python - <<'PY'
import torch
import torchvision
import timm
import mmcv
import mmcv.ops
import mmengine

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("mmcv:", mmcv.__version__)
print("mmengine:", mmengine.__version__)
print("Environment complete")
PY