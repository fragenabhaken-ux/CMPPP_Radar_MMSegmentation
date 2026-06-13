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

 