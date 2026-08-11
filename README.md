# CMPPP Radar MMSegmentation

This repository contains the code developed for the Master's thesis on camera-conditioned automotive radar point-cloud generation using conditional marked Poisson point processes (CMPPPs).

All files newly introduced for the Master's thesis use the prefix `ppp`.

## Relevant Branches

The main branches used during development are:

- **Working branch on JUPITER:**  
  https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/final_jupiter_runs

- **Working branch for local WSL evaluation of generated metric samples:**  
  https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/juwels_stage

- **Cleaned branch for further use by Jonas/Tobi:**  
  https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/juwels_stage

The following notes refer primarily to the cleaned branch.

## Maintained Models

Only the Transformer-based architectures have been maintained in the cleaned version of the repository, including the corresponding code for:

- training,
- evaluation and metric-sample generation,
- metric evaluation, and
- demo-video generation.

The maintained model configurations correspond to:

- `ViT-Enc-Dec-LIIF-BEV`
- `ViT-Enc-BEV` without a pretrained DINOv3 backbone
- `ViT-Enc-BEV` with a pretrained DINOv3 backbone
- `ViT-Enc-Img`

## Model Configurations

The relevant configuration directories are:

- **ViT-Enc-Dec-LIIF-BEV**  
  https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/configs/ppp_finals_TrafoEncDec

- **ViT-Enc-BEV without pretrained DINOv3 backbone**  
  https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/configs/ppp_finals_TrafoEnc_2_BEVpolDom

- **ViT-Enc-BEV with pretrained DINOv3 backbone**  
  https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/configs/ppp_finals_TrafoEnc_BEVpolDom

- **ViT-Enc-Img**  
  https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/configs/ppp_finals_TrafoEnc_CamDom

## JUPITER / Slurm Jobs

The Slurm batch scripts used on JSC JUPITER are located in:

https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/jsc_jupiter

Setting up the Python environment on JUPITER is nontrivial. The required setup procedure is documented here:

https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/CMPPP_Radar_MMSeg/jsc_jupiter/README.md

> **Note:** According to Jonas, the Slurm jobs used for environment setup can alternatively be executed using a GPU on an entry node.

## Model Checkpoints

The checkpoints of the final model runs have been backed up on JSC at:

```text
/p/data1/nxtaim/users/huber7/CMPPP_Radar_MMSeg
