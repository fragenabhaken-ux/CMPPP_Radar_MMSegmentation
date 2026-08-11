# CMPPP Radar MMSegmentation

This repository contains the code developed for the Master's thesis on camera-conditioned automotive radar point-cloud generation using conditional marked Poisson point processes (CMPPPs).

All files newly introduced for the Master's thesis use the prefix `ppp`. The original Repo can be found [here](https://github.com/open-mmlab/mmsegmentation).

## Relevant Branches

The main branches used during development are:

- **Working branch on JUPITER:**
  [final_jupiter_runs](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/final_jupiter_runs)

- **Working branch for local WSL evaluation of generated metric samples:**  
  [juwels_stage](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/juwels_stage)

- **Cleaned branch for further use by Jonas/Tobi:**  
  [CMPPP_Radar_MMSeg](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg)

The following notes refer primarily to the cleaned branch [CMPPP_Radar_MMSeg](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg).

## Maintained Models

Only the Transformer-based architectures have been maintained in the cleaned version of the repository, including the corresponding code for:

- training,
- metric-sample generation,
- metric evaluation, and
- demo-video sample generation.

The maintained model configurations correspond to:

- `ViT-Enc-Dec-LIIF-BEV`
- `ViT-Enc-BEV` without a pretrained DINOv3 backbone
- `ViT-Enc-BEV` with a pretrained DINOv3 backbone
- `ViT-Enc-Img`

## Model Configurations

The relevant configuration directories are:

- **ViT-Enc-Dec-LIIF-BEV**  
  [ppp_finals_TrafoEncDec](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/configs/ppp_finals_TrafoEncDec)

- **ViT-Enc-BEV without pretrained DINOv3 backbone**  
  [ppp_finals_TrafoEnc_2_BEVpolDom](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/configs/ppp_finals_TrafoEnc_2_BEVpolDom)

- **ViT-Enc-BEV with pretrained DINOv3 backbone**  
  [ppp_finals_TrafoEnc_BEVpolDom](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/configs/ppp_finals_TrafoEnc_BEVpolDom)

- **ViT-Enc-Img**  
  [ppp_finals_TrafoEnc_CamDom](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/configs/ppp_finals_TrafoEnc_CamDom)

## JUPITER / Slurm Jobs

The Slurm batch scripts used on JSC JUPITER are located in the folder [jsc_jupiter](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/CMPPP_Radar_MMSeg/).

Setting up the Python environment on JUPITER is nontrivial. The required setup procedure is documented [here](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/CMPPP_Radar_MMSeg/jsc_jupiter/README.md).

> **Note:** According to Jonas, the Slurm jobs used for environment setup can alternatively be executed using a GPU on an entry node.

Slurm jobs for:
- [training](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/CMPPP_Radar_MMSeg/jsc_jupiter/train_ppp.sbatch)
- [metric sample generation](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/CMPPP_Radar_MMSeg/jsc_jupiter/test_ppp_metric_eval.sbatch)
- [demo-vid sample generation](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/CMPPP_Radar_MMSeg/jsc_jupiter/test_ppp_demo_vid.sbatch)

## Model Checkpoints

The checkpoints of the final model runs have been backed up on JSC at:

```text
/p/data1/nxtaim/users/huber7/CMPPP_Radar_MMSeg
```

## Metric Evaluation

Metric evaluation uses a two-stage workflow:

1. Run inference/evaluation on JSC and generate the metric samples.
2. Download the generated metric samples from scratch to the local machine.

### BEV Models
3. Evaluate the samples using the local WSL repository on the `juwels_stage` branch by first entering the paths to the metrics samples in the [config](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/juwels_stage/ppp_metric_eval/metric_eval_config.py) and then running the [jupyter notebook](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/juwels_stage/ppp_metric_eval/compare_models.ipynb).

The corresponding metric-evaluation code is located on the juwels_stage branch in:

https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/juwels_stage/ppp_metric_eval

### Image-Domain Models

3. Evaluate the samples using the local WSL repository on the `juwels_stage` branch by first entering the paths to the metrics samples in the [config](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/juwels_stage/camera_domain_metric_eval/metric_eval_config.py) and then running the [jupyter notebook](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/juwels_stage/camera_domain_metric_eval/compare_camera_models.ipynb).

The corresponding metric-evaluation code is located on the juwels_stage branch in:

https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/juwels_stage/camera_domain_metric_eval

This evaluation code essentially performs the same procedure but operates directly in the image domain.

## Demo Video Generation

Demo-video generation follows a workflow similar to metric evaluation:

1. Generate the individual demo-video frames on JSC by running the slurm job [test_ppp_demo_vid.sbath](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/blob/CMPPP_Radar_MMSeg/jsc_jupiter/test_ppp_demo_vid.sbatch).
2. Download the generated frames to the local machine from scratch.
3. Generate the final video locally using WSL.

Only the folder containing the actual generated frames needs to be downloaded.

After downloading the frames, navigate to the MMSeg repository in WSL on the branch [juwels_stage](https://github.com/fragenabhaken-ux/CMPPP_Radar_MMSegmentation/tree/juwels_stage) and execute:

```bash
bash ppp_make_demo_vid_local_finalize.sh "<path-to-frame-directory>"
```

For example:

```bash
bash ppp_make_demo_vid_local_finalize.sh \
"/mnt/c/Users/uik04723/Desktop/demo_frames_polar_dense_six_bev/intensity_with_gt"
```

The script combines the generated frames into the final demo video.
