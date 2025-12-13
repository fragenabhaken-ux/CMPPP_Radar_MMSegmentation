#!/bin/bash
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate seg_clone_2311
python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py
