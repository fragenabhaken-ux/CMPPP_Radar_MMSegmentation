#!/bin/bash
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate seg
python tools/train.py configs/mytest/fast_scnn_ppp_cityscapes_256x512.py
