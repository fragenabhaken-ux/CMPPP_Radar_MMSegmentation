#!/bin/bash
export PYTHONPATH=$HOME/repositories/mmsegmentation:$PYTHONPATH
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate seg_clone_2311
# conda activate seg_marked
# python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py
# python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py --resume
python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512_test.py
