#!/bin/bash
export PYTHONPATH=$HOME/repositories/mmsegmentation:$PYTHONPATH
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate seg_clone_2311
# conda activate seg_marked
python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py
# python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py --resume
# python tools/test.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py work_dirs/fast_scnn_ppp_sys100heatmaps_256x512/best_ppp_count_abs_error_iter_100.pth
# python tools/test.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py work_dirs/fast_scnn_ppp_sys100heatmaps_256x512/iter_80000.pth

