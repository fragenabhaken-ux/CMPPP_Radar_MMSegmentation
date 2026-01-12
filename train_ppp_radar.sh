#!/bin/bash
export PYTHONPATH=$HOME/repositories/mmsegmentation:$PYTHONPATH
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate seg_clone_2311
# conda activate seg_marked
# python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512_cropbug.py
# python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py
# python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py --resume
# python tools/test.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py work_dirs/fast_scnn_ppp_sys100heatmaps_256x512/best_ppp_count_abs_error_iter_100.pth
# python tools/test.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py work_dirs/fast_scnn_ppp_sys100heatmaps_256x512/iter_80000.pth
# python tools/train.py configs/mytest/fast_scnn_ppp_sys100heatmaps_512x1536.py
# python tools/test.py configs/mytest/fast_scnn_ppp_sys100heatmaps_256x512.py work_dirs/fast_scnn_ppp_sys100heatmaps_256x512/iter_28000.pth
# python tools/train.py configs/mytest/fast_scnn_ppp_debug.py
# python tools/test.py configs/mytest/fast_scnn_ppp_debug.py work_dirs/fast_scnn_ppp_debug/best_ppp_count_abs_error_iter_32000.pth
# python tools/train.py configs/mytest/fast_scnn_mppp_256x512_crop.py
# python tools/train.py configs/mytest/fast_scnn_mppp_544x1792_nocrop.py
# python tools/train.py configs/mytest/fast_scnn_mppp_408x1344_nocrop.py
# python tools/test.py configs/mytest/fast_scnn_mppp_256x512_crop.py work_dirs/fast_scnn_mppp_256x512_crop/best_ppp_count_abs_error_iter_36000.pth
python tools/train.py configs/mytest/fast_scnn_mppp_256x512_crop_normfix.py