"""
Hier habe ich die Lernraten-Range-Tests für verschiedene Trainingsläufe geplottet. 
Ich habe sowohl die rohen Verluste als auch eine geglättete Version (Rolling Mean) geplottet. 
Die x-Achse ist logarithmisch skaliert.
Beim Training habe ich dafür die Lernrate exponentiell erhöht.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

paths = [
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13509726/20260319_153153/vis_data/20260319_153153.json", ''), #"AdamW"
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13509738/20260319_153342/vis_data/20260319_153342.json", "SGD"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13510460/20260319_160745/vis_data/20260319_160745.json", "SGD 2"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13512283/20260319_174103/vis_data/20260319_174103.json", "load from 40k"), #AdamW,
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13512851/20260319_181241/vis_data/20260319_181241.json", "load from 40k"), #AdamW, 
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/LRRangeTest_13527872/20260320_120205/vis_data/20260320_120205.json", "load from 40k"), #AdamW,
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13748777_upernet_r50_5k_LRrangetest/20260422_171653/vis_data/20260422_171653.json", "upernet err too short"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13751804_upernet_r50_5k_LRrangetest/20260422_200001/vis_data/20260422_200001.json", "UPerNet (R50, no pretraining)"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13751806_upernet_r50pretrained_5k_LRrangetest/20260422_201246/vis_data/20260422_201246.json", "UPerNet (R50, ImageNet pretrained)"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13751807_resnet34_fcn_5k_LRrangetest/20260422_210156/vis_data/20260422_210156.json", "ResNet-34, FCN"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13753649_unet_5k/20260422_232714/vis_data/20260422_232714.json", "U-Net"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13754310_resnet34_fcn_5k_LRrangetest/20260423_001201/vis_data/20260423_001201.json", "ResNet-34, FCN"),
    # ("/p/scratch/nxtaim-1/huber7/data1/260519_dinov3_5k_LR/13863109_dinov3_LRRangeTest/20260518_224130/vis_data/20260518_224130.json", "DINOv3, ViT-S/16"),
    ("/p/scratch/nxtaim-1/huber7/work_dirs/13865606_dinov3_LRRangeTest/20260519_144334/vis_data/20260519_144334.json", "DINOv3, ViT-S/16"),
]

# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_AdamW.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_SGD.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_SGD_2.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_AdamW_load40k.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_upernet.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_upernet_pretrained.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_resnet34_fcn.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_unet.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_resnet34_fcn_V2.png"
plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_vit_s_16_V2.png"

plt.figure(figsize=(10, 6))

for path, info in paths:
    try:
        df = pd.read_json(path, lines=True).sort_values("step")
        df = df[df["loss"].notna() & df["lr"].notna()].copy()

        # --- smooth ---
        # df["loss_smooth"] = df["loss"].rolling(700, min_periods=10).mean()
        # df["loss_smooth"] = df["loss"].rolling(50, min_periods=10, center=False, win_type="gaussian").mean(std=10)
        df["loss_smooth"] = df["loss"].rolling(10, min_periods=10, center=True, ).mean()
        # df["loss_smooth"] = df["loss"].rolling(200, min_periods=10, center=True, ).mean()
        # df["loss_smooth"] = df["loss"].ewm(span=60, adjust=False).mean()
        

        # --- raw loss (faint) ---
        plt.plot(df["lr"], df["loss"], alpha=0.3, label="Loss")

        # --- smoothed loss (main signal) ---
        plt.plot(df["lr"], df["loss_smooth"], label="Rolling mean", linewidth=2)

    except Exception as e:
        print(f"ERROR: {path} -> {e}")



import matplotlib.ticker as mticker

plt.xscale("log")
ax = plt.gca()

ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=15))
ax.xaxis.set_minor_locator(
    mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=1000)
)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())

plt.ylim(7, 14)

plt.xlabel("Learning rate (log scale)")
plt.ylabel("PPP loss")
plt.title(f"Learning rate range test, {info}")
plt.legend()

ax.grid(True, which="major", axis="x", alpha=0.8)
ax.grid(True, which="minor", axis="x", alpha=0.35)
ax.grid(True, which="major", axis="y", alpha=0.8)

plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()