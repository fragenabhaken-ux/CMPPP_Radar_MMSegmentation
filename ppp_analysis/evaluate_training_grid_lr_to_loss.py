"""
Hier habe ich die Lernraten-Range-Tests für verschiedene Trainingsläufe geplottet. 
Ich habe sowohl die rohen Verluste als auch eine geglättete Version (Rolling Mean) geplottet. 
Die x-Achse ist logarithmisch skaliert.
Beim Training habe ich dafür die Lernrate exponentiell erhöht.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"Iter\(train\)\s*\[\s*(?P<step>\d+)/(?P<max_step>\d+)\].*?"
    r"\blr:\s*(?P<lr>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
    r".*?"
    r"(?<!\.)\bloss:\s*(?P<loss>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)

OPTIONAL_PATTERNS = {
    "loss_ppp": re.compile(r"\bdecode\.loss_ppp:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I),
    "loss_depth": re.compile(r"\bdecode\.loss_depth:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I),
    "loss_velocity": re.compile(r"\bdecode\.loss_velocity:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I),
    "grad_norm": re.compile(r"\bgrad_norm:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I),
}


def read_mmengine_log(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue

            row = {
                "step": int(m.group("step")),
                "max_step": int(m.group("max_step")),
                "lr": float(m.group("lr")),
                "loss": float(m.group("loss")),
            }

            for col, pat in OPTIONAL_PATTERNS.items():
                mm = pat.search(line)
                row[col] = float(mm.group(1)) if mm else None

            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values("step").drop_duplicates("step", keep="last")
    df = df[df["loss"].notna() & df["lr"].notna()].copy()

    return df

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
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13865606_dinov3_LRRangeTest/20260519_144334/vis_data/20260519_144334.json", "DINOv3, ViT-S/16"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13878705_dinov3_LRRangeTest/20260522_163612/vis_data/20260522_163612.json", "DINOv3, ViT-S/16"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13879418_dinov3_arch_adjust_RangeTest/20260522_232924/vis_data/20260522_232924.json", "DINOv3, ViT-S/16"),
    # ("/p/scratch/nxtaim-1/huber7/data1/260529_dinov3_LoRA/13900059_ppp_dinov3_LoRA_rope_big_neck_LRrangetest/20260529_181549/vis_data/20260529_181549.json", "DINOv3, LoRA"),
    # ("/p/scratch/nxtaim-1/huber7/data1/260529_dinov3_LoRA/13904112_ppp_dinov3_LoRA_x2cap_rope_big_neck_LRrangetest/20260531_173840/20260531_173840.log", "DINOv3, LoRA x2cap"),
    # ("/p/scratch/nxtaim-1/huber7/data1/260529_dinov3_LoRA/13904115_ppp_dinov3_mid_LoRA_rope_big_neck_LRrangetest/20260531_174013/20260531_174013.log", "DINOv3, mid LoRA"),
    # ("/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13908049_ppp_dinov3_rope_big_neck_noCrop_5k_LRRangeTest/20260601_181047/20260601_181047.log", "DINOv3, DINOv3 noCrop"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13909791_ppp_dinov3_LoRA_rope_big_neck_noCrop_5k_LRRangeTest/20260602_003031/20260602_003031.log", "DINOv3, LoRA noCrop"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13909769_ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_5k_LRRangeTest/20260601_231457/20260601_231457.log", "DINOv3, multitask_head noCrop"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13909769_ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_5k_LRRangeTest/20260601_231457/20260601_231457.log", "DINOv3, LoRA noCrop"),
    # ("/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13914801_ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_5k_LRRangeTest/20260602_230737/20260602_230737.log", "DINOv3, multitask_two_depth_head noCrop"),
    # ("/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13917240_ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_5k_LRRangeTest/20260603_114353/20260603_114353.log", "DINOv3, mid LoRA noCrop"),
    # ("/p/scratch/nxtaim-1/huber7/data1/260601_dinov3_noCrop/13917268_ppp_dinov3_big_LoRA_rope_big_neck_noCrop_5k_LRRangeTest/20260603_115605/20260603_115605.log", "DINOv3, big LoRA noCrop"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/13939976_ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_LRRangeTest/20260606_015237/20260606_015237.log", "DINOv3, V2"),
    # ("/e/scratch/nxtaim-1/huber7/work_dirs/858748_ppp_unet_5k_LRrangetest/20260613_213709/20260613_213709.log", "U-Net, Jupiter"),
    ("/e/scratch/nxtaim-1/huber7/work_dirs/881695_ppp_dinov3_LoRA_V3_5k_LRrangetest/20260615_164049/20260615_164049.log", "Trafo V3, Jupiter"),
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
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_vit_s_16_V3.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_lora.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_lora_x2cap.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_mid_lora.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_noCrop.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_lora_noCrop.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_multitask_head_noCrop.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_LoRA_noCrop.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_multitask_two_depth_head_noCrop.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_mid_LoRA_noCrop.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_big_LoRA_noCrop.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_dinov3_V2.png"
# plot_file = "/e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/jupiter_lr_range_plot_unet.png"
plot_file = "/e/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/jupiter_lr_range_plot_trafo_v3.png"

plt.figure(figsize=(10, 6))

for path, info in paths:
    try:
        df = read_mmengine_log(path)

        if df.empty:
            print(f"WARNING: no training loss rows found in {path}")
            continue

        df["loss_smooth"] = (
            df["loss"]
            .rolling(10, min_periods=10, center=True)
            .mean()
        )

        # --- smooth ---
        # df["loss_smooth"] = df["loss"].rolling(700, min_periods=10).mean()
        # df["loss_smooth"] = df["loss"].rolling(50, min_periods=10, center=False, win_type="gaussian").mean(std=10)
        # df["loss_smooth"] = df["loss"].rolling(10, min_periods=10, center=True, ).mean()
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

plt.ylim(10, 400)

plt.xlabel("Learning rate (log scale)")
plt.ylabel("loss (PPP + depth)")
plt.title(f"Learning rate range test, {info}")
plt.legend()

ax.grid(True, which="major", axis="x", alpha=0.8)
ax.grid(True, which="minor", axis="x", alpha=0.35)
ax.grid(True, which="major", axis="y", alpha=0.8)

plt.tight_layout()
plt.savefig(plot_file, dpi=150)
plt.close()