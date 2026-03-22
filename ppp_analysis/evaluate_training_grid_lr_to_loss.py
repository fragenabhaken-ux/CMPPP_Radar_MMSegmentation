import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

paths = [
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13509726/20260319_153153/vis_data/20260319_153153.json", ''), #"AdamW"
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13509738/20260319_153342/vis_data/20260319_153342.json", "SGD"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13510460/20260319_160745/vis_data/20260319_160745.json", "SGD 2"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13512283/20260319_174103/vis_data/20260319_174103.json", "load from 40k"), #AdamW,
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13512851/20260319_181241/vis_data/20260319_181241.json", "load from 40k"), #AdamW, 
    ("/p/scratch/nxtaim-1/huber7/work_dirs/LRRangeTest_13527872/20260320_120205/vis_data/20260320_120205.json", "load from 40k"), #AdamW, 
]

# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_AdamW.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_SGD.png"
# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_SGD_2.png"
plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/lr_range_plot_AdamW_load40k.png"

plt.figure(figsize=(10, 6))

for path, info in paths:
    try:
        df = pd.read_json(path, lines=True).sort_values("step")
        df = df[df["loss"].notna() & df["lr"].notna()].copy()

        # --- smooth ---
        # df["loss_smooth"] = df["loss"].rolling(700, min_periods=10).mean()
        # df["loss_smooth"] = df["loss"].rolling(50, min_periods=10, center=False, win_type="gaussian").mean(std=10)
        # df["loss_smooth"] = df["loss"].rolling(10, min_periods=10, center=True, ).mean()
        df["loss_smooth"] = df["loss"].rolling(500, min_periods=10, center=True, ).mean()
        # df["loss_smooth"] = df["loss"].ewm(span=60, adjust=False).mean()
        

        # --- raw loss (faint) ---
        plt.plot(df["lr"], df["loss"], alpha=0.3, label="Loss")

        # --- smoothed loss (main signal) ---
        plt.plot(df["lr"], df["loss_smooth"], label="Rolling mean (window=10)", linewidth=2)

    except Exception as e:
        print(f"ERROR: {path} -> {e}")

# --- LOG SCALE ---
plt.xscale("log")

# --- LIMIT Y-AXIS ---
# plt.ylim(9, 14)
plt.ylim(7.8, 8.2)

plt.xlabel("Learning rate (log scale)")
plt.ylabel("PPP loss")
plt.title(f"Learning rate range test {info}")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()

plt.savefig(plot_file, dpi=150)
plt.close()

print(f"Saved plot to: {plot_file}")