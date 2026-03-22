import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import importlib.util
import os

def get_config_path(json_path):
    vis_data_dir = os.path.dirname(json_path)
    return os.path.join(vis_data_dir, "config.py")

def load_config(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg

def get_max_norm(cfg):
    optim_wrapper = getattr(cfg, "optim_wrapper", None)

    if optim_wrapper is None:
        return None

    clip_grad = optim_wrapper.get("clip_grad", None)

    if clip_grad is None:
        return None

    return clip_grad.get("max_norm", None)

# paths = [
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13494254/20260318_212633/vis_data/20260318_212633.json", "LR: 1e-4"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13494258/20260318_212637/vis_data/20260318_212637.json", "LR: 5e-4"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13494259/20260318_212636/vis_data/20260318_212636.json", "LR: 1e-3"),
# ]

# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/loss_plot_lr.png"

# paths = [
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13493437/20260318_202511/vis_data/20260318_202511.json", "Clip norm: Off"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13493442/20260318_202515/vis_data/20260318_202515.json", "Clip norm: 1 (note: This is also the LR baseline)"),
    # ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13493459/20260318_203220/vis_data/20260318_203220.json", "Clip norm: 10"),
# ]

# plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/loss_plot_grad_clip.png"

paths = [
    ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13494254/20260318_212633/vis_data/20260318_212633.json", "LR: 1e-4"),
    ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13494258/20260318_212637/vis_data/20260318_212637.json", "LR: 5e-4"),
    ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13494259/20260318_212636/vis_data/20260318_212636.json", "LR: 1e-3"),
    ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13493437/20260318_202511/vis_data/20260318_202511.json", "Clip norm: Off"),
    ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13493442/20260318_202515/vis_data/20260318_202515.json", "Clip norm: 1 (note: This is also the LR baseline)"),
    ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13493459/20260318_203220/vis_data/20260318_203220.json", "Clip norm: 10"),
    ("/p/scratch/nxtaim-1/huber7/work_dirs/ppp_train_13507298/20260319_140402/vis_data/20260319_140402.json", "SGD, LR: 1e-3"),
]

plot_file = "/p/project1/nxtaim-1/huber7/repos/mmsegmentation_ppp/ppp_analysis/loss_plot_all.png"

plt.figure(figsize=(10, 6))

for path, info in paths:
    try:
        df = pd.read_json(path, lines=True).sort_values("step")
        df = df[df["loss"].notna()].copy()

        # LR
        lr = df["lr"].dropna().iloc[0]

        # --- LOAD CONFIG ---
        config_path = get_config_path(path)
        cfg = load_config(config_path)
        max_norm = get_max_norm(cfg)

        # REMOVE outliers
        q99 = df["loss"].quantile(0.99)
        df = df[df["loss"] < q99]

        # smooth
        y = df["loss"].rolling(20, min_periods=1).mean()

        # label
        label = f"lr={lr:g}, clip={max_norm}, info={info}"

        plt.plot(df["step"], y, label=label)

    except Exception as e:
        print(f"ERROR: {path} -> {e}")


plt.xlabel("step")
plt.ylabel("loss (smoothed)")
plt.title("Loss over step")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(plot_file, dpi=150)
plt.close()

print(f"Saved plot to: {plot_file}")