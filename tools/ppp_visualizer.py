#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------
#  Extract Centers from Cityscapes InstanceIds
# -------------------------------------------------------------
# THING_CLASSES = {24, 25, 26, 27, 28, 31, 32, 33}
THING_CLASSES = {26}


def extract_centers_from_label(instance_map):
    """
    instance_map: (H,W) numpy array with Cityscapes instanceIds
    Returns: list of (cx,cy)
    """
    centers = []

    unique_ids = np.unique(instance_map)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue

        class_id = inst_id // 1000
        if class_id not in THING_CLASSES:
            continue

        ys, xs = np.where(instance_map == inst_id)
        if len(xs) == 0:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())
        centers.append((cx, cy))

    return centers


# -------------------------------------------------------------
#  Heatmap Normalization
# -------------------------------------------------------------
def normalize_heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    maxv = heatmap.max()
    if maxv > 0:
        heatmap = heatmap / maxv
    return heatmap


# -------------------------------------------------------------
#  Visualization
# -------------------------------------------------------------
def visualize(image, centers_gt, heatmap):
    """
    image: RGB image (H,W,3)
    centers_gt: list[(cx,cy),...]
    heatmap: (H,W) float np array
    """

    heatmap_norm = normalize_heatmap(heatmap)

    plt.figure(figsize=(16, 7))

    # ---------------- Original + GT Centers ----------------
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    for (cx, cy) in centers_gt:
        plt.scatter(cx, cy,
                    c='red',
                    s=120,
                    marker='x',
                    linewidths=2)
    plt.title("Original Image + GT Centers")
    plt.axis("off")

    # ---------------- PPP Heatmap ----------------
    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.4)
    plt.imshow(heatmap_norm, cmap='jet', alpha=0.6)
    for (cx, cy) in centers_gt:
        plt.scatter(cx, cy,
                    c='white',
                    s=120,
                    marker='x',
                    linewidths=2)
    plt.title("PPP Heatmap + GT Centers")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to Cityscapes leftImg8bit image")
    parser.add_argument("--gt", required=True, help="GT instanceIds PNG")
    parser.add_argument("--logits", required=True, help="Saved PPP heatmap logits (.pt)")
    args = parser.parse_args()

    # ----- Load Image -----
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ----- Load GT instanceIds -----
    gt_raw = cv2.imread(args.gt, cv2.IMREAD_UNCHANGED)
    if gt_raw is None:
        raise FileNotFoundError(args.gt)

    centers_gt = extract_centers_from_label(gt_raw)

    # ----- Load PPP heatmap logits -----
    logits = torch.load(args.logits)

    if isinstance(logits, dict) and "logits" in logits:
        logits = logits["logits"]

    logits = logits.squeeze().cpu().float().numpy()  # (H,W)

    # ----- Visualize -----
    visualize(img, centers_gt, logits)


if __name__ == "__main__":
    main()
