import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import os

# ----------------------------------------------------
# Dateipfade
# ----------------------------------------------------
img_path = "/home/lorenz-linux/repositories/mmsegmentation/data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
label_path = "/home/lorenz-linux/repositories/mmsegmentation/data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_instanceIds.png"

# ----------------------------------------------------
# Laden
# ----------------------------------------------------
img = np.array(Image.open(img_path))
label = np.array(Image.open(label_path))

# ----------------------------------------------------
# 1:1 Kopie der Logik aus PPPHead.extract_centers_from_label
# ----------------------------------------------------
thing_classes = {
    24, 25, 26, 27, 28, 31, 32, 33
}

def extract_centers_from_label_np(instance_mask):
    """
    instance_mask: (H,W) numpy array mit Cityscapes instanceIds
    return: Liste [(cx, cy), ...]
    """
    centers = []
    unique_ids = np.unique(instance_mask)

    for inst_id in unique_ids:
        if inst_id == 0:
            continue

        class_id = inst_id // 1000
        if class_id not in thing_classes:
            continue

        ys, xs = np.where(instance_mask == inst_id)
        if len(xs) == 0:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())
        centers.append((cx, cy))

    return centers


# ----------------------------------------------------
# Zentren berechnen
# ----------------------------------------------------
centers = extract_centers_from_label_np(label)

print("Gefundene Instanz-Zentren:", len(centers))
print("Beispiele:", centers[:10])

# ----------------------------------------------------
# Plot
# ----------------------------------------------------
plt.figure(figsize=(16, 7))

# Originalbild
plt.subplot(1, 2, 1)
plt.title("Original Image + PPP Centers")
plt.imshow(img)
for (cx, cy) in centers:
    plt.scatter(cx, cy, c='red', s=80, marker='x')
plt.axis("off")

# Labelbild
plt.subplot(1, 2, 2)
plt.title("GT InstanceIds + PPP Centers")
plt.imshow(label, cmap='nipy_spectral')
for (cx, cy) in centers:
    plt.scatter(cx, cy, c='white', s=80, marker='x')
plt.axis("off")

plt.tight_layout()

out_file = "ppp_centers_debug.png"
plt.savefig(out_file, dpi=200)
print(f"Gespeichert als: {out_file}")
