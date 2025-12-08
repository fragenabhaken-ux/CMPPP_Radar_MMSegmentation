import re
import matplotlib.pyplot as plt

logfile = "work_dirs/fast_scnn_ppp_cityscapes_256x512/20251208_214835/20251208_214835.log"

iters = []
losses = []

pattern = re.compile(r"Iter\(train\)\s*\[\s*(\d+)/\d+\]\s.*loss:\s*([-\d\.]+)")

with open(logfile, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            it = int(m.group(1))
            loss = float(m.group(2))
            iters.append(it)
            losses.append(loss)

plt.figure(figsize=(12,5))
plt.plot(iters, losses, label="loss", linewidth=1.2, alpha=0.5)

# --- Smooth the loss curve (EWMA)
alpha = 0.01
smoothed = []
current = None

for x in losses:
    if current is None:
        current = x
    else:
        current = alpha * x + (1 - alpha) * current
    smoothed.append(current)

plt.plot(iters, smoothed, label="loss (smoothed)", linewidth=2)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.legend()
plt.show()
