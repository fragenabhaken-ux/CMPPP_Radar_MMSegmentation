import re
import matplotlib.pyplot as plt

logfile = "work_dirs/fast_scnn_ppp_cityscapes_256x512/20251205_232828/20251205_232828.log"

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
plt.plot(iters, losses, label="loss", linewidth=1.5)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.legend()
plt.show()
