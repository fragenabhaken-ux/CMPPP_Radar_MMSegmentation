# mmseg/engine/hooks/loss_plot_hook.py

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import matplotlib.pyplot as plt
import os
from collections import defaultdict


@HOOKS.register_module()
class LossPlotHook(Hook):
    """
    Records all loss terms from outputs and plots them in stacked subplots.
    """

    def __init__(self, out_file="losses.png", smooth_alpha=0.05):
        self.out_file = out_file
        self.smooth_alpha = smooth_alpha
        self.iters = []
        self.losses = defaultdict(list)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if outputs is None:
            return

        self.iters.append(runner.iter)

        for k, v in outputs.items():
            if "loss" in k:
                self.losses[k].append(v.item())

    def _smooth(self, values, alpha):
        smoothed = []
        cur = None
        for x in values:
            cur = x if cur is None else alpha * x + (1 - alpha) * cur
            smoothed.append(cur)
        return smoothed

    def after_train(self, runner):
        if not self.losses:
            return

        keys = list(self.losses.keys())
        n = len(keys)

        fig, axes = plt.subplots(
            n, 1,
            figsize=(12, 5.0 * n),
            sharex=True
        )

        if n == 1:
            axes = [axes]

        for ax, key in zip(axes, keys):
            raw = self.losses[key]
            smooth = self._smooth(raw, self.smooth_alpha)

            ax.plot(self.iters, raw, alpha=0.35, label="raw")
            ax.plot(self.iters, smooth, linewidth=2, label="smoothed")

            ax.set_ylabel(key)
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend()

        axes[-1].set_xlabel("Iteration")
        fig.suptitle("Training Losses (log scale)", y=0.995)

        out_path = os.path.join(runner.work_dir, self.out_file)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()



# from mmengine.hooks import Hook
# from mmengine.registry import HOOKS
# import matplotlib.pyplot as plt
# import os

# @HOOKS.register_module()
# class LossPlotHook(Hook):
#     """
#     Automatically records training loss and plots it at the end of training.
#     """

#     def __init__(self, out_file="loss.png", smooth_alpha=0.05):
#         self.out_file = out_file
#         self.smooth_alpha = smooth_alpha
#         self.iters = []
#         self.losses = []

#     def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
#         if outputs is None or "loss" not in outputs:
#             return
#         loss = outputs["loss"].item()
#         self.iters.append(runner.iter)
#         self.losses.append(loss)

#     def after_train(self, runner):
#         if not self.losses:
#             return

#         plt.figure(figsize=(10, 4))
#         plt.plot(self.iters, self.losses, label="loss", alpha=0.4)

#         # EWMA smoothing
#         alpha = self.smooth_alpha
#         smoothed = []
#         cur = None
#         for x in self.losses:
#             cur = x if cur is None else alpha * x + (1 - alpha) * cur
#             smoothed.append(cur)

#         plt.plot(self.iters, smoothed, label="loss (smoothed)", linewidth=2)
#         plt.xlabel("Iteration")
#         plt.ylabel("Loss")
#         plt.title("Training Loss")
#         plt.grid(True)
#         plt.legend()

#         out_path = os.path.join(runner.work_dir, self.out_file)
#         plt.savefig(out_path, dpi=150, bbox_inches="tight")
#         plt.close()
