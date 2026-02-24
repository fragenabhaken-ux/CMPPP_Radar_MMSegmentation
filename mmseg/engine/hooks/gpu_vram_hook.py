from mmengine.hooks import Hook
from mmengine.dist import is_main_process, get_rank
import torch
import socket
import os
import time
import json

from mmseg.registry import HOOKS

@HOOKS.register_module()
class VRAMHook(Hook):
    def __init__(self, interval=50, reset_peak=True, log_all_ranks=False):
        self.interval = interval
        self.reset_peak = reset_peak
        self.log_all_ranks = log_all_ranks

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter % self.interval != 0:
            return

        print( "gpu_vram_hook : ",
        "host", socket.gethostname(),
        "RANK", os.environ.get("RANK"),
        "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
        "SLURM_PROCID", os.environ.get("SLURM_PROCID"),
        "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch.cuda.device_count", torch.cuda.device_count(),
        "current_device", torch.cuda.current_device() if torch.cuda.is_available() else None,
        flush=True
        )

        if (not self.log_all_ranks) and (not is_main_process()):
            print("gpu_vram_hook skipping on rank ", get_rank())
            return

        dev = torch.device("cuda", torch.cuda.current_device())
        alloc = torch.cuda.max_memory_allocated(dev) / (1024**2)
        reserv = torch.cuda.max_memory_reserved(dev) / (1024**2)
        total = torch.cuda.get_device_properties(dev).total_memory / (1024**2)
        free_approx = total - reserv

        # ---- publish to MessageHub (this is what LoggerHook/Visualizer exports) ----
        hub = runner.message_hub

        if self.log_all_ranks:
            r = get_rank()
            # include rank in the *tag* to avoid mixing ranks into one curve
            hub.update_scalar(f'train/vram_alloc_mb_rank{r}', float(alloc), count=1)
            hub.update_scalar(f'train/vram_reserved_mb_rank{r}', float(reserv), count=1)
            hub.update_scalar(f'train/vram_free_mb_rank{r}', float(free_approx), count=1)
        else:
            hub.update_scalar('train/vram_alloc_mb', float(alloc), count=1)
            hub.update_scalar('train/vram_reserved_mb', float(reserv), count=1)
            hub.update_scalar('train/vram_free_mb', float(free_approx), count=1)

        # (optional) still print a human-readable line
        prefix = f"[rank {get_rank()} | {dev}] " if self.log_all_ranks else ""
        runner.logger.info(
            f"{prefix}VRAM peak: allocated={alloc:.0f}MB reserved={reserv:.0f}MB "
            f"total={total:.0f}MB free≈{free_approx:.0f}MB"
        )

        if self.reset_peak:
            torch.cuda.reset_peak_memory_stats(dev)