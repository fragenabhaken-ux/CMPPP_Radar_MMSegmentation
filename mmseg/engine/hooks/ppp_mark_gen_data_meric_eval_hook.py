"""
PPPMarkGenDataMetricEvalHook
============================
Lightweight hook for generating an offline metric-evaluation dataset.

Each GPU / distributed rank writes **one** HDF5 file containing all samples
that rank processed.  Running on 4 GPUs therefore produces 4 files:

    {log_dir}/metric_eval_data/{phase}/rank00.h5
    {log_dir}/metric_eval_data/{phase}/rank01.h5
    {log_dir}/metric_eval_data/{phase}/rank02.h5
    {log_dir}/metric_eval_data/{phase}/rank03.h5

Inside each file, every sample is stored as a named group:

    /sample_000000/img           – (H, W, 3) uint8
    /sample_000000/pred_lambda   – (H, W)    float32   λ(x) / (H*W)
    /sample_000000/gt_support    – (H, W)    float32

The index in the group name is the per-rank write counter (0-based), not a
global sample index.

No PNG figures, no JSON metadata, no simulated point clouds.

Usage in config
---------------
    dict(
        type='PPPMarkGenDataMetricEvalHook',
        max_samples_test=500,
        img_mean=data_rbg_mean,
        img_std=data_rbg_std,
    )
"""

import os

import h5py
import numpy as np
import torch
from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS


@HOOKS.register_module()
class PPPMarkGenDataMetricEvalHook(Hook):
    """
    Accumulates (img, pred_lambda_mass, gt_support) into one HDF5 file per
    rank.  Designed for test / val runs; train is a no-op.

    Parameters
    ----------
    max_samples_test : int
        Stop after this many samples *per rank* during test.
    max_samples_val : int
        Stop after this many samples *per rank* during val (0 = disabled).
    img_mean : list[float] | None
        Per-channel mean used during normalisation (length 3).
    img_std : list[float] | None
        Per-channel std used during normalisation.
    """

    def __init__(
        self,
        max_samples_test: int = 500,
        max_samples_val: int = 0,
        img_mean=None,
        img_std=None,
    ):
        self.max_samples = {
            "val":  int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.img_mean = (
            np.asarray(img_mean, dtype=np.float32)[:, None, None]
            if img_mean is not None else None
        )
        self.img_std = (
            np.asarray(img_std, dtype=np.float32)[:, None, None]
            if img_std is not None else None
        )

        # Per-rank write counter
        self._written: dict[str, int] = {"val": 0, "test": 0}

        # One open HDF5 file handle per phase; None means not yet opened.
        self._h5: dict[str, "h5py.File | None"] = {"val": None, "test": None}

    # ------------------------------------------------------------------
    # Epoch-level open / close
    # ------------------------------------------------------------------

    def before_val_epoch(self, runner) -> None:
        self._close_h5("val")
        self._written["val"] = 0

    def before_test_epoch(self, runner) -> None:
        self._close_h5("test")
        self._written["test"] = 0

    def after_val_epoch(self, runner) -> None:
        self._close_h5("val")

    def after_test_epoch(self, runner) -> None:
        self._close_h5("test")

    # ------------------------------------------------------------------
    # Iteration callbacks
    # ------------------------------------------------------------------

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._maybe_run(runner, data_batch, phase="val")

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._maybe_run(runner, data_batch, phase="test")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _maybe_run(self, runner, data_batch, phase: str) -> None:
        if not isinstance(data_batch, dict):
            return
        if self.max_samples[phase] <= 0:
            return
        if self._written[phase] >= self.max_samples[phase]:
            return

        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        was_training = model.training
        model.eval()

        proc = model.data_preprocessor(data_batch)
        inputs  = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            preds = model.decode_head(model.extract_feat(inputs))

        remaining = self.max_samples[phase] - self._written[phase]
        n_save = min(remaining, preds.size(0))

        h5f = self._get_h5(runner, phase)

        for i in range(n_save):
            self._append_sample(
                h5f=h5f,
                img_tensor=inputs[i],
                pred=preds[i],
                sample=data_samples[i],
                local_index=self._written[phase] + i,
            )

        h5f.flush()
        self._written[phase] += n_save

        if was_training:
            model.train()

    # ------------------------------------------------------------------
    # HDF5 file management
    # ------------------------------------------------------------------

    def _get_out_dir(self, runner, phase: str) -> str:
        d = os.path.join(runner.log_dir, "metric_eval_data", phase)
        os.makedirs(d, exist_ok=True)
        return d

    def _get_h5(self, runner, phase: str) -> "h5py.File":
        """Return the open HDF5 file for this rank/phase, opening it on first call."""
        if self._h5[phase] is None:
            out_dir = self._get_out_dir(runner, phase)
            rank = get_rank()
            filepath = os.path.join(out_dir, f"rank{rank:02d}.h5")
            self._h5[phase] = h5py.File(filepath, "w")
        return self._h5[phase]

    def _close_h5(self, phase: str) -> None:
        if self._h5[phase] is not None:
            try:
                self._h5[phase].close()
            except Exception:
                pass
            self._h5[phase] = None

    # ------------------------------------------------------------------
    # Per-sample writing
    # ------------------------------------------------------------------

    def _append_sample(
        self,
        h5f: "h5py.File",
        img_tensor: torch.Tensor,
        pred: torch.Tensor,
        sample,
        local_index: int,
    ) -> None:
        # ---- decode prediction ----
        pred_lambda_theta = torch.exp(pred[0])

        # ---- GT support mask ----
        gt = sample.gt_sem_seg.data.to(pred.device)
        gt_support = gt[0].cpu().numpy().astype(np.float32)

        # ---- normalise λ to mass per pixel ----
        meta = sample.metainfo
        ori_shape = meta.get("ori_shape")
        if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
            H, W = int(ori_shape[0]), int(ori_shape[1])
        else:
            H, W = pred_lambda_theta.shape[-2:]

        pred_lambda_mass = (
            pred_lambda_theta.detach().cpu().numpy() / float(H * W)
        ).astype(np.float32)

        # ---- de-normalise image ----
        img = self._denorm_img(img_tensor.detach().cpu().numpy())

        # ---- write group into HDF5 ----
        grp = h5f.create_group(f"sample_{local_index:06d}")
        grp.create_dataset("img",         data=img,              compression="gzip", compression_opts=4)
        grp.create_dataset("pred_lambda", data=pred_lambda_mass, compression="gzip", compression_opts=4)
        grp.create_dataset("gt_support",  data=gt_support,       compression="gzip", compression_opts=4)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _denorm_img(self, x: np.ndarray) -> np.ndarray:
        """(C, H, W) float array → (H, W, C) uint8."""
        x = x.astype(np.float32)
        if self.img_mean is not None and self.img_std is not None:
            x = x * self.img_std + self.img_mean
            x = (x + 1.0) * 127.5
        else:
            if x.min() >= -1.01 and x.max() <= 1.01:
                x = (x + 1.0) * 127.5
            elif x.max() <= 1.01:
                x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return np.transpose(x, (1, 2, 0))
