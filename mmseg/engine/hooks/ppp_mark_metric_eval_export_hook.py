"""
PPPMarkMetricEvalExportHook
===========================
Lightweight hook for generating an offline metric-evaluation dataset.

Each GPU / distributed rank writes **one** HDF5 file containing all samples
that rank processed. Running on 4 GPUs therefore produces 4 files:

    {log_dir}/metric_eval_data/{phase}/rank00.h5
    {log_dir}/metric_eval_data/{phase}/rank01.h5
    {log_dir}/metric_eval_data/{phase}/rank02.h5
    {log_dir}/metric_eval_data/{phase}/rank03.h5

Inside each file, every sample is stored as a named group:

    /sample_000000/img             – (H, W, 3) uint8
    /sample_000000/pred_lambda     – (H, W)    float32   lambda(x) / (H*W)
    /sample_000000/pred_depth      – (H, W)    float32   exp(raw depth channel)
    /sample_000000/pred_velocity   – (H, W)    float32   decoded velocity
    /sample_000000/gt_support      – (H, W)    float32
    /sample_000000/gt_depth        – (H, W)    float32
    /sample_000000/gt_velocity     – (H, W)    float32

The index in the group name is the per-rank write counter (0-based), not a
global sample index. Metadata such as img_path and shapes are written as HDF5
attributes on the sample group.

No PNG figures, no JSON metadata, no simulated point clouds.

Usage in config
---------------
    custom_hooks = [
        dict(
            type='PPPMarkMetricEvalExportHook',
            max_samples_test=500,
            max_samples_val=0,
            img_mean=data_rbg_mean,
            img_std=data_rbg_std,
        )
    ]
"""

import os
from typing import Any

import h5py
import numpy as np
import torch
from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS


@HOOKS.register_module()
class PPPMarkMetricEvalExportHook(Hook):
    """
    Export compact HDF5 files for offline metric evaluation.

    The hook assumes the PPP mark model predicts either three channels:

        pred[0] = log intensity / log lambda parameter
        pred[1] = log depth / transformed via exp(pred[1])
        pred[2] = raw velocity

    or four channels for the two-depth head:

        pred[0] = log intensity / log lambda parameter
        pred[1] = raw_d0
        pred[2] = raw_delta
        pred[3] = raw velocity

    Velocity decoding follows model.decode_head.velocity_mode:

        signed:
            pred_velocity = raw_velocity ** 3
            gt_velocity   = gt[2]

        absolute:
            pred_velocity = abs(raw_velocity)
            gt_velocity   = abs(gt[2])

    The ground-truth segmentation tensor contains three channels:

        gt[0] = support mask
        gt[1] = depth
        gt[2] = velocity

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
    require_three_channels : bool
        If True, raise a clear error when pred or gt has fewer than three
        channels. This is useful for avoiding silent incomplete exports.
    """

    def __init__(
        self,
        max_samples_test: int = 500,
        max_samples_val: int = 0,
        img_mean=None,
        img_std=None,
        require_three_channels: bool = True,
    ):
        self.max_samples = {
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }
        self.require_three_channels = bool(require_three_channels)

        self.img_mean = (
            np.asarray(img_mean, dtype=np.float32)[:, None, None]
            if img_mean is not None else None
        )
        self.img_std = (
            np.asarray(img_std, dtype=np.float32)[:, None, None]
            if img_std is not None else None
        )

        # Per-rank write counter.
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

    def after_val_epoch(self, runner, metrics=None, **kwargs) -> None:
        self._close_h5("val")

    def after_test_epoch(self, runner, metrics=None, **kwargs) -> None:
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

        velocity_mode = getattr(model.decode_head, "velocity_mode", "signed")
        if velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                f"velocity_mode must be 'signed' or 'absolute', got {velocity_mode!r}."
            )

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            preds = model.decode_head(model.extract_feat(inputs))

        remaining = self.max_samples[phase] - self._written[phase]
        n_save = min(remaining, preds.size(0))

        h5f = self._get_h5(runner, phase)
        h5f.attrs["velocity_mode"] = velocity_mode

        for i in range(n_save):
            self._append_sample(
                h5f=h5f,
                img_tensor=inputs[i],
                pred=preds[i],
                sample=data_samples[i],
                local_index=self._written[phase] + i,
                velocity_mode=velocity_mode,
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
            self._h5[phase].attrs["phase"] = phase
            self._h5[phase].attrs["rank"] = int(rank)
            self._h5[phase].attrs["format"] = "ppp_mark_metric_eval_v1"
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
        velocity_mode: str,
    ) -> None:
        if self.require_three_channels:
            self._assert_three_channels(pred, name="prediction")

        # ---- decode prediction ----
        # Channel convention:
        #   pred[0] = log intensity / log lambda parameter
        #   pred[1] -> depth = exp(pred[1])
        #
        # Normal heads:
        #   pred[2] -> raw velocity
        #
        # Two-depth heads:
        #   pred[1] -> raw_d0
        #   pred[2] -> raw_delta
        #   pred[3] -> raw velocity
        pred_lambda_theta = torch.exp(pred[0])
        pred_depth = torch.exp(pred[1]).detach().cpu().numpy().astype(np.float32)

        if pred.shape[0] == 4:
            raw_velocity = pred[3]
            pred_velocity_channel = 3
        else:
            raw_velocity = pred[2]
            pred_velocity_channel = 2

        if velocity_mode == "signed":
            pred_velocity_tensor = raw_velocity ** 3
        elif velocity_mode == "absolute":
            pred_velocity_tensor = torch.abs(raw_velocity)
        else:
            raise RuntimeError(f"Unsupported velocity_mode={velocity_mode!r}")

        pred_velocity = (
            pred_velocity_tensor.detach().cpu().numpy().astype(np.float32)
        )

        # ---- GT channels ----
        gt = sample.gt_sem_seg.data.to(pred.device)
        if self.require_three_channels:
            self._assert_three_channels(gt, name="ground truth")

        gt_support = gt[0].detach().cpu().numpy().astype(np.float32)
        gt_depth = gt[1].detach().cpu().numpy().astype(np.float32)

        if velocity_mode == "signed":
            gt_velocity_tensor = gt[2]
        elif velocity_mode == "absolute":
            gt_velocity_tensor = torch.abs(gt[2])
        else:
            raise RuntimeError(f"Unsupported velocity_mode={velocity_mode!r}")

        gt_velocity = (
            gt_velocity_tensor.detach().cpu().numpy().astype(np.float32)
        )

        # ---- normalise lambda to mass per pixel ----
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

        self._write_attrs(grp, meta=meta, local_index=local_index)
        grp.attrs["velocity_mode"] = velocity_mode
        grp.attrs["pred_velocity_channel"] = int(pred_velocity_channel)

        self._create_dataset(grp, "img", img)
        self._create_dataset(grp, "pred_lambda", pred_lambda_mass)
        self._create_dataset(grp, "pred_depth", pred_depth)
        self._create_dataset(grp, "pred_velocity", pred_velocity)
        self._create_dataset(grp, "gt_support", gt_support)
        self._create_dataset(grp, "gt_depth", gt_depth)
        self._create_dataset(grp, "gt_velocity", gt_velocity)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_three_channels(x: torch.Tensor, name: str) -> None:
        if x.ndim < 3 or x.shape[0] < 3:
            raise ValueError(
                f"Expected {name} tensor with at least 3 channels, "
                f"but got shape {tuple(x.shape)}."
            )

    @staticmethod
    def _create_dataset(grp: "h5py.Group", name: str, data: np.ndarray) -> None:
        grp.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=4,
        )

    def _write_attrs(self, grp: "h5py.Group", meta: dict[str, Any], local_index: int) -> None:
        grp.attrs["local_index"] = int(local_index)
        grp.attrs["rank"] = int(get_rank())

        # Keep attributes simple and robust. h5py attributes do not like every
        # Python object type, so we stringify complex metadata.
        for key in [
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ]:
            if key in meta and meta[key] is not None:
                grp.attrs[key] = self._attr_value(meta[key])

    @staticmethod
    def _attr_value(value: Any):
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
            return value
        if isinstance(value, (tuple, list)) and all(
            isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))
            for v in value
        ):
            return np.asarray(value)
        return str(value)

    def _denorm_img(self, x: np.ndarray) -> np.ndarray:
        """(C, H, W) float array -> (H, W, C) uint8."""
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