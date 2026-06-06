"""
PPPMarkMetricEvalExportTwoDepthHook
===================================

Lightweight hook for generating an offline metric-evaluation dataset
for the two-depth PPP mark head.

Each GPU / distributed rank writes one HDF5 file containing all samples
that rank processed. Running on 4 GPUs therefore produces 4 files:

    {log_dir}/metric_eval_data_twodepth/{phase}/rank00.h5
    {log_dir}/metric_eval_data_twodepth/{phase}/rank01.h5
    {log_dir}/metric_eval_data_twodepth/{phase}/rank02.h5
    {log_dir}/metric_eval_data_twodepth/{phase}/rank03.h5

Inside each file, every sample is stored as a named group:

    /sample_000000/img                       – (H, W, 3) uint8
    /sample_000000/pred_lambda              – (H, W)    float32   lambda(x) / (H*W)

    /sample_000000/pred_depth_d0            – (H, W)    float32   exp(raw_d0)
    /sample_000000/pred_depth_near          – (H, W)    float32   clamp(d0 - delta, min=depth_eps)
    /sample_000000/pred_depth_far           – (H, W)    float32   d0 + delta
    /sample_000000/pred_depth_delta         – (H, W)    float32   softplus(raw_delta)

    /sample_000000/pred_depth_best          – (H, W)    float32   near/far depth closer to GT depth
    /sample_000000/pred_depth_best_abs_error– (H, W)    float32   min(abs(near-gt), abs(far-gt))
    /sample_000000/pred_depth_best_choice   – (H, W)    uint8     0=near, 1=far

    /sample_000000/pred_velocity            – (H, W)    float32   raw_velocity ** 3

    /sample_000000/gt_support               – (H, W)    float32
    /sample_000000/gt_depth                 – (H, W)    float32
    /sample_000000/gt_velocity              – (H, W)    float32

The index in the group name is the per-rank write counter, not a global
sample index. Metadata such as img_path and shapes are written as HDF5
attributes on the sample group.

No PNG figures, no JSON metadata, no simulated point clouds.

Usage in config
---------------

    custom_hooks = [
        dict(
            type='PPPMarkMetricEvalExportTwoDepthHook',
            max_samples_test=500,
            max_samples_val=0,
            img_mean=data_rbg_mean,
            img_std=data_rbg_std,
            depth_eps=1e-3,
        )
    ]
"""

from __future__ import annotations

import os
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS


@HOOKS.register_module()
class PPPMarkMetricEvalExportTwoDepthHook(Hook):
    """
    Export compact HDF5 files for offline metric evaluation of the two-depth head.

    Prediction channel convention
    -----------------------------
    The hook assumes the two-depth PPP mark model predicts exactly four channels:

        pred[0] = log intensity / log lambda parameter
        pred[1] = raw central depth parameter raw_d0
        pred[2] = raw depth separation parameter raw_delta
        pred[3] = signed cube-root velocity / transformed via pred[3] ** 3

    The depth parametrization is:

        d0        = exp(raw_d0)
        delta     = softplus(raw_delta)
        d_near    = clamp(d0 - delta, min=depth_eps)
        d_far     = d0 + delta

    Ground truth channel convention
    -------------------------------
    The ground-truth segmentation tensor must contain three channels:

        gt[0] = support mask
        gt[1] = depth
        gt[2] = velocity

    Parameters
    ----------
    max_samples_test : int
        Stop after this many samples per rank during test.
    max_samples_val : int
        Stop after this many samples per rank during val. 0 disables val export.
    img_mean : list[float] | None
        Per-channel mean used during normalisation.
    img_std : list[float] | None
        Per-channel std used during normalisation.
    depth_eps : float
        Minimum value used for clamping d_near.
    require_four_pred_channels : bool
        If True, raise a clear error when prediction has fewer than four channels.
    require_three_gt_channels : bool
        If True, raise a clear error when GT has fewer than three channels.
    """

    def __init__(
        self,
        max_samples_test: int = 500,
        max_samples_val: int = 0,
        img_mean=None,
        img_std=None,
        depth_eps: float = 1e-3,
        require_four_pred_channels: bool = True,
        require_three_gt_channels: bool = True,
    ):
        self.max_samples = {
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.depth_eps = float(depth_eps)
        self.require_four_pred_channels = bool(require_four_pred_channels)
        self.require_three_gt_channels = bool(require_three_gt_channels)

        self.img_mean = (
            np.asarray(img_mean, dtype=np.float32)[:, None, None]
            if img_mean is not None
            else None
        )
        self.img_std = (
            np.asarray(img_std, dtype=np.float32)[:, None, None]
            if img_std is not None
            else None
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

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
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
        d = os.path.join(runner.log_dir, "metric_eval_data_twodepth", phase)
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
            self._h5[phase].attrs["format"] = "ppp_mark_metric_eval_twodepth_v1"
            self._h5[phase].attrs["prediction_channels"] = (
                "pred[0]=log_lambda, "
                "pred[1]=raw_d0, "
                "pred[2]=raw_delta, "
                "pred[3]=raw_velocity"
            )
            self._h5[phase].attrs["depth_parametrization"] = (
                "d0=exp(raw_d0), "
                "delta=softplus(raw_delta), "
                "d_near=clamp(d0-delta, min=depth_eps), "
                "d_far=d0+delta"
            )
            self._h5[phase].attrs["depth_eps"] = float(self.depth_eps)

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
        if self.require_four_pred_channels:
            self._assert_min_channels(pred, min_channels=4, name="prediction")

        # ---- GT channels ----
        gt = sample.gt_sem_seg.data.to(pred.device)

        if self.require_three_gt_channels:
            self._assert_min_channels(gt, min_channels=3, name="ground truth")

        gt_support_t = gt[0]
        gt_depth_t = gt[1]
        gt_velocity_t = gt[2]

        # ---- decode prediction ----
        # Channel convention:
        #   pred[0] = log intensity / log lambda parameter
        #   pred[1] = raw_d0
        #   pred[2] = raw_delta
        #   pred[3] = raw_velocity
        pred_lambda_theta_t = torch.exp(pred[0])

        raw_d0_t = pred[1]
        raw_delta_t = pred[2]
        raw_velocity_t = pred[3]

        pred_depth_d0_t = torch.exp(raw_d0_t)
        pred_depth_delta_t = F.softplus(raw_delta_t)

        pred_depth_near_t = torch.clamp(
            pred_depth_d0_t - pred_depth_delta_t,
            min=self.depth_eps,
        )
        pred_depth_far_t = pred_depth_d0_t + pred_depth_delta_t

        pred_velocity_t = raw_velocity_t ** 3

        # Best two-depth choice relative to GT depth.
        # This is mainly useful for offline diagnostics and mirrors the loss logic:
        #
        #   min(|d_near - gt_depth|, |d_far - gt_depth|)
        #
        # We store the full map, but for actual metric aggregation you should
        # usually apply gt_support as mask.
        depth_err_near_t = torch.abs(pred_depth_near_t - gt_depth_t)
        depth_err_far_t = torch.abs(pred_depth_far_t - gt_depth_t)

        pred_depth_best_choice_t = (depth_err_far_t < depth_err_near_t).to(torch.uint8)
        pred_depth_best_t = torch.where(
            pred_depth_best_choice_t.bool(),
            pred_depth_far_t,
            pred_depth_near_t,
        )
        pred_depth_best_abs_error_t = torch.minimum(depth_err_near_t, depth_err_far_t)

        # ---- normalise lambda to mass per pixel ----
        meta = sample.metainfo
        ori_shape = meta.get("ori_shape")

        if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
            H, W = int(ori_shape[0]), int(ori_shape[1])
        else:
            H, W = pred_lambda_theta_t.shape[-2:]

        pred_lambda_mass = (
            pred_lambda_theta_t.detach().cpu().numpy() / float(H * W)
        ).astype(np.float32)

        # ---- convert tensors to numpy ----
        pred_depth_d0 = self._to_float32_np(pred_depth_d0_t)
        pred_depth_near = self._to_float32_np(pred_depth_near_t)
        pred_depth_far = self._to_float32_np(pred_depth_far_t)
        pred_depth_delta = self._to_float32_np(pred_depth_delta_t)
        pred_depth_best = self._to_float32_np(pred_depth_best_t)
        pred_depth_best_abs_error = self._to_float32_np(pred_depth_best_abs_error_t)
        pred_depth_best_choice = (
            pred_depth_best_choice_t.detach().cpu().numpy().astype(np.uint8)
        )
        pred_velocity = self._to_float32_np(pred_velocity_t)

        gt_support = self._to_float32_np(gt_support_t)
        gt_depth = self._to_float32_np(gt_depth_t)
        gt_velocity = self._to_float32_np(gt_velocity_t)

        # ---- de-normalise image ----
        img = self._denorm_img(img_tensor.detach().cpu().numpy())

        # ---- write group into HDF5 ----
        grp = h5f.create_group(f"sample_{local_index:06d}")

        self._write_attrs(grp, meta=meta, local_index=local_index)

        self._create_dataset(grp, "img", img)

        self._create_dataset(grp, "pred_lambda", pred_lambda_mass)

        self._create_dataset(grp, "pred_depth_d0", pred_depth_d0)
        self._create_dataset(grp, "pred_depth_near", pred_depth_near)
        self._create_dataset(grp, "pred_depth_far", pred_depth_far)
        self._create_dataset(grp, "pred_depth_delta", pred_depth_delta)

        self._create_dataset(grp, "pred_depth_best", pred_depth_best)
        self._create_dataset(
            grp,
            "pred_depth_best_abs_error",
            pred_depth_best_abs_error,
        )
        self._create_dataset(
            grp,
            "pred_depth_best_choice",
            pred_depth_best_choice,
        )

        self._create_dataset(grp, "pred_velocity", pred_velocity)

        self._create_dataset(grp, "gt_support", gt_support)
        self._create_dataset(grp, "gt_depth", gt_depth)
        self._create_dataset(grp, "gt_velocity", gt_velocity)

        # Useful per-sample scalar attributes for quick inspection.
        support_mask = gt_support > 0.0
        n_points = int(support_mask.sum())

        grp.attrs["gt_count"] = n_points
        grp.attrs["pred_expected_count"] = float(pred_lambda_mass.sum())

        if n_points > 0:
            grp.attrs["mean_depth_best_l1_on_support"] = float(
                pred_depth_best_abs_error[support_mask].mean()
            )
            grp.attrs["mean_depth_near_l1_on_support"] = float(
                np.abs(pred_depth_near[support_mask] - gt_depth[support_mask]).mean()
            )
            grp.attrs["mean_depth_far_l1_on_support"] = float(
                np.abs(pred_depth_far[support_mask] - gt_depth[support_mask]).mean()
            )
            grp.attrs["mean_velocity_l1_on_support"] = float(
                np.abs(pred_velocity[support_mask] - gt_velocity[support_mask]).mean()
            )

            chosen_far_ratio = float(pred_depth_best_choice[support_mask].mean())
            grp.attrs["chosen_far_ratio_on_support"] = chosen_far_ratio
        else:
            grp.attrs["mean_depth_best_l1_on_support"] = np.nan
            grp.attrs["mean_depth_near_l1_on_support"] = np.nan
            grp.attrs["mean_depth_far_l1_on_support"] = np.nan
            grp.attrs["mean_velocity_l1_on_support"] = np.nan
            grp.attrs["chosen_far_ratio_on_support"] = np.nan

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_min_channels(
        x: torch.Tensor,
        min_channels: int,
        name: str,
    ) -> None:
        if x.ndim < 3 or x.shape[0] < min_channels:
            raise ValueError(
                f"Expected {name} tensor with at least {min_channels} channels, "
                f"but got shape {tuple(x.shape)}."
            )

    @staticmethod
    def _to_float32_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _create_dataset(grp: "h5py.Group", name: str, data: np.ndarray) -> None:
        grp.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=4,
        )

    def _write_attrs(
        self,
        grp: "h5py.Group",
        meta: dict[str, Any],
        local_index: int,
    ) -> None:
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
            "video_name",
            "video_frame_idx",
            "global_idx",
            "frame_idx",
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
        """
        Convert image tensor from (C, H, W) float array to (H, W, C) uint8.

        This follows the same convention as the existing demo/export hooks:
        if img_mean and img_std are given, first invert normalisation and then
        map from [-1, 1] to [0, 255].
        """
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