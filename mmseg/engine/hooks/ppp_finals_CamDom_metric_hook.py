"""
PPPMarkMetricEvalHook
===========================
Lightweight hook for generating an offline metric-evaluation dataset for the
four-channel image-domain marked PPP model.

Each GPU / distributed rank writes one HDF5 file containing all samples that
rank processed:

    {log_dir}/metric_eval_data/{phase}/rank00.h5
    {log_dir}/metric_eval_data/{phase}/rank01.h5
    ...

Current prediction-channel convention
-------------------------------------

    pred[0] = log intensity parameter
    pred[1] = raw depth, decoded with exp
    pred[2] = raw velocity
    pred[3] = raw RCS

Current ground-truth convention
-------------------------------

    gt[0] = support mask
    gt[1] = depth
    gt[2] = velocity
    gt[3] = RCS

Inside each sample group the hook stores:

    /sample_000000/img             (H, W, 3) uint8
    /sample_000000/pred_lambda     (H, W)    float32
    /sample_000000/pred_depth      (H, W)    float32
    /sample_000000/pred_velocity   (H, W)    float32
    /sample_000000/pred_rcs        (H, W)    float32
    /sample_000000/gt_support      (H, W)    float32
    /sample_000000/gt_depth        (H, W)    float32
    /sample_000000/gt_velocity     (H, W)    float32
    /sample_000000/gt_rcs          (H, W)    float32

``pred_lambda`` is the predicted intensity mass per image cell,
``exp(pred[0]) / (H * W)``. Velocity and RCS are exported after applying the
same inverse links used by the current visualization hook.

Usage in config
---------------

    custom_hooks = [
        dict(
            type='PPPMarkMetricEvalExportHook',
            max_samples_test=10000,
            max_samples_val=0,
            img_mean=data_rbg_mean,
            img_std=data_rbg_std,
            rcs_link='cube',
        )
    ]
"""

from __future__ import annotations

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
class PPPMarkMetricEvalHook(Hook):
    """Export compact HDF5 files for offline marked-PPP evaluation.

    Parameters
    ----------
    max_samples_test : int
        Maximum number of samples written by each rank during testing.
    max_samples_val : int
        Maximum number of samples written by each rank during validation.
        Set to zero to disable validation export.
    img_mean : list[float] | None
        Per-channel image mean used by the data preprocessor.
    img_std : list[float] | None
        Per-channel image standard deviation used by the data preprocessor.
    rcs_link : {"identity", "cube", "linear_cube"}
        Inverse link applied to the raw RCS prediction. This matches
        ``PPPMarkVisHookLight``.
    require_four_channels : bool
        Require both prediction and ground truth to contain support/intensity,
        depth, velocity, and RCS channels. Enabled by default to prevent a
        legacy three-channel model from silently producing incomplete files.
    """

    def __init__(
        self,
        max_samples_test: int = 500,
        max_samples_val: int = 0,
        img_mean=None,
        img_std=None,
        rcs_link: str = "cube",
        require_four_channels: bool = True,
    ) -> None:
        self.max_samples = {
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }
        self.require_four_channels = bool(require_four_channels)

        self.rcs_link = str(rcs_link).lower()
        if self.rcs_link not in {"identity", "cube", "linear_cube"}:
            raise ValueError(
                f"Unsupported rcs_link={rcs_link!r}. "
                "Use 'identity', 'cube', or 'linear_cube'."
            )

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

        # One open HDF5 handle per phase. None means not yet opened.
        self._h5: dict[str, h5py.File | None] = {"val": None, "test": None}

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

    def after_val_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        self._maybe_run(runner, data_batch, phase="val")

    def after_test_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
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

        model = (
            runner.model.module
            if is_model_wrapper(runner.model)
            else runner.model
        )
        was_training = model.training
        model.eval()

        try:
            velocity_mode = getattr(
                model.decode_head,
                "velocity_mode",
                "signed",
            )
            if velocity_mode not in {"signed", "absolute"}:
                raise ValueError(
                    "velocity_mode must be 'signed' or 'absolute', "
                    f"got {velocity_mode!r}."
                )

            proc = model.data_preprocessor(data_batch)
            inputs = proc["inputs"]
            data_samples = proc["data_samples"]

            with torch.no_grad():
                preds = model.decode_head(model.extract_feat(inputs))

            if not torch.is_tensor(preds) or preds.ndim != 4:
                raise TypeError(
                    "Expected decode-head output with shape (N, C, H, W), "
                    f"got {type(preds)!r} with shape "
                    f"{getattr(preds, 'shape', None)!r}."
                )

            remaining = self.max_samples[phase] - self._written[phase]
            n_save = min(remaining, preds.size(0), len(data_samples))

            h5f = self._get_h5(runner, phase)
            h5f.attrs["velocity_mode"] = velocity_mode
            h5f.attrs["rcs_link"] = self.rcs_link
            h5f.attrs["prediction_channel_convention"] = (
                "0=log_lambda,1=raw_depth,2=raw_velocity,3=raw_rcs"
            )
            h5f.attrs["ground_truth_channel_convention"] = (
                "0=support,1=depth,2=velocity,3=rcs"
            )

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
        finally:
            if was_training:
                model.train()

    # ------------------------------------------------------------------
    # HDF5 file management
    # ------------------------------------------------------------------

    def _get_out_dir(self, runner, phase: str) -> str:
        out_dir = os.path.join(
            runner.log_dir,
            "metric_eval_data",
            phase,
        )
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _get_h5(self, runner, phase: str) -> h5py.File:
        """Return this rank's open HDF5 file for the requested phase."""
        if self._h5[phase] is None:
            out_dir = self._get_out_dir(runner, phase)
            rank = get_rank()
            filepath = os.path.join(out_dir, f"rank{rank:02d}.h5")

            h5f = h5py.File(filepath, "w")
            h5f.attrs["phase"] = phase
            h5f.attrs["rank"] = int(rank)
            h5f.attrs["format"] = "ppp_mark_metric_eval_v2_rcs"
            h5f.attrs["has_rcs"] = True
            self._h5[phase] = h5f

        return self._h5[phase]

    def _close_h5(self, phase: str) -> None:
        h5f = self._h5[phase]
        if h5f is not None:
            try:
                h5f.close()
            except Exception:
                pass
            self._h5[phase] = None

    # ------------------------------------------------------------------
    # Per-sample writing
    # ------------------------------------------------------------------

    def _append_sample(
        self,
        h5f: h5py.File,
        img_tensor: torch.Tensor,
        pred: torch.Tensor,
        sample,
        local_index: int,
        velocity_mode: str,
    ) -> None:
        minimum_channels = 4 if self.require_four_channels else 3
        self._assert_channels(
            pred,
            minimum=minimum_channels,
            name="prediction",
        )

        # Current standard head:
        #   pred[0] = log intensity
        #   pred[1] = raw depth
        #   pred[2] = raw velocity
        #   pred[3] = raw RCS
        pred_lambda_theta = torch.exp(pred[0])
        pred_depth_tensor = torch.exp(pred[1])
        raw_velocity = pred[2]

        if velocity_mode == "signed":
            pred_velocity_tensor = raw_velocity ** 3
        elif velocity_mode == "absolute":
            pred_velocity_tensor = torch.abs(raw_velocity)
        else:
            raise RuntimeError(
                f"Unsupported velocity_mode={velocity_mode!r}."
            )

        pred_rcs_tensor = None
        if pred.shape[0] >= 4:
            pred_rcs_tensor = self._decode_rcs(pred[3])
        elif self.require_four_channels:
            raise ValueError(
                "RCS export is enabled, but prediction channel 3 is missing."
            )

        pred_depth = self._to_float32_numpy(pred_depth_tensor)
        pred_velocity = self._to_float32_numpy(pred_velocity_tensor)
        pred_rcs = (
            self._to_float32_numpy(pred_rcs_tensor)
            if pred_rcs_tensor is not None
            else None
        )

        # Ground truth:
        #   gt[0] = support
        #   gt[1] = depth
        #   gt[2] = velocity
        #   gt[3] = RCS
        gt = sample.gt_sem_seg.data.to(pred.device)
        self._assert_channels(
            gt,
            minimum=minimum_channels,
            name="ground truth",
        )

        gt_support = self._to_float32_numpy(gt[0])
        gt_depth = self._to_float32_numpy(gt[1])

        if velocity_mode == "signed":
            gt_velocity_tensor = gt[2]
        elif velocity_mode == "absolute":
            gt_velocity_tensor = torch.abs(gt[2])
        else:
            raise RuntimeError(
                f"Unsupported velocity_mode={velocity_mode!r}."
            )
        gt_velocity = self._to_float32_numpy(gt_velocity_tensor)

        gt_rcs = None
        if gt.shape[0] >= 4:
            # RCS targets are already stored in physical/data units. No inverse
            # link is applied to the ground truth.
            gt_rcs = self._to_float32_numpy(gt[3])
        elif self.require_four_channels:
            raise ValueError(
                "RCS export is enabled, but ground-truth channel 3 is missing."
            )

        # Convert the intensity field into mass per full-image cell.
        meta = sample.metainfo
        ori_shape = meta.get("ori_shape")
        if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
            height = int(ori_shape[0])
            width = int(ori_shape[1])
        else:
            height, width = pred_lambda_theta.shape[-2:]

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid original image shape {(height, width)}."
            )

        pred_lambda_mass = self._to_float32_numpy(
            pred_lambda_theta / float(height * width)
        )

        img = self._denorm_img(
            img_tensor.detach().cpu().numpy(),
        )

        grp = h5f.create_group(f"sample_{local_index:06d}")
        self._write_attrs(
            grp,
            meta=meta,
            local_index=local_index,
        )
        grp.attrs["velocity_mode"] = velocity_mode
        grp.attrs["rcs_link"] = self.rcs_link
        grp.attrs["pred_velocity_channel"] = 2
        grp.attrs["pred_rcs_channel"] = 3
        grp.attrs["gt_velocity_channel"] = 2
        grp.attrs["gt_rcs_channel"] = 3

        self._create_dataset(grp, "img", img)
        self._create_dataset(grp, "pred_lambda", pred_lambda_mass)
        self._create_dataset(grp, "pred_depth", pred_depth)
        self._create_dataset(grp, "pred_velocity", pred_velocity)
        if pred_rcs is not None:
            self._create_dataset(grp, "pred_rcs", pred_rcs)

        self._create_dataset(grp, "gt_support", gt_support)
        self._create_dataset(grp, "gt_depth", gt_depth)
        self._create_dataset(grp, "gt_velocity", gt_velocity)
        if gt_rcs is not None:
            self._create_dataset(grp, "gt_rcs", gt_rcs)

    # ------------------------------------------------------------------
    # Decoding and utilities
    # ------------------------------------------------------------------

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> torch.Tensor:
        """Apply the same RCS inverse link as the visualization hook."""
        if self.rcs_link == "identity":
            return raw_rcs
        if self.rcs_link == "cube":
            return raw_rcs ** 3
        if self.rcs_link == "linear_cube":
            return raw_rcs + raw_rcs ** 3
        raise RuntimeError(f"Unsupported rcs_link={self.rcs_link!r}.")

    @staticmethod
    def _to_float32_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _assert_channels(
        x: torch.Tensor,
        minimum: int,
        name: str,
    ) -> None:
        if x.ndim < 3 or x.shape[0] < minimum:
            raise ValueError(
                f"Expected {name} tensor with at least {minimum} channels, "
                f"but got shape {tuple(x.shape)}."
            )

    @staticmethod
    def _create_dataset(
        grp: h5py.Group,
        name: str,
        data: np.ndarray,
    ) -> None:
        grp.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=4,
        )

    def _write_attrs(
        self,
        grp: h5py.Group,
        meta: dict[str, Any],
        local_index: int,
    ) -> None:
        grp.attrs["local_index"] = int(local_index)
        grp.attrs["rank"] = int(get_rank())

        # Include stable SYS100 identifiers so exports from different model
        # runs can be matched and verified sample by sample.
        for key in [
            "sample_id",
            "frame_idx",
            "global_idx",
            "video_path",
            "video_name",
            "video_frame_idx",
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "doppler_period",
            "ego_longitudinal_velocity",
            "ego_yaw_rate",
        ]:
            if key in meta and meta[key] is not None:
                grp.attrs[key] = self._attr_value(meta[key])

    @staticmethod
    def _attr_value(value: Any):
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                np.integer,
                np.floating,
                np.bool_,
            ),
        ):
            return value

        if isinstance(value, (tuple, list)) and all(
            isinstance(
                element,
                (
                    int,
                    float,
                    bool,
                    np.integer,
                    np.floating,
                    np.bool_,
                ),
            )
            for element in value
        ):
            return np.asarray(value)

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            if value.dtype.kind in {"b", "i", "u", "f", "S", "U"}:
                return value

        return str(value)

    def _denorm_img(self, x: np.ndarray) -> np.ndarray:
        """Convert a (C, H, W) float image to (H, W, C) uint8."""
        x = x.astype(np.float32, copy=False)

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