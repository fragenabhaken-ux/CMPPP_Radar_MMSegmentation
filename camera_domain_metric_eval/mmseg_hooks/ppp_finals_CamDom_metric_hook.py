"""HDF5 export hook for four-channel camera-domain marked PPP models."""

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
class PPPMarkMetricEvalExportHook(Hook):
    """Export raw camera-domain predictions for offline metric evaluation.

    Prediction channels:
        0: log intensity density parameter
        1: raw depth
        2: raw radial velocity
        3: raw RCS

    Ground-truth channels:
        0: binary support
        1: depth
        2: radial velocity
        3: RCS

    The exported ``pred_lambda`` is expected-count mass per full-image cell:

        exp(pred[0]) / (H_0 W_0).

    The hook deliberately exports dense maps only. Final metric computation and
    plotting are performed by the offline codebase.
    """

    def __init__(
        self,
        max_samples_test: int = 500,
        max_samples_val: int = 0,
        img_mean=None,
        img_std=None,
        depth_link: str = "exp",
        velocity_link: str = "cube",
        rcs_link: str = "cube",
        require_four_channels: bool = True,
        require_full_image_shape: bool = True,
    ) -> None:
        self.max_samples = {
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }
        self.require_four_channels = bool(require_four_channels)
        self.require_full_image_shape = bool(require_full_image_shape)

        self.depth_link = str(depth_link).lower()
        self.velocity_link = str(velocity_link).lower()
        self.rcs_link = str(rcs_link).lower()

        if self.depth_link not in {"identity", "exp", "softplus"}:
            raise ValueError(f"Unsupported depth_link={depth_link!r}")
        if self.velocity_link not in {"identity", "cube"}:
            raise ValueError(f"Unsupported velocity_link={velocity_link!r}")
        if self.rcs_link not in {"identity", "cube", "linear_cube"}:
            raise ValueError(f"Unsupported rcs_link={rcs_link!r}")

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

        self._written = {"val": 0, "test": 0}
        self._h5: dict[str, h5py.File | None] = {"val": None, "test": None}

    def before_val_epoch(self, runner) -> None:
        self._reset("val")

    def before_test_epoch(self, runner) -> None:
        self._reset("test")

    def after_val_epoch(self, runner, metrics=None, **kwargs) -> None:
        self._close_h5("val")

    def after_test_epoch(self, runner, metrics=None, **kwargs) -> None:
        self._close_h5("test")

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        self._maybe_export(runner, data_batch, phase="val")

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        self._maybe_export(runner, data_batch, phase="test")

    def _reset(self, phase: str) -> None:
        self._close_h5(phase)
        self._written[phase] = 0

    def _maybe_export(self, runner, data_batch, phase: str) -> None:
        if not isinstance(data_batch, dict):
            return
        if self.max_samples[phase] <= 0:
            return
        if self._written[phase] >= self.max_samples[phase]:
            return

        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        was_training = model.training
        model.eval()

        try:
            velocity_mode = str(
                getattr(model.decode_head, "velocity_mode", "signed")
            ).lower()
            if velocity_mode not in {"signed", "absolute"}:
                raise ValueError(f"Unsupported velocity_mode={velocity_mode!r}")

            processed = model.data_preprocessor(data_batch)
            inputs = processed["inputs"]
            data_samples = processed["data_samples"]

            with torch.no_grad():
                predictions = model.decode_head(model.extract_feat(inputs))

            if not torch.is_tensor(predictions) or predictions.ndim != 4:
                raise TypeError(
                    "Expected raw decode-head tensor with shape [N,C,H,W], got "
                    f"{type(predictions)!r} / {getattr(predictions, 'shape', None)!r}"
                )

            remaining = self.max_samples[phase] - self._written[phase]
            n_save = min(remaining, predictions.shape[0], len(data_samples))
            h5f = self._get_h5(runner, phase)
            h5f.attrs["velocity_mode"] = velocity_mode
            h5f.attrs["depth_link"] = self.depth_link
            h5f.attrs["velocity_link"] = self.velocity_link
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
                    prediction=predictions[i],
                    sample=data_samples[i],
                    local_index=self._written[phase] + i,
                    velocity_mode=velocity_mode,
                )
            h5f.flush()
            self._written[phase] += n_save
        finally:
            if was_training:
                model.train()

    def _append_sample(
        self,
        *,
        h5f: h5py.File,
        img_tensor: torch.Tensor,
        prediction: torch.Tensor,
        sample,
        local_index: int,
        velocity_mode: str,
    ) -> None:
        minimum_channels = 4 if self.require_four_channels else 3
        self._assert_channels(prediction, minimum_channels, "prediction")

        gt = sample.gt_sem_seg.data.to(prediction.device)
        self._assert_channels(gt, minimum_channels, "ground truth")

        if tuple(prediction.shape[-2:]) != tuple(gt.shape[-2:]):
            raise ValueError(
                "Prediction and GT spatial shapes differ: "
                f"{tuple(prediction.shape[-2:])} vs {tuple(gt.shape[-2:])}"
            )

        meta = sample.metainfo
        ori_shape = meta.get("ori_shape")
        if isinstance(ori_shape, (tuple, list, np.ndarray)) and len(ori_shape) >= 2:
            full_height = int(ori_shape[0])
            full_width = int(ori_shape[1])
        else:
            full_height, full_width = map(int, prediction.shape[-2:])

        if full_height <= 0 or full_width <= 0:
            raise ValueError(f"Invalid full-image shape {(full_height, full_width)}")

        prediction_shape = tuple(map(int, prediction.shape[-2:]))
        full_shape = (full_height, full_width)
        if self.require_full_image_shape and prediction_shape != full_shape:
            raise ValueError(
                "Metric export requires full-image prediction grids. "
                f"Prediction shape={prediction_shape}, ori_shape={full_shape}. "
                "Set the evaluation data preprocessor to the full output size."
            )

        log_lambda = prediction[0]
        if not torch.isfinite(log_lambda).all():
            raise ValueError("Predicted log intensity contains NaN or Inf")
        pred_lambda_mass = torch.exp(log_lambda) / float(full_height * full_width)

        pred_depth = self._decode_depth(prediction[1])
        pred_velocity = self._decode_velocity(prediction[2], velocity_mode)
        pred_rcs = (
            self._decode_rcs(prediction[3])
            if prediction.shape[0] >= 4
            else None
        )

        gt_support = gt[0]
        gt_depth = gt[1]
        gt_velocity = torch.abs(gt[2]) if velocity_mode == "absolute" else gt[2]
        gt_rcs = gt[3] if gt.shape[0] >= 4 else None

        unique_support = torch.unique(gt_support.detach())
        valid_binary = torch.isclose(unique_support, torch.zeros_like(unique_support)) | torch.isclose(
            unique_support,
            torch.ones_like(unique_support),
        )
        if not bool(valid_binary.all()):
            raise ValueError(
                "Camera-domain GT support must be binary, found values "
                f"{unique_support.detach().cpu().tolist()}"
            )

        arrays = {
            "pred_lambda": pred_lambda_mass,
            "pred_depth": pred_depth,
            "pred_velocity": pred_velocity,
            "gt_support": gt_support,
            "gt_depth": gt_depth,
            "gt_velocity": gt_velocity,
        }
        if pred_rcs is not None:
            arrays["pred_rcs"] = pred_rcs
        if gt_rcs is not None:
            arrays["gt_rcs"] = gt_rcs

        for name, tensor in arrays.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf")

        group = h5f.create_group(f"sample_{local_index:06d}")
        self._write_attrs(group, meta, local_index)
        group.attrs["velocity_mode"] = velocity_mode
        group.attrs["depth_link"] = self.depth_link
        group.attrs["velocity_link"] = self.velocity_link
        group.attrs["rcs_link"] = self.rcs_link
        group.attrs["prediction_height"] = prediction_shape[0]
        group.attrs["prediction_width"] = prediction_shape[1]
        group.attrs["full_image_height"] = full_height
        group.attrs["full_image_width"] = full_width
        group.attrs["pred_expected_count"] = float(pred_lambda_mass.sum().item())
        group.attrs["gt_count"] = int((gt_support > 0).sum().item())

        self._create_dataset(group, "img", self._denorm_img(img_tensor.detach().cpu().numpy()))
        for name, tensor in arrays.items():
            self._create_dataset(group, name, self._to_numpy(tensor))

    def _decode_depth(self, raw: torch.Tensor) -> torch.Tensor:
        if self.depth_link == "identity":
            return raw
        if self.depth_link == "exp":
            return torch.exp(raw)
        if self.depth_link == "softplus":
            return F.softplus(raw)
        raise RuntimeError("Unreachable")

    def _decode_velocity(self, raw: torch.Tensor, mode: str) -> torch.Tensor:
        value = raw if self.velocity_link == "identity" else raw.pow(3)
        return value.abs() if mode == "absolute" else value

    def _decode_rcs(self, raw: torch.Tensor) -> torch.Tensor:
        if self.rcs_link == "identity":
            return raw
        if self.rcs_link == "cube":
            return raw.pow(3)
        if self.rcs_link == "linear_cube":
            return raw + raw.pow(3)
        raise RuntimeError("Unreachable")

    def _get_out_dir(self, runner, phase: str) -> str:
        path = os.path.join(runner.log_dir, "metric_eval_data", phase)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_h5(self, runner, phase: str) -> h5py.File:
        if self._h5[phase] is None:
            path = os.path.join(self._get_out_dir(runner, phase), f"rank{get_rank():02d}.h5")
            h5f = h5py.File(path, "w")
            h5f.attrs["phase"] = phase
            h5f.attrs["rank"] = int(get_rank())
            h5f.attrs["format"] = "ppp_camera_metric_eval_v3"
            h5f.attrs["has_rcs"] = True
            self._h5[phase] = h5f
        return self._h5[phase]

    def _close_h5(self, phase: str) -> None:
        h5f = self._h5[phase]
        if h5f is not None:
            h5f.close()
            self._h5[phase] = None

    @staticmethod
    def _assert_channels(tensor: torch.Tensor, minimum: int, name: str) -> None:
        if tensor.ndim < 3 or tensor.shape[0] < minimum:
            raise ValueError(
                f"Expected {name} with at least {minimum} channels, got {tuple(tensor.shape)}"
            )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _create_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
        group.create_dataset(name, data=data, compression="gzip", compression_opts=4)

    def _write_attrs(self, group: h5py.Group, meta: dict[str, Any], local_index: int) -> None:
        group.attrs["local_index"] = int(local_index)
        group.attrs["rank"] = int(get_rank())
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
            "dopplerRange",
            "doppler_range",
            "ego_longitudinal_velocity",
            "ego_yaw_rate",
        ]:
            if key in meta and meta[key] is not None:
                group.attrs[key] = self._attr_value(meta[key])

    @staticmethod
    def _attr_value(value: Any):
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
            return value
        if isinstance(value, (tuple, list)):
            return np.asarray(value)
        if isinstance(value, np.ndarray):
            return value.item() if value.ndim == 0 else value
        return str(value)

    def _denorm_img(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        if self.img_mean is not None and self.img_std is not None:
            image = image * self.img_std + self.img_mean
            image = (image + 1.0) * 127.5
        elif image.min() >= -1.01 and image.max() <= 1.01:
            image = (image + 1.0) * 127.5
        elif image.max() <= 1.01:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return np.transpose(image, (1, 2, 0))
